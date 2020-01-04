# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import itertools

def get_intersection(region1, region2):
    """ Get intersection of two segments
    """
    start1, end1 = region1
    start2, end2 = region2
    start = max(start1, start2)
    end = min(end1, end2)

    return start, end


def get_iou(pred, gt):
    """ Get tIoU of two segments
    """
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end - start + end_pred - start_pred)
    iou = float(intersection) / (union + 1e-8)

    return iou


def get_miou(predictions, groundtruths):
    """ Get mean IoU
    """
    ious = []
    for idx in groundtruths.keys():
        pred = predictions[idx][0]
        ious.append(get_iou(pred['timestamp'], groundtruths[idx]['timestamp']))

    miou = sum(ious) / len(ious)

    return miou

def nms_detections(proposals, overlap=0.7):
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    proposals: list of item, each item is a dict containing 'timestamp' and 'score' field
    overlap: iou threshold
    Returns
    -------
    new proposals with only the proposals selected after non-maximum suppression.
    """
    if len(proposals) == 0:
        return proposals

    props = np.array([item['timestamp'] for item in proposals])
    scores = np.array([item['score'] for item in proposals])
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]

    out_proposals = []
    for idx in range(nms_props.shape[0]):
        prop = nms_props[idx].tolist()
        score = float(nms_scores[idx])
        out_proposals.append({'timestamp': prop, 'score': score})

    return out_proposals


def get_recall_at_k(predictions, groundtruths, iou_threshold=0.5, max_proposal_num=5):
    """ Get R@k for all predictions
    R@k: Given k proposals, if there is at least one proposal has higher tIoU than iou_threshold, R@k=1; otherwise R@k=0
    The predictions should have been sorted by confidence
    """
    hit = np.zeros(shape=(len(groundtruths.keys()),), dtype=np.float32)

    for idd, idx in enumerate(groundtruths.keys()):
        if idx in predictions.keys():
            preds = predictions[idx][:max_proposal_num]
            for pred in preds:
                if get_iou(pred['timestamp'], groundtruths[idx]['timestamp']) >= iou_threshold:
                    hit[idd] = 1.

    avg_recall = np.sum(hit) / len(hit)
    return avg_recall


def evaluation_metric_util(options, data_provision, sess, inputs, outputs, interactor_inputs=None,
                      interactor_outputs=None, proposal_inputs=None, proposal_outputs=None, split='val'):
    """
    Metric evaluation (recall at k proposals)
    :param options: hyper parameters
    :param data_provision: data interface
    :param sess: tensorflow session
    :param inputs: input placeholders for graph1
    :param outputs: output placeholders for graph1
    :param interactor_inputs: input placeholders for graph2
    :param interactor_outputs: output placeholders for graph2
    :param proposal_inputs: input placeholders for graph3
    :param proposal_outputs: output placeholders for graph3
    :param split: data split for evaluation
    :return: evaluated metrics
    """
    eval_batch_size = options['eval_batch_size']
    unique_anno_ids = data_provision.get_ids(split)
    anchors = data_provision.get_anchors()
    grounding = data_provision.get_grounding(split)

    print('Predicting proposal scores ...')

    count = 0

    # output data, for evaluation
    out_data = {'results': {}}
    results = {}

    for batch_data in data_provision.iterate_batch(split, eval_batch_size):
        video_feats = batch_data['video_feat']
        video_feat_mask = batch_data['video_feat_mask']
        max_feat_len = video_feat_mask.shape[-1]
        this_batch_size = video_feat_mask.shape[0]
        zero_state = np.zeros(shape=(this_batch_size, options['rnn_size']))
        video_c_state = video_h_state = zero_state
        interactor_c_state = interactor_h_state = zero_state
        interactor_states = []  # interactor states before self attention
        print('max_feat_len: {}'.format(max_feat_len))

        for video_feat_id in range(max_feat_len):
            print('Loop: {}'.format(video_feat_id))
            video_feat = video_feats[:, video_feat_id]
            batch_data['video_feat'] = video_feat
            batch_data['video_c_state'] = video_c_state
            batch_data['video_h_state'] = video_h_state
            batch_data['interactor_c_state'] = interactor_c_state
            batch_data['interactor_h_state'] = interactor_h_state

            feed_dict = {}
            for key, value in batch_data.items():
                if key not in inputs:
                    continue
                feed_dict[inputs[key]] = value

            video_c_state, video_h_state, interactor_c_state, interactor_h_state = \
                sess.run([outputs['video_c_state'], outputs['video_h_state'],
                          outputs['interactor_c_state'], outputs['interactor_h_state']], feed_dict=feed_dict)
            interactor_states.append(interactor_h_state)

        interactor_states = np.stack(interactor_states, axis=1)

        feed_dict = {interactor_inputs['interactor_states']: interactor_states,
                     interactor_inputs['mask']: video_feat_mask}
        interactor_states_selfatt = sess.run(interactor_outputs['interactor_states_selfatt'], feed_dict=feed_dict)
        feed_dict = {proposal_inputs['interactor_states']: interactor_states,
                     proposal_inputs['interactor_states_selfatt']: interactor_states_selfatt}

        if options['predict_boundary']:
            proposal_scores, boundary_scores = sess.run([proposal_outputs['proposal_scores'],
                                                         proposal_outputs['boundary_scores']], feed_dict=feed_dict)
        else:
            proposal_scores = sess.run(proposal_outputs['proposal_scores'], feed_dict=feed_dict)

        feat_lens = np.sum(video_feat_mask, axis=-1)
        for sample_id in range(this_batch_size):
            unique_anno_id = unique_anno_ids[count]
            feat_len = feat_lens[sample_id]
            # small gap (in seconds) due to feature resolution
            gap = 0.5
            result = []

            for i in range(feat_len):
                for j in range(len(anchors)):
                    # calculate time stamp from feature id
                    end_feat = i + 0.5
                    start_feat = end_feat - anchors[j]
                    end_time = options['feature_to_second'] * end_feat
                    start_time = options['feature_to_second'] * start_feat

                    if start_time < 0. - options['feature_to_second']*gap:
                        continue

                    start_time = max(0., start_time)

                    start_feat_id = int(start_feat)
                    end_feat_id = int(end_feat)

                    proposal_score = float(proposal_scores[sample_id, i, j])

                    if options['predict_boundary']:
                        left_boundary_score = float(boundary_scores[sample_id, start_feat_id, 0])
                        right_boundary_score = float(boundary_scores[sample_id, end_feat_id, 0])
                        boundary_score = 0.5 * (left_boundary_score + right_boundary_score)
                        score = 0.5 * (proposal_score + boundary_score)
                    else:
                        score = proposal_score

                    result.append({'timestamp': [start_time, end_time],
                                   'score': score})

            print('Number of proposals (before post-processing): %d' % len(result))

            result = sorted(result, key=lambda x: x['score'], reverse=True)

            # non-maximum suppresion
            result = nms_detections(result, overlap=options['nms_threshold'])
            print('Number of proposals (after nms): %d' % len(result))

            result = sorted(result, key=lambda x: x['score'], reverse=True)

            result = result[:10]

            print('#{}, {}'.format(count, unique_anno_id))
            print('sentence query:')
            sentence_query = grounding[unique_anno_id]['raw_sentence']
            print(sentence_query)
            print('result (top 10):')
            print(result[:10])
            print('ground-truth:')
            print(grounding[unique_anno_id]['timestamp'])

            results[unique_anno_id] = result

            if (count + 1) % 10 == 0:
                print('Processed %d items' % (count + 1))

            count = count + 1

    out_data['results'] = results

    print('Evaluating ...')
    recall_at_k = get_recall_at_k(results, grounding, options['tiou_measure'], options['max_proposal_num'])

    print('R@{}, tIoU={}: {}'.format(options['max_proposal_num'], options['tiou_measure'], recall_at_k))

    return out_data, recall_at_k


def eval_result(result_file, gt_file):
    """
    Calculate mIoU, recalls for a given result file
    :param result_file: input .json result file
    :param gt_file: ground-truth file
    :return: None
    """
    results = json.load(open(result_file, 'r'))['results']
    groundtruth_data = json.load(open(gt_file, 'r'))
    video_ids = list(groundtruth_data.keys())

    out_grounding_data = {}

    for video_id in video_ids:
        gd = groundtruth_data[video_id]
        for anno_id in range(len(gd['timestamps'])):
            unique_anno_id = video_id + '-' + str(anno_id)
            out_grounding_data[unique_anno_id] = {
                'video_id': video_id,
                'anno_id': anno_id,
                'timestamp': gd['timestamps'][anno_id],
                'sentence': gd['encoded_sentences'][anno_id],
                'raw_sentence': gd['sentences'][anno_id]}

    groundtruth_data = out_grounding_data

    miou = get_miou(results, groundtruth_data)
    print('mIoU: {}'.format(miou))

    for iou, max_proposal_num in list(itertools.product([0.7, 0.5, 0.3], [1, 5])):
        recall = get_recall_at_k(results, groundtruth_data, iou_threshold=iou, max_proposal_num=max_proposal_num)
        print('R@{}, IoU={}: {}'.format(max_proposal_num, iou, recall))

    return


def format_loss_output(val_loss_list):
    print_info = ''
    if len(val_loss_list) >= 1:
        print_info = 'loss: {}'.format(val_loss_list[0])
    if len(val_loss_list) >= 2:
        print_info += ' , proposal_loss: {}'.format(val_loss_list[1])
    if len(val_loss_list) >= 3:
        print_info += ' , boundary_loss: {}'.format(val_loss_list[2])

    print_info += '\n'
    print(print_info)


def mkdirs(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

