# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
import numpy as np
from opt import default_options
from data_provider import DataProvision
from util import nms_detections, get_recall_at_k, mkdirs
import tensorflow as tf
import sys

# set default encoding
#reload(sys)
#sys.setdefaultencoding('utf-8')

#np.set_printoptions(threshold='nan')


def test(options):

    # print variable names
    for v in tf.trainable_variables():
        print(v.name)
        print(v.get_shape())

    print('Loading data ...')
    data_provision = DataProvision(options)

    batch_size = 400
    split = 'test'

    test_batch_generator = data_provision.iterate_batch(split, batch_size)
    unique_anno_ids = data_provision.get_ids(split)
    anchors = data_provision.get_anchors()
    grounding = data_provision.get_grounding(split)

    print('Start to predict ...')
    t0 = time.time()

    count = 0

    # output data, for evaluation
    out_data = {}
    out_data['results'] = {}
    results = {}

    for batch_data in test_batch_generator:
        video_feats = batch_data['video_feat']
        video_feat_mask = batch_data['video_feat_mask']
        feat_lens = np.sum(video_feat_mask, axis=-1)
        this_batch_size = video_feat_mask.shape[0]

        for sample_id in range(this_batch_size):
            unique_anno_id = unique_anno_ids[count]
            feat_len = feat_lens[sample_id]
            # small gap (in seconds) due to feature resolution
            gap = 0.5

            print('%d-th video-query: %s, feat_len: %d'%(count, unique_anno_id, feat_len))
            
            result = []
            scores = np.random.random(size=(feat_len, options['num_anchors']))
            for i in range(feat_len):
                for j in range(options['num_anchors']):
                    # calculate time stamp from feature id
                    end_feat = i + 0.5
                    start_feat = end_feat - anchors[j]
                    end_time = options['feature_to_second'] * end_feat
                    start_time = options['feature_to_second'] * start_feat

                    if start_time < 0. - gap:
                        continue

                    start_time = max(0., start_time)

                    result.append({'timestamp': [start_time, end_time], 'score': scores[i, j]})

            print('Number of proposals (before post-processing): %d' % len(result))

            result = sorted(result, key=lambda x: x['score'], reverse=True)

            # non-maximum suppresion
            result = nms_detections(result, overlap=options['nms_threshold'])
            print('Number of proposals (after nms): %d'%len(result))

            result = sorted(result, key=lambda x: x['score'], reverse=True)
            result = result[:10]
            
            print('#{}, {}'.format(count, unique_anno_id))
            print('sentence query:')
            sentence_query = grounding[unique_anno_id]['raw_sentence']
            print(sentence_query)
            print('result (top 10):')
            print(result)
            print('groundtruth:')
            print(grounding[unique_anno_id]['timestamp'])

            results[unique_anno_id] = result

            count = count + 1

    out_data['results'] = results

    out_json_file = 'results/random_anchor_predict_proposals_%s_nms_%.2f.json'%(split, options['nms_threshold'])

    mkdirs(os.path.dirname(out_json_file))

    print('Writing result json file ...')
    with open(out_json_file, 'w') as fid:
        json.dump(out_data, fid)

    print('Evaluating ...')
    recall_at_k = get_recall_at_k(results, grounding, options['tiou_measure'], options['max_proposal_num'])

    print('Recall at {}: {}'.format(options['max_proposal_num'], recall_at_k))
    
    print('Total running time: %f seconds.'%(time.time()-t0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    options = default_options()
    for key, value in options.items():
        if type(value) == bool:
            parser.add_argument('--%s' % key, action='store_true')
        else:
            parser.add_argument('--%s' % key, dest=key, type=type(value), default=None)
    args = parser.parse_args()
    args = vars(args)
    for key, value in args.items():
        if value:
            options[key] = value

    test(options)
