# -*- coding: utf-8 -*-

import random
import numpy as np
import os
import h5py
import json
from util import get_iou, get_intersection
import time

#np.set_printoptions(threshold=np.inf)

FEATURE_FILE_MAPPING = {
    'tacos': 'tacos_c3d_fc6_nonoverlap.hdf5',
    'charades': 'charades_c3d_fc6_nonoverlap.hdf5',
    'activitynet_captions': 'activitynet_c3d_fc6_stride_2s.hdf5'
}

class DataProvision:
    def __init__(self, options):

        self._options = options
        self._splits = ['train', 'val', 'test']

        self._ids = {}  # video ids + annotation id
        self._sizes = {}  # size of train/val/test split data
        self._grounding = {}  # grounding data

        self._data_path = './datasets/{}/data/save/'.format(self._options['dataset'])

        self._anchors = json.load(open(os.path.join(self._data_path, 'anchors.json')))  # anchor data
        self._num_anchors = len(self._anchors)  # number of anchors

        assert(self._num_anchors == self._options['num_anchors'])

        print('Data Size:')
        for split in self._splits:
            grounding_data = json.load(open(os.path.join(self._data_path, '{}.json'.format(split)), 'r'))

            video_ids = list(grounding_data.keys())

            out_grounding_data = {}

            for video_id in video_ids:
                gd = grounding_data[video_id]
                for anno_id in range(len(gd['timestamps'])):
                    unique_anno_id = video_id + '-' + str(anno_id)
                    out_grounding_data[unique_anno_id] = {'video_id': video_id, 'anno_id': anno_id,
                                                          'timestamp': gd['timestamps'][anno_id],
                                                          'sentence': gd['encoded_sentences'][anno_id],
                                                          'raw_sentence': gd['sentences'][anno_id]}

            self._ids[split] = list(out_grounding_data.keys())
            self._sizes[split] = len(self._ids[split])
            self._grounding[split] = out_grounding_data

            print('%s-split: %d' % (split, self._sizes[split]))

        # feature dictionary
        print('Loading features ...')
        feature_data_path = './datasets/{}/features/{}'.format(self._options['dataset'],
                                                               FEATURE_FILE_MAPPING[self._options['dataset']])
        features = h5py.File(feature_data_path, 'r')
        self._feature_ids = features.keys()
        self._features = {video_id: features[video_id][self._options['feature_name']].value for video_id in
                          self._feature_ids}

        # load weight data
        print('Loading weight data ...')
        self._proposal_weight = json.load(open(os.path.join(self._data_path, 'weights_anchor.json')))

        if self._options['predict_boundary']:
            self._boundary_weight = json.load(open(os.path.join(self._data_path, 'weights_boundary.json')))
            self._boundary_weight[0] /= self._boundary_weight[1]
            self._boundary_weight[1] /= 1.

        # when using tensorflow built-in function: tf.nn.weighted_cross_entropy_with_logits()
        for i in range(len(self._proposal_weight)):
            self._proposal_weight[i][0] /= self._proposal_weight[i][1]
            self._proposal_weight[i][1] = 1.

        print('Loading Glove pretrained word embedding ...')
        glove_vocab_file = os.path.join(self._data_path, '{}_glove_embeds.npy'.format(self._options['dataset']))
        self._glove = np.load(open(glove_vocab_file, 'rb'))

        print('Done loading.')

    def get_size(self, split):
        return self._sizes[split]

    def get_ids(self, split):
        return self._ids[split]

    def get_grounding(self, split):
        return self._grounding[split]

    def get_anchors(self):
        return self._anchors

    def process_batch_data(self, batch_data, max_len=0):

        data_length = []
        for data in batch_data:
            data_length.append(data.shape[0])
        max_length = max(data_length)

        if max_len:
            max_length = max_len

        dim = batch_data[0].shape[1]

        out_batch_data = np.zeros(shape=(len(batch_data), max_length, dim), dtype='float32')
        out_batch_data_mask = np.zeros(shape=(len(batch_data), max_length), dtype='int32')

        for i, data in enumerate(batch_data):
            if max_len:
                effective_len = min(data.shape[0], max_len)
            else:
                effective_len = data.shape[0]
            out_batch_data[i, :effective_len, :] = data[:effective_len]
            out_batch_data_mask[i, :effective_len] = 1

        out_batch_data = np.asarray(out_batch_data, dtype='float32')
        out_batch_data_mask = np.asarray(out_batch_data_mask, dtype='int32')

        return out_batch_data, out_batch_data_mask

    def iterate_batch(self, split, batch_size):

        ids = list(self._ids[split])

        if split == 'train':
            print('Randomly shuffle training data ...')
            random.shuffle(ids)

        current = 0

        while True:
            batch_feature = []
            batch_sentence = []
            batch_sentence_bw = []
            batch_proposal = []
            batch_boundary = []

            # anchor mask: to mask out neither positive nor negative samples
            batch_anchor_mask = []

            max_sample_len = 0
            for sample_id in range(batch_size):
                unique_id = ids[sample_id + current]

                vid = self._grounding[split][unique_id]['video_id']
                grounding = self._grounding[split][unique_id]
                anno_id = grounding['anno_id']
                timestamp = grounding['timestamp']
                sentence = grounding['sentence']
                raw_sentence = grounding['raw_sentence']
                n_anchors = self._num_anchors

                if self._options['dataset'] == 'activitynet_captions':
                    vid = 'v_' + vid
                feature = self._features[vid]
                feature_len = feature.shape[0]

                # sampling
                if split == 'train':
                    sample_len = self._options['sample_len']
                else:
                    sample_len = feature_len

                max_sample_len = max(sample_len, max_sample_len)

                start_feat_id = random.randint(0, max((feature_len - sample_len), 0))
                end_feat_id = min(start_feat_id + sample_len, feature_len)
                feature = feature[start_feat_id:end_feat_id]

                # get word embedding for all words in the sentence
                sentence_embed = np.stack(
                    [self._glove[word_id] for word_id in sentence[:self._options['max_sentence_len']]])

                if self._options['bidirectional_lstm_sentence']:
                    sentence_embed_bw = sentence_embed[::-1]
                    batch_sentence_bw.append(sentence_embed_bw)

                batch_feature.append(feature)
                batch_sentence.append(sentence_embed)

                # generate proposal ground-truth data
                gt_proposal = np.zeros(shape=(sample_len, n_anchors), dtype=np.int32)
                gt_boundary = np.zeros(shape=(sample_len, 1), dtype=np.int32)

                anchor_mask = np.ones(shape=(sample_len, n_anchors), dtype=np.float32)

                gt_start_time, gt_end_time = timestamp
                gt_start_feature, gt_end_feature = \
                    int(gt_start_time // self._options['feature_to_second']), \
                    int(gt_end_time // self._options['feature_to_second'])

                start_point = max((gt_start_feature + gt_end_feature) // 2, 0)
                end_point = gt_end_feature + (gt_end_feature - gt_start_feature + 1)

                # only need to check whether proposals that have end point falling at the region of
                # (feat_check_start, feat_check_end) are "correct" proposals
                feat_check_start, feat_check_end = get_intersection((start_point, end_point),
                                                                      (start_feat_id, end_feat_id))
                for feat_id in range(feat_check_start, feat_check_end):
                    for anchor_id, anchor in enumerate(self._anchors):
                        end_feat = feat_id + 0.5
                        start_feat = end_feat - anchor
                        end_time = self._options['feature_to_second'] * end_feat
                        start_time = self._options['feature_to_second'] * start_feat
                        tiou = get_iou((start_time, end_time), (gt_start_time, gt_end_time))

                        if tiou > self._options['proposal_tiou_threshold']:
                            gt_proposal[feat_id - start_feat_id, anchor_id] = 1
                        elif tiou < self._options['negative_tiou_threshold']:
                            gt_proposal[feat_id - start_feat_id, anchor_id] = 0
                        else:
                            anchor_mask[feat_id - start_feat_id, anchor_id] = 0

                if gt_start_feature in range(start_feat_id, end_feat_id):
                    gt_boundary[gt_start_feature - start_feat_id] = 1

                if gt_end_feature in range(start_feat_id, end_feat_id):
                    gt_boundary[gt_end_feature - start_feat_id] = 1

                batch_proposal.append(gt_proposal)
                batch_boundary.append(gt_boundary)
                batch_anchor_mask.append(anchor_mask)

            batch_feature, batch_feature_mask = self.process_batch_data(batch_feature, max_sample_len)
            batch_sentence, batch_sentence_mask = self.process_batch_data(batch_sentence,
                                                                          self._options['max_sentence_len'])
            if self._options['bidirectional_lstm_sentence']:
                batch_sentence_bw, _ = self.process_batch_data(batch_sentence_bw, self._options['max_sentence_len'])
            batch_proposal = np.array(batch_proposal)
            batch_boundary = np.array(batch_boundary)

            batch_anchor_mask = np.array(batch_anchor_mask)

            # serve as a tuple
            batch_data = {'video_feat': batch_feature, 'video_feat_mask': batch_feature_mask, \
                          'sentence': batch_sentence, 'sentence_mask': batch_sentence_mask, \
                          'proposal': batch_proposal, 'proposal_weight': np.array(self._proposal_weight), \
                          'anchor_mask': batch_anchor_mask}
            if self._options['bidirectional_lstm_sentence']:
                batch_data['sentence_bw'] = batch_sentence_bw

            if self._options['predict_boundary']:
                batch_data['boundary'] = batch_boundary
                batch_data['boundary_weight'] = np.array([self._boundary_weight])

            yield batch_data

            current = current + batch_size

            if split != 'train' and current + batch_size > self._options['eval_batch_num']:
                current = 0
                break

            if current + batch_size > self.get_size(split):
                # at the end of list, shuffle it
                if split == 'train':
                    print('Randomly shuffle training data ...')
                    random.shuffle(ids)
                    print('The new shuffled ids are:')
                    print('%s, %s, %s, ..., %s' % (ids[0], ids[1], ids[2], ids[-1]))
                    time.sleep(3)
                    current = 0
                else:
                    if current < self.get_size(split):
                        # few samples left, so use smaller batch
                        batch_size = self.get_size(split) - current
                    else:
                        current = 0
                        break
