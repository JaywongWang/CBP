# -*- coding: utf-8 -*-

'''
This script is to calculate weights for the defined anchors
the video is decoded at fps=16
'''

import os
import json
import h5py
import random


def get_iou(pred, gt):
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
    iou = float(intersection) / (union + 1e-8)

    return iou

def get_intersection(region1, region2):
    start1, end1 = region1
    start2, end2 = region2
    start = max(start1, start2)
    end = min(end1, end2)

    return (start, end)

data_source = './data/'
feature_path = './features/charades_c3d_fc6_nonoverlap.hdf5'
feature_name = 'c3d_fc6_features'
features = h5py.File(feature_path, 'r')

splits = {'train': 'train', 'test': 'test'}

'''
feature_resolution = 16
# designed anchors
# anchor length is measured in frame number
anchors = list(range(feature_resolution*3, feature_resolution*(22+1), feature_resolution))
n_anchors = len(anchors)
'''

# designed anchors
# anchor length is measured in feature number
anchors = list(range(3, 23))
n_anchors = len(anchors)

print('Anchors are: (in feature number)')
print(anchors)
with open(os.path.join(data_source, 'anchors.json'), 'w') as fid:
    json.dump(anchors, fid)

count_positives = [0 for _ in range(n_anchors)]
count_negatives = [0 for _ in range(n_anchors)]
weights = [[0., 0.] for _ in range(n_anchors)]

# encode proposal information
split = 'train'
split_data = json.load(open(os.path.join(data_source, '%s.json' % splits[split])))
video_ids = split_data.keys()

sample_num = 10
sample_len = 50

# segment dictionary, each value stores all sampled segments for the video
sum_video_length = 0

for index, vid in enumerate(video_ids):
    print('Processing video id {}'.format(vid))
    data = split_data[vid]
    feature_len = features[vid][feature_name].value.shape[0]
    n_annos = len(data['sentences'])
    print('Video length: {}'.format(feature_len))
    print('{} sentences for video {}'.format(n_annos, vid))

    this_sample_len = min(feature_len, sample_len)
    sum_video_length += n_annos*sample_num*this_sample_len
    for idx in range(n_annos):
        
        #gt_start_frame, gt_end_frame = data['framestamps'][idx]
        gt_start_time, gt_end_time = data['timestamps'][idx]

        for sample_id in range(sample_num):
            start_feat_id = random.randint(0, max((feature_len - sample_len), 0))
            end_feat_id = min(start_feat_id + sample_len, feature_len)

            for feat_id in range(start_feat_id, end_feat_id):
                end = feat_id + 0.5
                for anchor_id, anchor in enumerate(anchors):
                    pred_start, pred_end = end - anchor, end
                    start_time, end_time = pred_start, pred_end   # when fps=16 and every 16 non-overlap frames correspond to one feature
                    tiou = get_iou((start_time, end_time), (gt_start_time, gt_end_time))
                    if tiou > 0.7:
                        count_positives[anchor_id] += 1
                    elif tiou < 0.3:
                        count_negatives[anchor_id] += 1
                    else:
                        pass


print('Calculating anchor weights ...')
for i in range(n_anchors):
    # weight for negative label
    weights[i][1] = 1.0
    # weight for positive label
    weights[i][0] = count_negatives[i] / count_positives[i]

# avoid too large/small weights
for i in range(n_anchors):
    if weights[i][0] > 100.:
        weights[i][0] = 100.

print('Writing ...')
with open(os.path.join(data_source, 'weights_anchor.json'), 'w') as fid:
    json.dump(weights, fid)


