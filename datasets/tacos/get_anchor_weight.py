# -*- coding: utf-8 -*-

'''
This script is to calculate weights for the defined anchors
the video is decoded at 16 fps
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
feature_path = './features/tacos_c3d_fc6_nonoverlap.hdf5'
feature_name = 'c3d_fc6_features'
features = h5py.File(feature_path, 'r')

splits = {'train': 'train', 'val': 'val', 'test': 'test'}

'''
n_anchors = 32
feature_resolution = 16
# designed anchors
# anchor length is measured in frame number 
anchors = list(range(feature_resolution, (n_anchors+1)*feature_resolution, feature_resolution))
'''
n_anchors = 32
# designed anchors
# anchor length is measured in feature number
anchors = list(range(1, n_anchors+1))


print('Anchors are: (in feature number)')
print(anchors)
with open(os.path.join(data_source, 'anchors.json'), 'w') as fid:
    json.dump(anchors, fid)

count_anchors = [0 for _ in range(n_anchors)]
weights = [[0., 0.] for _ in range(n_anchors)]

# encode proposal information
split = 'train'
split_data = json.load(open(os.path.join(data_source, '%s.json' % splits[split])))
video_ids = split_data.keys()

sample_num = 100
sample_len = 60

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
        
        gt_start_time, gt_end_time = data['timestamps'][idx]
        gt_start_feature, gt_end_feature = int(gt_start_time), int(gt_end_time)

        assert(gt_end_feature <= feature_len)
        
        start_point = max((gt_start_feature + gt_end_feature) / 2, 0)
        end_point = gt_end_feature + (gt_end_feature - gt_start_feature + 1)
        end_point = min(end_point, feature_len)

        for sample_id in range(sample_num):
            start_feat_id = random.randint(0, max((feature_len - sample_len), 0))
            end_feat_id = min(start_feat_id + sample_len, feature_len)

            feat_check_start, feat_check_end = get_intersection((start_point, end_point), (start_feat_id, end_feat_id))
            feat_check_start, feat_check_end = int(feat_check_start), int(feat_check_end)
            
            if feat_check_start > feat_check_end:
                continue

            for feat_id in range(feat_check_start, feat_check_end):
                end = feat_id + 0.5
                for anchor_id, anchor in enumerate(anchors):
                    pred_start, pred_end = end - anchor, end
                    start_time, end_time = pred_start, pred_end
                    tiou = get_iou((start_time, end_time), (gt_start_time, gt_end_time))
                    if tiou > 0.5:
                        count_anchors[anchor_id] += 1

print('Calculating anchor weights ...')
for i in range(n_anchors):
    # weight for negative label
    weights[i][1] = count_anchors[i] / float(sum_video_length)
    # weight for positive label
    weights[i][0] = 1. - weights[i][1]


print('Writing ...')
with open(os.path.join(data_source, 'weights_anchor.json'), 'w') as fid:
    json.dump(weights, fid)


