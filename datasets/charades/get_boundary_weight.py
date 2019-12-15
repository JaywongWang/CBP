'''
This script is to calculate weights for the boundary prediction

the video is decoded at 16 fps
'''

import os
import h5py
import json
import math
import numpy as np
import h5py
import random
from sklearn.cluster import KMeans

# count the boundaries that are contained in region2 for region1 
def get_num_boundary(region1, region2):
    start1, end1 = region1
    start2, end2 = region2

    count = 0
    if start1 >= start2 and start1 <= end2:
        count += 1
    if end1 >= start2 and end1 <= end2:
        count += 1

    return count


feature_path = './features/charades_c3d_fc6_nonoverlap.hdf5'
features = h5py.File(feature_path, 'r')

splits = {'train': 'train', 'test': 'test'}

c3d_resolution = 16

count_boundary = 0.
weights = [0., 0.]

# encode proposal information
data_source = './data/'
split = 'train'
split_data = json.load(open(os.path.join(data_source, '%s.json'%splits[split])))
video_ids = split_data.keys()

sample_num = 5
sample_len = 50

# segment dictionary, each value stores all sampled segments for the video
sum_video_length = 0
for index, vid in enumerate(video_ids):
    print('Processing video id {}'.format(vid))
    data = split_data[vid]
    feature_len = features[vid]['c3d_fc6_features'].value.shape[0]
    n_annos = len(data['sentences'])
    print('Video length: {}'.format(feature_len))
    print('{} sentences for video {}'.format(n_annos, vid))

    this_sample_len = min(feature_len, sample_len)
    sum_video_length += n_annos*sample_num*this_sample_len
    for idx in range(n_annos):
        
        gt_start_frame, gt_end_frame = data['framestamps'][idx]
        gt_start_feature, gt_end_feature = gt_start_frame//c3d_resolution, gt_end_frame//c3d_resolution

        for sample_id in range(sample_num):
            start_feat_id = random.randint(0, max((feature_len - sample_len), 0))
            end_feat_id = min(start_feat_id + sample_len, feature_len)
            
            count_boundary += get_num_boundary((gt_start_feature, gt_end_feature), (start_feat_id, end_feat_id-1))

print('Calculating boundary weights ...')
# weight for negative label
weights[1] = count_boundary / float(sum_video_length)
# weight for positive label
weights[0] = 1. - weights[1]


print('Writing ...')
with open(os.path.join(data_source, 'weights_boundary.json'), 'w') as fid:
    json.dump(weights, fid)


