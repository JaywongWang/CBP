'''
This script is to calculate weights for the boundary prediction
'''

import os
import json
import h5py


# count the boundaries of region1 that are contained in region2
def get_num_boundary(region1, region2, interval):
    start1, end1 = region1
    start2, end2 = region2

    count = 0
    for gap in range(interval):
        if start2 <= start1 + gap <= end2:
            count += 1
        if start2 <= end1 + gap <= end2:
            count += 1

    return count


feature_path = './features/activitynet_c3d_fc6_stride_2s.hdf5'
feature_name = 'c3d_fc6_features'
features = h5py.File(feature_path, 'r')

splits = {'train': 'train', 'val': 'val'}

count_boundary = 0.
weights = [0., 0.]

# encode proposal information
data_source = './data/'
split = 'train'
split_data = json.load(open(os.path.join(data_source, '%s.json' % splits[split])))
video_ids = split_data.keys()

sample_num = 1
# sample_len = 300
boundary_interval = 1

# segment dictionary, each value stores all sampled segments for the video
sum_video_length = 0
for index, vid in enumerate(video_ids):
    print('Processing video id {}'.format(vid))
    data = split_data[vid]
    feature_len = features['v_' + vid][feature_name].value.shape[0]
    n_annos = len(data['sentences'])
    print('Video length: {}'.format(feature_len))
    print('{} sentences for video {}'.format(n_annos, vid))

    this_sample_len = feature_len
    sum_video_length += n_annos * sample_num * this_sample_len
    for idx in range(n_annos):

        gt_start_feature, gt_end_feature = data['featstamps'][idx]

        for sample_id in range(sample_num):
            start_feat_id = 0
            end_feat_id = feature_len - 1

            count_boundary += get_num_boundary((gt_start_feature, gt_end_feature), (start_feat_id, end_feat_id),
                                               boundary_interval)

print('Calculating boundary weights ...')
# weight for negative label
weights[1] = count_boundary / float(sum_video_length)
# weight for positive label
weights[0] = 1. - weights[1]

print('Writing ...')
with open(os.path.join(data_source, 'weights_boundary.json'), 'w') as fid:
    json.dump(weights, fid)
