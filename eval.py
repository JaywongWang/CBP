# -*- coding: utf-8 -*-

'''
Result evaluation
'''

import argparse
from util import eval_result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default='./results/charades/predict_proposals_test_nms_0.55.json')
    parser.add_argument('--gt_file', type=str, default='./datasets/charades/data/save/test.json')
    args = parser.parse_args()

    eval_result(args.result_file, args.gt_file)
