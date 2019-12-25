# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
from opt import default_options
from data_provider import DataProvision
from model import CBP
from util import evaluation_metric_util, mkdirs
import argparse
import tensorflow as tf
import sys

# set default encoding
#reload(sys)
#sys.setdefaultencoding('utf-8')

#np.set_printoptions(threshold='nan')


def test(options):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])
    sess = tf.InteractiveSession(config=sess_config)

    # build model
    print('Building model ...')
    model = CBP(options)
    inputs, outputs = model.build_inference()
    interactor_inputs, interactor_outputs = model.build_interactor_self_attention_inference()
    proposal_inputs, proposal_outputs = model.build_proposal_prediction_inference()

    # print variable names
    for v in tf.trainable_variables():
        print(v.name)
        print(v.get_shape())

    print('Loading data ...')
    data_provision = DataProvision(options)

    print('Restoring model from %s' % options['init_from'])
    saver = tf.train.Saver()
    saver.restore(sess, options['init_from'])

    split = 'test'
    print('Start to predict ...')
    t0 = time.time()

    out_data, recall_at_k = evaluation_metric_util(
        options, data_provision, sess, inputs, outputs,
        interactor_inputs=interactor_inputs, interactor_outputs=interactor_outputs,
        proposal_inputs=proposal_inputs, proposal_outputs=proposal_outputs, split=split)

    out_json_file = './results/%d/predict_proposals_%s_nms_%.2f.json' % (
        options['train_id'], split, options['nms_threshold'])

    mkdirs(os.path.dirname(out_json_file))

    print('Writing result json file ...')
    with open(out_json_file, 'w') as fid:
        json.dump(out_data, fid)

    print('Total running time: %f seconds.' % (time.time() - t0))


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
        if value is not None:
            options[key] = value

    test(options)
