# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import numpy as np
from collections import OrderedDict
import tensorflow as tf
from opt import default_options
from data_provider import DataProvision
from model import CBP
from util import evaluation_metric_util, format_loss_output, mkdirs
import sys

# set default encoding
reload(sys)
sys.setdefaultencoding('utf-8')


def evaluation_metric(options, data_provision, sess, inputs, outputs, interactor_inputs=None,
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
    :return: evaluated metrics
    """

    out_data, recall_at_k = evaluation_metric_util(
        options, data_provision, sess, inputs, outputs,
        interactor_inputs=interactor_inputs, interactor_outputs=interactor_outputs,
        proposal_inputs=proposal_inputs, proposal_outputs=proposal_outputs, split=split)

    res_file = 'results/%d/temp_grounding_result.json' % options['train_id']
    if not os.path.exists(os.path.dirname(res_file)):
        os.makedirs(os.path.dirname(res_file))

    print('Writing result json file ...')
    with open(res_file, 'w') as fid:
        json.dump(out_data, fid)

    return recall_at_k

def train(options):
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])
    sess = tf.InteractiveSession(config=sess_config)

    print('Loading data ...')
    data_provision = DataProvision(options)

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    init_epoch = options['init_epoch']
    lr_init = options['learning_rate']
    status_file = options['status_file']
    lr = lr_init
    lr_decay_factor = options['lr_decay_factor']
    n_epoch_to_decay = options['n_epoch_to_decay'] # when to decay the lr
    next_epoch_to_decay = n_epoch_to_decay.pop()

    n_iters_per_epoch = data_provision.get_size('train') // batch_size
    eval_in_iters = int(n_iters_per_epoch / float(options['n_eval_per_epoch']))

    #############################################
    # build model #

    print('Building model for training ...')
    model = CBP(options)
    inputs, outputs = model.build_train()
    t_loss = outputs['loss']
    t_proposal_loss = outputs['proposal_loss']
    t_loss_list = [t_loss, t_proposal_loss]
    if options['predict_boundary']:
        t_boundary_loss = outputs['boundary_loss']
        t_loss_list.append(t_boundary_loss)
    t_reg_loss = outputs['reg_loss']

    print('Building model for inference ...')
    i_inputs, i_outputs = model.build_inference(reuse=True)
    interactor_inputs, interactor_outputs = model.build_interactor_self_attention_inference(reuse=True)
    proposal_inputs, proposal_outputs = model.build_proposal_prediction_inference(reuse=True)
    
    t_summary = tf.summary.merge_all()
    t_lr = tf.placeholder(tf.float32)
    
    if options['solver'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=t_lr)
    elif options['solver'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=t_lr)
    elif options['solver'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=t_lr, momentum=options['momentum'])
    elif options['solver'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=t_lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=t_lr)
    
    # gradient clipping option
    if options['clip_gradient_norm'] < 0:
        train_op = optimizer.minimize(t_loss + options['reg'] * t_reg_loss)
    else:
        gvs = optimizer.compute_gradients(t_loss + options['reg'] * t_reg_loss)
        clip_grad_var = [(tf.clip_by_norm(grad, options['clip_gradient_norm']), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var)

    # save summary data
    train_summary_writer = tf.summary.FileWriter(os.path.dirname(options['status_file']), sess.graph)

    # initialize all variables
    tf.global_variables_initializer().run()

    ## test model variable shape
    if 'print_debug' in options.keys() and options['print_debug']:
        print('*********** Variable Shape *************')
        for v in tf.trainable_variables():
            print('%s:' % v.name)
            print(v.get_shape())

        if 'test_tensors' in options:
            print('********** Tensor Shape ************')
            tf_graph = tf.get_default_graph()
            for t_name in options['test_tensors']:
                t = tf_graph.get_tensor_by_name('%s:0' % t_name)
                print('%s: ' % t_name)
                print(t.get_shape())

    # for saving and restoring checkpoints during training
    saver = tf.train.Saver(max_to_keep=100, write_version=1)

    # initialize model from a given checkpoint path
    if options['init_from']:
        print('Init model from %s' % options['init_from'])
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print('Restoring parameters ...')
        restore_vars = [v for v in restore_vars if any([v.name.startswith(module_name)
                                                        for module_name in options['init_module']])]
        saver_tmp = tf.train.Saver(var_list=restore_vars)
        saver_tmp.restore(sess, options['init_from'])

    # save loss/evaluation history
    json_worker_status = OrderedDict()
    json_worker_status['options'] = options
    json_worker_status['history'] = []
    json_worker_status['eval_results'] = []
    json.dump(json_worker_status, open(options['status_file'], 'w'))

    if options['eval_init']:
        print('Evaluating performance ...')
        evaluation_metric(options, data_provision, sess, i_inputs, i_outputs,
                          interactor_inputs, interactor_outputs, proposal_inputs, proposal_outputs)

    t0 = time.time()
    eval_id = 0
    train_batch_generator = data_provision.iterate_batch('train', batch_size)
    # all saved checkpoint file names
    checkpoint_filenames = []
    total_iter = 0
    for epoch in range(init_epoch, max_epochs):
        
        # manually set when to decay learning rate
        if not options['auto_lr_decay']:
            if epoch == next_epoch_to_decay:
                if len(n_epoch_to_decay) == 0:
                    next_epoch_to_decay = -1
                else:
                    next_epoch_to_decay = n_epoch_to_decay.pop()

                print('Decaying learning rate ...')
                lr *= lr_decay_factor

        print('epoch: %d/%d, lr: %.1E (%.1E)'%(epoch, max_epochs, lr, lr_init))
        for iter in range(n_iters_per_epoch):
            batch_data = next(train_batch_generator)
            feed_dict = {
                t_lr: lr,
                inputs['dropout']: options['dropout']
            }
            for key, value in batch_data.items():
                if key not in inputs:
                    continue
                feed_dict[inputs[key]] = value

            if options['predict_boundary']:
                _, summary, loss, proposal_loss, boundary_loss, reg_loss = \
                    sess.run([train_op, t_summary, t_loss, t_proposal_loss, t_boundary_loss, t_reg_loss], feed_dict=feed_dict)
            else:
                _, summary, loss, proposal_loss, reg_loss = \
                    sess.run([train_op, t_summary, t_loss, t_proposal_loss, t_reg_loss], feed_dict=feed_dict)
            
            if iter == 0 and epoch == init_epoch:
                smooth_loss = proposal_loss
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * proposal_loss
            
            if iter % options['n_iters_display'] == 0:
                if options['predict_boundary']:
                    print('iter: %d, epoch: %d/%d, \n'
                          'lr: %.1E, loss: %.4f, proposal_loss: %.4f, boundary_loss: %.4f, reg_loss: %.4f' %
                          (iter, epoch, max_epochs, lr, loss, proposal_loss, boundary_loss, reg_loss))
                else:
                    print('iter: %d, epoch: %d/%d, \nlr: %.1E, loss: %.4f, proposal_loss: %.4f, reg_loss: %.4f' %
                          (iter, epoch, max_epochs, lr, loss, proposal_loss, reg_loss))
                train_summary_writer.add_summary(summary, iter + epoch * n_iters_per_epoch)
                jstatus = OrderedDict()
                jstatus['epoch'] = (epoch, max_epochs)
                jstatus['iter'] = (iter, n_iters_per_epoch)
                jstatus['proposal_loss'] = (float(proposal_loss), float(smooth_loss), float(reg_loss))
                json_worker_status['history'].append(jstatus)

            # every 30 secs write once
            if (time.time() - t0) / 60.0 > 0.5:
                t0 = time.time()
                json.dump(json_worker_status, open(status_file, 'w'))
            
            if (total_iter+1) % eval_in_iters == 0:
                print('Evaluating model performance ...')
                recall_at_k = evaluation_metric(options, data_provision, sess,
                                                i_inputs, i_outputs, interactor_inputs,
                                                interactor_outputs, proposal_inputs, proposal_outputs, split='test')

                jeval_results = OrderedDict()
                jeval_results['lr'] = lr
                jeval_results['recall_at_k'] = recall_at_k
                json_worker_status['eval_results'].append(jeval_results)
                json.dump(json_worker_status, open(status_file, 'w'))

                checkpoint_path = '%sepoch%02d_rec%.2f_%02d_lr%f.ckpt' % \
                                      (options['ckpt_prefix'], epoch, 100.*recall_at_k,
                                       eval_id, lr)

                saver.save(sess, checkpoint_path)
                checkpoint_filenames.append(checkpoint_path)
                
                eval_id = eval_id + 1

                # automatically lower learning rate
                if options['auto_lr_decay']:
                    # review val loss history or score history
                    eval_results = json_worker_status['eval_results']
                    view_end_eval_id = eval_id
                    view_start_eval_id = view_end_eval_id - options['observe_patience']
                    view_start_epoch_id = (view_end_eval_id + init_epoch*options['n_eval_per_epoch'] -
                                           options['observe_patience']) // options['n_eval_per_epoch']

                    review_results = [result['recall_at_k'] for result in eval_results[view_start_eval_id: view_end_eval_id]]
                    best_result = max(review_results)
                    
                    if view_start_eval_id >= 0:
                        # if the eval result does improve
                        if review_results.index(best_result) == 0:
                            # go back to the state of view_start_eval_id, and lower learning rate
                            print('Init model from %s ...' % checkpoint_filenames[view_start_eval_id])
                            saver.restore(sess, checkpoint_filenames[view_start_eval_id])
                            print('Decaying learning rate ...')
                            lr *= lr_decay_factor
                
                if lr < options['min_lr']:
                    print('Reach minimum learning rate. Done training.')
                    return

            total_iter += 1
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = default_options()
    for key, value in options.items():
        parser.add_argument('--%s' % key, dest=key, type=type(value), default=None)
    args = parser.parse_args()
    args = vars(args)
    for key, value in args.items():
        if value is not None:
            options[key] = value

    options['ckpt_prefix'] = './checkpoints/' + str(options['train_id']) + '/'
    options['status_file'] = options['ckpt_prefix'] + 'status.json'

    mkdirs(options['ckpt_prefix'])

    train(options)

