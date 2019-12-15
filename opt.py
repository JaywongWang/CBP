# -*- coding: utf-8 -*-

from collections import OrderedDict

def default_options():

    options = OrderedDict()
    
    ### DATA ###
    options['dataset'] = 'charades'  # dataset name: 'tacos', 'charades', 'activitynet_captions'
    options['feature_name'] = 'c3d_fc6_features'   # feature name
    options['feature_to_second'] = 1    # how many seconds correspond to one feature
    options['init_from'] = ''   # initialized weights
    options['init_module'] = ['sentence_encoding', 'interactor']  # initialized modules

    ### MODEL CONFIG ###
    options['video_feat_dim'] = 4096  # dim of video snippet feature
    options['word_embed_size'] = 300  # use Glove pretrained word embedding
    options['attention_hidden_size'] = 512 # neuron size for attention hidden layer
    options['filter_number'] = 512    # filter number for the first cnn layer
    options['rnn_size'] = 512        # number of rnn hidden neurons
    options['max_sentence_len'] = 10  # maximum length of the sentences
    options['adaptive_sent_states'] = False  # sequence length input for dynamic_rnn
    options['sample_len'] = 50   # sampling video length, measured in seconds
    options['predict_boundary'] = False   # whether add a branch to predict boundary
    options['selfatt_restricted'] = False   # restricted self attention
    options['restricted_neighbors'] = 50     # restricted self attention
    options['num_anchors'] = 20            # number of anchors
    options['anchor_mask'] = False         # use anchor mask
    options['bidirectional_lstm_sentence'] = False   # bidirectional lstm for sentence modeling

    ### OPTIMIZATION ###
    # gpus
    options['gpu_id'] = 0    # gpu id
    options['train_id'] = 1    # train id
    options['weight_boundary'] = 8.    # weighting for boundary prediction branch
    options['dropout'] = 0.5      # dropout
    options['solver'] = 'adam'      # 'sgd', momentum', 'adam', 'rmsprop', 'sgd_nestreov_momentum' 
    options['momentum'] = 0.9     # only valid when solver is set to momentum optimizer
    options['batch_size'] = 64   # batch size
    options['eval_batch_size'] = 3720    # batch size for evaluation
    options['eval_batch_num'] = 10000  # eval batch number for each evaluation (to speed up the evaluation process)
    options['learning_rate'] = 1e-3    # learning rate
    options['lr_decay_factor'] = 0.1   # learning rate decay
    options['n_epoch_to_decay'] = list(range(1, 100, 1))[::-1]
    options['auto_lr_decay'] = True  # automatically decay learning rate
    options['observe_patience'] = 5   # observation patience for decaying learning rate
    options['min_lr'] = 1e-5    # minimum learning rate
    options['reg'] = 1e-5       # regularization
    options['max_epochs'] = 100   # max number of epochs
    options['init_epoch'] = 0     # initial epoch
    options['n_eval_per_epoch'] = 0.2  # number of evaluations per epoch
    options['eval_init'] = False  # evaluate the initialized model
    options['clip_gradient_norm'] = 100.0  # gradient clipping
    options['log_input_min'] = 1e-20     # minimum input to the log() function
    options['proposal_tiou_threshold'] = 0.85   # tiou threshold to positive samples
    options['negative_tiou_threshold'] = 0.15   # tiou threshold to negative samples
    options['nms_threshold'] = 0.55            # threshold for non-maximum suppression

    ### INFERENCE ###
    options['max_proposal_num'] = 5   # number of proposals for evaluation
    options['tiou_measure'] = 0.7      # evaluate recall@k based on a given tIoU threshold

    # logging
    options['n_iters_display'] = 1    # display frequency

    # debug
    options['print_debug'] = True
    options['test_tensors'] = ['video_feat', 'proposal', 'proposal_weight']

    return options
