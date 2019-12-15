# -*- coding: utf-8 -*-

'''
Model Implementation
'''

import tensorflow as tf
import math


class CBP(object):

    def __init__(self, options):
        self.options = options

    def self_attention_matmul(self, inputs, mask, dim):
        """ Self attention module
        """
        '''
        num = tf.shape(inputs)[1]
        inputs_reshape = tf.reshape(inputs, [-1, dim])
        linear_map = tf.get_variable('linear_map', [dim, 512],
                                     initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(dim)))
        inputs_map = tf.matmul(inputs_reshape, linear_map)
        inputs_map = tf.reshape(inputs_map, [-1, num, 512])
        relevance = 1. / math.sqrt(dim) * tf.matmul(inputs_map, tf.transpose(inputs_map, perm=[0, 2, 1]))
        '''
        num = tf.shape(inputs)[1]
        inputs_reshape = tf.reshape(inputs, [-1, dim])
        query_map = tf.get_variable('query_map', [dim, 256],
                                    initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(dim)))
        key_map = tf.get_variable('key_map', [dim, 256],
                                  initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(dim)))
        #value_map = tf.get_variable('value_map', [dim, dim],
        #                            initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(dim)))

        query = tf.matmul(inputs_reshape, query_map)
        key = tf.matmul(inputs_reshape, key_map)
        query = tf.reshape(query, [-1, num, 256])
        key = tf.reshape(key, [-1, num, 256])
        relevance = 1. / math.sqrt(dim) * tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))

        # use while_loop to get masked softmax results
        batch_size = tf.shape(inputs)[0]
        ind = tf.constant(0)
        masked_attention = tf.fill([0, num, num], 0.)
        mask = tf.to_int32(tf.reduce_sum(mask, axis=-1))

        def condition(ind, data, mask):
            return tf.less(ind, batch_size)

        def body(ind, data, mask):
            pick_num = mask[ind]
            attention = tf.nn.softmax(relevance[ind, :pick_num, :pick_num], dim=-1)
            attention = tf.concat([attention, tf.zeros(shape=[num - pick_num, pick_num])], axis=0)
            attention = tf.concat([attention, tf.zeros(shape=[num, num - pick_num])], axis=-1)
            data = tf.concat([data, tf.expand_dims(attention, axis=0)], axis=0)
            return tf.add(ind, 1), data, mask

        _, masked_attention, _ = tf.while_loop(condition, body, loop_vars=[ind, masked_attention, mask],
                                               shape_invariants=[ind.get_shape(), tf.TensorShape([None, None, None]),
                                                                 mask.get_shape()])

        outputs = tf.matmul(masked_attention, inputs)
        #outputs = tf.matmul(masked_attention, tf.reshape(tf.matmul(inputs_reshape, value_map), [-1, num, dim]))

        outputs = tf.concat([outputs, inputs], axis=-1)

        return outputs

    def self_attention_matmul_restricted(self, inputs, mask, dim, restricted=10):
        """ Restricted self attention module
        """

        num = tf.shape(inputs)[1]
        inputs_reshape = tf.reshape(inputs, [-1, dim])
        linear_map = tf.get_variable('linear_map', [dim, 512],
                                     initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(dim)))
        inputs_map = tf.matmul(inputs_reshape, linear_map)
        inputs_map = tf.reshape(inputs_map, [-1, num, 512])
        relevance = 1. / math.sqrt(dim) * tf.matmul(inputs_map, tf.transpose(inputs_map, perm=[0, 2, 1]))
        '''
        num = tf.shape(inputs)[1]
        inputs_reshape = tf.reshape(inputs, [-1, dim])
        query_map = tf.get_variable('query_map', [dim, 256], 
                                    initializer=tf.random_normal_initializer(stddev=1./math.sqrt(dim)))
        key_map = tf.get_variable('key_map', [dim, 256], 
                                    initializer=tf.random_normal_initializer(stddev=1./math.sqrt(dim)))
        query = tf.matmul(inputs_reshape, query_map)
        key = tf.matmul(inputs_reshape, key_map)
        query = tf.reshape(query, [-1, num, 256])
        key = tf.reshape(key, [-1, num, 256])
        relevance = 1./math.sqrt(dim)*tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))
        '''

        # use while_loop to get masked softmax results
        batch_size = tf.shape(inputs)[0]
        ind = tf.constant(0)
        masked_attention = tf.fill([0, num, num], 0.)
        mask = tf.to_int32(tf.reduce_sum(mask, axis=-1))

        def condition_help(ind, feat_id, out_attention, relevance, mask):
            return tf.less(feat_id, mask[ind])

        def body_help(ind, feat_id, out_attention, relevance, mask):
            start_feat_id = tf.maximum(0, feat_id - restricted // 2)
            end_feat_id = tf.minimum(mask[ind], feat_id + restricted // 2)
            part_softmax = tf.nn.softmax(relevance[feat_id, start_feat_id:end_feat_id])
            left_padding = tf.zeros(shape=[start_feat_id, ], dtype=tf.float32)
            right_padding = tf.zeros(shape=[mask[ind] - end_feat_id, ], dtype=tf.float32)
            out_softmax = tf.expand_dims(tf.concat([left_padding, part_softmax, right_padding], axis=0), axis=0)
            out_attention = tf.concat([out_attention, out_softmax], axis=0)

            return ind, tf.add(feat_id, 1), out_attention, relevance, mask

        def condition(ind, data, mask):
            return tf.less(ind, batch_size)

        def body(ind, data, mask):
            pick_num = mask[ind]
            out_attention = tf.fill([0, pick_num], 0.)
            feat_id = tf.constant(0)
            _, _, out_attention, _, _ = tf.while_loop(condition_help, body_help,
                                                      loop_vars=[ind, feat_id, out_attention, relevance[ind], mask],
                                                      shape_invariants=[ind.get_shape(), feat_id.get_shape(),
                                                                        tf.TensorShape([None, None]),
                                                                        tf.TensorShape([None, None]), mask.get_shape()])
            out_attention = tf.concat([out_attention, tf.zeros(shape=[num - pick_num, pick_num])], axis=0)
            out_attention = tf.concat([out_attention, tf.zeros(shape=[num, num - pick_num])], axis=-1)
            data = tf.concat([data, tf.expand_dims(out_attention, axis=0)], axis=0)

            return tf.add(ind, 1), data, mask

        _, masked_attention, _ = tf.while_loop(condition, body, loop_vars=[ind, masked_attention, mask],
                                               shape_invariants=[ind.get_shape(), tf.TensorShape([None, None, None]),
                                                                 mask.get_shape()])

        outputs = tf.matmul(masked_attention, inputs)

        outputs = tf.concat([outputs, inputs], axis=-1)

        return outputs

    def build_inference(self, reuse=False):
        """
        Build inference model for generating next states
        """

        inputs = {}
        outputs = {}

        video_feat = tf.placeholder(tf.float32, [None, self.options['video_feat_dim']], name='video_feat')
        sentence = tf.placeholder(tf.float32, [None, self.options['max_sentence_len'], self.options['word_embed_size']])
        sentence_mask = tf.placeholder(tf.float32, [None, None])

        if self.options['bidirectional_lstm_sentence']:
            sentence_bw = tf.placeholder(tf.float32,
                                         [None, self.options['max_sentence_len'], self.options['word_embed_size']])
            inputs['sentence_bw'] = sentence_bw

        video_c_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])
        video_h_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])

        interactor_c_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])
        interactor_h_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])

        inputs['video_feat'] = video_feat
        inputs['sentence'] = sentence
        inputs['sentence_mask'] = sentence_mask
        inputs['video_c_state'] = video_c_state
        inputs['video_h_state'] = video_h_state
        inputs['interactor_c_state'] = interactor_c_state
        inputs['interactor_h_state'] = interactor_h_state

        video_state = tf.nn.rnn_cell.LSTMStateTuple(video_c_state, video_h_state)
        interactor_state = tf.nn.rnn_cell.LSTMStateTuple(interactor_c_state, interactor_h_state)

        batch_size = tf.shape(video_feat)[0]

        rnn_cell_sentence = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_interator = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )

        with tf.variable_scope('sentence_encoding', reuse=reuse) as sentence_scope:
            if self.options['adaptive_sent_states']:
                sequence_length = tf.reduce_sum(sentence_mask, axis=-1)
            else:
                sequence_length = tf.fill([batch_size, ], self.options['max_sentence_len'])
            
            initial_state = rnn_cell_sentence.zero_state(batch_size=batch_size, dtype=tf.float32)

            sentence_states, sentence_final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell_sentence,
                inputs=sentence,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=tf.float32
            )

            if self.options['bidirectional_lstm_sentence']:
                rnn_cell_sentence_bw = tf.contrib.rnn.LSTMCell(
                    num_units=self.options['rnn_size'],
                    state_is_tuple=True,
                    initializer=tf.orthogonal_initializer()
                )
                with tf.variable_scope('sentence_bw') as scope:
                    sentence_states_bw, sentence_final_state_bw = tf.nn.dynamic_rnn(
                        cell=rnn_cell_sentence_bw,
                        inputs=sentence_bw,
                        sequence_length=sequence_length,
                        initial_state=initial_state,
                        dtype=tf.float32
                    )
                    sentence_states_bw = tf.reverse_sequence(sentence_states_bw,
                                                             seq_lengths=tf.to_int32(sequence_length), seq_axis=1)
                sentence_states = tf.concat([sentence_states, sentence_states_bw], axis=-1)

        with tf.variable_scope('interactor', reuse=reuse) as interactor_scope:
            sentence_states_reshape = tf.reshape(sentence_states, [-1, (
                    1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])

            # get video state
            with tf.variable_scope('video_rnn') as video_rnn_scope:
                _, video_state = rnn_cell_video(inputs=video_feat, state=video_state)

            video_c_state, video_h_state = video_state

            # calculate attention over words
            # use a one-layer network to do this
            with tf.variable_scope('word_attention', reuse=reuse) as attention_scope:
                h_states = tf.tile(tf.concat([interactor_h_state, video_h_state], axis=-1),
                                   [1, self.options['max_sentence_len']])
                h_states = tf.reshape(h_states, [-1, 2 * self.options['rnn_size']])

                attention_input = tf.concat([h_states, sentence_states_reshape], axis=-1)

                attention_layer1 = tf.contrib.layers.fully_connected(
                    inputs=attention_input,
                    num_outputs=self.options['attention_hidden_size'],
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.contrib.layers.xavier_initializer()
                )
                attention_layer2 = tf.contrib.layers.fully_connected(
                    inputs=attention_layer1,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer()
                )

            # reshape to match
            attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_sentence_len']])
            attention_score = tf.nn.softmax(attention_reshape, dim=-1)
            attention_score = tf.reshape(attention_score, [-1, 1, self.options['max_sentence_len']])

            # attended word feature
            attended_word_feature = tf.matmul(attention_score,
                                              sentence_states)  # already support batch matrix multiplication in v1.0
            attended_word_feature = tf.reshape(attended_word_feature, [-1, (
                    1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])

            # calculate next interator state
            interactor_input = tf.concat([video_h_state, attended_word_feature], axis=-1)

            with tf.variable_scope('interactor_rnn') as interactor_rnn_scope:
                _, interactor_state = rnn_cell_interator(inputs=interactor_input, state=interactor_state)
            interactor_c_state, interactor_h_state = interactor_state

        outputs['video_c_state'] = video_c_state
        outputs['video_h_state'] = video_h_state
        outputs['interactor_c_state'] = interactor_c_state
        outputs['interactor_h_state'] = interactor_h_state

        return inputs, outputs

    def build_interactor_self_attention_inference(self, reuse=False):
        """
        Build inference model for generating self attended interactor states
        """

        inputs = {}
        outputs = {}
        interactor_states = tf.placeholder(tf.float32, [None, None, self.options['rnn_size']])
        inputs['interactor_states'] = interactor_states
        mask = tf.placeholder(tf.float32, [None, None])
        inputs['mask'] = mask

        with tf.variable_scope('interactor', reuse=reuse) as interactor_scope:
            with tf.variable_scope('selfatt_interactor'):
                if self.options['selfatt_restricted']:
                    interactor_states_selfatt = self.self_attention_matmul_restricted(interactor_states, mask,
                                                                                      self.options['rnn_size'],
                                                                                      self.options[
                                                                                          'restricted_neighbors'])
                else:
                    interactor_states_selfatt = self.self_attention_matmul(interactor_states, mask,
                                                                           self.options['rnn_size'])

        outputs['interactor_states_selfatt'] = interactor_states_selfatt

        return inputs, outputs

    def build_proposal_prediction_inference(self, reuse=False):
        """
        Build inference model for generating proposal & boundary predictions
        """

        inputs = {}
        outputs = {}

        interactor_states = tf.placeholder(tf.float32, [None, None, self.options['rnn_size']])
        inputs['interactor_states'] = interactor_states
        interactor_states_selfatt = tf.placeholder(tf.float32, [None, None, 2 * self.options['rnn_size']])
        inputs['interactor_states_selfatt'] = interactor_states_selfatt

        max_feat_len = tf.shape(interactor_states)[1]

        interactor_states_selfatt = tf.reshape(interactor_states_selfatt, [-1, 2 * self.options['rnn_size']])

        with tf.variable_scope('interactor', reuse=reuse) as interactor_scope:
            with tf.variable_scope('predict_proposal'):
                logit_outputs = tf.contrib.layers.fully_connected(
                    inputs=interactor_states_selfatt,
                    num_outputs=self.options['num_anchors'],
                    activation_fn=None
                )
                logit_outputs = tf.reshape(logit_outputs, [-1, max_feat_len, self.options['num_anchors']])

                # score
                proposal_scores = tf.sigmoid(logit_outputs, name='proposal_scores')
                outputs['proposal_scores'] = proposal_scores

            if self.options['predict_boundary']:
                with tf.variable_scope('predict_boundary') as boundary_scope:
                    boundary_outputs = tf.contrib.layers.fully_connected(
                        inputs=interactor_states_selfatt,
                        num_outputs=1,
                        activation_fn=None
                    )

                    boundary_outputs = tf.reshape(boundary_outputs, [-1, max_feat_len, 1])

                    # score
                    boundary_scores = tf.nn.sigmoid(boundary_outputs, name='boundary_scores')
                    outputs['boundary_scores'] = boundary_scores

        return inputs, outputs

    def build_train(self):
        """
        Build training model
        """

        inputs = {}
        outputs = {}

        video_feat = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat')
        video_feat_mask = tf.placeholder(tf.float32, [None, None])
        anchor_mask = tf.placeholder(tf.float32, [None, None, self.options['num_anchors']])
        sentence = tf.placeholder(tf.float32, [None, None, self.options['word_embed_size']])
        sentence_mask = tf.placeholder(tf.float32, [None, None])

        if self.options['bidirectional_lstm_sentence']:
            sentence_bw = tf.placeholder(tf.float32,
                                         [None, self.options['max_sentence_len'], self.options['word_embed_size']])
            inputs['sentence_bw'] = sentence_bw

        inputs['video_feat'] = video_feat
        inputs['video_feat_mask'] = video_feat_mask
        inputs['anchor_mask'] = anchor_mask
        inputs['sentence'] = sentence
        inputs['sentence_mask'] = sentence_mask

        ## proposal, densely annotated
        proposal = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal')
        inputs['proposal'] = proposal

        ## weighting for positive/negative labels (solve imblance data problem)
        proposal_weight = tf.placeholder(tf.float32, [self.options['num_anchors'], 2], name='proposal_weight')
        inputs['proposal_weight'] = proposal_weight

        if self.options['predict_boundary']:
            boundary = tf.placeholder(tf.int32, [None, None, 1], name='boundary')
            inputs['boundary'] = boundary
            boundary_weight = tf.placeholder(tf.float32, [1, 2], name='boundary_weight')
            inputs['boundary_weight'] = boundary_weight

        # fc dropout
        dropout = tf.placeholder(tf.float32)
        inputs['dropout'] = dropout

        # get batch size, which is a scalar tensor
        batch_size = tf.shape(video_feat)[0]

        rnn_cell_sentence = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_interator = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )

        rnn_cell_sentence = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_sentence,
            input_keep_prob=1.0 - dropout,
            output_keep_prob=1.0 - dropout
        )
        rnn_cell_video = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_video,
            input_keep_prob=1.0 - dropout,
            output_keep_prob=1.0 - dropout
        )
        rnn_cell_interator = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_interator,
            input_keep_prob=1.0 - dropout,
            output_keep_prob=1.0 - dropout
        )

        with tf.variable_scope('sentence_encoding') as sentence_scope:
            if self.options['adaptive_sent_states']:
                sequence_length = tf.reduce_sum(sentence_mask, axis=-1)
            else:
                sequence_length = tf.fill([batch_size, ], self.options['max_sentence_len'])
            initial_state = rnn_cell_sentence.zero_state(batch_size=batch_size, dtype=tf.float32)

            sentence_states, sentence_final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell_sentence,
                inputs=sentence,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=tf.float32
            )

            if self.options['bidirectional_lstm_sentence']:
                rnn_cell_sentence_bw = tf.contrib.rnn.LSTMCell(
                    num_units=self.options['rnn_size'],
                    state_is_tuple=True,
                    initializer=tf.orthogonal_initializer()
                )
                with tf.variable_scope('sentence_bw') as scope:
                    sentence_states_bw, sentence_final_state_bw = tf.nn.dynamic_rnn(
                        cell=rnn_cell_sentence_bw,
                        inputs=sentence_bw,
                        sequence_length=sequence_length,
                        initial_state=initial_state,
                        dtype=tf.float32
                    )
                    sentence_states_bw = tf.reverse_sequence(sentence_states_bw,
                                                             seq_lengths=tf.to_int32(sequence_length), seq_axis=1)
                sentence_states = tf.concat([sentence_states, sentence_states_bw], axis=-1)

        proposal_outputs = tf.fill([batch_size, 0, self.options['num_anchors']], 0.)
        boundary_outputs = tf.fill([batch_size, 0, 1], 0.)
        interactor_states = tf.fill([batch_size, 0, self.options['rnn_size']], 0.)

        with tf.variable_scope('interactor') as interactor_scope:
            interactor_state = rnn_cell_interator.zero_state(batch_size=batch_size, dtype=tf.float32)
            video_state = rnn_cell_video.zero_state(batch_size=batch_size, dtype=tf.float32)
            sentence_states_reshape = tf.reshape(sentence_states, [-1, (
                    1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])
            for i in range(self.options['sample_len']):
                if i > 0:
                    interactor_scope.reuse_variables()

                # get video state
                with tf.variable_scope('video_rnn') as video_rnn_scope:
                    _, video_state = rnn_cell_video(inputs=video_feat[:, i, :], state=video_state)

                # calculate attention over words
                # use a one-layer network to do this
                with tf.variable_scope('word_attention') as attention_scope:
                    h_states = tf.tile(tf.concat([interactor_state[1], video_state[1]], axis=-1),
                                       [1, self.options['max_sentence_len']])
                    h_states = tf.reshape(h_states, [-1, 2 * self.options['rnn_size']])

                    attention_input = tf.concat([h_states, sentence_states_reshape], axis=-1)

                    attention_layer1 = tf.contrib.layers.fully_connected(
                        inputs=attention_input,
                        num_outputs=self.options['attention_hidden_size'],
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer()
                    )
                    attention_layer2 = tf.contrib.layers.fully_connected(
                        inputs=attention_layer1,
                        num_outputs=1,
                        activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer()
                    )

                # reshape to match
                attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_sentence_len']])
                attention_score = tf.nn.softmax(attention_reshape, axis=-1)
                attention_score = tf.reshape(attention_score, [-1, 1, self.options['max_sentence_len']])

                # attended word feature
                attended_word_feature = tf.matmul(attention_score, sentence_states)
                attended_word_feature = tf.reshape(attended_word_feature, [-1, (
                        1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])

                # calculate next interator state
                interactor_input = tf.concat([video_state[1], attended_word_feature], axis=-1)

                with tf.variable_scope('interactor_rnn') as interactor_rnn_scope:
                    _, interactor_state = rnn_cell_interator(inputs=interactor_input, state=interactor_state)

                    interactor_states = tf.concat([interactor_states, tf.expand_dims(interactor_state[1], axis=1)],
                                                  axis=1)

        with tf.variable_scope('interactor', reuse=False) as interactor_scope:
            with tf.variable_scope('selfatt_interactor') as scope:
                if self.options['selfatt_restricted']:
                    interactor_states_selfatt = self.self_attention_matmul_restricted(interactor_states,
                                                                                      video_feat_mask,
                                                                                      self.options['rnn_size'],
                                                                                      self.options[
                                                                                          'restricted_neighbors'])
                else:
                    interactor_states_selfatt = self.self_attention_matmul(interactor_states, video_feat_mask,
                                                                           self.options['rnn_size'])

            interactor_states_selfatt = tf.reshape(interactor_states_selfatt, [-1, 2 * self.options['rnn_size']])

            with tf.variable_scope('predict_proposal') as proposal_scope:
                proposal_outputs = tf.contrib.layers.fully_connected(
                    inputs=interactor_states_selfatt,
                    num_outputs=self.options['num_anchors'],
                    activation_fn=None
                )

            if self.options['predict_boundary']:
                with tf.variable_scope('predict_boundary') as boundary_scope:
                    boundary_outputs = tf.contrib.layers.fully_connected(
                        inputs=interactor_states_selfatt,
                        num_outputs=1,
                        activation_fn=None
                    )

        proposal_outputs = tf.reshape(proposal_outputs, [-1, self.options['num_anchors']])

        # weighting positive samples
        proposal_weight0 = tf.reshape(proposal_weight[:, 0], [-1, self.options['num_anchors']])
        # weighting negative samples
        proposal_weight1 = tf.reshape(proposal_weight[:, 1], [-1, self.options['num_anchors']])

        # tile
        proposal_weight0 = tf.tile(proposal_weight0, [tf.shape(proposal_outputs)[0], 1])
        proposal_weight1 = tf.tile(proposal_weight1, [tf.shape(proposal_outputs)[0], 1])

        # get weighted sigmoid xentropy loss
        # use tensorflow built-in function
        # weight1 will be always 1.
        proposal = tf.reshape(proposal, [-1, self.options['num_anchors']])
        proposal_loss_term = tf.nn.weighted_cross_entropy_with_logits(
            targets=tf.to_float(proposal), logits=proposal_outputs, pos_weight=proposal_weight0)

        if self.options['anchor_mask']:
            proposal_loss_term = tf.reshape(anchor_mask, [-1, self.options['num_anchors']]) * proposal_loss_term

        proposal_loss_term = tf.reduce_sum(proposal_loss_term, axis=-1)
        proposal_loss_term = tf.reshape(proposal_loss_term, [-1])

        video_feat_mask = tf.reshape(video_feat_mask, [-1])
        proposal_loss = tf.reduce_sum((video_feat_mask * proposal_loss_term)) / tf.to_float(
            tf.reduce_sum(video_feat_mask))

        # summary data, for visualization using Tensorboard
        tf.summary.scalar('proposal_loss', proposal_loss)

        # outputs from proposal module
        outputs['proposal_loss'] = proposal_loss

        if self.options['predict_boundary']:
            boundary_outputs = tf.reshape(boundary_outputs, [-1, 1])
            # weighting positive samples
            boundary_weight0 = tf.reshape(boundary_weight[:, 0], [-1, 1])
            # weighting negative samples
            boundary_weight1 = tf.reshape(boundary_weight[:, 1], [-1, 1])

            # tile
            boundary_weight0 = tf.tile(boundary_weight0, [tf.shape(boundary_outputs)[0], 1])
            boundary_weight1 = tf.tile(boundary_weight1, [tf.shape(boundary_outputs)[0], 1])

            # get weighted sigmoid xentropy loss
            # use tensorflow built-in function
            # weight1 will be always 1.
            boundary = tf.reshape(boundary, [-1, 1])
            boundary_loss_term = tf.nn.weighted_cross_entropy_with_logits(
                targets=tf.to_float(boundary), logits=boundary_outputs, pos_weight=boundary_weight0)
            boundary_loss_term = tf.reduce_sum(boundary_loss_term, axis=-1)
            boundary_loss_term = tf.reshape(boundary_loss_term, [-1])

            boundary_loss = tf.reduce_sum((video_feat_mask * boundary_loss_term)) \
                            / tf.to_float(tf.reduce_sum(video_feat_mask))

            # summary data, for visualization using Tensorboard
            tf.summary.scalar('boundary_loss', boundary_loss)

            # outputs from proposal module
            outputs['boundary_loss'] = boundary_loss

        outputs['loss'] = outputs['proposal_loss']
        if self.options['predict_boundary']:
            outputs['loss'] += self.options['weight_boundary'] * outputs['boundary_loss']

        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        outputs['reg_loss'] = reg_loss

        return inputs, outputs
