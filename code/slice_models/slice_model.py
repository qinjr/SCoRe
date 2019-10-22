import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

TRAIN_NEG_SAMPLE_NUM = 1
TEST_NEG_SAMPLE_NUM = 9

'''
Slice Based Models: RRN, GCMC
'''
class SliceBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        # reset graph
        tf.reset_default_graph()

        self.obj_per_time_slice = obj_per_time_slice

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_1hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, item_fnum], name='user_1hop_ph')
            self.user_2hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, user_fnum], name='user_2hop_ph')
            
            self.item_1hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, user_fnum], name='item_1hop_ph')
            self.item_2hop_ph = tf.placeholder(tf.int32, [None, max_time_len, self.obj_per_time_slice, item_fnum], name='item_2hop_ph')
            
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')
            self.length_ph = tf.placeholder(tf.int32, [None,], name='length_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
            # neg sample num
            self.neg_sample_num_reshape = tf.placeholder(tf.int32, [2,])

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
            self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

            B, T, K, _ = self.user_1hop_ph.get_shape().as_list()
            # user interaction set and co-interaction set
            self.user_1hop = tf.nn.embedding_lookup(self.emb_mtx, self.user_1hop_ph)
            self.user_1hop = tf.reshape(self.user_1hop, [-1, T, K, item_fnum * eb_dim])
            self.user_2hop = tf.nn.embedding_lookup(self.emb_mtx, self.user_2hop_ph)
            self.user_2hop = tf.reshape(self.user_2hop, [-1, T, K, user_fnum * eb_dim])

            # item interaction set and co-interaction set
            self.item_1hop = tf.nn.embedding_lookup(self.emb_mtx, self.item_1hop_ph)
            self.item_1hop = tf.reshape(self.item_1hop, [-1, T, K, user_fnum * eb_dim])
            self.item_2hop = tf.nn.embedding_lookup(self.emb_mtx, self.item_2hop_ph)
            self.item_2hop = tf.reshape(self.item_2hop, [-1, T, K, item_fnum * eb_dim])
            
            # target item and target user
            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
            self.target_item = tf.reshape(self.target_item, [-1, item_fnum * eb_dim])
            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            self.target_user = tf.reshape(self.target_user, [-1, user_fnum * eb_dim])
            
    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1,])
    
    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_bprloss(self):
        self.y_pred_reshape = tf.reshape(self.y_pred, self.neg_sample_num_reshape)
        self.y_pred_pos = tf.tile(tf.expand_dims(self.y_pred_reshape[:, 0], 1), [1, self.neg_sample_num_reshape[1] - 1])
        self.y_pred_neg = self.y_pred_reshape[:, 1:]
        self.loss = tf.sigmoid(self.y_pred_pos - self.y_pred_neg)
        self.loss = -tf.log(tf.clip_by_value(self.loss, 1e-10, 1))
        self.loss = tf.reduce_mean(self.loss)
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
    
    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.user_1hop_ph : batch_data[0],
                self.user_2hop_ph : batch_data[1],
                self.item_1hop_ph : batch_data[2],
                self.item_2hop_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.length_ph : batch_data[7],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8,
                self.neg_sample_num_reshape : [-1, 1 + TRAIN_NEG_SAMPLE_NUM]
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.user_1hop_ph : batch_data[0],
                self.user_2hop_ph : batch_data[1],
                self.item_1hop_ph : batch_data[2],
                self.item_2hop_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.length_ph : batch_data[7],
                self.reg_lambda : reg_lambda,
                self.keep_prob : 1.,
                self.neg_sample_num_reshape : [-1, 1 + TEST_NEG_SAMPLE_NUM]
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

class RRN(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(RRN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        user_side = tf.reduce_sum(self.user_1hop, axis=2)
        item_side = tf.reduce_sum(self.item_1hop, axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')

        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()
        # self.build_bprloss()


class GCMC(SliceBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(GCMC, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)

        user_1hop_li = tf.layers.dense(self.user_1hop, self.user_1hop.get_shape().as_list()[-1], activation=None, use_bias=False)
        item_1hop_li = tf.layers.dense(self.item_1hop, self.item_1hop.get_shape().as_list()[-1], activation=None, use_bias=False)

        # sum pooling
        user_1hop_seq_sum = tf.nn.relu(tf.reduce_sum(user_1hop_li, axis=2))
        item_1hop_seq_sum = tf.nn.relu(tf.reduce_sum(item_1hop_li, axis=2))

        user_1hop_seq = tf.layers.dense(user_1hop_seq_sum, user_1hop_seq_sum.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)
        item_1hop_seq = tf.layers.dense(item_1hop_seq_sum, item_1hop_seq_sum.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=False)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_1hop_seq, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru1')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_1hop_seq, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru2')
        # pred
        self.y_pred_pos = tf.exp(tf.reduce_sum(tf.layers.dense(item_side_final_state, hidden_size, use_bias=False) * user_side_final_state, axis=1))
        self.y_pred_neg = tf.exp(tf.reduce_sum(tf.layers.dense(item_side_final_state, hidden_size, use_bias=False) * user_side_final_state, axis=1))
        self.y_pred = self.y_pred_pos / (self.y_pred_pos + self.y_pred_neg)

        self.build_logloss()
        # self.build_bprloss()


