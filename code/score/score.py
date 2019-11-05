import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

TRAIN_NEG_SAMPLE_NUM = 1
TEST_NEG_SAMPLE_NUM = 99

'''
SCOREBASE Models: SCORE
'''
class SCOREBASE(object):
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
            # regularization term
            self.reg_lambda = tf.placeholder(tf.float32, [], name='lambda')
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
    
    def build_bprloss(self):
        self.y_pred_reshape = tf.reshape(self.y_pred, self.neg_sample_num_reshape)
        self.y_pred_pos = tf.tile(tf.expand_dims(self.y_pred_reshape[:, 0], 1), [1, self.neg_sample_num_reshape[1] - 1])
        self.y_pred_neg = self.y_pred_reshape[:, 1:]
        self.loss = tf.sigmoid(self.y_pred_pos - self.y_pred_neg)
        self.loss = -tf.log(tf.clip_by_value(self.loss, 1e-10, 1))
        self.loss = tf.reduce_mean(self.loss)

    def build_l2norm(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

    def build_train_step(self):
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
    
    def lrelu(self, x, alpha=0.2):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    
    def co_attention(self, seq1, seq2, target_t):
        _, T, K, __ = seq1.get_shape().as_list()
        # tile target_t and seq1/seq2
        target = tf.expand_dims(tf.expand_dims(target_t, 2), 2)
        target = tf.tile(target, [1, 1, K, K, 1])
        seq1_tile = tf.tile(tf.expand_dims(seq1, 3), [1, 1, 1, K, 1])
        seq2_tile = tf.tile(tf.expand_dims(seq2, 3), [1, 1, 1, K, 1])

        inp = tf.concat([target, seq1_tile, seq2_tile], axis=-1)
        relateness = tf.layers.dense(inp, 1, activation=tf.nn.relu, use_bias=True) #[B, T, K, K, 1]
        atten = tf.nn.softmax(tf.reshape(relateness, [-1, T, K * K]))
        atten = tf.reshape(atten, [-1, T, K, K])
        seq1_weights = tf.expand_dims(tf.reduce_sum(atten, axis=3), axis=3)
        seq2_weights = tf.expand_dims(tf.reduce_sum(atten, axis=2), axis=3)

        seq1_result = tf.reduce_sum(seq1 * seq1_weights, axis=2)
        seq2_result = tf.reduce_sum(seq2 * seq2_weights, axis=2)

        relateness = tf.reshape(relateness, [-1, T, K, K])
        atten_info = tf.concat([tf.reduce_sum(relateness, axis=3), tf.reduce_sum(relateness, axis=2)], axis=2)
        return seq1_result, seq2_result, atten_info
    
    def attention(self, key, value, query, mask):
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1]) # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis = -1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None) #[B, T, 1]

        mask = tf.equal(mask, tf.ones_like(mask)) #[B, T, 1]
        paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.reshape(tf.where(mask, fc3, paddings), [-1, max_len])) #[B, T]
        
        # atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        # atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return tf.expand_dims(score, 2)

class SCORE(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(SCORE, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        self.mask = tf.expand_dims(tf.sequence_mask(self.length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])

        user_1hop_seq, item_2hop_seq, atten_info_item = self.co_attention(self.user_1hop, self.item_2hop, self.target_item_t)
        user_2hop_seq, item_1hop_seq, atten_info_user = self.co_attention(self.user_2hop, self.item_1hop, self.target_user_t)
        atten_info = tf.concat([atten_info_item, atten_info_user], axis=2)
        
        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')
        
        self.query = tf.concat([self.target_user, self.target_item], axis=1)
        self.key = tf.concat([user_side_rep_t, item_side_rep_t, atten_info], axis=2)
        self.value = tf.concat([user_side_rep_t, item_side_rep_t], axis=2)
        score = self.attention(self.key, self.value, self.query, self.mask)
        user_final_state = tf.reduce_sum(user_side_rep_t * score, axis=1)
        item_final_state = tf.reduce_sum(item_side_rep_t * score, axis=1)
        
        inp = tf.concat([user_final_state, item_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.build_train_step()
    
# Ablation models: RIA, RCA, SCORE_USER, SCORE_ITEM
class RIA(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(RIA, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        self.mask = tf.expand_dims(tf.sequence_mask(self.length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])

        user_1hop_seq, item_2hop_seq, atten_info_item = self.co_attention(self.user_1hop, self.item_2hop, self.target_item_t)
        user_2hop_seq, item_1hop_seq, atten_info_user = self.co_attention(self.user_2hop, self.item_1hop, self.target_user_t)
        atten_info = atten_info_item + atten_info_user
        
        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            _, user_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            _, item_side_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')
        
        inp = tf.concat([user_side_final_state, item_side_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.build_train_step()

class RCA(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(RCA, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        self.mask = tf.expand_dims(tf.sequence_mask(self.length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])

        user_1hop_seq = tf.reduce_sum(self.user_1hop, axis=2)
        user_2hop_seq = tf.reduce_sum(self.user_2hop, axis=2)
        item_1hop_seq = tf.reduce_sum(self.item_1hop, axis=2)
        item_2hop_seq = tf.reduce_sum(self.item_2hop, axis=2)
        
        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')
        
        self.query = tf.concat([self.target_user, self.target_item], axis=1)
        self.key = tf.concat([user_side_rep_t, item_side_rep_t], axis=2)
        self.value = tf.concat([user_side_rep_t, item_side_rep_t], axis=2)
        score = self.attention(self.key, self.value, self.query, self.mask)
        user_final_state = tf.reduce_sum(user_side_rep_t * score, axis=1)
        item_final_state = tf.reduce_sum(item_side_rep_t * score, axis=1)
        
        inp = tf.concat([user_final_state, item_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.build_train_step()
        
class SCORE_USER(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(SCORE_USER, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        self.mask = tf.expand_dims(tf.sequence_mask(self.length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])

        user_1hop_seq, item_2hop_seq, atten_info_item = self.co_attention(self.user_1hop, self.item_2hop, self.target_item_t)
        user_2hop_seq, item_1hop_seq, atten_info_user = self.co_attention(self.user_2hop, self.item_1hop, self.target_user_t)
        atten_info = tf.concat([atten_info_item, atten_info_user], axis=2)
        
        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')
        
        self.query = tf.concat([self.target_user, self.target_item], axis=1)
        self.key = tf.concat([user_side_rep_t, item_side_rep_t, atten_info], axis=2)
        self.value = tf.concat([user_side_rep_t, item_side_rep_t], axis=2)
        score = self.attention(self.key, self.value, self.query, self.mask)
        user_final_state = tf.reduce_sum(user_side_rep_t * score, axis=1)

        inp = tf.concat([user_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.build_train_step()

class SCORE_ITEM(SCOREBASE):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, 
                obj_per_time_slice, user_fnum, item_fnum):
        super(SCORE_ITEM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
        self.mask = tf.expand_dims(tf.sequence_mask(self.length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, axis=1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, axis=1), [1, max_time_len, 1])

        user_1hop_seq, item_2hop_seq, atten_info_item = self.co_attention(self.user_1hop, self.item_2hop, self.target_item_t)
        user_2hop_seq, item_1hop_seq, atten_info_user = self.co_attention(self.user_2hop, self.item_1hop, self.target_user_t)
        atten_info = tf.concat([atten_info_item, atten_info_user], axis=2)
        
        user_side = tf.concat([user_1hop_seq, user_2hop_seq], axis=2)
        item_side = tf.concat([item_1hop_seq, item_2hop_seq], axis=2)

        # RNN
        with tf.name_scope('rnn'):
            user_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=user_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_user_side')
            item_side_rep_t, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=item_side, 
                                                        sequence_length=self.length_ph, dtype=tf.float32, scope='gru_item_side')
        
        self.query = tf.concat([self.target_user, self.target_item], axis=1)
        self.key = tf.concat([user_side_rep_t, item_side_rep_t, atten_info], axis=2)
        self.value = tf.concat([user_side_rep_t, item_side_rep_t], axis=2)
        score = self.attention(self.key, self.value, self.query, self.mask)
        item_final_state = tf.reduce_sum(item_side_rep_t * score, axis=1)

        inp = tf.concat([item_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        # build loss
        self.build_logloss()
        self.build_l2norm()
        self.build_train_step()