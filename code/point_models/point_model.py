import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np


'''
Point Based Models: GRU4Rec
'''
class PointBaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_seq_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum], name='user_seq_ph')
            self.user_seq_length_ph = tf.placeholder(tf.int32, [None,], name='user_seq_length_ph')
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
        
        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
            self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
            self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
            self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

            self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_seq_ph)
            self.user_seq = tf.reshape(self.user_seq, [-1, max_time_len, item_fnum * eb_dim])
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

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
    
    def build_bprloss(self):
        self.pred_reshape = tf.reshape(self.y_pred, [-1, 1 + 1])
        self.pred_pos = self.pred_reshape[:, 0]
        self.pred_neg = self.pred_reshape[:, 1]
        self.loss = tf.reduce_mean(tf.log(tf.nn.sigmoid(self.pred_pos - self.pred_neg)))
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.user_seq_ph : batch_data[0],
                self.user_seq_length_ph : batch_data[1],
                self.target_user_ph : batch_data[2],
                self.target_item_ph : batch_data[3],
                self.label_ph : batch_data[4],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.user_seq_ph : batch_data[0],
                self.user_seq_length_ph : batch_data[1],
                self.target_user_ph : batch_data[2],
                self.target_item_ph : batch_data[3],
                self.label_ph : batch_data[4],
                self.reg_lambda : reg_lambda,
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

class GRU4Rec(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        super(GRU4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)

        # GRU
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_length_ph, dtype=tf.float32, scope='gru1')
        
        inp = tf.concat([user_seq_final_state, self.target_item, self.target_user], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

class Caser(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        super(Caser, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
        
        with tf.name_scope('user_seq_cnn'):
            # horizontal filters
            filters_user = 4
            h_kernel_size_user = [10, eb_dim * item_fnum]
            v_kernel_size_user = [self.user_seq.get_shape().as_list()[1], 1]

            self.user_seq = tf.expand_dims(self.user_seq, 3)
            conv1 = tf.layers.conv2d(self.user_seq, filters_user, h_kernel_size_user)
            max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
            user_hori_out = tf.reshape(max1, [-1, filters_user]) #[B, F]

            # vertical
            conv2 = tf.layers.conv2d(self.user_seq, filters_user, v_kernel_size_user)
            conv2 = tf.reshape(conv2, [-1, eb_dim * item_fnum, filters_user])
            user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, eb_dim * item_fnum])

            inp = tf.concat([user_hori_out, user_vert_out, self.target_item, self.target_user], axis=1)

        # fully connected layer
        self.build_fc_net(inp)
        self.build_logloss()


class SVDpp(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        super(SVDpp, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)

        with tf.name_scope('user_feature_rep'):
            self.user_feat_w_list = []
            for i in range(user_fnum):
                self.user_feat_w_list.append(tf.get_variable('user_feat_w_%d'%i, [], initializer=tf.truncated_normal_initializer))
            self.target_user_rep = self.target_user[:, :eb_dim] * self.user_feat_w_list[0]
            for i in range(1, user_fnum):
                self.target_user_rep += self.target_user[:,i*eb_dim:(i+1)*eb_dim] * self.user_feat_w_list[i]

        with tf.name_scope('item_feature_rep'):
            self.item_feat_w_list = []
            for i in range(item_fnum):
                self.item_feat_w_list.append(tf.get_variable('item_feat_w_%d'%i, [], initializer=tf.truncated_normal_initializer))
            self.target_item_rep = self.target_item[:, :eb_dim] * self.item_feat_w_list[0]
            self.user_seq_rep = self.user_seq[:, :, :eb_dim] * self.item_feat_w_list[0]
            for i in range(1, item_fnum):
                self.target_item_rep += self.target_item[:,i*eb_dim:(i+1)*eb_dim] * self.item_feat_w_list[i]
                self.user_seq_rep += self.user_seq[:, :, i*eb_dim:(i+1)*eb_dim] * self.item_feat_w_list[i]
        
        # prediction
        self.user_seq_mask = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32), 2)
        self.user_seq_rep = self.user_seq_rep * self.user_seq_mask
        self.neighbor = tf.reduce_sum(self.user_seq_rep, axis=1)
        self.norm_neighbor = self.neighbor / tf.sqrt(tf.expand_dims(tf.norm(self.user_seq_rep, 1, (1, 2)), 1))

        self.latent_score = tf.reduce_sum(self.target_item_rep * (self.target_user_rep + self.norm_neighbor), 1)
        self.y_pred = tf.nn.sigmoid(self.latent_score)
        
        self.build_logloss()

class DELF(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        super(DELF, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)

        with tf.name_scope('inputs_item'):
            self.item_seq_ph = tf.placeholder(tf.int32, [None, max_time_len, user_fnum], name='item_seq_ph')
            self.item_seq_length_ph = tf.placeholder(tf.int32, [None,], name='item_seq_length_ph')
        
        with tf.name_scope('embedding_item'):
            self.item_seq = tf.nn.embedding_lookup(self.emb_mtx, self.item_seq_ph)
            self.item_seq = tf.reshape(self.item_seq, [-1, max_time_len, user_fnum * eb_dim])
        
        # sequence mask for user seq and item seq
        self.user_seq_mask = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32), 2)
        self.item_seq_mask = tf.expand_dims(tf.sequence_mask(self.item_seq_length_ph, max_time_len, dtype=tf.float32), 2)

        self.user_rep_item_base = self.attention(self.user_seq, self.user_seq, self.target_item, self.user_seq_mask)
        self.item_rep_user_base = self.attention(self.item_seq, self.item_seq, self.target_user, self.item_seq_mask)
    
        # pairwise interaction layer
        self.inter1 = tf.concat([self.target_user, self.target_item], axis=1)
        self.inter2 = tf.concat([self.user_rep_item_base, self.item_rep_user_base], axis=1)
        self.inter3 = tf.concat([self.target_user, self.item_rep_user_base], axis=1)
        self.inter4 = tf.concat([self.target_item, self.user_rep_item_base], axis=1)

        # fusion layer
        f1 = self.fusion_mlp(self.inter1)
        f2 = self.fusion_mlp(self.inter2)
        f3 = self.fusion_mlp(self.inter3)
        f4 = self.fusion_mlp(self.inter4)

        f = f1 + f2 + f3 + f4
        self.y_pred = tf.reshape(tf.layers.dense(f, 1, activation=tf.sigmoid), [-1,])
        self.build_logloss()        

    def fusion_mlp(self, inp):
        fc1 = tf.layers.dense(inp, 20, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 4, activation=tf.nn.relu)
        return fc2

    def attention(self, key, value, query, mask):
        # key, value: [B, T, D], query: [B, D], mask: [B, T, 1]
        _, max_len, k_dim = key.get_shape().as_list()
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1]) # [B, T, D]
        key = tf.layers.dense(key, k_dim, activation=tf.nn.tanh)

        paddings = (1 - mask) * (-2 ** 32 + 1)
        attention = tf.nn.softmax(tf.expand_dims(tf.reduce_sum(queries * key * mask, axis=2), 2) + paddings, dim=1)
        output = tf.reduce_sum(value * attention, axis=1)
        return output
    
    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.user_seq_ph : batch_data[0],
                self.user_seq_length_ph : batch_data[1],
                self.item_seq_ph : batch_data[2],
                self.item_seq_length_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.user_seq_ph : batch_data[0],
                self.user_seq_length_ph : batch_data[1],
                self.item_seq_ph : batch_data[2],
                self.item_seq_length_ph : batch_data[3],
                self.target_user_ph : batch_data[4],
                self.target_item_ph : batch_data[5],
                self.label_ph : batch_data[6],
                self.reg_lambda : reg_lambda,
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss

class DEEMS(DELF):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        super(DEEMS, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)

        # GRU
        with tf.name_scope('rnn'):
            _, user_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_seq, 
                                                        sequence_length=self.user_seq_length_ph, dtype=tf.float32, scope='gru1')
            _, item_seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_seq, 
                                                        sequence_length=self.item_seq_length_ph, dtype=tf.float32, scope='gru2')
        
        inp_user = tf.concat([user_seq_final_state, self.target_user], axis=1)
        inp_item = tf.concat([item_seq_final_state, self.target_item], axis=1)
        
        self.y_pred_user = self.build_fc_net(inp_user)
        self.y_pred_item = self.build_fc_net(inp_item)
        self.y_pred = 0.5 * (self.y_pred_user + self.y_pred_item)

        self.build_logloss()
        self.loss += 0.05 * tf.reduce_sum((self.y_pred_item - self.y_pred_user) ** 2)
    
    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp)
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu)
        dp1 = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu)
        dp2 = tf.nn.dropout(fc2, self.keep_prob)
        fc3 = tf.layers.dense(dp2, 1, activation=None)
        # output
        y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1,])
        return y_pred
    
class SASRec(PointBaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum):
        super(SASRec, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
        self.user_seq = self.multihead_attention(self.normalize(self.user_seq), self.user_seq)

        self.mask = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.mask_1 = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph - 1, max_time_len, dtype=tf.float32), axis=-1)
        self.get_mask = self.mask - self.mask_1
        self.seq_rep = self.user_seq * self.mask
        self.final_pred_rep = tf.reduce_sum(self.user_seq * self.mask, axis=1)

        # pos and neg for sequence
        self.pos = self.user_seq[:, 1:, :]
        self.neg = self.user_seq[:, 2:, :]
        
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, 1), [1, max_time_len, 1])

        self.pos_seq_rep = tf.concat([self.seq_rep[:, 1:, :], self.pos, self.target_user_t[:, 1:, :]], axis=2)
        self.neg_seq_rep = tf.concat([self.seq_rep[:, 2:, :], self.neg, self.target_user_t[:, 2:, :]], axis=2)
        
        self.preds_pos = self.build_fc_net(self.pos_seq_rep)
        self.preds_neg = self.build_fc_net(self.neg_seq_rep)
        self.label_pos = tf.ones_like(self.preds_pos)
        self.label_neg = tf.zeros_like(self.preds_neg)

        self.loss = tf.losses.log_loss(self.label_pos, self.preds_pos) + tf.losses.log_loss(self.label_neg, self.preds_neg)

        # prediction for target user and item
        inp = tf.concat([self.final_pred_rep, self.target_item, self.target_user], axis=1)
        self.y_pred = self.build_fc_net(inp)
        self.y_pred = tf.reshape(self.y_pred, [-1,])
        
        self.loss += tf.losses.log_loss(self.label_ph, self.y_pred)
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_fc_net(self, inp):
        with tf.variable_scope('prediction_layer'):
            fc1 = tf.layers.dense(inp, 200, activation=tf.nn.relu, name='fc1', reuse=tf.AUTO_REUSE)
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2', reuse=tf.AUTO_REUSE)
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 1, activation=tf.sigmoid, name='fc3', reuse=tf.AUTO_REUSE)
        return fc3

    def multihead_attention(self,
                            queries, 
                            keys, 
                            num_units=None, 
                            num_heads=2, 
                            scope="multihead_attention", 
                            reuse=None):
        '''Applies multihead attention.
        
        Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
        A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]
            
            # Linear projections
            # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
    
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
            
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
            
            # Dropouts
            outputs = tf.nn.dropout(outputs, self.keep_prob)
                
            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                
            # Residual connection
            outputs += queries
                
            # Normalize
            #outputs = normalize(outputs) # (N, T_q, C)
    
        return outputs

    def normalize(self,
              inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
        '''Applies layer normalization.
        
        Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        
        Returns:
        A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs
