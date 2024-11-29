import tensorflow as tf
import time
import config
import os
import pickle
import numpy as np
import tqdm
import utils
from sklearn.metrics import precision_score, f1_score
from differential_privacy.privacy_accountant.tf import accountant
from differential_privacy.optimizer import base_dp_optimizer_MomentAcc
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
np_config.enable_numpy_behavior()
sess = tf.compat.v1.Session()
change=0.0
_,_,_,_,egs=utils.read_graph(config.train_file)
accountant = accountant.GaussianMomentsAccountant(config.n_node)
global_step = tf.Variable(0, name="global_step", trainable=False)
def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.compat.v1.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.compat.v1.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class Discriminator():
    def __init__(self, n_node, node_emd_init, config,clipping,sigma):
            self.n_node = n_node
            self.sigma=sigma
            self.clipping=clipping
            self.emd_dim = config.n_emb
            self.node_emd_init = node_emd_init

        #with tf.compat.v1.variable_scope('disciminator',reuse=tf.compat.v1.AUTO_REUSE):
            if node_emd_init:
                self.node_embedding_matrix = tf.compat.v1.get_variable(name='dis_node_embedding',
                                                                   shape=self.node_emd_init.shape,
                                                                   initializer=tf.constant_initializer(
                                                                       self.node_emd_init),
                                                                   trainable=True)
            else:
                self.node_embedding_matrix = tf.compat.v1.get_variable(name='dis_node_embedding',
                                                                   shape=[2, self.n_node, self.emd_dim],
                                                                   initializer=tf.keras.initializers.GlorotNormal(),
                                                                   trainable=True)

            self.pos_node_ids = tf.compat.v1.placeholder(tf.int32, shape=[None])
            self.pos_node_neighbor_ids = tf.compat.v1.placeholder(tf.int32, shape=[None])
            self.fake_node_embedding = tf.compat.v1.placeholder(tf.float32, shape=[2, 2, None, self.emd_dim])
            _node_embedding_matrix = []

            #with tf.GradientTape() as disc_tape:

            for i in range(2):
                    _node_embedding_matrix.append(tf.reshape(tf.nn.embedding_lookup(self.node_embedding_matrix, tf.constant([i])), [-1, self.emd_dim]))

            self.pos_node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[0], self.pos_node_ids)
            self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[1], self.pos_node_neighbor_ids)
            pos_score = tf.matmul(self.pos_node_embedding, self.pos_node_neighbor_embedding, transpose_b=True)
            self.pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score))
            _neg_loss = [0, 0, 0, 0]
            node_id = [self.pos_node_ids, self.pos_node_neighbor_ids]
            for i in range(2):
                    for j in range(2):
                        node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[j], node_id[i])
                        _fake_node_embedding = tf.reshape(tf.nn.embedding_lookup(self.fake_node_embedding, tf.constant([i])),[2, -1, self.emd_dim])
                        _fake_node_embedding = tf.reshape(tf.nn.embedding_lookup(_fake_node_embedding, tf.constant([j])),[-1, self.emd_dim])
                        neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True)
                        _neg_loss[i * 2 + j] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_score), logits=neg_score))


            self.neg_loss = _neg_loss
            self.loss = self.pos_loss + self.neg_loss[0] * config.neg_weight[0] + self.neg_loss[1] * config.neg_weight[1] + \
            self.neg_loss[2] * config.neg_weight[2] + self.neg_loss[3] * config.neg_weight[3]
        #optimizer = tf.compat.v1.train.AdamOptimizer(config.lr_dis,config.epsilon)
            optimizer=base_dp_optimizer_MomentAcc.DPGradientDescentGaussianOptimizer(
                    accountant,
                    l2_norm_clip=self.clipping,
                    noise_multiplier=self.sigma,
                    num_microbatches=config.num_microbatches,
                    learning_rate=config.lr_dis)
            disc_vars=[self.node_embedding_matrix]
            print("disc_vars\n",disc_vars)
            self.d_updates=optimizer.minimize(self.loss,global_step=global_step,var_list=disc_vars)























