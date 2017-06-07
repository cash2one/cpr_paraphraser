# -*- coding: UTF-8 -*-

import tensorflow as tf
from prepare import Vocabulary
import cPickle as pickle

class CpRModel:

    def __init__(self, batch_size, sample_size, word_vec_size=100, z_size=100, with_rnn=True, alpha=0, rnn_layer=3):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.word_vec_size = word_vec_size
        self.z_size = z_size
        self.alpha = alpha

        self.vocab = pickle.load(open('data/ch_char_vocab.pkl'))
        self.vocab_length = len(self.vocab)

        self.input = tf.placeholder(tf.int32, [self.batch_size, self.sample_size])
        self.output = tf.placeholder(tf.int32, [self.batch_size, self.sample_size])
        
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.anneal_rate = tf.placeholder(tf.float32, [])

        with tf.name_scope('cnn'):
            embedding = tf.get_variable('embedding', [self.vocab_length, self.word_vec_size])
            input_embed = tf.nn.embedding_lookup(embedding, self.input)

            conv1 = self.conv1d('conv1', input_embed, 5, 128)
            conv2 = self.conv1d('conv2', conv1, 5, 256)
            conv3 = self.conv1d('conv3', conv2, 5, 512)

            conv_out = self.full_layer('full1', tf.reshape(conv3, [self.batch_size, 512*self.sample_size]), self.z_size * 2)


        with tf.name_scope('vae'):
            self.mu = conv_out[:, :self.z_size]
            self.lv = conv_out[:, self.z_size:]
            eps = tf.random_normal(self.lv.get_shape().as_list())
            latent = self.mu + eps * tf.exp(self.lv * 0.5)

            self.kl_cost = tf.reduce_mean(-0.5 * (1 + self.lv - self.mu**2 -tf.exp(self.lv)))

        with tf.name_scope('decnn'):
            # latent = tf.reshape(latent, [self.batch_size, self.z_size, 1])
            deconv_in = tf.reshape(self.full_layer('full2', latent, self.sample_size*512), conv3.get_shape().as_list())
            
            deconv1 = self.deconv1d('deconv1', deconv_in, 5, 256)
            deconv2 = self.deconv1d('deconv2', deconv1, 5, 128)
            deconv3 = self.deconv1d('deconv3', deconv2, 5, self.word_vec_size)

        if not with_rnn:
            with tf.name_scope('cost'):
                result_vec = self.full_layer('output', tf.reshape(deconv3, [-1, self.word_vec_size]), self.vocab_length, active_func='softmax')
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= result_vec, labels = self.reshape(tf.one_hot(self.output, depth=self.vocab_length), result_vec.get_shape().as_list())))
                self.cost += self.anneal_rate * self.kl_cost
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            self.result = tf.argmax(tf.reshape(result_vec, [self.batch_size, self.sample_size, self.vocab_length]), axis=2)
        
        else:
            with tf.name_scope('cnn_to_rnn'):
                # mixture = tf.concat([deconv3, input_embed], -1)
                # mixture = tf.reshape(mixture, [-1, self.word_vec_size * 2])
                # rnn_input = self.full_layer('rnn_input_mixture', mixture, self.word_vec_size)
                rnn_input = input_embed
            with tf.name_scope('rnn'):
                self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.word_vec_size) for _ in xrange(rnn_layer)])
                self.init_cell_states = self.cell.zero_state(self.batch_size, dtype=tf.float32)

                output, last_state = tf.nn.dynamic_rnn(self.cell, tf.reshape(rnn_input, [self.batch_size, self.sample_size, self.word_vec_size]), initial_state=self.init_cell_states)
                self.last_state = last_state
                logits = self.full_layer('rnn_to_outsize', tf.reshape(output, [-1, self.word_vec_size]), self.vocab_length, active_func='none')
                
            self.result = tf.argmax(tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.sample_size, -1]), axis=2)

            with tf.name_scope('cost'):
                self.cost = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [logits], 
                    [tf.reshape(self.output, [-1])], 
                    [tf.ones([self.batch_size*self.sample_size], dtype=tf.float32)])
                self.cost = tf.reduce_sum(self.cost) / self.batch_size# + self.anneal_rate * self.kl_cost

                if self.alpha > 0.0:
                    result_cnn = self.full_layer('result_cnn', tf.reshape(deconv3, [-1, self.word_vec_size]), self.vocab_length, active_func='softmax')
                    aux = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= result_cnn, labels = tf.reshape(tf.one_hot(self.output, depth=self.vocab_length), result_cnn.get_shape().as_list())))
                    # self.cost += self.alpha * aux
            with tf.name_scope('optimizer'):
                tvar = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvar), clip_norm=5)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(grads, tvar))


    # convolution
    def conv1d(self, name, input, kernal_size, depth):
        w = tf.get_variable(name = name+'_w', shape = [kernal_size, input.get_shape().as_list()[-1], depth])
        b = tf.get_variable(name = name+'_b', shape = [depth])
        o1 = tf.nn.bias_add(tf.nn.convolution(input, w, padding='SAME'), b)
        o2 = self.batch_normalization(o1)
        return tf.nn.relu(o2, name = name)

    # full connected layer
    def full_layer(self, name, input, out_size, active_func='relu'):
        w = tf.get_variable(name = name+'_w', shape = [input.get_shape().as_list()[-1], out_size])
        b = tf.get_variable(name = name+'_b', shape = [out_size])
        out = tf.matmul(input, w) + b
        if active_func == 'relu':
            return tf.nn.relu(out, name = name)
        elif active_func == 'softmax':
            return tf.nn.softmax(out, name = name)
        elif active_func == 'none':
            return out
        else:
            raise NotImplementedError('not supported function')

    # deconvolution
    def deconv1d(self, name, input, kernal_size, depth):
        in_channel = input.get_shape().as_list()[-1]
        width = input.get_shape().as_list()[-2]
        input = tf.reshape(input, [self.batch_size, 1, width, in_channel])
        w = tf.get_variable(name = name+'_w', shape = [1, kernal_size, depth, in_channel])
        b = tf.get_variable(name = name+'_b', shape = [depth])
        o1 = tf.nn.bias_add(tf.nn.conv2d_transpose(input, w, output_shape=[self.batch_size, 1, width, depth], strides=[1, 1, 1, 1]) ,b)
        o1 = tf.reshape(o1, [self.batch_size, width, depth])
        o2 = self.batch_normalization(o1)
        return tf.nn.relu(o2, name = name)

    # batch normalization
    def batch_normalization(self, input):
        scale = tf.Variable(tf.ones([input.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([input.get_shape()[-1]]))
        mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, scale, 0.001)
