# Taken from SELU implementation from https://github.com/bioinf-jku/SNNs/blob/master/selu.py
import os

import tensorflow as tf
import random

'''
Tensorflow Implementation of the Scaled ELU function and Dropout
Taken from https://github.com/hunkim/DeepLearningZeroToAll
'''
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


DIRNAME = os.path.dirname(os.path.realpath(__file__)) + '/saved_graphs'

"""
Basic LSTM Cell
"""


def lstm_cell(hidden_size):
    return rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)


class RNN:
    def __init__(self, sess, name, learn_rate, hidden_size, sequence_length, stack_num):
        self.sess = sess
        self.name = name
        self._build_net(learn_rate, hidden_size, sequence_length, stack_num)
        self.saver = tf.train.Saver()

    def _build_net(self, learn_rate, hidden_size, sequence_length, stack_num):
        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, sequence_length, 11])
            self.Y = tf.placeholder(tf.float32, [None, 1])  # Team score

            multi_cells = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(stack_num)], state_is_tuple=True)

            outputs, _states = tf.nn.dynamic_rnn(
                multi_cells,
                self.X,
                dtype=tf.float32)

            self.hypothesis = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)

            tf.add_to_collection("logit", self.hypothesis)
        # define cost/loss & optimizer
        # TODO Having two outputs? What would be the best cost function? or Is this cost suitable
        self.cost = tf.reduce_sum(tf.square(self.hypothesis - self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.cost)

        # Test model and check accuracy
        # predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, self.Y), dtype=tf.float32))
        self.accuracy = tf.reduce_mean(tf.abs(self.hypothesis - self.Y))

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data})

    def save(self):
        if not os.path.exists(DIRNAME):
            os.makedirs(DIRNAME)
        self.saver.save(self.sess, DIRNAME + '/' + self.name + '.ckpt')
        self.saver.export_meta_graph(DIRNAME + '/' + self.name + '.meta')

    def predict(self, x_test):
        self.saver.restore(self.sess, DIRNAME + '/' + self.name + '.ckpt')
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test})

    @property
    def get_sess(self):
        return self.sess

    @property
    def get_logit(self):
        return self.hypothesis


class Runner:
    def __init__(self):
        tf.set_random_seed(777)  # reproducibility

    def train_run(self, model, x_train, y_train, training_epoch):
        model.get_sess.run(tf.global_variables_initializer())
        for epoch in range(training_epoch):
            c, _ = model.train(x_train, y_train)
            if epoch % 200 == 0:
                print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(c))

    def get_accuracy(self, model, x_test, y_test):
        return model.get_accuracy(x_test, y_test)

    def predict(self, model, x_test):
        return model.predict(x_test)
