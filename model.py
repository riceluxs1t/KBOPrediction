# Taken from SELU implementation from https://github.com/bioinf-jku/SNNs/blob/master/selu.py
import tensorflow as tf
import random
'''
Tensorflow Implementation of the Scaled ELU function and Dropout
Taken from https://github.com/hunkim/DeepLearningZeroToAll
'''
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.ops import math_ops, random_ops, array_ops
from tensorflow.python.layers import utils

from constants.py import *


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def dropout_selu(
    x, 
    keep_prob, 
    alpha= -1.7580993408473766, 
    fixedPointMean=0.0, 
    fixedPointVar=1.0, 
    noise_shape=None, 
    seed=None, 
    name=None, 
    training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 40]) # TODO : Format data correctly.
            self.Y = tf.placeholder(tf.float32, [None, 1]) # Win : 1 Lose : 0

            # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # weights & bias for nn layers
            W1 = tf.get_variable("W1", shape=[40, 512],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([512]))
            L1 = selu(tf.matmul(X, W1) + b1)
            L1 = dropout_selu(L1, keep_prob=keep_prob)

            W2 = tf.get_variable("W2", shape=[512, 512],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([512]))
            L2 = selu(tf.matmul(L1, W2) + b2)
            L2 = dropout_selu(L2, keep_prob=keep_prob)

            W3 = tf.get_variable("W3", shape=[512, 512],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([512]))
            L3 = selu(tf.matmul(L2, W3) + b3)
            L3 = dropout_selu(L3, keep_prob=keep_prob)

            W4 = tf.get_variable("W4", shape=[512, 512],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([512]))
            L4 = selu(tf.matmul(L3, W4) + b4)
            L4 = dropout_selu(L4, keep_prob=keep_prob)

            W5 = tf.get_variable("W5", shape=[512, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([1]))
            self.hypothesis = tf.matmul(L4, W5) + b5 # Probability of winning

        # define cost/loss & optimizer
        self.cost = tf.reduce_sum(tf.square(self.hypothesis - Y)) # TODO: Need to find a more suiting convex function.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        # Test model and check accuracy
        correct_prediction = tf.equal(
            tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=KEEP_RATE):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
    
    @property
    def get_sess(self):
        return self.sess

class Runner:

    def __init__(self):
        tf.set_random_seed(777)  # reproducibility

    def train_run(self, model, x_train, y_train):
        model.get_sess.run(tf.global_variables_initializer())
        for epoch in xrange(TRAINING_EPOCHS):
            feed_dict = {X: x_train, Y: y_train, keep_prob: KEEP_RATE}
            c, _ = model.train([cost, optimizer], feed_dict=feed_dict)

            print ('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))

        #TODO: Save frozen graph of the tf after training.

    def get_accuracy(self, model, x_test, y_test):
        return model.get_accuracy(x_test, y_test)

    def predict(self, model, x_test):
        return model.predict(x_test)

