import numbers
import os
import random
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.ops import math_ops, random_ops, array_ops
from tensorflow.python.layers import utils
from constants import *


class SeLuModel:

    def __init__(self, sess, name, learn_rate, sequence_length):
        """
        sess = Tensorflow Session
        name = Model name
        learn_rate = Train learning rate
        sequence_length = The number of previous games to look at for each game.
        """
        self.sess = sess
        self.name = name
        self._build_net(learn_rate, sequence_length)
        self.saver = tf.train.Saver()
        self.dirname = os.path.dirname(os.path.realpath(__file__)) + '/saved_graphs'

    def _build_net(self, learn_rate, sequence_length):
        """
        learn_rate = Train learning rate
        sequence_length = The number of previous games to look at for each game.
        Construct a Neural Network.
        Each input data of home team and away team will pass through auto encoder/decoder to 
        reduce the hidden factor to the game among different teams.
        Then, the inputs go through layers of NN to output the predicted result of the game.
        """
        with tf.variable_scope(self.name):
            # input place holders
            self.X_home = tf.placeholder(tf.float32, [None, 11 * sequence_length])
            self.X_away = tf.placeholder(tf.float32, [None, 11 * sequence_length])
            self.Y = tf.placeholder(tf.float32, [None, 2])

            self.keep_prob = tf.placeholder(tf.float32)

            # weights & bias for nn layers
            W1h_encoder = tf.get_variable("W1_home_encoder", shape=[11 * sequence_length, 11],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1h_encoder = tf.Variable(tf.random_normal([11]), name='b1_home_encoder')
            L1h_encoder = self.selu(tf.matmul(self.X_home, W1h_encoder) + b1h_encoder)
            L1h_encoder = self.dropout_selu(L1h_encoder, keep_prob=self.keep_prob)

            W1h_decoder = tf.get_variable("W1_home_decoder", shape=[11, 11 * sequence_length],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1h_decoder = tf.Variable(tf.random_normal([11 * sequence_length]), name='b1_home_decoder')
            L1h_decoder = self.selu(tf.matmul(L1h_encoder, W1h_decoder) + b1h_decoder)
            L1h_decoder = self.dropout_selu(L1h_decoder, keep_prob=self.keep_prob)

            W1a_encoder = tf.get_variable("W1_away_encoder", shape=[11 * sequence_length, 11],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1a_encoder = tf.Variable(tf.random_normal([11]), name='b1_away_encoder')
            L1a_encoder = self.selu(tf.matmul(self.X_away, W1a_encoder) + b1a_encoder)
            L1a_encoder = self.dropout_selu(L1a_encoder, keep_prob=self.keep_prob)

            W1a_decoder = tf.get_variable("W1_away_decoder", shape=[11, 11 * sequence_length],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1a_decoder = tf.Variable(tf.random_normal([11 * sequence_length]), name='b1_away_decoder')
            L1a_decoder = self.selu(tf.matmul(L1a_encoder, W1h_decoder) + b1h_decoder)
            L1a_decoder = self.dropout_selu(L1h_decoder, keep_prob=self.keep_prob)

            M = tf.concat([L1h_decoder, L1a_decoder], 1)

            W2 = tf.get_variable("W2", shape=[11 * sequence_length * 2, 11 * sequence_length],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([11 * sequence_length]), name='b2')
            L2 = self.selu(tf.matmul(M, W2) + b2)
            L2 = self.dropout_selu(L2, keep_prob=self.keep_prob)

            W3 = tf.get_variable("W3", shape=[11 * sequence_length, int(11 * sequence_length / 2)],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([int(11 * sequence_length / 2)]), name='b3')
            L3 = self.selu(tf.matmul(L2, W3) + b3)
            L3 = self.dropout_selu(L3, keep_prob=self.keep_prob)

            W4 = tf.get_variable("W4", shape=[int(11 * sequence_length / 2), 2],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([2]), name='b4')

            self.hypothesis = tf.matmul(L3, W4) + b4 # Probability of winning

            tf.add_to_collection("logit", self.hypothesis)
        # define cost/loss & optimizer
        self.cost = tf.reduce_sum(tf.square(self.hypothesis - self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.cost)

    def get_accuracy(self, x_test_home, x_test_away, y_test, keep_prop=1.0):
        """
        The predictions from x_test_home and x_test_away are mapped to 1 or 0 depending on whether the
        home team wins or not. Then it is compared with y_test which is the ground truth.
        """
        predict = tf.map_fn(
            lambda x: x[0] > x[1],
            self.sess.run(
                self.hypothesis, 
                feed_dict={
                self.X_home: x_test_home, 
                self.X_away: x_test_away, 
                self.Y: y_test, 
                self.keep_prob: keep_prop}
            ), 
            dtype=bool)

        real = tf.map_fn(
            lambda x: x[0] > x[1],
            y_test,
            dtype=bool)

        return self.sess.run(
            tf.divide(
                tf.reduce_sum(tf.cast(tf.equal(predict, real), dtype=tf.int32)), len(y_test)))

    def train(self, x_test_home, x_test_away, y_data, keep_prop):
        """
        x* : The training data
        y_data : The training ground truth
        keep_prop : 1 - drop rate
        """
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X_home: x_test_home, self.X_away: x_test_away, self.Y: y_data, self.keep_prob: keep_prop})

    def save(self):
        """
        Saves the trained data to the given directory.
        """
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.saver.save(self.sess, self.dirname + '/' + self.name + '.ckpt')
        self.saver.export_meta_graph(self.dirname + '/' + self.name + '.meta')

    def predict(self, x_test_home, x_test_away, keep_prop=1.0):
        """
        Predicts the result based on the trained meta data.
        """
        self.saver.restore(self.sess, self.dirname + '/' + self.name + '.ckpt')
        return self.sess.run(self.hypothesis, feed_dict={
            self.X_home: x_test_home, self.X_away: x_test_away, self.keep_prob: keep_prop})
    
    @property
    def get_sess(self):
        return self.sess

    def selu(self, x):
        with ops.name_scope('elu') as scope:
            return SCALE * tf.where(x>=0.0, x, ALPHA * tf.nn.elu(x))

    def dropout_selu(
        self,
        x, 
        keep_prob, 
        alpha= DROP_ALPHA, 
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


class Runner:

    def __init__(self):
        tf.set_random_seed(777)  # reproducibility

    def train_run(self, model, x_train_home, x_train_away, y_train, training_epoch, keep_prob):
        """
        training_epoch : The number of iteration to train the data.
        keep_prob : 1 - drop rate
        Trains the model.
        """
        model.get_sess.run(tf.global_variables_initializer())
        for epoch in range(training_epoch):
            c, _ = model.train(x_train_home, x_train_away, y_train, keep_prob)
            if epoch % 200 == 0:
                print ('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(c))

    def get_accuracy(self, model, x_train_home, x_train_away, y_test):
        return model.get_accuracy(x_train_home, x_train_away, y_test)

    def predict(self, model, x_test_home, x_test_away):
        return model.predict(x_test_home, x_test_away)
