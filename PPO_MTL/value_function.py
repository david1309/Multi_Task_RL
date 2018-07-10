"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()

        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()

        # overrides the TF's "default_graph" to define within it operations and tensors
        with self.g.as_default():

            # PLACEHOLDERS
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            self.task_ph = tf.placeholder(tf.int32, (), 'task_valfunc')

            # NN Architecture
            hid1_size = 64
            hid2_size = 64
            hid3_task_size = 64
            
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined


            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1_valfunc")
            
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2_valfunc")


            # Task specific Hidden Layers ("Heads")
            h3_t1 = tf.layers.dense(out, hid3_task_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3_t1_valfunc")

            h3_t2 = tf.layers.dense(out, hid3_task_size, tf.tanh,
                          kernel_initializer=tf.random_normal_initializer(
                              stddev=np.sqrt(1 / hid2_size)), name="h3_t2_valfunc")

            h3_t3 = tf.layers.dense(out, hid3_task_size, tf.tanh,
                  kernel_initializer=tf.random_normal_initializer(
                      stddev=np.sqrt(1 / hid2_size)), name="h3_t3_valfunc")

            # Task specific Output Layer (Linear layer: activation=None --> linear)
            f1_m = lambda: tf.layers.dense(h3_t1, 1, kernel_initializer=tf.random_normal_initializer(
                                            stddev=np.sqrt(1 / hid3_task_size)), name="means_t1_valfunc")

            f2_m = lambda: tf.layers.dense(h3_t2, 1, kernel_initializer=tf.random_normal_initializer(
                                            stddev=np.sqrt(1 / hid3_task_size)), name="means_t2_valfunc")

            f3_m = lambda: tf.layers.dense(h3_t3, 1, kernel_initializer=tf.random_normal_initializer(
                                            stddev=np.sqrt(1 / hid3_task_size)))#, name="means_t3_valfunc")

            out = tf.case({tf.equal(self.task_ph, 0): f1_m, tf.equal(self.task_ph, 1): f2_m, tf.equal(self.task_ph, 2): f3_m},\
                                 name= "case_means_valfunc")

            self.out = tf.squeeze(out)

            print('\nValue Network Params -- h1: {}, h2: {}, h3_task: {}, lr: {:.3g}'
              .format(hid1_size, hid2_size, hid3_task_size, self.lr))

            # LOSS && OPTIMIZER
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

            # Session and Variables Initializers
            self.var_init = tf.global_variables_initializer()


            # Need to create the Saver() Op over here (and not in train.py) since
            # for it to work it requires variables to already exist ... max_to_keep= None
            # to store all the checkpoints that I want without deleting old checkpoints
            # TODO: None makes the checkpoint text file to only have the last checkpoint,
            # you can still restore all other checkpoints but I am not sure how this affects 
            # the restorer
            self.tf_saver = tf.train.Saver(max_to_keep=None) # Enables to safe/restore TF graphs

    def fit(self, task, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x, task)  # get NN prediction to check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)

        # Set Replay Buffer
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])

        # Store current data batch for next call to fit
        self.replay_buffer_x = x 
        self.replay_buffer_y = y

        # Train Network
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train) # Remove wrong correlations assumtions
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end],
                             self.task_ph: task
                             }

                self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        y_hat = self.predict(x, task)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x, task):
        """ Predict method """
        feed_dict = {self.obs_ph: x, self.task_ph: task}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.var_init)# var_init def. in _build_graph as global_variables_initializer

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
