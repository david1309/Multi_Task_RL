"""
State - Value function Neural Network 
Architecture is built for Multi Task Learning , where each of the NN "heads" 
predicts the value function for a task 

Adapted by David Alvarez Charris (david13.ing@gmail.com)

Original code: Patrick Coady (pat-coady.github.io)
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn_ops import leaky_relu
from tensorflow.python.ops.nn_ops import relu
from collections import OrderedDict
from sklearn.utils import shuffle


class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, dims_core_hid, dims_head_hid, num_tasks, act_func_name = "tan"):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
        """
        self.replay_buffer_x = [None]*num_tasks
        self.replay_buffer_y = [None]*num_tasks
        self.obs_dim = obs_dim
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()

        # NN architecture parameters
        self.dims_core_hid = dims_core_hid
        self.dims_head_hid = dims_head_hid

        # NN Setup
        act_dict = {"tan": tf.tanh, "relu": relu, "lrelu": leaky_relu}
        self.act_func = act_dict[act_func_name]
        self.num_tasks = num_tasks

        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()

        # overrides the TF's "default_graph" to define within it operations and tensors
        with self.g.as_default():
            self._placeholders() # defines all input placeholders 
            self._value_nn() # defines architecture of Value Network and its outputs (values)
            self._loss_train_op()
            self.summary_op = tf.summary.merge_all() # for TensorBoard visualization
            self.var_init = tf.global_variables_initializer()

            # Need to create the Saver() Op over here (and not in train.py) since
            # for it to work it requires variables to already exist ... max_to_keep= None
            # to store all the checkpoints that I want without deleting old checkpoints
            # TODO: None makes the checkpoint text file to only have the last checkpoint,
            # you can still restore all other checkpoints but I am not sure how this affects 
            # the restorer
            self.tf_saver = tf.train.Saver(max_to_keep=None) # Enables to safe/restore TF graphs

    def _placeholders(self):
        """ Input placeholders"""

        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
        self.dis_rew_ph = tf.placeholder(tf.float32, (None,), 'dis_rew_valfunc')
        self.task_ph = tf.placeholder(tf.int32, (), 'task_valfunc')

    def case_fn(self):
        """"
        Helper function used to return on the "function (fn) " parameter required by the tf.case Op.
        """        
        self.num_call_case_fn += 1

        # TF internal workings make the first function passed to the tf.case be called twice (see internal code of.case, tf.cond)
        # This "if" forces the first function to be returned for the first 2 __calls__        
        if self.num_call_case_fn != 2: self.which_case +=1

        return self.case_layers_list[self.which_case-1]

    def _value_nn(self):
        """ Value Network: alue function approximation 

        parametrizes a NN that given a set of observations "s" it predicts what is the value of such state V(s)
        """
        
        # General NN Setup
        self.lr = 1e-2 / np.sqrt(self.dims_core_hid[2]) # TODO: 1e-2  MAGIC NUMBER empirically determined
        self.case_layers_list = [] # stores the last layer of each head which will be passed to the tf.case
        self.which_case = 0 # keeps track of which function (fn) of tf.case has been called (only used in case_fn)
        self.num_call_case_fn = 0 # keeps track of how many times case_fn has been called (only used in case_fn)        
        cond_list = [] # stores the conditional statement (predicate) for the tf.case


        with tf.variable_scope('value_NN'):    


            # ****** Value prediction
            # Core Block: Common Hidden Layers
            with tf.variable_scope('Core'):
                h_core = self.obs_ph

                for hid in range(len(self.dims_core_hid)-1):
                    h_core = tf.layers.dense(h_core, self.dims_core_hid[hid+1], self.act_func,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.dims_core_hid[hid])), name="h{}_core".format(hid+1))


            # Heads Blocks: Task specific Hidden Layers 
            with tf.variable_scope('Heads'):   

                for head in range(self.num_tasks):
                    with tf.variable_scope('head_{}'.format(head+1)):
                        h_head = h_core

                        # Create all hidden layers of current head
                        for hid in range(len(self.dims_head_hid)-1):
                            h_head = tf.layers.dense(h_head, self.dims_head_hid[hid+1], self.act_func,
                                      kernel_initializer=tf.random_normal_initializer(
                                          stddev=np.sqrt(1 / self.dims_head_hid[hid])), name="h{}_head_{}".format(hid+1, head+1))                            

                        # Final dense layer for current head
                        dense = tf.layers.dense(h_head, 1,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/self.dims_head_hid[-1])), 
                                name="dense_head_{}".format(head+1)) 

                        # store it to use it in switch case
                        self.case_layers_list.append(dense) 

                    cond_list.append(tf.equal(self.task_ph, head))

                # Automatically built case
                vals_case_dict = OrderedDict(zip(cond_list, [self.case_fn] * len(cond_list)))
                vals_cased = tf.case(vals_case_dict, name= "case_values")
                self.vals_pred = tf.squeeze(vals_cased)


        print('\nValue Network Params -- core_hidden: {}, head_hidden: {}, act.func: {}, lr: {:.3g}'
              .format(self.dims_core_hid[1:], self.dims_head_hid[1:], self.act_func.__name__, self.lr))

    def _loss_train_op(self):
        """
        Defines MSE loss between predicted values (vals_pred) and the ground truth values (disconunted rewards)
        """
        self.loss = tf.reduce_mean(tf.square(self.vals_pred - self.dis_rew_ph))  # squared loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)  

        # Summaries for TensorBoard
        tf.summary.scalar('valFunc_loss', self.loss) # Summaries for TensorBoard

    def fit(self, task, observes, dis_rew, logger):
        """
        Update value function based on observations and disccounted rewards (which are the target values)

        Args:            
            task: interger indicating which task data is being used
            observes: observations, shape = (N, obs_dim)
            dis_rew: disccounted sum of rewards (target value), shape = (N, act_dim)
            logger: Logger object, see utils.py
        """
        
        # Initialize training variables
        num_batches = max(observes.shape[0] // 256, 1) # TODO: Harcoded max num batches
        batch_size = observes.shape[0] // num_batches

        # Fin out initial variance
        vals_pred = self.predict(observes, task)  # get NN prediction to check explained variance prior to update
        old_exp_var = 1 - np.var(dis_rew - vals_pred)/np.var(dis_rew)

        # Set Replay Buffer
        if self.replay_buffer_x[task] is None:
            x_train, y_train = observes, dis_rew
        else:
            x_train = np.concatenate([observes, self.replay_buffer_x[task]])
            y_train = np.concatenate([dis_rew, self.replay_buffer_y[task]])

        # Store current data batch for next call to fit (buffers)
        self.replay_buffer_x[task] = observes 
        self.replay_buffer_y[task] = dis_rew

        # Train Network
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train) # Remove wrong correlations assumptions

            for j in range(num_batches):                
                # Configure Feed Dict
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.dis_rew_ph: y_train[start:end],
                             self.task_ph: task
                             }

                # Run Optimizer
                self.sess.run(self.train_op, feed_dict=feed_dict)

        # Logging: once Optimized, compute new MSE 
        vals_pred = self.predict(observes, task)
        loss = np.mean(np.square(vals_pred - dis_rew)) # explained variance after update
        exp_var = 1 - np.var(dis_rew - vals_pred) / np.var(dis_rew)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

        # Update TensorBoard through its op.
        summary_updated = self.sess.run(self.summary_op, feed_dict)

        return summary_updated

    def predict(self, obs, task):
        """ Given an observation predict its value  """
        feed_dict = {self.obs_ph: obs, self.task_ph: task} 

        return np.squeeze(  self.sess.run(self.vals_pred , feed_dict=feed_dict)  )

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.var_init)# var_init def. in _build_graph as global_variables_initializer

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
