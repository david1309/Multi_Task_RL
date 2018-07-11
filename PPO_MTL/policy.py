"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import tensorflow as tf


class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, clipping_range=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence ("distace") between pi_old and pi_new
            hid1_mult: multiplying this by obs_dim the size of the 1st hidden layer is determined
            policy_logvar: natural log of initial policy variance
        """
        self.beta = [1.0]*3  # Beta: gain of D_KL divergance loss term
        self.eta = 50  # Eta: gain of the loss term controling that D_KL doesn't exceed KL_targ (hinge loss)
        self.kl_targ = kl_targ  # Target value for the KL Divergance between pi_old and pi_new
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.epochs = 20 # Trainning Epochs
        self.lr = None
        self.lr_multiplier = [1.0]*3  # dynamically adjust lr based on value of D_KL
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping_range = clipping_range
        
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()

        # overrides the TF's "default_graph" to define within it operations and tensors
        with self.g.as_default():
            self._placeholders() # defines all input placeholders 
            self._policy_nn() # defines architecture of Policy Network and its outputs (means & log_vars)
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
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

        # indicator of which task is being optimized
        self.task_ph = tf.placeholder(tf.int32, (), 'task')

        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')

        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')

        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'alpha')

        # log_vars and means with pi_old (previous step's policy parameters), used
        # to construct old distribution and compute D_KL w/ new distribution.:
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Policy Network: policy function approximation 

        Policy parametrizes the means and variances of a Gaussian distribution over each action dimension.
        NN only outputs means of distro. based on observation, while (log) variances are computed by an
        additional Trainable variables ("log_vars")
        """
        # Hidden sizes  
        hid1_size = 64  
        hid2_size = 64 
        hid3_task_size = 64

        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 MAGIC NUMBER empirically determined

        # Common Hidden Layers
        h1 = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        h2 = tf.layers.dense(h1, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")

        # Task specific Hidden Layers ("Heads")
        h3_t1 = tf.layers.dense(h2, hid3_task_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3_t1")

        h3_t2 = tf.layers.dense(h2, hid3_task_size, tf.tanh,
                      kernel_initializer=tf.random_normal_initializer(
                          stddev=np.sqrt(1 / hid2_size)), name="h3_t2")

        h3_t3 = tf.layers.dense(h2, hid3_task_size, tf.tanh,
              kernel_initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(1 / hid2_size)), name="h3_t3")

        # Task specific Output Layer (Linear layer: activation=None --> linear)
        f1_m = lambda: tf.layers.dense(h3_t1, self.act_dim,
                                         kernel_initializer=tf.random_normal_initializer(
                                             stddev=np.sqrt(1 / hid3_task_size)), name="means_t1")

        f2_m = lambda: tf.layers.dense(h3_t2, self.act_dim,
                                 kernel_initializer=tf.random_normal_initializer(
                                     stddev=np.sqrt(1 / hid3_task_size)), name="means_t2")

        f3_m = lambda: tf.layers.dense(h3_t3, self.act_dim,
                         kernel_initializer=tf.random_normal_initializer(
                             stddev=np.sqrt(1 / hid3_task_size)))#, name="mt3")

        self.means = tf.case({tf.equal(self.task_ph, 0): f1_m, tf.equal(self.task_ph, 1): f2_m, tf.equal(self.task_ph, 2): f3_m},\
                             name= "case_means")


        # Task specific Log Variances Variables
        # logvar_speed is used to 'fool' gradient descent into making faster updates to log-variances, 
        #since the model is now predicting 'logvar_speed' variances for each action dimension
        logvar_speed = (10 * hid2_size) // 48 # MAGIC NUMBER

        # log_vars is a trainnable variable predicting logvar_speed variances (rows) for each action dimension (columns)
        log_vars_t1 = tf.get_variable('logvars_t1', (logvar_speed, self.act_dim), tf.float32, tf.constant_initializer(0.0))
        log_vars_t2 = tf.get_variable('logvars_t2', (logvar_speed, self.act_dim), tf.float32, tf.constant_initializer(0.0))
        log_vars_t3 = tf.get_variable('logvars_t3', (logvar_speed, self.act_dim), tf.float32, tf.constant_initializer(0.0))

        f1_v = lambda: tf.reduce_sum(log_vars_t1, axis=0) + self.policy_logvar
        f2_v = lambda: tf.reduce_sum(log_vars_t2, axis=0) + self.policy_logvar
        f3_v = lambda: tf.reduce_sum(log_vars_t3, axis=0) + self.policy_logvar

        self.log_vars = tf.case({tf.equal(self.task_ph, 0): f1_v, tf.equal(self.task_ph, 1): f2_v, tf.equal(self.task_ph, 2): f3_v},\
                        name= "case_logvars")


        print('\nPolicy Network Params -- h1: {}, h2: {}, h3_task: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_task_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate new and old log probabilities of actions based on the log PDF parametrized by NN.
        The NN is parametrizing a Gaussian Distro N(actions; means, var) ; var = exp(log_vars) 
        Therefore:
        Log(N) = -1/2*log(2*pi*var) -1/2* ((actions-means)^2 / var) 
        Log(N) ~ -1/2*log(var) - 1/2*((actions-means)^2 / var)
        """

        # New Log PDF
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph-self.means) / tf.exp(self.log_vars), axis=1)
        self.logp = logp

        # Old Log PDF
        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from Multivariate Gaussian (parametrized by NN) based on Standard Multivar Normal : 
        action ~ N(actions; means, var) = means + sqrt(var)*N(0,1) """

        # self.sampled_act = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,)))
        self.sampled_act = self.means+tf.sqrt(tf.exp(self.log_vars))*tf.random_normal(shape=(self.act_dim,))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new), making sure the Policy update is not to large
            3) Hinge loss on [D_KL - kl_targ]^2, kicks in when D_KL exceeds the target value

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        if self.clipping_range is not None:
            print('Loss: setting up loss with clipping objective')
            pg_ratio = tf.exp(self.logp - self.logp_old)
            clipped_pg_ratio = tf.clip_by_value(pg_ratio,1-self.clipping_range[0],1+self.clipping_range[1])
            surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio, self.advantages_ph * clipped_pg_ratio)
            self.loss = -tf.reduce_mean(surrogate_loss)

        else:
            print('Loss: setting up loss with KL penalty')
            loss1 = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.loss = loss1 + loss2 + loss3

        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.var_init)# var_init def. in _build_graph as global_variables_initializer

    def sample(self, obs, task):
        """Draw action sample from policy distribution given observation"""
        feed_dict = {self.obs_ph: obs, self.task_ph: task}

        # sampled_act --> Op. defined in _sample
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, task, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:            
            task: interget indicating which task data is being used
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """

        # Initialize Feed dicts and variables
        feed_dict = {self.task_ph: task, # Used to determine which "head" to optimize
                     self.obs_ph: observes, # Used for feedforward pass of Policy Network
                     self.act_ph: actions, # Used to compute log Probability of actions
                     self.advantages_ph: advantages, # Used to compute loss 1 - Policy Gradient loss
                     self.beta_ph: self.beta[task], # Loss gains and learning rate
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier[task]}

        # Find out what are the initial means and logvars predicted by NN and trainnable variable
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np # Used to compute old logp_old and KL
        feed_dict[self.old_means_ph] = old_means_np # Used to compute old logp_old and KL
        loss, kl, entropy = 0, 0, 0


        # Train NN
        for e in range(self.epochs):            

            # Run Optimizer
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)

            # Once Optimized, compute loss, KL Divergance and Entropy
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)

            # Stop Optimizing (early stopping) if Policy update steps are too large ! (D_KL diverges)
            if kl > self.kl_targ * 4:  break

        # Update Learning Rate and KL loss 2 (beta from loss 2)
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  #  Too big steps --> Increase beta to reach D_KL target
            self.beta[task] = np.minimum(35, 1.5 * self.beta[task]) 
            if self.beta[task] > 30 and self.lr_multiplier[task] > 0.1:
                self.lr_multiplier[task] /= 1.5

        elif kl < self.kl_targ / 2: # Too small steps --> Decrease beta to reach D_KL target
            self.beta[task] = np.maximum(1 / 35, self.beta[task] / 1.5) 
            if self.beta[task] < (1 / 30) and self.lr_multiplier[task] < 10:
                self.lr_multiplier[task] *= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta[task],
                    '_lr_multiplier': self.lr_multiplier[task]})


    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
