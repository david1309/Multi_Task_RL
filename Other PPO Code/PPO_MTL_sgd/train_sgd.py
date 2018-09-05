"""
PPO: Proximal Policy Optimization for Multi Task Learning done through
     Hard Parameter Sharing Neural Networks Architecture

Multi Task, Multi Domain, and strong code improments done by David Alvarez Charris (david13.ing@gmail.com)

Original code: Patrick Coady (pat-coady.github.io)

----------------------------------------------------------------

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/).
"""
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime, timedelta
import os
import argparse
import signal
import pickle # to save Scaler
import tensorflow as tf # need to be imported here to run summary ops
import sys # used to save command line arguments
import pandas as pd # to modify and overwrite the aux_logs



def init_gym(env_name, task_param = None):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")
        task_param: parameter to create a task specific environment

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = np.prod(env.observation_space.shape) # env.observation_space.shape[0]
    act_dim = np.prod(env.action_space.shape) #env.action_space.shape[0]

    if task_param != None: # TODO: dont hardcore which aspect of env. to modify but rather receive it as input argument
    	env.env.world.gravity = (-task_param,-10)

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, task, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension to a similar range
        task: int indicating which head (task specific hidden layer) of the policy to use
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """

    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], [] # Store trajectory of episode
    done = False
    step = 0.0
    scale, offset = scaler.get() # standardize observations based on runnin mean (offset) and std (scale)
    scale[-1] = 1.0  # don't scale time the "step" additional feature
    offset[-1] = 0.0 # don't offset time the "step" additional feature
    
    # Start Env. Simulation
    while not done:
        if animate: env.render()

        # Modify Observation: Additional feature + standardizing based on running mean / std
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step as additonal feature (accelerates learning)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)

        # Act based on Policy Network
        action = policy.sample(obs, task).reshape((1, -1)).astype(np.float32)


        # self.means + tf.sqrt(tf.exp(self.log_vars))*tf.random_normal(shape=(self.act_dim,))
        feed_dict = {policy.obs_ph: obs, policy.task_ph: task}
        pol_means = policy.sess.run(policy.means, feed_dict=feed_dict)
        pol_logvars = policy.sess.run(policy.log_vars, feed_dict=feed_dict)
        # print("\n\n *** RUN EPISODE *** ")
        # print("Policy Means: {}".format(pol_means))
        # print("Policy Log Vars: {}".format(pol_logvars))
        # print("Sqrt Log Vars: {}".format(  np.sqrt(np.exp(pol_logvars))  ))

        if np.any(np.isnan(action)): print("\n\n NaN Action: {} \n\n".format(action))
        action[np.isnan(action)] = 0 

        actions.append(action)

        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))

        if not isinstance(reward, float): reward = np.float32(reward) # np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    # print("Run Episode Episodes: {}".format(len(observes)))
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, time_steps_batch, task, animate=False):
    """ Run policy and collect data for time steps

    Args:
        env: ai gym environments 
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension to a similar range
        logger: logger object, used to save stats from time steps
        time_steps_batch: total time steps to run
        task: int indicating which head (task specific hidden layer) of the policy to use
        animate: boolean, True uses env.render() method to animate episode


    Returns: list of trajectory dictionaries, list length = number of episodes needed to reach time_steps_batch
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []

    while total_steps < time_steps_batch:

        # Run a single episode
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, task, animate)
        total_steps += observes.shape[0]
        # print("Trajectory Length: {}".format(len(trajectories)))
        # print("Total Steps: {}".format(total_steps))

        if total_steps > time_steps_batch:
            step_diff = total_steps - time_steps_batch 
            observes, actions, rewards, unscaled_obs = \
            observes[0:-step_diff], actions[0:-step_diff], rewards[0:-step_diff], unscaled_obs[0:-step_diff]
            total_steps -=step_diff

        # print("Total Steps: {}".format(total_steps))
        # Store Trajectory
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)

    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:

        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)

        # Include Discounted Reward in Trajectory Dictionary
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func, task):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations and returns predicted state value
        task: int indicating which head (task specific hidden layer) of the Value. func to use

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes, task)
        # print("\n\n *** ADD VALUE *** ")
        # print("Observes Shape: {}".format(observes.shape))        
        # print("Values Shape: {}".format(values.shape))
        # print("Some Values: {}\n\n".format(values[[0,-1]]))


        # Include Value Prediction in Trajectory Dictionary
        trajectory['values'] = values


def add_gae(trajectories, gamma, lamda):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lamda: lambda (see paper).
            lamda=0 : use TD residuals
            lamda=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:

        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']        

        # Compute Temporal Differences = -V(st) + TD_target ;
        # TD_target = rt + gamma*V(st+1) + gamma^2*V(st+2) + ...
        values = trajectory['values']
        # print("\n\n *** ADD GAE *** ")
        # print("Values Shape: {}".format(values.shape))
        # print("values[1:]: {}".format(values[1:]))
        # print("Some Values: {}\n\n".format(values[[0,-1]]))
        #shifting values so that first entry is V(st+1) and last entry has V(sN) = 0
        tds = - values + (rewards + np.append(values[1:] * gamma, 0))

        # Include Advantages in Trajectory Dictionary
        advantages = discount(tds, gamma * lamda)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, time_step):
    """ Log various batch statistics """
    logger.log({'_TimeStep': time_step,

                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew), 

                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_obs': np.mean(observes),

                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),

                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0))               
                })


# Other Useful Methods
def sim_agent(env, policy, task, scaler, num_episodes_sim=1, animate=False, save_video=False, out_dir='./video'):
    """ Simulates trainned agent (Policy) in given environment (env)

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        task: int indicating which head (task specific hidden layer) of the policy to use
        num_episodes_sim (int): number of episodes  to simulate
        animate (bool): determines if video should be rendered in window
        save_video (bool): enables saving video and other stats of simulated episodes

    Returns:
        mean_reward_episodes (double): Mean reward obtained across all episodes
        if save_video=True, stores videos and stats in folder determined by 'out_dir'

    """    
    
    # Monitoring Config
    if save_video: 
        if not os.path.exists(out_dir):  os.makedirs(out_dir) # create directory if it doesn't exist
        env = wrappers.Monitor(env, out_dir, force=True) # Used to save log data and video

    # Simulate each Episode
    episodes_tot_reward = []
    for episode in range(num_episodes_sim):

        obs = env.reset()
        reward_sum = 0
        done = False
        step = 0.0
        scale, offset = scaler.get() # standardize observations 
        scale[-1] = 1.0  # don't scale time the "step" additional feature
        offset[-1] = 0.0 # don't offset time the "step" additional feature

        # Start Env. Simulation
        while not done:
            if animate: env.render()

            # Modify Observation: Additoinal feature + standardizing based on running mean / std
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1) # add time step as additonal feature 
            obs = (obs - offset) * scale  # center and scale observations

            # Act based on Policy Network
            action = policy.sample(obs, task).reshape((1, -1)).astype(np.float32)
            obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
            reward_sum += reward
            step += 1e-3  # increment time step feature

        # Accumualte info for Episode    
        episodes_tot_reward.append(reward_sum)

    # Get Stats over all episodes    
    mean_reward_episodes = np.mean(episodes_tot_reward)    

    return mean_reward_episodes

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


# Main Train Execution
def main(env_name, max_time_steps, time_steps_batch, time_steps_mini_batch, gamma, lamda, kl_targ, clipping_range, pol_loss_type, init_pol_logvar, animate,\
        save_video, save_rate, num_episodes_sim, task_params, task_name, dims_core_hid, dims_head_hid, act_func_name,\
        time_step_to_load, now_to_load):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        max_time_steps: maximum number of time steps to run
        gamma: reward discount factor (float)
        lamda: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        clipping_range: max value to clip the policy gradient ratio
        pol_loss_type: string determining which type of loss to use for the Policy Network
        time_steps_batch: number of time steps per policy training batch
        init_pol_logvar: natural log of initial policy variance
        save_video: Boolean determining if videos of the agent will be saved
        save_rate: Int determining how often to save videos for
        num_episodes_sim: Number of episodes to simulate/save videos for
        task_params: list of parameters to modify each environment for a different task
        task_name: name user assigns to the task being used to modify the environment
    """


    # ****************  Environment Initialization and Paths  ***************
    task_params_str = ''.join(str(e) +', ' for e in task_params)
    num_tasks = len(task_params)
    envs = [None]*num_tasks
    scalers = [None]*num_tasks
    loggers = [None]*num_tasks

    print ("\n\n------ PATHS: ------")
    start_time = datetime.now()
    if time_step_to_load == None: now = start_time.strftime("%b-%d_%H:%M:%S") # If NOT loading from Checkpoint -> used to  create unique directories
    else: 
        assert now_to_load != None,\
            "\n\nWARNING: Date time to load ({}) was not provided. Please provide a valid date time of an experiment".format(now_to_load)
        now = now_to_load
    logs_path = os.path.join('log-files', env_name, task_name, task_params_str, now)

    for task in range(num_tasks):
        # Create task specific environment 
        envs[task], obs_dim, act_dim = init_gym(env_name, task_param = task_params[task])
        obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

        # Create task specific Paths and logger object
        loggers[task] = Logger(logname= [env_name, task_name, task_params_str], now=now, \
                               logname_file= "_{}_{}".format(task_name, task_params[task])) 

        if time_step_to_load == None: # If NOT loading from Checkpoint
            scalers[task] = Scaler(obs_dim)            

            # Auxiliary saver (becase logger sometimes fails or takes to much time)
            with open(logs_path + '/aux_{}_{}.txt'.format(task_name, task_params[task]), 'w') as f: 
                f.write("_TimeStep" + "  " + "_MeanReward")

        
    aigym_path= os.path.join('./videos', env_name, task_name, task_params_str, now) # videos folders 
    agent_path = os.path.join('agents', env_name , task_name, task_params_str, now) # agent / policy folders  
    if time_step_to_load == None: # If NOT loading from Checkpoint 
        os.makedirs(agent_path)
        with open(agent_path + '/commandline_args.txt', 'w') as f: f.write(' '.join(sys.argv[1:]))  # save commandline command
        with open(logs_path + '/commandline_args.txt', 'w') as f: f.write(' '.join(sys.argv[1:]))  # save commandline command

    print("\nPath for Saved Videos : {}".format(aigym_path)) 
    print("Path for Saved Agents: {}\n".format(agent_path))    


    # ****************  Initialize Policy, Value Networks and Scaler  ***************
    print ("\n\n------ NEURAL NETWORKS: ------")
    dims_core_hid.insert(0, obs_dim) # Modify dims list to have the size of the layer 'n-1' at position '0'
    dims_head_hid.insert(0, dims_head_hid[-1])
    
    val_func = NNValueFunction(obs_dim, dims_core_hid, dims_head_hid, num_tasks, time_steps_mini_batch)
    policy = Policy(obs_dim, act_dim, dims_core_hid, dims_head_hid, num_tasks, time_steps_mini_batch, pol_loss_type = pol_loss_type)

    # Load from Checkpoint:
    # Validate intented time step to load OR get last time step number if no target time step was provided 
    if time_step_to_load != None:
        load_agent_path = agent_path # agent / policy folders
        saved_ep_list = [file.split(".")[0].split("_")[-1] for file in os.listdir(load_agent_path) if "policy" in file]

        if time_step_to_load == -1: # Get last saved time step
            time_step_to_load = sorted([int(ep_string) for ep_string in saved_ep_list])[-1]

        else: # Validate if time_step_to_load was indeed saved 
            assert str(time_step_to_load) in saved_ep_list,\
            "\n\nWARNING: Time Step you want to load ({}) was not stored during trainning".format(time_step_to_load)

        # Load Policy Network's Ops and Variables & Load Scaler Object
        policy.tf_saver.restore(policy.sess, "{}/policy_ep_{}".format(load_agent_path, time_step_to_load)) 
        val_func.tf_saver.restore(val_func.sess, "{}/val_func_ep_{}".format(load_agent_path, time_step_to_load))
        scalers = pickle.load(open("{}/scalers_ep_{}.p".format(load_agent_path, time_step_to_load), 'rb'))         
        print("\n\n ---- CHECKPOINT LOAD:  Time Step Loaded **{}**".format(time_step_to_load))

        # Delete extra epochs that where logged to the auxiliary logs
        for task in range(num_tasks):
            aux_log_path = logs_path + '/aux_{}_{}.txt'.format(task_name, task_params[task])
            aux_log = pd.read_table(aux_log_path, delim_whitespace=True)
            idx_to_cut = aux_log.index[aux_log["_TimeStep"] == time_step_to_load ].tolist()[0]
            aux_log[0:idx_to_cut+1].to_csv(aux_log_path, header=True, index=False, sep=' ', mode='w') # overwrite trimmed aux_log


    # If NOT loading from Checkpoint: run some time steps to initialize scalers and create Tensor board dirs
    elif time_step_to_load == None:
        for task in range(num_tasks): run_policy(envs[task], policy, scalers[task], loggers[task], time_steps_batch=int(time_steps_batch/3), task=task)  

        # Tensor Board writer
        os.makedirs(agent_path + '/tensor_board/policy')
        os.makedirs(agent_path + '/tensor_board/valFunc')

    tb_pol_writer = tf.summary.FileWriter(agent_path + '/tensor_board/policy', graph=policy.g)
    tb_val_writer = tf.summary.FileWriter(agent_path + '/tensor_board/valFunc', graph=val_func.g)


    # ****************  Start Training  ***************
    print ("\n\n------ TRAINNING: ------")
    animate = True if animate == "True" else False
    save_video = True if save_video == "True" else False
    saver_offset = save_rate
    killer = GracefulKiller()

    if time_step_to_load == None: time_step = 0
    else: time_step = time_step_to_load
    
    # Time steps are counted across all tasks i.e. N time steps indicates each tasks has been runned for N times
    while time_step < max_time_steps and not killer.kill_now:

        # ****************  Obtain data (train set)  ***************         
        observes_all = [None]*num_tasks
        actions_all = [None]*num_tasks
        advantages_all = [None]*num_tasks
        disc_sum_rew_all = [None]*num_tasks

        time_step += time_steps_batch
        for task in range(num_tasks):

            # Obtain 'time_steps_batch' trajectories and add additional intermediate calculations
            trajectories = run_policy(envs[task],policy, scalers[task], loggers[task],time_steps_batch=time_steps_batch,task=task,animate=animate)

            add_value(trajectories, val_func, task)  # add estimated values to trajectories
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lamda)  # calculate advantage

            # Concatenate all time steps into single NumPy arrays
            observes_all[task], actions_all[task], advantages_all[task], disc_sum_rew_all[task] = build_train_set(trajectories)

            # print("Observes Shape: {}".format(observes_all[task].shape))
            # print("Actions Shape: {}\n\n".format(actions_all[task].shape))
            # print("Advantage Shape: {}\n\n".format(advantages_all[task].shape))

            # Logging Stats
            log_batch_stats(observes_all[task], actions_all[task], advantages_all[task], disc_sum_rew_all[task], \
                            loggers[task], time_step)

        # ****************  Update Policy and Value Networks  ***************
        # print ("*************************************")
        for task in range(num_tasks):
            pol_summary = policy.update(task, observes_all[task], actions_all[task], advantages_all[task], loggers[task])  # update policy
            val_summary = val_func.fit(task, observes_all[task], disc_sum_rew_all[task], loggers[task])  # update value function
            # Auxiliary saver (because logger sometimes fails or takes to much time)
            with open(logs_path + '/aux_{}_{}.txt'.format(task_name, task_params[task]), 'a') as f: 
                f.write("\n" + str(loggers[task].log_entry['_TimeStep']) + "  " + str(loggers[task].log_entry['_MeanReward'])) 
            loggers[task].write(display=False)  # write logger results to file and stdout

            tb_pol_writer.add_summary(pol_summary, global_step=time_step)
            tb_val_writer.add_summary(val_summary, global_step=time_step)


        # ****************  Storing NN and Videos  ***************
        # Store Policy, Value Network and Scaler: every 'save_rate'  or in first/last time steps
        if time_step >= saver_offset or time_step >=max_time_steps or time_step <=time_steps_batch*1.5 or killer.kill_now:
        # TODO: Make saving agent/video a method so that it can be called in killer.kill_now 
            saver_offset += save_rate
            policy.tf_saver.save(policy.sess, "{}/policy_ep_{}".format(agent_path, time_step)) # Save Policy Network
            val_func.tf_saver.save(val_func.sess, "{}/val_func_ep_{}".format(agent_path, time_step)) # Save Value Network
            pickle.dump(scalers, open("{}/scalers_ep_{}.p".format(agent_path, time_step), 'wb'))            
            print ("---- Saved Agent at Time Step {} ----".format(time_step))

            # Save video of current agent/policy
            if save_video: 
                print ("---- Saving Video at Time Step {} ----".format(time_step))
                for task in range(num_tasks):
                    _ = sim_agent(envs[task], policy, task, scalers[task], num_episodes_sim, save_video=True, 
                                    out_dir=aigym_path + "/vid_ts_{}/{}_{}".format(time_step, task_name, task_params[task]))
                    envs[task].close() # closes window open by monitor wrapper
                    envs[task], _, _ = init_gym(env_name,task_param=task_params[task]) # Recreate env as it was killed
            print("\n\n")

            # If Ctrl + C is Pressed, ask user if Trainning shall be terminated
            if killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    break
                killer.kill_now = False


    # ****************  Terminate Variables  **************
    for task in range(num_tasks):
        envs[task].close()
        loggers[task].close()
    policy.close_sess()
    val_func.close_sess()

    # Save elapsed time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    timedelta(0, 8, 562000)
    delta_time = divmod(elapsed_time.days * 86400 + elapsed_time.seconds, 60)
    delta_str = "Elapsed Time: {} min {} seconds".format(delta_time[0], delta_time[1])
    # save elapsed time, 'a' to append not overwrite
    with open(agent_path + '/commandline_args.txt', 'a') as f: f.write('\n\n' + delta_str) 
    with open(logs_path + '/commandline_args.txt', 'a') as f: f.write('\n\n' + delta_str)  


# Example of how to Train Agent:
# python train_sgd.py BipedalWalker-v2 --task_params 1 2 3 --task_name Wind -dcore 64 64 -dhead 64 --pol_loss_type kl --max_time_steps 10000 --time_steps_batch 1024 --time_steps_mini_batch G6 --save_video False --save_rate 10000

# Example how to reload a agent from Checkpoint:
# python train_sgd.py BipedalWalker-v2 --time_step_to_load -1 --now_to_load Jul-25_23:42:17 --task_params 1 2 3 --task_name Wind -dcore 64 32 16 -dhead 128 64 16 --pol_loss_type both --max_time_steps 100000 --time_steps_batch 1000 --save_video False --save_rate 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')

    parser.add_argument('-maxT', '--max_time_steps', type=int, help='Total Number of time steps [10000000]',
                        default=10000000)
    parser.add_argument('-b', '--time_steps_batch', type=int,
                help='Number of time steps used for each training batch [5000]', default=4096) # 32000
    parser.add_argument('-minib', '--time_steps_mini_batch', type=int,
                help='Number of time steps used for each mini batch [256]', default=256) # 5000


    parser.add_argument('-g', '--gamma', type=float, help='Discount factor [0.995]', default=0.995)
    parser.add_argument('-l', '--lamda', type=float, 
                        help='Lambda for Generalized Advantage Estimation [0.98]', default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value [0.01]', default=0.01)
    parser.add_argument('-clip', '--clipping_range', type=float, help='clipping range of Policy gradient ratio [0.2]', default=0.2)
    parser.add_argument('-ltype', '--pol_loss_type', type=str, 
                         help='Determines which loss to use in the Policy Network (kl, clip, both)[kl]',  default="kl")
    parser.add_argument('-v', '--init_pol_logvar', type=float,
                        help='Initial policy log-variance [1.0]',
                        default=1.0)
    parser.add_argument('-a', '--animate', type=str,  help='Determines if simulation will be rendered [False]',
                        default="False")
    parser.add_argument('-svid', '--save_video', type=str,  
                        help='Determines if videos will be saved every "save_rate" time steps [False]', default="False")
    parser.add_argument('-svidr', '--save_rate', type=int,  
                        help='Every how many Time steps to save things (e.g. models, videos) [100000]', default=100000)
    parser.add_argument('-nsim', '--num_episodes_sim', type=int,  
                        help='Nuber of Episodes to simulate / save videos for [1]', default=1)

    # MTL Params
    parser.add_argument('-tp', '--task_params', nargs='+', type=float, help='List of parameters for each task [None]', default=None)

    parser.add_argument('-tn', '--task_name', type=str, help='Name of task being solved [default_task]', default="default_task")

    parser.add_argument('-dcore', '--dims_core_hid', nargs='+', type=int, help='List specifying the dimension of each CORE'\
                        'hidden layer [64, 64]', default=[64,64])

    parser.add_argument('-dhead', '--dims_head_hid', nargs='+', type=int, help='List specifying the dimension of each HEAD'\
                        'hidden layer [64]', default=[64])

    parser.add_argument('-act', '--act_func_name', type=str, help='Name of activation function to use (tan, relu, lrelu) [tan]',\
                         default="tan")

    # Reload from checkpoint
    parser.add_argument('-tsload', '--time_step_to_load', type=int,
                        help='Time step of saved agent to load; default is NOT to load anything [None]', default=None)

    parser.add_argument('-nowload', '--now_to_load', type=str, help='Date time of agent to load (e.g.Jun-13_03:02:11)', default=None)


    args = parser.parse_args()


    # Train Policy and Value Network with parsed parameters
    main(**vars(args))
