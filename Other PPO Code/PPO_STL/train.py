"""
PPO: Proximal Policy Optimization for Single Task Learning

Adapted by David Alvarez Charris (david13.ing@gmail.com)

Original code: Patrick Coady (pat-coady.github.io)
--------------------------------------------------------------

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
from datetime import datetime
import os
import argparse
import signal
import pickle # to save Scaler


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


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
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

        # Modify Observation: Additoinal feature + standardizing based on running mean / std
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step as additonal feature (accelerates learning)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)

        # Act based on Policy Network
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))

        if not isinstance(reward, float): reward = np.float32(reward) # np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes, animate=False):
    """ Run policy and collect data for episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run
        animate: boolean, True uses env.render() method to animate episode

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):

        # Run a single episode
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, animate)
        total_steps += observes.shape[0]

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


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)

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


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_Episode': episode,

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
def sim_agent(env, policy,scaler, num_episodes_sim=1, animate=False, save_video=False, out_dir='./video'):
    """ Simulates trainned agent (Policy) in given environment (env)

    Args:
        env: ai gym environment
        policy: policy object with sample() method
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
            action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
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
def main(env_name, num_episodes, gamma, lamda, kl_targ, batch_size, hid1_mult, init_pol_logvar, animate,\
        save_video, num_episodes_sim, task_params, task_name):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lamda: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        init_pol_logvar: natural log of initial policy variance
        save_video: Boolean determining if videos of the agent will be saved
        num_episodes_sim: Number of episodes to simulate/save videos for
        task_params: list of parameters to modify each environment for a different task
        task_name: name user assigns to the task being used to modify the environment
    """

    # ****************  Environment Initialization and Paths  ***************
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

    # Paths
    print ("\n\n---- PATHS: ----")
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now) # logger object
    aigym_path= os.path.join('./videos', env_name, task_name, now) # videos folders 
    agent_path = os.path.join('agents', env_name, now) # agent / policy folders
    os.makedirs(agent_path)
    print("Path for Saved Videos: {}".format(aigym_path))
    print("Path for Saved Agents: {}\n".format(agent_path))
    
    # Initialize Policy, Value Networks and Scaler
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, init_pol_logvar)
    run_policy(env, policy, scaler, logger, episodes=5) # run some episodes to initialize scaler  

    # Start Trainning
    animate = True if animate == "True" else False
    save_video = True if save_video == "True" else False
    saver_perc = int(num_episodes*0.02) # determinines when the agent and video should be saved
    saver_offset = saver_perc
    killer = GracefulKiller()
    episode = 0
    
    while episode < num_episodes:

        # Obtain 'batch_size' trajectories and add additional intermediate calculations
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size, animate=animate)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lamda)  # calculate advantage

        # Concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)

        # Logging Stats
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)

        # Update Policy and Value Networks
        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout

        # Store Policy, Value Network and Scaler: every 20% of total episodes or in first/last episode
        if episode >= saver_offset or episode >=num_episodes or episode <=batch_size or killer.kill_now:
        # TODO: Make saving agent/video a method so that it can be called in killer.kill_now 
            saver_offset += saver_perc
            policy.tf_saver.save(policy.sess, "{}/policy_ep_{}".format(agent_path, episode)) # Save Policy Network
            val_func.tf_saver.save(val_func.sess, "{}/val_func_ep_{}".format(agent_path, episode)) # Save Value Network
            pickle.dump(scaler, open("{}/scaler_ep_{}.p".format(agent_path, episode), 'wb'))            
            print ("---- Saved Agent at Episode {} ----".format(episode))

            # Save video of current agent/policy
            if save_video: 
                print ("---- Saving Video at Episode {} ----".format(episode))
                _ = sim_agent(env, policy, scaler, num_episodes_sim, save_video=True, 
                                out_dir=aigym_path+"/vid_ep_{}/{}_{}".format(episode, task_name, task))
                env.close() # closes window open by monitor wrapper
                env, _, _ = init_gym(env_name) # Recreate env as it is killed when saving videos
            print("\n\n")

            # If Ctrl + C is Pressed, ask user if Trainning shall be terminated
            if killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    break
                killer.kill_now = False
    
    # Terminate Sessions
    env.close()
    logger.close()
    policy.close_sess()
    val_func.close_sess()


# Example of how to Train Agent:
# python train.py Pendulum-v0 --num_episodes 1000 --batch_size 20
# python train.py BipedalWalker-v2 --num_episodes 10000 --batch_size 20
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Total Number of episodes [1000]',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor [0.995]', default=0.995)
    parser.add_argument('-l', '--lamda', type=float, 
                        help='Lambda for Generalized Advantage Estimation [0.98]', default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value [0.003]', default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes used for each training batch [20]', default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(hidden1 size = hid1_mult*observation dimension) [10]', default=10)
    parser.add_argument('-v', '--init_pol_logvar', type=float,
                        help='Initial policy log-variance [1.0]',
                        default=-1.0)
    parser.add_argument('-a', '--animate', type=str,  help='Determines if simulation will be rendered [False]',
                        default="False")
    parser.add_argument('-svid', '--save_video', type=str,  
                        help='Determines if videos will be saved every 20% episodes [False]',default="False")
    parser.add_argument('-nsim', '--num_episodes_sim', type=int,  
                        help='Nuber of Episodes to simulate / save videos for [2]', default=1)

    args = parser.parse_args()

    # Train Policy and Value Network with parsed parameters
    main(**vars(args))
