"""
PPO: Proximal Policy Optimization

Code used to load and test in an environment an already trained
agent (Policy)

Heavily inspired by Patrick Coady (pat-coady.github.io) code

"""
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from utils import Logger, Scaler
import os
import argparse
import pickle # to load Scaler


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.env.world.gravity = (0,-10)

    return env, obs_dim, act_dim


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


# Main Test Execution
def main(env_name, now, episode_to_load, gamma, lamda, kl_targ, hid1_mult, init_pol_logvar, animate,\
        save_video, num_episodes_sim):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        now: Data time string indicating when the agent was created
        episode_to_load: Epoch of saved agent to load
        gamma: reward discount factor (float)
        lamda: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        init_pol_logvar: natural log of initial policy variance
        save_video: Boolean determining if videos of the agent will be saved
        num_episodes_sim: Number of episodes to simulate/save videos for
    """

    # Environment Initialization
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

    # Initialize Policy, Value Networks and Scaler (their vars and ops values will then be loaded)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, init_pol_logvar)

    # Validate intented episode to load OR get last episode number if no target load episode was provided 
    load_agent_path = os.path.join('agents', env_name, now) # agent / policy folders
    saved_ep_list = [file.split(".")[0].split("_")[-1] for file in os.listdir(load_agent_path) if "policy" in file]

    if episode_to_load == None: # Get last saved episode
        episode_to_load = sorted([int(ep_string) for ep_string in saved_ep_list])[-1]
    else: # Validate if episode_to_load was indeed saved
        assert str(episode_to_load) in saved_ep_list,\
        "\n\nWARNING: Episode you want to load ({}) was not stored during trainning".format(episode_to_load)
    
    # Paths
    print ("\n\n---- PATHS: ----")    
    logger = Logger(logname=env_name, now= now + "_Test", 
                    logname_file="_loadedEp_{}".format(episode_to_load)) # logger object
    aigym_path = os.path.join('./videos', env_name, now + "_Test") # videos folders
    print("Path for Saved Videos: {}".format(aigym_path))

    # Load Policy Network's Ops and Variables & Load Scaler Object
    policy.tf_saver.restore(policy.sess, "{}/policy_ep_{}".format(load_agent_path, episode_to_load)) 
    # val_func.tf_saver.restore(val_func.sess, "{}/val_func_ep_{}".format(load_agent_path, episode_to_load))
    scaler = pickle.load(open("{}/scaler_ep_{}.p".format(load_agent_path, episode_to_load), 'rb'))         
    print("\n\nAgent Loaded ! - Episoded Loaded:{}".format(episode_to_load))


    # TEST: Start Simulating Agent
    animate = True if animate == "True" else False
    save_video = True if save_video == "True" else False
    
    mean_reward_episodes = sim_agent(env, policy, scaler, num_episodes_sim, animate=animate , 
                                    save_video=save_video, 
                                    out_dir=aigym_path+"/vid_loadedEp_{}".format(episode_to_load))
    env.close() # closes window open by monitor wrapper

    # Logging Test Stats
    logger.log({'_MeanReward': mean_reward_episodes,'_Episode': num_episodes_sim})
    logger.write(display=True)  # write logger results to file and stdout
    
    # Terminate Sessions
    logger.close()
    policy.close_sess()


# Example of how to Test Agent:
# python test.py BipedalWalker-v2 Jun-13_03:02:11 --animate True --save_video True --num_episodes_sim 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Test policy on OpenAI Gym environment '
                                                  'trainned using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('now', type=str, help='Data time when agent was created (e.g.Jun-13_03:02:11)')
    parser.add_argument('-e', '--episode_to_load', type=int,
                        help='Episode of saved agent to load; default is last saved [None]', default=None)

    parser.add_argument('-g', '--gamma', type=float, help='Discount factor [0.995]', default=0.995)
    parser.add_argument('-l', '--lamda', type=float, 
                        help='Lambda for Generalized Advantage Estimation [0.98]', default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value [0.003]', default=0.003)
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
