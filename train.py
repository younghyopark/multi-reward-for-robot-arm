import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
from rl_modules.ddpg_agent_mgda import ddpg_agent as ddpg_agent_mgda
import random
import torch
# from knockknock import email_sender



# import register
# from registration import make


"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env, args):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'num_reward':env.num_reward,
            'temp': args.softmax_temperature
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

# @email_sender(recipient_emails=["notify.younghyopark@gmail.com"], sender_emails=["notify.younghyopark@gmail.com"])
def launch(args):
    # create the ddpg_agent
    env = gym.make('gym_multiRL:MultiRL{}'.format(args.env_name))
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env, args)
    # create the ddpg agent to interact with the environment 
    # if args.actor_loss_type=='mgda':
    #     ddpg_trainer = ddpg_agent_mgda(args, env, env_params)
    # else:
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()

    launch(args)
