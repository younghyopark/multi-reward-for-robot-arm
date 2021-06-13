from rl_modules.models import base_actor,end_actor,base_critic,end_critic
import torch.nn as nn


def get_actor_model(args,env_params):
    
    model = {}
    model['rep'] = base_actor(env_params)
    if args.cuda:
        model['rep'].cuda()
    task = list(range(env_params['num_reward']))
    for t in task:
        model[t] = end_actor(env_params)
        if args.cuda:
            model[t].cuda()
    return model

def get_critic_model(args,env_params):
    
    model = {}
    model['rep'] = base_critic(env_params)
    if args.cuda:
        model['rep'].cuda()
    task = list(range(env_params['num_reward']))
    for t in task:
        model[t] = end_actor(env_params)
        if args.cuda:
            model[t].cuda()
    return model