import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.action_out = nn.Linear(256, env_params['action'])
        init = torch.rand(4)
        # self.weights = nn.Parameter(init, requires_grad=True)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions



class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.num_reward = env_params['num_reward']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.intermediate_q_out = nn.Linear(256, self.num_reward)
        self.softmin = nn.Softmin(dim=1)
        self.temp = env_params['temp']

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.intermediate_q_out(x)

        return x

    def softmin_forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.intermediate_q_out(x)
        # print(x.shape)
        y = self.softmin(x/self.temp)
#         q_value = self.q_out(x)
        # print(y.shape)
        return (x*y).sum(axis=1)#q_value



    def deep_forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.intermediate_q_out(x)
        x = self.out_fc1(x)
        # x = self.out_fc2(x)
        # x = self.out_fc3(x)

        return x

    # def multi_to_single(self, x):
    #     x = self.out_fc1(x)
    #     # x = self.out_fc2(x)
    #     # x = self.out_fc3(x)

    #     return x


class base_critic(nn.Module):
    def __init__(self, env_params):
        super(base_critic, self).__init__()
        self.max_action = env_params['action_max']
        self.num_reward = env_params['num_reward']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.intermediate_q_out = nn.Linear(256, 50)
        # self.softmin = nn.Softmin(dim=1)
        # self.temp = env_params['temp']

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.intermediate_q_out(x))

        return x




class end_critic(nn.Module):
    def __init__(self, env_params):
        super(end_critic, self).__init__()
        self.max_action = env_params['action_max']
        self.num_reward = env_params['num_reward']
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 1)
        # self.fc3 = nn.Linear(256, 256)
        # self.intermediate_q_out = nn.Linear(256, self.num_reward)
        # self.softmin = nn.Softmin(dim=1)
        # self.temp = env_params['temp']

    def forward(self, x, actions):
        # x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc3(x))
        # x = self.intermediate_q_out(x)

        return x
        






class base_actor(nn.Module):
    def __init__(self, env_params):
        super(base_actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        # self.action_out = nn.Linear(256, env_params['action'])
        # init = torch.rand(4)
        # self.weights = nn.Parameter(init, requires_grad=True)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # actions = self.max_action * torch.tanh(self.action_out(x))

        return x


class end_actor(nn.Module):
    def __init__(self, env_params):
        super(end_actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(128,128)
        # self.fc2 = nn.Linear()
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.action_out = nn.Linear(128, env_params['action'])
        # init = torch.rand(4)
        # self.weights = nn.Parameter(init, requires_grad=True)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions
