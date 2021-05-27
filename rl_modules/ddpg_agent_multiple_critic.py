import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from tqdm import trange
from tensorboardX import SummaryWriter

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network

        self.actor_network = actor(env_params)
        sync_networks(self.actor_network)

        self.critic_network = dict()
        for i in range(self.env.num_reward):
            self.critic_network[i] = critic(env_params)

            # sync the networks across the cpus
            sync_networks(self.critic_network[i])

        # build up the target network
        self.actor_target_network=actor(env_params)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        self.critic_target_network = dict()
        for i in range(self.env.num_reward):
            self.critic_target_network[i] = critic(env_params)

            # load the weights into the target networks
            self.critic_target_network[i].load_state_dict(self.critic_network[i].state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.actor_target_network.cuda()
            for i in range(self.env.num_reward):
                self.critic_network[i].cuda()
                self.critic_target_network[i].cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = dict()
        for i in range(self.env.num_reward):
            
            self.critic_optim[i] = torch.optim.Adam(self.critic_network[i].parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        print(MPI.COMM_WORLD.Get_rank())
        # self.defined=False

        # while self.defined==False:
        if MPI.COMM_WORLD.Get_rank()==0:
            
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

            self.result_dir = f'./learning_curves/{args.env_name}/{self.args.run_name}'
            if not os.path.isdir(self.result_dir):
                os.makedirs(self.result_dir, exist_ok=True)
                print(f'creating {self.result_dir}')
            self.writer = SummaryWriter(logdir=self.result_dir)

            # self.defined=True


    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in trange(self.args.n_epochs):
            print('starting {}th epoch training'.format(epoch))
            for _ in trange(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                for i in range(self.env.num_reward):
                    self._soft_update_target_network(self.critic_target_network[i], self.critic_network[i])
            # start to do the evaluation
            success_rate, reward_components = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model_{}_{}.pt'.format(self.args.run_name,epoch))
                
                self.writer.add_scalar('success_rate', success_rate,epoch)

                for i in range(self.env.num_reward):
                    self.writer.add_scalar('rewards/number_{}'.format(i), reward_components[i],epoch)



        # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
        #                     self.model_path + '/model_{}.pt'.format(self.args.run_name))

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        for i in range(self.env.num_reward):
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).reshape(transitions['r'].shape[0],-1)[:,i]
            # print(r_tensor.shape)
            if self.args.cuda:
                inputs_norm_tensor = inputs_norm_tensor.cuda()
                inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
                actions_tensor = actions_tensor.cuda()
                r_tensor = r_tensor.cuda()
            # calculate the target Q value function
            with torch.no_grad():
                # do the normalization
                # concatenate the stuffs
                actions_next = self.actor_target_network(inputs_next_norm_tensor)
                q_next_value = self.critic_target_network[i](inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                # print('r_tensor_shape :', r_tensor.shape)
                # print('q_next_value :', q_next_value.shape)
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                # clip the q value
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            # the q loss
            # if self.args.ddpg_vq_version=='ver3':
            #     real_q_value = self.critic_network.deep_forward(inputs_norm_tensor, actions_tensor)
            # else:
            real_q_value = self.critic_network[i](inputs_norm_tensor, actions_tensor)

            # print('target_q_value :', target_q_value.shape)
            # print('real_q_value :', real_q_value.shape)
            # print((target_q_value - real_q_value).shape)
            # print(( (target_q_value - real_q_value).pow(2)).shape)
            # print((target_q_value - real_q_value).pow(2).mean().shape)
            if self.args.critic_loss_type=='MSE':
                critic_loss = (target_q_value - real_q_value).pow(2).mean()
            # elif self.args.critic_loss_type=='max':
            #     critic_loss, _ = torch.max((target_q_value - real_q_value).pow(2),dim=1)
            #     critic_loss = torch.mean(critic_loss)
            elif self.args.critic_loss_type=='MAE':
                critic_loss = (target_q_value - real_q_value).abs().mean()

                # print(critic_loss.shape)
                # .mean()
                        # update the critic_network
            self.critic_optim[i].zero_grad()
            critic_loss.backward()
            sync_grads(self.critic_network[i])
            self.critic_optim[i].step()


    #         print('critic_loss :',critic_loss.shape)
            # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        update_index_sampling_prob = []
        for i in range(self.env.num_reward):
            update_index_sampling_prob.append(self.critic_network[i](inputs_norm_tensor, actions_real).mean().data.cpu().numpy().item())
        update_index_sampling_prob = torch.Tensor(np.array(update_index_sampling_prob))
        update_index_sampling_prob = torch.nn.Softmin(dim=0)(update_index_sampling_prob/self.args.softmax_temperature).numpy()


        if self.args.actor_loss_type=='default':
            actor_loss = -(self.critic_network(inputs_norm_tensor, actions_real)).mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        elif self.args.actor_loss_type=='min':
            actor_loss = -(self.critic_network(inputs_norm_tensor, actions_real)).min(axis=1)[0].mean()
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        elif self.args.actor_loss_type=='softmin':
            # print((self.critic_network.softmin_forward(inputs_norm_tensor, actions_real)).shape)
            actor_loss = -(self.critic_network.softmin_forward(inputs_norm_tensor, actions_real)).mean()
            # print(actor_loss.shape)
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        elif self.args.actor_loss_type=='random':
            # print((self.critic_network.softmin_forward(inputs_norm_tensor, actions_real)).shape)
            update_index = np.random.choice(self.env.num_reward, p= update_index_sampling_prob)
            actor_loss = -(self.critic_network[update_index](inputs_norm_tensor, actions_real)).mean()
            # print(actor_loss)
            actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
            # print(actor_loss)
            # for i in range(self.env.num_reward):
            #     self.writer.add_scalar('update_probability/number_{}'.format(i), update_index_sampling_prob[i])

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()


    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_reward_components=dict()
        per_reward_components=dict()
        local_reward_components =dict()
        global_reward_components =dict()
        for i in range(self.env.num_reward):
            total_reward_components[i]=[]
        for _ in range(self.args.n_test_rollouts):
            for i in range(self.env.num_reward):
                per_reward_components[i]=[]
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward_new, _, info = self.env.step(actions)
                for i in range(self.env.num_reward):
                    if self.env.num_reward==1:
                        per_reward_components[i].append(reward_new)
                    else:
                        per_reward_components[i].append(reward_new[0,i])
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            for i in range(self.env.num_reward):
                total_reward_components[i].append(per_reward_components[i])
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        for i in range(self.env.num_reward):
                total_reward_components[i]=np.array(total_reward_components[i])
                local_reward_components[i] = np.mean(total_reward_components[i][:, -1])
                global_reward_components[i] = MPI.COMM_WORLD.allreduce(local_reward_components[i], op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_reward_components
