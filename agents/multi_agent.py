#!/usr/bin/env python
# coding=utf8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

from agents.replay_buffer import *
from agents.evaluation import Evaluation
from agents.mixing_network import VdnMixingNetwork, QmixMixingNetwork, QtranBaseMixingNetwork



"""Agent's Neural Network"""
class DQN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, action_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class DRQN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, action_dim)

        if args.use_orthogonal_init:
            print("------ Use orthogonal initialization ------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h



"""Multi Agent Network"""
class MultiAgent(object):
    def __init__(self, args):
        self.args = args
        # Device
        self.device = args.device
        
        if args.load_nn:
            print("LOAD!")
            self.q_network.load_state_dict(torch.load(args.nn_file))

        # Multi-Agent parameters
        self.n_predator = args.n_predator
        self.n_prey = args.n_prey
        self.map_size = args.map_size
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        # Agent network type
        self.agent_network = args.agent_network
        self.rnn_hidden_dim = args.rnn_hidden_dim
            
        # Mixing network type
        self.mixing_network = args.mixing_network
        
        # Input dimension
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id

        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------ Add last action ------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------ Add agent id ------")
            self.input_dim += self.n_predator
        
        # Main network, target network
        if self.agent_network == 'mlp':
            print("------ Use MLP DQN agent ------")
            self.q_network = DQN(args, self.input_dim, self.action_dim).to(self.device)
            self.target_q_network = DQN(args, self.input_dim, self.action_dim).to(self.device)
        elif self.agent_network == 'rnn':
            print("------ Use DRQN agent ------")
            self.q_network = DRQN(args, self.input_dim, self.action_dim).to(self.device)
            self.target_q_network = DRQN(args, self.input_dim, self.action_dim).to(self.device)
        else:
            print("Agent netowrk type should be mlp, rnn")
            sys.exit()
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        if self.mixing_network == "vdn":
            print("------ Use VDN ------")
            self.mixing_net = VdnMixingNetwork().to(self.device)
            self.target_mixing_net = VdnMixingNetwork().to(self.device)
        elif self.mixing_network == "qmix":
            print("------ Use QMIX ------")
            self.mixing_net = QmixMixingNetwork(args).to(self.device)
            self.target_mixing_net = QmixMixingNetwork(args).to(self.device)
        elif self.mixing_network == "qtran-base":
            print("------ Use QTRAN-base ------")
            self.mixing_net = QtranBaseMixingNetwork(args).to(self.device)
            self.target_mixing_net = QtranBaseMixingNetwork(args).to(self.device)
        else:
            print("Mixing network type should be vdn, qmix, qtran-base")
            sys.exit()
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # Optimizer parameters
        self.batch_size = args.batch_size
        self.parameters = list(self.q_network.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr, weight_decay=1e-4)
        self.use_grad_clip = args.use_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # QTRAN-base parameters
        self.lambda_opt = args.lambda_opt
        self.lambda_nopt = args.lambda_nopt

        # Reinforcement learning parameters
        self.train_step = 0
        self.target_update_period = args.target_update_period
        self.use_hard_update = args.use_hard_update
        self.tau = args.tau
        self.gamma = args.df # discount factor
        
        self.scenario = args.scenario
        self.seed = args.seed
        self.save_period = args.save_period

    def choose_action(self, h_state, obs_n, last_onehot_a_n, epsilon, state, prey_agent, train=True):
        with torch.no_grad():
            # Action of predator
            inputs = []
            inputs.append(torch.tensor(obs_n, dtype=torch.float32))
            if self.add_last_action:
                last_onehot_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                inputs.append(last_onehot_a_n)
            if self.add_agent_id:
                inputs.append(torch.eye(self.n_predator))                
            inputs = torch.cat([x for x in inputs], dim=-1).to(self.device)  # inputs.shape=(N,inputs_dim)
            q_values_n, h_state = self.q_network(inputs, h_state.to(self.device))
            a_n = q_values_n.argmax(dim=-1).cpu().numpy()            

            if train and np.random.rand() < epsilon:
                a_n = [np.random.choice(self.action_dim) for _ in range(self.n_predator)]

            # Action of prey
            prey_a_n = []
            for i in range(self.n_prey):
                prey_a_n.append(prey_agent.choose_action(state, i))

            return a_n, h_state, prey_a_n

    def train(self, replay_buffer):
        self.train_step += 1
        
        # Get mini batch
        mini_batch, max_episode_len = replay_buffer.sample()
        
        obs_n = mini_batch['obs_n'].to(self.device)
        s = mini_batch['s'].to(self.device) # s.shape=(batch_size,max_episode_len+1,N,state_dim)
        last_onehot_a_n = mini_batch['last_onehot_a_n'].to(self.device)
        a_n = mini_batch['a_n'].to(self.device)
        r = mini_batch['r'].to(self.device)
        done = mini_batch['done'].to(self.device)
        active = mini_batch['active'].to(self.device)

        # Pre-processing input
        inputs = self.get_inputs(obs_n, last_onehot_a_n) # inputs.shape=(batch_size,max_episode_len,N,input_dim)

        # Initialize hidden & cell state
        h_state = torch.zeros([self.batch_size*self.n_predator, self.rnn_hidden_dim]).to(self.device)
        target_h_state = torch.zeros([self.batch_size*self.n_predator, self.rnn_hidden_dim]).to(self.device)

        q_evals, q_targets = [], []
        h_evals, h_targets = [], []
        for t in range(max_episode_len):  # t=0,1,2,...(max_episode_len-1)
            q_eval, h_state = self.q_network(inputs[:, t].reshape(-1, self.input_dim).to(self.device), h_state)  # q_eval.shape=(batch_size*N,action_dim)
            q_target, target_h_state = self.target_q_network(inputs[:, t + 1].reshape(-1, self.input_dim).to(self.device), target_h_state)
            q_evals.append(q_eval.reshape(self.batch_size, self.n_predator, -1)) # q_eval.shape=(batch_size,N,action_dim)
            q_targets.append(q_target.reshape(self.batch_size, self.n_predator, -1)) 
            
            if self.mixing_network == "qtran-base":
                h_evals.append(h_state.reshape(self.batch_size, self.n_predator, -1)) # h_eval.shape=(batch_size,N,hidden_dim)
                h_targets.append(target_h_state.reshape(self.batch_size, self.n_predator, -1))

        # Stack them according to the time (dim=1)
        q_evals = torch.stack(q_evals, dim=1).to(self.device) # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_targets = torch.stack(q_targets, dim=1).to(self.device)
        
        if self.mixing_network == "qtran-base":
            h_evals = torch.stack(h_evals, dim=1).to(self.device) # h_evals.shape=(batch_size,max_episode_len,N,hidden_dim)
            h_targets = torch.stack(h_targets, dim=1).to(self.device)

        # a_n.shape=(batch_size,max_episode_len,N)
        q_eval_n = torch.gather(q_evals, dim=-1, index=a_n.unsqueeze(-1)).squeeze(-1).to(self.device) # q_eval_n.shape(batch_size,max_episode_len,N)

        with torch.no_grad():
            q_target_n, target_max_a_n = q_targets.max(dim=-1) # q_targets.shape=(batch_size,max_episode_len,N)

        # Compute q_joint using mixing network, q_joint.shape=(batch_size,max_episode_len,1)
        if self.mixing_network == "vdn":
            q_joint = self.mixing_net(q_eval_n)
            target_q_joint = self.target_mixing_net(q_target_n.to(self.device)) # targets.shape=(batch_size, 1)
        elif self.mixing_network == "qmix":
            q_joint = self.mixing_net(q_eval_n, s[:, :-1])
            target_q_joint = self.target_mixing_net(q_target_n.to(self.device), s[:, 1:])
        elif self.mixing_network == "qtran-base":
            q_joint, v_joint = self.mixing_net(h_evals, last_onehot_a_n[:, 1:])
            with torch.no_grad():
                onehot_placeholder = torch.zeros([self.batch_size, max_episode_len, self.n_predator, self.action_dim]).to(self.device)
                onehot_target_max_a_n = onehot_placeholder.scatter(3, target_max_a_n.unsqueeze(-1), 1).to(self.device)
                target_q_joint, _ = self.target_mixing_net(h_targets, onehot_target_max_a_n)
        else:
            sys.exit()

        if self.mixing_network == "qtran-base":
            td_target = r + self.gamma * target_q_joint * (1 - done) # td_target.shape=(batch_size,max_episode_len,1)
            td_error = (q_joint - td_target.detach())
            mask_td_error = td_error * active
            td_loss = (mask_td_error ** 2).sum() / active.sum()

            # Optimal action loss
            max_q_n, max_q_a_n = q_evals.max(dim=-1) # max_q_a.shape=(batch_size,max_episode_len,N)
            onehot_placeholder = torch.zeros([self.batch_size, max_episode_len, self.n_predator, self.action_dim]).to(self.device)
            onehot_max_q_a_n = onehot_placeholder.scatter(3, max_q_a_n.unsqueeze(-1), 1).to(self.device)
            max_q_joint, _ = self.mixing_net(h_evals, onehot_max_q_a_n)
            opt_error = max_q_n.sum(dim=-1, keepdim=True) - max_q_joint.detach() + v_joint
            masked_opt_error = opt_error * active
            opt_loss = (masked_opt_error ** 2).sum() / active.sum()
            
            # Non-optimal action loss
            nopt_values = q_eval_n.sum(dim=-1, keepdim=True) - q_joint.detach() + v_joint # Don't use target networks here either
            nopt_error = nopt_values.clamp(max=0)
            masked_nopt_error = nopt_error * active
            nopt_loss = (masked_nopt_error ** 2).sum() / active.sum()

            loss = td_loss + (self.lambda_opt * opt_loss) + (self.lambda_nopt * nopt_loss)
        else:
            td_target = r + self.gamma * target_q_joint * (1 - done) # td_target.shape=(batch_size,max_episode_len,1)
            td_error = (q_joint - td_target.detach())
            mask_td_error = td_error * active
            loss = (mask_td_error ** 2).sum() / active.sum()

        # Optimize        
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()

        # Target network update
        if self.use_hard_update:
            if self.train_step % self.target_update_period == 0:
                print(">>> hard update")
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        else:
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.mixing_net.parameters(), self.target_mixing_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.train_step % self.save_period == 0:
            self.save_model(self.train_step)
            
        return loss.data.item()

    def get_inputs(self, obs_n, last_onehot_a_n):
        inputs = []
        inputs.append(obs_n)
        
        if self.add_last_action:
            inputs.append(last_onehot_a_n)
        if self.add_agent_id:
            batch_size = obs_n.shape[0]
            max_episode_len = obs_n.shape[1]
            agent_id_one_hot = torch.eye(self.n_predator).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_episode_len, 1, 1).to(self.device)
            inputs.append(agent_id_one_hot)
        
        inputs = torch.cat(inputs, dim=-1)
        return inputs
    
    def save_model(self, train_step):
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(cwd, "model")):
            os.makedirs(os.path.join(cwd, "model"))
        
        filename = "./model/{}_penalty_{}_{}_{}_seed_{}_{}_{}_step_{}.pkl".format(
                    self.scenario, self.args.penalty, self.args.n_predator, self.args.n_prey,
                    self.seed, self.args.mixing_network, self.args.agent_network, train_step)
        with open(filename, 'wb') as f:
            torch.save(self.q_network.state_dict(), f)

    # -------------------------- Not used -------------------------- #
    def get_predator_pos(self, state):
        """
        return position of agent 1 and 2
        :param state: input is state
        :return:
        """
        state_list = list(np.array(state).ravel())
        return state_list.index(1), state_list.index(2)

    def get_pos_by_id(self, state, id):
        state_list = list(np.array(state).ravel())
        return state_list.index(id)