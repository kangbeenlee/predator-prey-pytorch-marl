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
from agents.mixing_network import VdnMixingNetwork, QmixMixingNetwork, QtranBaseMixingNetwork #, QtranAltMixingNetwork



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

class DRQN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, action_dim)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h

class DuelingDRQN(nn.Module):
    def __init__(self, args, state_dim, action_dim, h = 32):
        super(DuelingDRQN, self).__init__()
        self.device = 'cpu'
        if torch.cuda.is_available(): self.device = torch.cuda.current_device()

        self.h_dim = h
        self.act_dim = action_dim

        self.Head = nn.Sequential(nn.Linear(state_dim, h), nn.ReLU())
        self.LSTM = nn.LSTM(h, h, batch_first=True)

        self.V = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.A = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, action_dim))
        self.to(self.device)

    def forward(self, obs, hc):
        b = type(hc)==int
        if b: hc = self.init_hidden_state(hc, True)

        x = self.Head(obs.to(self.device))
        x, hc = self.LSTM(x, hc)
        v = self.V(x)
        a = self.A(x)
        # re-center (apx) advantage
        a = a - a.mean(dim=-1, keepdim=True)

        q = v + a
        return q if b else (q, hc) # dim = action dim (i.e. it computes Q(S, A) for all A) 


"""Multi Agent Network"""
class MultiAgent(object):
    def __init__(self, args, action_dim, obs_dim):
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
        self.obs_dim = obs_dim
        self.state_dim = 2 * (self.n_predator + self.n_prey)
        self.action_dim = action_dim
        self.joint_action_dim = self.action_dim * self.n_predator

        # Agent network type
        self.agent_network = args.agent_network
        self.rnn_hidden_dim = args.rnn_hidden_dim
        if self.agent_network == 'rnn':
            self.use_rnn = True
        else:
            self.use_rnn = False
            
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
        elif self.agent_network == 'deuling_rnn':
            print("------ Use DeulingDRQN agent ------")
        else:
            print("Agent netowrk type should be mlp, rnn, deuling_rnn")
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
        # elif self.mixing_network == "qtran-alt":
        #     print("------ Use QTRAN-alt ------")
        #     self.mixing_net = QmixMixingNetwork().to(self.device)
        #     self.target_mixing_net = QmixMixingNetwork().to(self.device)
        else:
            print("Mixing network type should be vdn, qmix, qtran-base, qtran-alt")
            sys.exit()
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # Optimizer parameters
        self.batch_size = args.batch_size
        self.parameters = list(self.q_network.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr, weight_decay=1e-4)
        self.use_grad_clip = args.use_grad_clip
        self.max_grad_norm = args.max_grad_norm

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
        
        mini_batch, max_episode_len = replay_buffer.sample()
        obs_n = mini_batch['obs_n'].to(self.device)
        s = mini_batch['s'].to(self.device) # s.shape=(batch_size,max_episode_len+1,N,state_dim)
        last_onehot_a_n = mini_batch['last_onehot_a_n'].to(self.device)
        a_n = mini_batch['a_n'].to(self.device)
        r = mini_batch['r'].to(self.device)
        done = mini_batch['done'].to(self.device)
        active = mini_batch['active'].to(self.device)

        inputs = self.get_inputs(obs_n, last_onehot_a_n) # inputs.shape=(batch_size,max_episode_len,N,input_dim)

        # Initialize hidden & cell state
        h_state = torch.zeros([self.batch_size*self.n_predator, self.rnn_hidden_dim]).to(self.device)
        target_h_state = torch.zeros([self.batch_size*self.n_predator, self.rnn_hidden_dim]).to(self.device)

        q_evals, q_targets = [], []
        for t in range(max_episode_len):  # t=0,1,2,...(max_episode_len-1)
            q_eval, h_state = self.q_network(inputs[:, t].reshape(-1, self.input_dim).to(self.device), h_state)  # q_eval.shape=(batch_size*N,action_dim)
            q_target, target_h_state = self.target_q_network(inputs[:, t + 1].reshape(-1, self.input_dim).to(self.device), target_h_state)
            q_evals.append(q_eval.reshape(self.batch_size, self.n_predator, -1))  # q_eval.shape=(batch_size,N,action_dim)
            q_targets.append(q_target.reshape(self.batch_size, self.n_predator, -1)) 

        # Stack them according to the time (dim=1)
        q_evals = torch.stack(q_evals, dim=1).to(self.device) # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_targets = torch.stack(q_targets, dim=1).to(self.device)

        # a_n.shape=(batch_size,max_episode_len,N)
        q_eval_n = torch.gather(q_evals, dim=-1, index=a_n.unsqueeze(-1)).squeeze(-1).to(self.device)  # q_evals.shape(batch_size,max_episode_len,N)

        with torch.no_grad():
            q_target_n = q_targets.max(dim=-1)[0].to(self.device) # q_targets.shape=(batch_size,max_episode_len,N)

        # Compute q_joint using mixing network, q_joint.shape=(batch_size,max_episode_len,1)
        if self.mixing_network == "vdn":
            q_joint = self.mixing_net(q_eval_n)
            target_q_joint = self.target_mixing_net(q_target_n) # targets.shape=(batch_size, 1)
        elif self.mixing_network == "qmix":
            q_joint = self.mixing_net(q_eval_n, s[:, :-1])
            target_q_joint = self.target_mixing_net(q_target_n, s[:, 1:])
        # elif self.mixing_network == "qtran-base":
        #     q_joint = self.mixing_net(onehot_a_n, hidden_q_n, hidden_v_n)
        #     target_q_joint = self.target_mixing_net(onehot_a_n, hidden_q_n, hidden_v_n)
        else: # qtran-alt
            pass
            sys.exit()

        # td_target.shape=(batch_size,max_episode_len,1)
        td_target = r + self.gamma * target_q_joint * (1 - done)

        td_error = (q_joint - td_target.detach())
        mask_td_error = td_error * active
        loss = (mask_td_error ** 2).sum() / active.sum()
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
            
        # if self.train_step % 10000 == 0 and self.cnt >= 10000:
        #     print("Penalty: ", args.penalty, "TD-error: ", self.cnt, self.TDerror/10000)
        #     self.TDerror = 0

        # if self.cnt % 10000 == 0 and self.cnt >= 10000:
        #     print("Penalty: ", args.penalty, "meanQ: ", self.cnt, self.TDerror2/10000, '%.3f' % self.beta)
        #     self.TDerror2 = 0
        
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