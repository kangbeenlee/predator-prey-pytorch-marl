import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class VdnMixingNetwork(nn.Module):
    def __init__(self):
        super(VdnMixingNetwork, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, max_episode_len, 1)

class QmixMixingNetwork(nn.Module):
    def __init__(self, args):
        super(QmixMixingNetwork, self).__init__()
        self.N = args.n_predator
        self.batch_size = args.batch_size
        self.state_dim = args.state_dim
        self.mixing_hidden_dim = args.mixing_hidden_dim
        """
        w1:(N, mixing_hidden_dim)
        b1:(1, mixing_hidden_dim)
        w2:(mixing_hidden_dim, 1)
        b2:(1, 1)
        """
        self.hyper_w1 = nn.Linear(self.state_dim, self.N * self.mixing_hidden_dim)
        self.hyper_w2 = nn.Linear(self.state_dim, self.mixing_hidden_dim * 1)
        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.mixing_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.mixing_hidden_dim, 1))

    def forward(self, q, s):
        # q.shape=(batch_size, max_episode_len, N)
        # s.shape=(batch_size, max_episode_len,state_dim)
        q = q.view(-1, 1, self.N)  # q.shape=(batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # s.shape=(batch_size * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(s))  # (batch_size * max_episode_len, N * mixing_hidden_dim)
        b1 = self.hyper_b1(s)  # (batch_size * max_episode_len, mixing_hidden_dim)
        w1 = w1.view(-1, self.N, self.mixing_hidden_dim)  # (batch_size * max_episode_len, N,  mixing_hidden_dim)
        b1 = b1.view(-1, 1, self.mixing_hidden_dim)  # (batch_size * max_episode_len, 1, mixing_hidden_dim)

        # torch.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, mixing_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s))  # (batch_size * max_episode_len, mixing_hidden_dim * 1)
        b2 = self.hyper_b2(s)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.mixing_hidden_dim, 1)  # (batch_size * max_episode_len, mixing_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_joint = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_joint = q_joint.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_joint


class QtranBaseMixingNetwork(nn.Module):
    def __init__(self, args):
        super(QtranBaseMixingNetwork, self).__init__()
        self.N = args.n_predator
        self.batch_size = args.batch_size
        self.action_dim = args.action_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.action_encoding_dim = self.action_dim + self.hidden_dim
        self.mixing_hidden_dim = args.mixing_hidden_dim

        self.joint_action_input_encoder = nn.Sequential(nn.Linear(self.action_encoding_dim, self.action_encoding_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(self.action_encoding_dim, self.action_encoding_dim))
        self.joint_action_value_network = nn.Sequential(nn.Linear(self.action_encoding_dim, self.mixing_hidden_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(self.mixing_hidden_dim, self.mixing_hidden_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(self.mixing_hidden_dim, 1))
        self.state_value_network = nn.Sequential(nn.Linear(self.hidden_dim, self.mixing_hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(self.mixing_hidden_dim, 1))

    # def forward(self, batch, hidden_states, actions=None):
    def forward(self, hidden_n, onehot_a_n):
        # hidden_n.shape=(batch_size,max_episode_len,N,hidden_dim)
        # onehot_a_n.shape(batch_size,max_episode_len,N,action_dim)
        max_episode_len = hidden_n.shape[1]
        
        hidden_n = hidden_n.reshape(self.batch_size*max_episode_len, self.N, -1) # hidden_n.shape=(batch_size*max_episode_len, N, hidden_dim)
        onehot_a_n = onehot_a_n.reshape(self.batch_size*max_episode_len, self.N, -1) # onehot_a_n.shape=(batch_size*max_episode_len, N, action_dim)
        
        joint_action_input = torch.cat([hidden_n, onehot_a_n], dim=-1) # joint_action_input.shape=(batch_size*max_episode_len, N, hidden_dim+action_dim)
        joint_action_input = self.joint_action_input_encoder(joint_action_input) # joint_action_input.shape=(batch_size*max_episode_len, N, hidden_dim+action_dim)
        
        input_q = joint_action_input.sum(dim=1) # input_q.shape=(batch_size*max_episode_len, hidden_dim+action_dim)
        input_v = hidden_n.sum(dim=1) # input_v.shape=(batch_size*max_episode_len, hidden_dim)

        q_joint = self.joint_action_value_network(input_q).view(self.batch_size, -1, 1)
        v_joint = self.state_value_network(input_v).view(self.batch_size, -1, 1)

        return q_joint, v_joint