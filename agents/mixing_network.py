import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class VdnMixingNetwork(nn.Module):
    def __init__(self):
        super(VdnMixingNetwork, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, 1)

class QmixMixingNetwork(nn.Module):
    def __init__(self, args):
        super(QmixMixingNetwork, self).__init__()
        self.N = args.n_predator
        self.batch_size = args.batch_size
        self.state_dim = 2 * (args.n_predator + args.n_prey)
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
        q = q.view(-1, 1, self.N)  # q.shape(batch_size, max_episode_len, N)
        s = s.reshape(-1, self.state_dim)  # s.shape(batch_size, max_episode_len, state_dim)

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
        self.state_dim = 2 * (args.n_predator + args.n_prey)
        self.agent_profile = self.env.get_agent_profile()
        self.action_dim = self.agent_profile["predator"]["act_dim"]
        self.action_encoding_dim = self.action_dim * 2
        self.mixing_hidden_dim = args.mixing_hidden_dim

        self.joint_action_input_encoder = nn.Sequential(nn.Linear(self.action_encoding_dim, self.action_encoding_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(self.action_encoding_dim, self.action_encoding_dim))
        self.joint_action_value_network = nn.Sequential(nn.Linear(self.action_encoding_dim, self.mixing_hidden_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(self.mixing_hidden_dim, self.mixing_hidden_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(self.mixing_hidden_dim, 1))
        self.state_value_network = nn.Sequential(nn.Linear(1, self.mixing_hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(self.mixing_hidden_dim, 1))

    # def forward(self, batch, hidden_states, actions=None):
    def forward(self, onehot_a_n, hidden_q_n, hidden_v_n):
        # s.shape=[batch_size, state_dim]
        # onehot_a_n.shape=[batch_size, N, action_dim]
        # hidden_q_n.shape=[batch_size, N, action_dim]
        # hidden_v_n.shape=[batch_size, N, 1]
        
        encoded_joint_action_input = torch.cat([hidden_q_n, onehot_a_n], dim=-1) # encoded_joint_action_input.shape=[batch_size, N, action_dim * 2]
        encoded_joint_action_input = self.joint_action_input_encoder(encoded_joint_action_input) # encoded_joint_action_input.shape=[batch_size, N, action_dim * 2]
        
        input_q = encoded_joint_action_input.sum(dim=1) # agent_state_action_encoding.shape=[batch_size, action_dim * 2]
        input_v = hidden_v_n.sum(dim=1) # agent_state_action_encoding.shape=[batch_size, 1]

        joint_q = self.joint_action_value_network(input_q)
        joint_v = self.state_value_network(input_v)

        return joint_q, joint_v
    
# class QtranAltMixingNetwork(nn.Module):
#     def __init__(self, args):
#         super(QtranAltMixingNetwork, self).__init__()
#         self.N = args.n_predator
#         self.state_dim = 2 * (args.n_predator + args.n_prey)
#         # self.agent_profile = self.env.get_agent_profile()
#         # self.action_dim = self.agent_profile["predator"]["act_dim"]
#         self.mixing_hidden_dim = args.mixing_hidden_dim
#         # self.action_encoding_dim = self.action_dim * 2        
        
#         self.args = args


#         # Q(s,-,u-i)
#         # Q takes [state, u-i, i] as input
#         joint_q_input_dim = self.state_dim + (self.N * self.action_dim) + self.N

#         self.joint_action_value_network = nn.Sequential(nn.Linear(joint_q_input_dim, self.mixing_hidden_dim),
#                                                         nn.ReLU(),
#                                                         nn.Linear(self.mixing_hidden_dim, self.mixing_hidden_dim),
#                                                         nn.ReLU(),
#                                                         nn.Linear(self.mixing_hidden_dim, self.action_dim))
#         self.state_value_network = nn.Sequential(nn.Linear(self.state_dim, self.mixing_hidden_dim),
#                                                  nn.ReLU(),
#                                                  nn.Linear(self.mixing_hidden_dim, 1))

#     def forward(self, batch, masked_actions=None):
#         bs = batch.batch_size
#         ts = batch.max_seq_length
#         # Repeat each state n_agents times
#         repeated_states = batch["state"].repeat(1, 1, self.N).view(-1, self.state_dim)

#         if masked_actions is None:
#             actions = batch["actions_onehot"].repeat(1, 1, self.N, 1)
#             agent_mask = (1 - torch.eye(self.N, device=batch.device))
#             agent_mask = agent_mask.view(-1, 1).repeat(1, self.action_dim)#.view(self.N, -1)
#             masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
#             masked_actions = masked_actions.view(-1, self.N * self.action_dim)

#         agent_ids = torch.eye(self.N, device=batch.device).unsqueeze(0).unsqueeze(0).repeat(bs, ts, 1, 1).view(-1, self.N)

#         inputs = torch.cat([repeated_states, masked_actions, agent_ids], dim=1)

#         joint_q = self.joint_action_value_network(inputs)

#         states = batch["state"].repeat(1,1,self.N).view(-1, self.state_dim)
#         joint_v = self.state_value_network(states)

#         return joint_q, joint_v