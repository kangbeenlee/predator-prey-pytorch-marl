import random
import torch
from collections import deque
import sys



class ReplayBuffer:
    """Replay buffer for mlp based DQN agent"""
    def __init__(self, args):
        self.replay_memory_capacity = args.memory_size  # capacity of experience replay memory (default 10,000) # b_size
        self.batch_size = args.batch_size  # size of minibatch from experience replay memory for updates (default 32) # m_size
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    # def sample_from_memory(self):
    #     return random.sample(self.replay_memory, self.batch_size)

    def sample_from_memory(self):
        minibatch = random.sample(self.replay_memory, self.batch_size)
        last_onehot_a_n_lst, obs_n_lst, onehot_a_n_lst, obs_n_prime_lst, s_lst, a_n_lst, r_lst, s_prime_lst, done_any_lst = [], [], [], [], [], [], [], [], []
        
        for experience in minibatch:
            last_onehot_a_n, obs_n, onehot_a_n, obs_n_prime, s, a_n, r, s_prime, done_any = experience
        
            last_onehot_a_n_lst.append(last_onehot_a_n)
            obs_n_lst.append(obs_n)
            onehot_a_n_lst.append(onehot_a_n)
            obs_n_prime_lst.append(obs_n_prime)
            s_lst.append(s)
            a_n_lst.append(a_n)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_any_lst.append([done_any])
            
        return torch.tensor(last_onehot_a_n_lst, dtype=torch.float32), \
               torch.tensor(obs_n_lst, dtype=torch.float32), \
               torch.tensor(onehot_a_n_lst, dtype=torch.float32), \
               torch.tensor(obs_n_prime_lst, dtype=torch.float32), \
               torch.tensor(s_lst, dtype=torch.float32), \
               torch.tensor(a_n_lst, dtype=torch.int64), \
               torch.tensor(r_lst, dtype=torch.float32), \
               torch.tensor(s_prime_lst, dtype=torch.float32), \
               torch.tensor(done_any_lst, dtype=torch.float32)
            
    def erase(self):
        self.replay_memory.popleft()
        
    def get_current_memory_size(self):
        return len(self.replay_memory)
    
    def get_batch_size(self):
        return self.batch_size