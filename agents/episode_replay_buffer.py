import numpy as np
import torch



class EpisodeReplayBuffer:
    def __init__(self, args, obs_dim, state_dim, action_dim):
        self.N = args.n_predator
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_limit = args.max_step
        self.buffer_size = args.memory_size
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.current_size = 0
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'done': np.ones([self.buffer_size, self.episode_limit, 1]),
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                       }
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs_n, s, last_onehot_a_n, a_n, r, done):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['done'][self.episode_num][episode_step] = done
        
        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_step(self, episode_step, obs_n, s):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        mini_batch = {}
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 's' or key == 'last_onehot_a_n':
                mini_batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32)
            elif key == 'a_n':
                mini_batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.int64)
            else:
                mini_batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32)

        return mini_batch, max_episode_len

    def __len__(self):
        return self.current_size