import numpy as np
import torch
import random
import time
import sys
import argparse

from envs.environment import MultiAgentEnv
import envs.scenarios as scenarios
from agents.trainer import Trainer



def make_env(scenario_name, args, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario class from script
    scenario = scenarios.load(scenario_name + ".py").Scenario(args)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world, 
                                   reward_callback=scenario.reward, 
                                   observation_callback=scenario.observation,
                                   done_callback=scenario.done)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("--gui", type=bool, default=False, help="Activate GUI")
    
    ## Environment
    # Scenario
    parser.add_argument("--scenario", type=str, default="pursuit", help="Scenario")
    parser.add_argument("--n_predator", type=int, default=2, help="Number of predators")
    parser.add_argument("--n_prey1", type=int, default=0, help="Number of preys 1")
    parser.add_argument("--n_prey2", type=int, default=1, help="Number of preys 2")
    parser.add_argument("--n_prey", type=int, default=1, help="Number of preys")
    
    # Observation
    parser.add_argument("--history_len", type=int, default=1, help="How many previous steps we look back")

    # Core
    parser.add_argument("--map_size", type=int, default=3, help="Size of the map")
    parser.add_argument("--render_every", type=float, default=1000, help="Render the nth episode")

    # Penalty
    parser.add_argument("--penalty", type=int, default=0, help="reward penalty (e.g. 5 and 10 mean 0.5 and 1.0 penalties individualy")
    
    
    ## Agent    
    # Agent network parameters
    parser.add_argument("--agent_network", type=str, default="rnn", help="Agent network type")
    parser.add_argument("--add_last_action", action="store_true", help="Use last action as agent network input")
    parser.add_argument("--add_agent_id", action="store_true", help="Use agent id as agent network input")
    parser.add_argument('--mlp_hidden_dim', type=int, default=64, help='DQN agent hidden dimension')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='RNN agent hidden dimension')
    parser.add_argument("--use_orthogonal_init", action="store_true", help="Use orthogonal initialization")

    # QTRAN-base parameters
    parser.add_argument("--lambda_opt", type=float, default=1.0, help="Weight constant for optimal action loss function")
    parser.add_argument("--lambda_nopt", type=float, default=1.0, help="Weight constant for non-optimal action loss function")

    # Mixing network parameters
    parser.add_argument("--mixing_network", type=str, default="vdn", help="Mixing network type")
    parser.add_argument('--mixing_hidden_dim', type=int, default=64, help='Mixing network hidden dimension')

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--step_size', default=10, type=int, help='Step size of optimizer scheduler')
    parser.add_argument("--gamma", type=float, default=0.5, help="Multiplicative factor of learning rate decay")
    parser.add_argument("--use_grad_clip", action="store_true", help="Use gradient clipping")
    parser.add_argument('--max_grad_norm', type=int, default=10, help='Maxinum gradient norm')
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--training_start", type=int, default=100, help="Number of steps before training [batch_size * pre_step]")
    parser.add_argument("--max_step", type=int, default=200, help="Maximum time step per episode")
    parser.add_argument("--training_step", type=int, default=10000000, help="Learning time step")
    parser.add_argument("--eval_episode", type=int, default=100, help="Evaluation time step")
    parser.add_argument("--testing_step", type=int, default=1000, help="Testing time step")

    # Reinforcement learning parameters
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=3000000, help="Number of steps over which epsilon is annealed to the final value")
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay memory size")
    parser.add_argument("--use_random_update", action="store_true", help="Use random update")
    parser.add_argument('--rollout_steps', default=20, type=int, help='Rollout steps for random update')
    parser.add_argument("--df", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--use_hard_update", action="store_true", help="Use hard target network update")
    parser.add_argument("--target_update_period", type=int, default=10000, help="Target network update period")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")

    # Train & Test parameters
    parser.add_argument("--train", action="store_true", help="Training or testing")
    parser.add_argument("--keyboard_input", action="store_true", help="Keyboard input test")

    # Trained model parameters
    parser.add_argument("--save_period", type=int, default=100000, help="Target network update period")
    parser.add_argument("--load_nn", action="store_true", help="Load nn from file or not")
    parser.add_argument("--nn_file", type=str, default="", help="The name of file for loading")
    
    args = parser.parse_args()

    # Add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    print("------ Use {} ------".format(args.device))

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(">>> Seed:", args.seed)

    # Print all set parameters
    agent_setting = "agentnetwork_" + args.agent_network + "_mixingnetwork_" + args.mixing_network + "_lr_" + str(args.lr) + "_batchsize_" + str(args.batch_size) \
                    + "_last_action_" + str(args.add_last_action) + "_agent_id_" + str(args.add_agent_id) + "_randomupdate_" + str(args.use_random_update)
    env_setting = "scenario_" + args.scenario + "_mapsize_" + str(args.map_size) + "_penalty_" + str(args.penalty)
    file_name = "n_" + str(args.n_predator) + "_" + agent_setting + "_" + env_setting
    print(">>> Setting:", file_name)

    # Load environment
    env = make_env(args.scenario, args)
    
    # Add environment information
    agent_profile = env.get_agent_profile()
    args.obs_dim = agent_profile["predator"]["obs_dim"][0]
    args.state_dim = 2 * (args.n_predator + args.n_prey)
    args.action_dim = agent_profile["predator"]["act_dim"]
    
    # Load trainer
    trainer = Trainer(args, env)

    # start learning
    if args.train:
        print(">>> Start training")
        start_time = time.time()
        trainer.learn()
        finish_time = time.time()
        # trainer.test()
        print("Training time (sec)", finish_time - start_time)
    else:
        trainer.test()

if __name__ == '__main__':
    main()