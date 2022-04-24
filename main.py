import argparse
import torch
import torch.optim as optim

from DQNModel import Actor
from envs import GoHighEnvVec
from ssbm_gym.ssbm_env import EnvVec, SSBMEnv
from train import train, pretrain

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=4, help='number of parallel workers')
parser.add_argument('--rollout-steps', type=int, default=600, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(4e7), help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter for GAE')
parser.add_argument('--lambd', type=float, default=1.00, help='lambda parameter for GAE')
parser.add_argument('--value_coeff', type=float, default=0.5, help='value loss coeffecient')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy loss coeffecient')
parser.add_argument('--grad_norm_limit', type=float, default=40., help='gradient norm clipping threshold')


args = parser.parse_args()

options = dict(
    render=False,
    player1='ai',
    player2='human',
    char1='fox',
    char2='fox',
    cpu2=3,
    stage='final_destination',
)


if __name__ == "__main__":
    #env =  GoHighEnvVec(args.num_workers, args.total_steps, options)
    env = EnvVec(SSBMEnv, args.num_workers, args.total_steps, options)
    print("Observation space " + str(env.observation_space.n))
    print("Action space " + str(env.action_space.n))
    net = Actor(env.observation_space.n, env.action_space.n)
    target_net = Actor(env.observation_space.n, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    buf, n_st = pretrain(args, net, target_net, optimizer, env)
    options['player2'] = 'cpu'
    training_env = EnvVec(SSBMEnv, args.num_workers, args.total_steps, options)
    train(args, net, target_net, optimizer, training_env, buf, n_st)