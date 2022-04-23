import argparse
import torch
import torch.optim as optim

from a2c_model import Actor
from envs import GoHighEnvVec
<<<<<<< HEAD
from train import train, pretrain
=======
from ssbm_gym.ssbm_env import EnvVec, SSBMEnv
from train import train
>>>>>>> a436fa5a76ae6441c88dbe8da993fb923de2223e

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
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
<<<<<<< HEAD
    pretrain_env = GoHighEnvVec(args.num_workers, args.total_steps, options)
=======
    env = EnvVec(SSBMEnv, args.num_workers, args.total_steps, options)
>>>>>>> a436fa5a76ae6441c88dbe8da993fb923de2223e

    net = Actor(pretrain_env.observation_space.n, pretrain_env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    n_steps = pretrain(args, net, optimizer, pretrain_env)

    options['player2'] = 'cpu'
    training_env = GoHighEnvVec(args.num_workers, args.total_steps, options)

    train(args, net, optimizer, training_env, n_steps)