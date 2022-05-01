import argparse
import torch
import torch.optim as optim

from StateNet import StateNet
from ppo_model import Actor
from envs import GoHighEnvVec
from ssbm_gym.ssbm_env import EnvVec, SSBMEnv
import train
import train_state

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=4, help='number of parallel workers')
parser.add_argument('--rollout-steps', type=int, default=600, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(4e7), help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter for GAE')
parser.add_argument('--lambd', type=float, default=1.00, help='lambda parameter for GAE')
parser.add_argument('--epsilon', type=float, default=0.10, help='epsilon parameter for PPO')
parser.add_argument('--value_coeff', type=float, default=0.5, help='value loss coeffecient')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy loss coeffecient')
parser.add_argument('--grad_norm_limit', type=float, default=40., help='gradient norm clipping threshold')
parser.add_argument('--state_prediction', type=bool, default=False, help='adds state predition model')
parser.add_argument('--state_offset', type=int, default=7, help='offset for state prediction, default 7')


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
    pretrain_env = EnvVec(SSBMEnv, args.num_workers, args.total_steps, options)

    if args.state_prediction:
        #double the input size
        print(pretrain_env.observation_space.n)
        print(pretrain_env.action_space.n)

        net = Actor(pretrain_env.observation_space.n * 2, pretrain_env.action_space.n)
        state_net = StateNet(pretrain_env.observation_space.n + pretrain_env.action_space.n, pretrain_env.observation_space.n)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        optimizer_state = optim.Adam(state_net.parameters(), lr=args.lr)

        n_steps = train_state.pretrain(args, net, optimizer, pretrain_env, state_net, optimizer_state)

        options['player2'] = 'cpu'
        train_env = EnvVec(SSBMEnv, args.num_workers, args.total_steps, options)

        train_state.train(args, net, optimizer, train_env, n_steps, state_net, optimizer_state)
        
    else:
        net = Actor(pretrain_env.observation_space.n, pretrain_env.action_space.n)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        n_steps = train.pretrain(args, net, optimizer, pretrain_env)

        options['player2'] = 'cpu'
        train_env = EnvVec(SSBMEnv, args.num_workers, args.total_steps, options)

        train.train(args, net, optimizer, train_env, n_steps)