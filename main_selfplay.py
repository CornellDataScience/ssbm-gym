"""
This is an example of environment definition and execution of the program.
Look inide envs.py to see how to create your own environment.
You can find how to create an action space inside ssbm_gym/spaces.py (MinimalActionSpace).
In this code player2 is set as "ai", to give an example of self play interfacing.
"""
from envs import SelfPlayEnv
import atexit
import platform
import random
import time

options = dict(
    render=True,
    player1='ai',
    player2='ai',
    char1='fox',
    char2='fox',
    stage='battlefield',
)
if platform.system() == 'Windows':
    options['windows'] = True


env = SelfPlayEnv(frame_limit=3600, options=options)
obs = env.reset()
atexit.register(env.close)

r = 0

if __name__ == "__main__":
    env = GoHighEnvVec(args.num_workers, args.total_steps, options)
    print("Action space " + str(env.action_space.n))
    net = Actor(env.observation_space.n, env.action_space.n)
    target_net = Actor(env.observation_space.n, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train(args, net, target_net, optimizer, env)
    
while True:
    action = tuple([random.randint(0, env.action_space.n - 1) for _ in range(2)])
    obs, reward, done, infos = env.step(action)
    r += reward
    if env.obs.frame % 60 == 0:
        print("Reward per second:", round(r, 4))
        r = 0
    if done:
        break

    