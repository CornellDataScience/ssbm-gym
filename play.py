import torch
from torch.distributions import Categorical
import os
from envs import GoHighEnv
import atexit
import platform
import random

checkpoint = torch.load(os.path.join("checkpoints", "agent.ckpt"))

net = checkpoint["model"]

options = dict(
    render=True,
    player1='ai',
    player2='human',
    char1='fox',
    char2='falco',
    stage='battlefield',
)
if platform.system == 'Windows':
    options["windows"] = True


env = GoHighEnv(frame_limit=7200, options=options)
atexit.register(env.close)

obs = env.reset()
with torch.no_grad():
    while True:
        obs = torch.tensor(obs)
        logps = net(obs)
        # argmax here 
        # actions = Categorical(logits=logps).sample().numpy()
        epsilon = .99

        generate = random.random())
        if (generate < 1 - epsilon):
            actions = torch.argmax(logps) 
        else: 
            # change this to be random later
            actions = torch.randint(0, 9, size = (4,))

        obs, reward, done, infos = env.step(actions)
        if done:
            break