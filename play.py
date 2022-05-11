import torch
from torch.distributions import Categorical
import os
from envs import GoHighEnv
import atexit
import platform

checkpoint = torch.load(os.path.join("checkpoints", "792302400.ckpt"))

net = checkpoint["model"]

options = dict(
    render=True,
    player1='human',
    player2='ai',
    char1='fox',
    char2='fox',
    stage='battlefield',
)
if platform.system() == 'Windows':
    options['windows'] = True

env = GoHighEnv(frame_limit=1234732, options=options)
atexit.register(env.close)

obs = env.reset()
with torch.no_grad():
    while True:
        obs = torch.tensor(obs)
        logps, _ = net(obs)
        actions = Categorical(logits=logps).sample().numpy()
        obs, reward, done, infos = env.step(actions)
        if done:
            break
