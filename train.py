import torch
import torch.nn as nn
from torch.distributions import Categorical
from statistics import mean, stdev
import time
import pandas as pd
import numpy as np

def train(params, net, optimizer, env):
    df = pd.DataFrame(columns = ["time", "reward_mean", "reward_std"])
    print("Resetting envs")
    obs = env.reset()
    print("Envs resetted")
    total_steps = 0
    # n_save = 50000
    n_save = 100
    while total_steps < params.total_steps:
        print("Total steps:", total_steps)
        # print("Gathering rollouts")
        steps, obs, _ = gather_rollout(params, net, env, obs)
        total_steps += params.num_workers * len(steps)
        final_obs = torch.tensor(obs)
        _, final_values = net(final_obs)
        steps.append((None, None, None, final_values))
        # print("Processing rollouts")
        actions, logps, values, returns, advantages = process_rollout(params, steps)

        # print("Updating network")
        update_network(params, net, optimizer, actions, logps, values, returns, advantages)

        if total_steps > n_save:
            _, _, to_print = gather_rollout(params, net, env, obs, prnt=True)
            df = df.append(to_print, ignore_index = True)
            save_model(net, optimizer, "checkpoints/" + str(total_steps) + ".ckpt")
            n_save += 250000
            df.to_csv('checkpoints/reward_'+str(n_save)+'.csv')

    env.close()


def gather_rollout(params, net, env, obs, prnt = False):
    steps = []
    ep_rewards = [0.] * params.num_workers
    t = time.time()
    for _ in range(params.rollout_steps):
        obs = torch.tensor(obs)
        logps, values = net(obs)
        actions = Categorical(logits=logps).sample()

        obs, rewards, dones, _ = env.step(actions.numpy())

        for i, done in enumerate(dones):
            ep_rewards[i] += rewards[i]
        
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        steps.append((rewards, actions, logps, values))

    if prnt:
        to_print = {"time": round(time.time() - t, 3), "reward_mean": round(mean(ep_rewards), 3), "reward_std":round(stdev(ep_rewards), 3)}
        return steps, obs, to_print
        print(to_print)
    return steps, obs, None


def process_rollout(params, steps):
    # bootstrap discounted returns with final value estimates
    _, _, _, last_values = steps[-1]
    returns = last_values.data

    advantages = torch.zeros(params.num_workers, 1)

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, actions, logps, values = steps[t]
        _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * params.gamma

        deltas = rewards + next_values.data * params.gamma - values.data
        advantages = advantages * params.gamma * params.lambd + deltas

        out[t] = actions, logps, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))


def update_network(params, net, optimizer, actions, logps, values, returns, advantages):
    # calculate action probabilities
    log_action_probs = logps.gather(1, actions.unsqueeze(-1))
    probs = logps.exp()
    policy_loss = (-log_action_probs * advantages).sum()
    value_loss = (.5 * (values - returns) ** 2.).sum()
    entropy_loss = (logps * probs).sum()

    loss = policy_loss + value_loss * params.value_coeff + entropy_loss * params.entropy_coeff
    loss.backward()

    nn.utils.clip_grad_norm_(net.parameters(), params.grad_norm_limit)
    optimizer.step()
    optimizer.zero_grad()



def save_model(net, optimizer, PATH):
    torch.save({
            'model': net,
            'optimizer': optimizer,
            }, PATH)