import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from statistics import mean, stdev
import time
import pandas as pd
import numpy as np

def pretrain(params, net, optimizer, env, state_net, optimizer_state):
    df = pd.DataFrame(columns = ["time", "reward_mean", "reward_std"])
    obs = env.reset()
    total_steps = 0
    n_save = 0
    start_time = time.time()
    latest_ckpt = 0

    action_buffer = []
    state_buffer = []

    # while total_steps < params.total_steps:
    while total_steps < 100000000:
        print("Total steps:", total_steps)

        # Gathering rollouts: for 600 steps, run the network in the environment without updating network
        steps, obs, _ = gather_rollout(params, net, env, obs, state_net, optimizer_state, action_buffer, state_buffer)
        total_steps += params.num_workers * len(steps)
        final_obs = torch.tensor(obs)
        _, final_values = net(final_obs)

        print("rollout gathered")

        # Append final values to steps without explicitly updating the resulting rewards, actions, logps
        steps.append((None, None, None, final_values))
        
        # Processing rollouts, get advantages
        actions, logps, values, returns, advantages = process_rollout(params, steps, state_net, optimizer_state, action_buffer, state_buffer)

        print("rollout processed")

        # Update network with process rollout results 
        # gradient ascent on advantage * policy gradient
        update_network(params, net, optimizer, actions, logps, values, returns, advantages)

        if total_steps > n_save:
            _, _, to_print = gather_rollout(params, net, env, obs, prnt=True)
            to_print["time"] = to_print["time"] - start_time
            print(to_print)
            df = df.append(to_print, ignore_index = True)
            save_model(net, optimizer, "checkpoints/" + str(total_steps) + ".ckpt")
            latest_ckpt = total_steps
            n_save += 250000
            df.to_csv('checkpoints/reward_'+str(n_save)+'.csv')

    print("pretrain complete")

    env.close()

    return latest_ckpt

def train(params, net, optimizer, env, n_steps, state_net, optimizer_state):
    df = pd.DataFrame(columns = ["time", "reward_mean", "reward_std"])
    obs = env.reset()
    total_steps = n_steps
    n_save = 250000
    start_time = time.time()

    # while total_steps < params.total_steps:
    while True:
        print("Total steps:", total_steps)

        # Gathering rollouts: for 600 steps, run the network in the environment without updating network
        steps, obs, _ = gather_rollout(params, net, env, obs, state_net, optimizer_state, action_buffer, state_buffer)
        total_steps += params.num_workers * len(steps)
        final_obs = torch.tensor(obs)
        _, final_values = net(final_obs)

        # Append final values to steps without explicitly updating the resulting rewards, actions, logps
        steps.append((None, None, None, final_values))
        
        # Processing rollouts, get advantages
        actions, logps, values, returns, advantages = process_rollout(params, steps)

        # Update network with process rollout results 
        # gradient ascent on advantage * policy gradient
        update_network(params, net, optimizer, actions, logps, values, returns, advantages)

        if total_steps > n_save:
            _, _, to_print = gather_rollout(params, net, env, obs, state_net, optimizer_state, action_buffer, state_buffer, prnt=True)
            to_print["time"] = to_print["time"] - start_time
            print(to_print)
            df = df.append(to_print, ignore_index = True)
            save_model(net, optimizer, "checkpoints/" + str(total_steps) + ".ckpt")
            n_save += 250000
            df.to_csv('checkpoints/reward_'+str(n_save)+'.csv')

    env.close()

def gather_rollout(params, net, env, obs, state_net, optimizer_state, action_buffer, state_buffer, prnt = False):
    """ Obs |> net -> action, values. Action |> env.step -> (reward, taken action (sampled from action_probs), action_probs, values ), obs"""
    steps = []
    ep_rewards = [0.] * params.num_workers

    for _ in range(params.rollout_steps):

        obs = torch.tensor(obs)

        if len(action_buffer) == params.state_offset:
            
            #predicting future state
            with torch.nograd():
                pred_obs = state_net(obs, action_buffer[-1])
        else:
            #might not work
            logps, values = net(obs + obs)

        actions = Categorical(logits=logps).sample() # 1 for each worker

        obs, rewards, dones, _ = env.step(actions.numpy())

        action_buffer.append(actions)
        state_buffer.append(obs)

        update_state_network(params, state_net, optimimer_state, action_buffer, state_buffer)

        for i, done in enumerate(dones):
            ep_rewards[i] += rewards[i]
        
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        steps.append((rewards, actions, logps, values))

    if prnt:
        to_print = {"time": time.time(), "reward_mean": round(mean(ep_rewards), 3), "reward_std":round(stdev(ep_rewards), 3)}
        return steps, obs, to_print

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

        returns = rewards + returns * params.gamma # discounted rewards

        deltas = rewards + next_values.data * params.gamma - values.data # reward + discounted difference of estimated value
        advantages = advantages * params.gamma * params.lambd + deltas # extended advantage estimator, see https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
 
        out[t] = actions, logps, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))


def update_network(params, net, optimizer, actions, logps, values, returns, advantages):
    # calculate action probabilities
    log_action_probs = logps.gather(1, actions.unsqueeze(-1))
    probs = logps.exp()
    # PPO loss from https://openai.com/blog/openai-baselines-ppo/

    ratios = torch.exp(logps[-1] - logps)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - params.epsilon, 1 + params.epsilon) * advantages

    policy_loss = (-torch.min(surr1, surr2)).mean()
    # policy_loss = (-log_action_probs * advantages).sum()
    value_loss = (.5 * (values - returns) ** 2.).sum()
    entropy_loss = (logps * probs).sum()

    loss = policy_loss + value_loss * params.value_coeff + entropy_loss * params.entropy_coeff
    loss.backward()

    nn.utils.clip_grad_norm_(net.parameters(), params.grad_norm_limit)
    optimizer.step()
    optimizer.zero_grad()

def update_state_network(params, state_net, optimimer_state, action_buffer, state_buffer):
    if len(action_buffer) == params.offset:
        pred = state_net(action_buffer[-1] + state_buffer[-1])

        loss = MSEloss(pred, state_buffer[0])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        action_buffer = action_buffer[-1]
        state_buffer = state_buffer[-1]


#need to be adjusted at some point
def save_model(net, optimizer, PATH):
    torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

def load_model(PATH, net_arg_1, net_arg_2, optim_lr):
    model = Actor(net_arg_1, net_arg_2)
    opt = optim.Adam(model.parameters(), optim_lr)

    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    opt.load_state_dict(ckpt['optimizer_state_dict'])
    return model, opt