import torch
import torch.nn as nn
from torch.distributions import Categorical
from statistics import mean, stdev
import time
import pandas as pd
import random
from buffer import ReplayBuffer

def train(params, net, target_net, optimizer, env):
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    df = pd.DataFrame(columns = ["time", "reward_mean", "reward_std"])
    print("Resetting envs")
    obs = env.reset()
    print("Envs resetted")
    total_steps = 0
    buffer = ReplayBuffer(maxsize=1000000)
    # n_save = 50000
    n_save = 100
    while total_steps < params.total_steps:
        print("Total steps:", total_steps)
        # print("Gathering rollouts")
        steps, new_obs, _ = gather_rollout(params, net, env, obs)
        rewards, actions, logps = steps[0]
        # rewards = torch.tensor([rewards])
        obs = torch.tensor(obs)
        actions = torch.tensor(actions)
       #  rewards = torch.tensor(rewards)
        buffer.add_experience(obs, actions, rewards, new_obs)
        total_steps += params.num_workers * len(steps)

        # print("Processing rollouts")
        # make a replay buffer
        # actions, logps, returns, advantages = process_rollout(params, steps)

        # convert things to tensors if needed
        
        
        
        # one step
        # add step to replay buffer
        # sample from buffer
        # use sample to update
        # print("Updating network")
        update_network(params, net, target_net, optimizer, buffer)
        if (total_steps+1)% 30 == 0:
            target_net.load_state_dict(net.state_dict())

        # reward + predicted for next step * gamma 
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
    for _ in range(1):
        obs = torch.tensor(obs)
        logps = net(obs)
        # epsilon argmax
        generate = random.random()
        # filler epsilon 
        epsilon = 0
        # need to get epsilon from somewhere
       # actions = Categorical(logits=logps).sample()
       # print(actions)
        #print(logps.shape)
        
        
        if (generate < 1 - epsilon):
            actions = torch.argmax(logps, dim = 1) 
        else: 
            # change this to be random later
            actions = torch.argmax(logps, dim = 1)
        print(actions)
        obs, rewards, dones, _ = env.step(actions.numpy())

        for i, done in enumerate(dones):
            ep_rewards[i] += rewards[i]
        
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        steps.append((rewards, actions, logps))
        obs = torch.tensor(obs)
      

    if prnt:
        to_print = {"time": round(time.time() - t, 3), "reward_mean": round(mean(ep_rewards), 3), "reward_std":round(stdev(ep_rewards), 3)}
        return steps, obs, to_print
        print(to_print)
    return steps, obs, None


def process_rollout(params, steps):
    # bootstrap discounted returns with final value estimates
    _, _, _, last_values = steps[-1]
    returns = last_values.data

    # uses advantages 
    advantages = torch.zeros(params.num_workers, 1)

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages

    # instead of computing advantage, we want to reward + gamma * max Q hat - Q
    # 
    for t in reversed(range(len(steps) - 1)):
        rewards, actions, logps, values = steps[t]
        _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * params.gamma

        deltas = rewards + next_values.data * params.gamma - values.data
        advantages = advantages * params.gamma * params.lambd + deltas

        out[t] = actions, logps, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))

def get_targets(self, rewards, gamma):
    '''
    def get_targets(self, rewards, next_states, dones):
        # TODO: Get the next Q function targets, as given by the Bellman optimality equation for Q functions.
        maxQs = self.get_max_q(next_states)
        dones = torch.unsqueeze(dones, dim = 1)
        maxQs = torch.unsqueeze(maxQs, dim = 1)
        targets = rewards + self.gamma * (maxQs * (1 - dones))
   
        return targets 
    '''


def update_network(params, net, target_net, optimizer, buffer):
    # calculate action probabilities
    #change batch size later as needed
    sample = buffer.sample(100)
    qVals = net((sample[0]))
    print(qVals.shape)
    actions = sample[1].unsqueeze(1)
    qVals = torch.gather(qVals, dim = 2, index = actions)

    # getting targetvals
    targetVals = target_net(sample[3])
    # if needed for targetVals 
    targetVals = torch.gather(targetVals, dim = 2, index = actions)

    optimizer.zero_grad()
    loss = torch.nn.MSELoss()(qVals, targetVals+sample[2])
    
    # gradient descent update 
    loss.backward()
    optimizer.step()



def save_model(net, optimizer, PATH):
    torch.save({
            'model': net,
            'optimizer': optimizer,
            }, PATH)