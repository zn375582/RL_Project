import gym
import timeit
import numpy as np
import matplotlib.pyplot as plt
from MarkovDecisionProcess import MarkovDecisionProcess as MDP
from Agent import *

# Set up environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
env.reset()
mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)


#Goal is to change reward and see if that improves the agent's success rate
def adjust_award(reward, done):
    #if reward == 0:
        #reward = -0.01
    if done:
        if reward < 1:
            reward = -1
    return reward

#See if chaning reward at all changes anything
def adjust_award_half(reward, done):
    #if reward == 0:
    #    reward = -0.01
    if done:
        if reward < 1:
            reward = -.5
    return reward


def run_experiment2(env, agent, num_runs=1, render=False):
    tot_reward = [0]
    success = 0
    tot_success = [0]
    for _ in range(num_runs):
        observation = 0
        done = False
        env.reset()
        render and env.render()
        reward_per_run = 0
        while not done:
            temp_success = success
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            reward_per_run += reward
            # Adjusting the original reward to help the agent learn faster
            adjusted_reward = adjust_award(reward_per_run, done)
            render and env.render()

            #Help understand what's going on
            if done:
                if reward == 1:
                    #print("episode: {}/{}, score: {:.2f} and goal has been found!".format(_, num_runs, adjusted_reward))
                    success += 1
                    #print("episode: {}/{}, score: {:.2f} Succeeded!!!! Success #: {}".format(_, num_runs, adjusted_reward,success))
                #else:
                    #print("episode: {}/{}, score: {:.2f} Fell in hole. Success #: {}".format(_, num_runs, adjusted_reward, success))
                tot_success.append(success)
                break
        env.close()
        tot_reward.append(adjusted_reward + tot_reward[-1])


    return tot_reward, success, tot_success



def run_experiment2_2(env, agent, num_runs=1, render=False):
    tot_reward = [0]
    success = 0
    tot_success = [0]
    for _ in range(num_runs):
        observation = 0
        done = False
        env.reset()
        render and env.render()
        reward_per_run = 0
        while not done:
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            reward_per_run += reward
            # Adjusting the original reward to help the agent learn faste
            adjusted_reward = adjust_award_half(reward_per_run, done)
            render and env.render()

            #Help understand what's going on
            if done:
                if reward == 1:
                    #print("episode: {}/{}, score: {:.2f} and goal has been found!".format(_, num_runs, adjusted_reward))
                    success += 1
                #else:
                    #print("episode: {}/{}, score: {:.2f}".format(_, num_runs, adjusted_reward))
                tot_success.append(success)
                break
        env.close()
        tot_reward.append(adjusted_reward + tot_reward[-1])
    return tot_reward, success, tot_success


agent = Agent(mdp, 1, 0.000001)
agent.value_iteration()

'''
num_runs = 10
cumulative_rewards, total_success, success_arr = run_experiment2(env, agent, num_runs)
print(cumulative_rewards)
print("Success without falling in hole: {}".format(total_success))
tot_holes = num_runs - total_success
print("Fell in {} holes".format(tot_holes))

#Print total reward for value iteration
reward = cumulative_rewards[len(cumulative_rewards) - 1]
print("Total reward: {}".format(reward))
print(success_arr)
'''

#Create graphs to compare experiments (1, 2, 2.2)
