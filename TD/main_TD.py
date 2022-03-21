# For comparison of TD Learning against our MDP on Frozen Lake
# Source Code from https://gym.openai.com/evaluations/eval_OyMhE4BARAmQDY8ixyZALQ/
import gym
from gym import wrappers
import numpy as np

from helper import *
from DeepTDLambdaLearner import DeepTDLambdaLearner

episodes = 1000

name_of_gym = 'FrozenLake-v1'
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
n_actions = env.action_space.n
n_states = env.observation_space.n

agent = DeepTDLambdaLearner(n_actions=n_actions, n_states=n_states)

# Iterate the game
for e in range(episodes):
    state = env.reset()
    state = package_state(state, name_of_gym)

    total_reward = 0
    done = False
    while not done:
        action, greedy = agent.get_e_greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        # env.render()

        next_state = package_state(next_state)

        # Tweaking the reward to help the agent learn faster
        tweaked_reward = tweak_reward(reward, done)

        agent.learn(state, action, next_state, tweaked_reward, greedy)

        state = next_state
        total_reward += tweaked_reward

        if done:
            if reward == 1:
                print("episode: {}/{}, score: {:.2f} and goal has been found!".format(e, episodes, total_reward))
            else:
                print("episode: {}/{}, score: {:.2f}".format(e, episodes, total_reward))
            break

    agent.reset_e_trace()

print(total_reward)
#print("Success without falling in hole: {}".format(total_success3))
#print("Success without falling in hole: {}".format(success_arr3))

#Print total reward for value iteration
#reward = cumulative_rewards3[len(cumulative_rewards3) - 1]
#print("Total reward: {}".format(reward))
# env.close()