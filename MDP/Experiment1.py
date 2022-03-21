import gym
import timeit
import numpy as np
import matplotlib.pyplot as plt
from MarkovDecisionProcess import MarkovDecisionProcess as MDP
from Agent import Agent


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
env.reset()
mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)

def run_experiment(env, agent, num_runs=1, render=False):
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
            render and env.render()

            # Help understand what's going on
            if done:
                if reward == 1:
                    # print("episode: {}/{}, score: {:.2f} and goal has been found!".format(_, num_runs, adjusted_reward))
                    success += 1
                    # print("episode: {}/{}, score: {:.2f} Succeeded!!!! Success #: {}".format(_, num_runs, adjusted_reward,success))
                # else:
                # print("episode: {}/{}, score: {:.2f} Fell in hole. Success #: {}".format(_, num_runs, adjusted_reward, success))
                tot_success.append(success)
        env.close()
        tot_reward.append(reward_per_run + tot_reward[-1])
    return tot_reward, success, tot_success


#agent1 = Agent(mdp, 1, 0.000001)
#agent1.value_iteration()

#agent2 = Agent(mdp, 1, 0.000001)
#agent2.policy_iteration()

#num_runs = 1000
#cumulative_rewards1, total_success = run_experiment(env, agent1, num_runs)
#cumulative_rewards2 = run_experiment(env, agent2, num_runs)
#print(cumulative_rewards1)
#print("Success without falling in hole: {}".format(total_success))
#tot_holes = num_runs - total_success
#print("Fell in {} holes".format(tot_holes))

#Print total reward for value iteration
#value_reward = cumulative_rewards1[len(cumulative_rewards1) - 1]
#print(value_reward)

#Policy iteration
#print(cumulative_rewards2)
#policy_reward = cumulative_rewards2[len(cumulative_rewards2) - 1]
#print(policy_reward)

#Hypthoses: Value iterations for finding the optimal value function will be best.
#MDP will be able to out perform human. Have better efficiency.
# Compare policies
'''
fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
ax.plot(range(num_runs+1), cumulative_rewards1, label="Value Iteration:")
ax.plot(range(num_runs+1), cumulative_rewards2, label="Policy Iteration:")
ax.grid(False)
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Cumulative reward')  # Add a y-label to the axes.
ax.set_title("Solving Frozen-Lake using MDP")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.show()

#Compare iterations... Did they find the optimal value function?
print(agent1.value_fn)
print(agent2.value_fn)
'''
# Grid to compare best spots to step on:
""""
X1 = np.reshape(agent2.value_fn, (4,4))
fig, ax = plt.subplots()
ax.imshow(X1, interpolation="nearest")
plt.show()

X2 = np.reshape(agent2.value_fn, (4,4))
fig, ax = plt.subplots()
ax.imshow(X2, interpolation="nearest")
plt.show()
"""

