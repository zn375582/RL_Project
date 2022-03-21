import gym
import timeit
import numpy as np
import matplotlib.pyplot as plt
from MarkovDecisionProcess import MarkovDecisionProcess as MDP
from Agent import *
from Experiment1 import *
from Experiment2 import *

num_runs = 10000

############################################################################

#Run Experiment 1 - Value Iteration:
agent1 = Agent(mdp, 1, 0.000001)
agent1.value_iteration()
cumulative_rewards1, total_success1, success_arr1 = run_experiment(env, agent1, num_runs)
print("Experiment 1.1: Value Iteration")
print(cumulative_rewards1)
print("Success without falling in hole: {}".format(total_success1))
print("Success without falling in hole: {}".format(success_arr1))

#Print total reward for value iteration
value_reward = cumulative_rewards1[len(cumulative_rewards1) - 1]
print(value_reward)


############################################################################

#Run Experiment 1 - Policy Iteration:
agent2 = Agent(mdp, 1, 0.000001)
agent2.policy_iteration()
cumulative_rewards2, total_success2, success_arr2 = run_experiment(env, agent2, num_runs)
print("\nExperiment 1.2: Policy Iteration")
print(cumulative_rewards2)
print("Success without falling in hole: {}".format(total_success2))
print("Success without falling in hole: {}".format(success_arr2))

#Print total reward for value iteration
policy_reward = cumulative_rewards2[len(cumulative_rewards2) - 1]
print(policy_reward)

############################################################################

#Run Experiment 2: -1 reward for falling in hole
agent3 = Agent(mdp, 1, 0.000001)
agent3.value_iteration()
cumulative_rewards3, total_success3, success_arr3 = run_experiment2(env, agent3, num_runs)
print("\nExperiment 2.1: -1 Reward")
print(cumulative_rewards3)
print("Success without falling in hole: {}".format(total_success3))
print("Success without falling in hole: {}".format(success_arr3))

#Print total reward for value iteration
reward = cumulative_rewards3[len(cumulative_rewards3) - 1]
print("Total reward: {}".format(reward))

############################################################################

#Run Experiment 2: -.5 reward for falling in hole
agent4 = Agent(mdp, 1, 0.000001)
agent4.value_iteration()
cumulative_rewards4, total_success4, success_arr4 = run_experiment2_2(env, agent4, num_runs)
print("\nExperiment 2.1: -.5 Reward")
print(cumulative_rewards4)
print("Success without falling in hole: {}".format(total_success4))
print("Success without falling in hole: {}".format(success_arr4))

#Print total reward for value iteration
reward = cumulative_rewards4[len(cumulative_rewards4) - 1]
print("Total reward: {}".format(reward))

#Create Graph with all data
fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
'''
ax.plot(range(num_runs+1), success_arr1, label="Value Iteration:")
#ax.plot(range(num_runs+1), success_arr2, label="Policy Iteration:")
ax.plot(range(num_runs+1), success_arr3, label="-1 Reward for Hole:")
ax.plot(range(num_runs+1), success_arr4, label="-0.5 Reward for Hole:")
'''
ax.plot(range(num_runs+1), cumulative_rewards1, label="Value Iteration:")
ax.plot(range(num_runs+1), cumulative_rewards2, label="Policy Iteration:")
ax.plot(range(num_runs+1), cumulative_rewards3, label="-1 Reward for Hole:")
ax.plot(range(num_runs+1), cumulative_rewards4, label="-0.5 Reward for Hole:")
ax.grid(False)
ax.set_xlabel('Episodes')  # Add an x-label to the axes.
ax.set_ylabel('Cumulative Reward')  # Add a y-label to the axes.
ax.set_title("Solving Frozen-Lake using MDP")  # Add a title to the axes.
ax.legend()  # Add a legend.
#Show final y value
result1 = cumulative_rewards1[num_runs-1]
result2 = cumulative_rewards2[num_runs-1]
result3 = cumulative_rewards3[num_runs]
result4 = cumulative_rewards4[num_runs]
for var in (result1, result3, result4): plt.annotate('%0.0f' % var, xy=(1, var), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.show()

#Compare iterations... Did they find the optimal value function?
#print(agent1.value_fn)
#print(agent2.value_fn)

