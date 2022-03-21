import numpy as np
import gym 
import random
import time
# from IPython.display import clear_output

# define reward adjusting method
def adjust_reward(reward, done):
	if reward == 0:
		reward = 0
	if done:
		if reward < 1:
			reward = -1
	return reward

env = gym.make("FrozenLake-v1")
env.reset()
env.render()

# CHECK THE n's
num_actions = env.action_space.n
num_states = env.observation_space.n
q_table = np.zeros((num_states, num_actions))

num_episodes = 10000
# if by max_num_steps, the agent has not reached the goal, reward of 0 is given
max_num_steps = 100
learning_rate = 0.1
discount_rate = 0.99

# Exploration rate has to initially be set to 1 because all Q-values are initially  0
# epsilon is the exploration rate, epsilon rate decreases over time
epsilon = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

#Count number of successful Episodes
num_successful_episodes = 0

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_num_steps):

        #Exploration-exploitation trade-off
        # Generate a random number and if this number > epsilon, then do exploitation
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > epsilon:
            #exploitation action
            action = np.argmax(q_table[state,:])
        else:
            #exploration action
            action = env.action_space.sample()
        
        # take step with action determined above
        new_state, reward, done, info = env.step(action)

        #adjust reward if doing reward adjustment experiment
        reward = adjust_reward(reward, done)

        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        #update state value to be new current state
        state = new_state
        #update tracking of the current episode's total rewards
        rewards_current_episode += reward

        if reward == 1:
            num_successful_episodes += 1

        if done == True:
            break

    # Exploration rate decay
    epsilon = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    # append total reward of the now done episode to the list of rewards from all episodes
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("Average reward per one thousand episodes:\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

#Print number of successful episodes
print("Number of successful episodes:\n")
print(num_successful_episodes)

# Print updated Q-table
# print("\nResulting Q-table\n")
# print(q_table)