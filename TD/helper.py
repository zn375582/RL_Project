# For comparison of TD Learning against our MDP on Frozen Lake
# Source Code from https://gym.openai.com/evaluations/eval_OyMhE4BARAmQDY8ixyZALQ/

import numpy as np

def tweak_reward(reward, done):
	if reward == 0:
		reward = -0.01		#Comment out to fully compare with
	if done:
		if reward < 1:
			reward = -1
	return reward

def package_state(state): # so that we can feed it into the tensorflow graph
    state = convert_to_one_hot(state, 16)
    state = state.reshape(1, -1)
    return state

def convert_to_one_hot(state_number, n_states):
	state = np.zeros((1,n_states))
	state[0][state_number] = 1
	return state
