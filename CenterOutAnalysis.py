from numpy import sin, linspace, pi
import matplotlib
import numpy as np
import tables
from matplotlib import pyplot as plt
from basicAnalysis import computeCursorPathLength



def CenterOut_withStressTrials(hdf_file):
	'''
	For CenterOut task with stress trials (joystickmulti_stress), the typical state sequence is the following:
	wait, target, hold, targ_transition, target, hold, targ_transition, reward
	'''

	hdf = tables.openFile(hdf_file)

	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	trial_type = hdf.root.task[:]['target_index']
	stress_type = hdf.root.task[:]['stress_trial']
	  
	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))   # total number of unique trials
	ind_hold_states = np.ravel(np.nonzero(state == 'hold'))   
	ind_target_states = np.ravel(np.nonzero(state == 'target'))
	ind_reward_states = np.ravel(np.nonzero(state == 'reward'))
	
	successful_stress_or_not = np.ravel(stress_type[state_time[ind_reward_states]])
	all_stress_or_not = np.ravel(stress_type[state_time[ind_wait_states]])

	num_trials = ind_wait_states.size
	total_states = state.size

	trial_success = np.zeros(num_trials)
	counter = 0 	# counter increments for all successful trials

	trial_success = np.array([(state[np.minimum(ind_wait_states[i]+7,total_states-1)] == 'reward') for i in range(num_trials)])

	hdf.close()
	return state_time, ind_wait_states, ind_reward_states, all_stress_or_not, successful_stress_or_not
