from numpy import sin, linspace, pi
import matplotlib
import numpy as np
import tables
from matplotlib import pyplot as plt


def FreeChoiceTaskPerformance(hdf_file):
	hdf = tables.openFile(hdf_file)

	# Task states
	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	# Target information: high-value target= targetH, low-value target= targetL
	targetH = hdf.root.task[:]['targetH']
	targetL = hdf.root.task[:]['targetL']
	# Reward schedules for each target
	reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
	reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
	# Trial type: instructed (1) or free-choice (2) trial 
	trial_type = hdf.root.task[:]['target_index']

	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
	#ind_target_states = np.ravel(np.nonzero(state == 'target'))
	ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
	ind_target_states = ind_check_reward_states - 3 # only look at target targets when the trial was successful (2 states before reward state)
	num_successful_trials = ind_check_reward_states.size
	target_times = state_time[ind_target_states]
	# creates vector same size of state vectors for comparison. instructed (1) and free-choice (2)
	instructed_or_freechoice = trial_type[state_time[ind_target_states]]
	# creates vector of same size of state vectors for comparision. (0) = small reward, (1) = large reward.
	rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
	rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]
	num_free_choice_trials = sum(instructed_or_freechoice) - num_successful_trials
	# creates vector of same size of target info: maxtrix of num_successful_trials x 3; (position_offset, reward_prob, left/right)
	targetH_info = targetH[state_time[ind_target_states]]
	targetL_info = targetL[state_time[ind_target_states]]

	# Initialize variables use for in performance computation
	counter = 0	# counter for counting number of free-choice trials
	time_counter = 0
	running_avg_length = 50
	target_all = np.zeros(ind_check_reward_states.size) # vector for recording target selection on all trials regardless of type
	reward_all = np.zeros(target_all.size)	# vector for recording whether rewards was allocated on all trials regardless of type
	target_freechoice = np.zeros(num_free_choice_trials)	# vector for recording target selection on free-choice trials only
	reward_freechoice = np.zeros(num_free_choice_trials)	# vector for recording reward allocation on free-choice trials only
	prob_choose_high_freechoice = np.zeros(target_freechoice.size)
	prob_choose_low_freechoice = np.zeros(target_freechoice.size)
	prob_reward_high_freechoice = np.zeros(target_freechoice.size)
	prob_reward_low_freechoice = np.zeros(target_freechoice.size)
	#prob_lowR_switchH = np.zeros(target_freechoice.size)
	#prob_lowNR_switchH = np.zeros(target_freechoice.size)
	#prob_highR_switchL = np.zeros(target_freechoice.size)
	#prob_highNR_switchL = np.zeros(target_freechoice.size)
	prob_choose_high_all = np.zeros(target_all.size)
	prob_choose_low_all = np.zeros(target_all.size)
	prob_reward_high_all = np.zeros(target_all.size)
	prob_reward_low_all = np.zeros(target_all.size)

	"""
	Find target choices for all (instructed and free-choice) and free-choice trials only.
	"""
	for i in range(0,ind_check_reward_states.size):
		target_state = state[ind_check_reward_states[i] - 2]
		trial = instructed_or_freechoice[i]
		if target_state == 'hold_targetL':
			target_all[i] = 1
			reward_all[i] = rewarded_reward_scheduleL[i]
		else:
			target_all[i] = 2
			reward_all[i] = rewarded_reward_scheduleH[i]

		#reward_all[i] = state[ind_check_reward_states[i]+1] == 'reward'
		if trial == 2:
			target_freechoice[counter] = target_all[i]
			reward_freechoice[counter] = reward_all[i]
			counter += 1

	"""
	Compute probabilities of target selection and reward.
	"""
	for i in range(0,target_freechoice.size):
		chosen_high_freechoice = target_freechoice[range(np.maximum(0,i - running_avg_length),i+1)] == 2
		chosen_low_freechoice = target_freechoice[range(np.maximum(0,i - running_avg_length),i+1)] == 1
		reward_high_freechoice = np.logical_and(chosen_high_freechoice,reward_freechoice[range(np.maximum(0,i - running_avg_length),i+1)])
		reward_low_freechoice = np.logical_and(chosen_low_freechoice,reward_freechoice[range(np.maximum(0,i - running_avg_length),i+1)])
    
		#prob_choose_high_freechoice[i] = float(sum(chosen_high_freechoice))/np.minimum(i+1,running_avg_length)
		#prob_choose_low_freechoice[i] = float(sum(chosen_low_freechoice))/np.minimum(i+1,running_avg_length)
		prob_choose_high_freechoice[i] = float(sum(chosen_high_freechoice))/chosen_high_freechoice.size
		prob_choose_low_freechoice[i] = float(sum(chosen_low_freechoice))/chosen_low_freechoice.size
		prob_reward_high_freechoice[i] = float(sum(reward_high_freechoice))/(sum(chosen_high_freechoice) + (sum(chosen_high_freechoice)==0))  # add logic statment to denominator so we never divide by 0
		prob_reward_low_freechoice[i] = float(sum(reward_low_freechoice))/(sum(chosen_low_freechoice) + (sum(chosen_low_freechoice)==0))

	for i in range(0,target_all.size):
		chosen_high_all = target_all[range(np.maximum(0,i - running_avg_length),i+1)] == 2
		chosen_low_all = target_all[range(np.maximum(0,i - running_avg_length),i+1)] == 1
		reward_high_all = np.logical_and(chosen_high_all,reward_all[range(np.maximum(0,i - running_avg_length),i+1)])
		reward_low_all = np.logical_and(chosen_low_all,reward_all[range(np.maximum(0,i - running_avg_length),i+1)])

		prob_choose_high_all[i] = float(sum(chosen_high_all))/np.minimum(i+1,running_avg_length)
		prob_choose_low_all[i] = float(sum(chosen_low_all))/np.minimum(i+1,running_avg_length)
		prob_reward_high_all[i] = float(sum(reward_high_all))/(sum(chosen_high_all) + (sum(chosen_high_all)==0))
		prob_reward_low_all[i] = float(sum(reward_low_all))/(sum(chosen_low_all) + (sum(chosen_low_all)==0))

	"""
	Plot results.
	"""
	plt.figure()
	plt.subplot(121)
	plt.plot(range(1,target_all.size+1),prob_choose_high_all,'b',label='High-value target')
	plt.plot(range(1,target_all.size+1),prob_choose_low_all,'r',label='Low-value target')
	plt.axis([1,target_all.size,0, 1])
	plt.xlabel('Trials')
	plt.ylabel('Probability of Target Selection')
	plt.legend()
	plt.title('All trials')

	plt.subplot(122)
	plt.plot(range(1,target_freechoice.size+1),prob_choose_high_freechoice,'b',label='High-value target')
	plt.plot(range(1,target_freechoice.size+1),prob_choose_low_freechoice,'r',label='Low-value target')
	plt.axis([1,target_freechoice.size, 0, 1])
	plt.xlabel('Trials')
	plt.title('Free-Choice Trials')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_targets_%s.png' % hdf_file[:-4])    # save this filetype for AI editing
	#plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_targets_%s.svg' % hdf_file[:-4])    # save this filetype for easy viewing
	plt.close()

	plt.figure()
	plt.subplot(121)
	plt.plot(range(1,target_all.size+1),prob_reward_high_all,'b',label='High-value target')
	plt.plot(range(1,target_all.size+1),prob_reward_low_all,'r',label='Low-value target')
	plt.axis([1,target_all.size,0, 1])
	plt.xlabel('Trials')
	plt.ylabel('Probability of Reward')
	plt.legend()
	plt.title('All Trials')

	plt.subplot(122)
	plt.plot(range(1,target_freechoice.size+1),prob_reward_high_freechoice,'b',label='High-value target')
	plt.plot(range(1,target_freechoice.size+1),prob_reward_low_freechoice,'r',label='Low-value target')
	plt.axis([1,target_freechoice.size, 0, 1])
	plt.xlabel('Trials')
	plt.legend()
	plt.title('Free-Choice Trials')

	plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_rewards_%s.png' % hdf_file[:-4])    # save this filetype for AI editing
	#plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_rewards_%s.svg' % hdf_file[:-4])    # save this filetype for easy viewing
	plt.close()

	hdf.close()