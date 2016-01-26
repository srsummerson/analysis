import numpy as np
import tables 
import matplotlib.pyplot as plt



def free_choice_behavior_analysis_old(hdf_file):
	# used when instructed trials came in blocks of three
	hdf = tables.openFile('hdf_file')
	# states
	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	# target info
	targetH = hdf.root.task[:]['targetH']
	targetL = hdf.root.task[:]['targetL']

	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
	ind_target_states = np.ravel(np.nonzero(state == 'target'))
	ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
	num_successful_trials = ind_check_reward_states.size
	target_times = state_time[ind_target_states]

	counter = 1
	time_counter = 1
	running_avg_length = 50



def free_choice_behavior_analysis(hdf_file):
	'''
	Extract data and initialize parameters.
	'''
	hdf = tables.openFile(hdf_file)
	# states
	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	# target info
	targetH = hdf.root.task[:]['targetH']
	targetL = hdf.root.task[:]['targetL']
	# reward schedules
	reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
	reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
	# instructed (1) or free-choice (2) trial
	trial_type = hdf.root.task[:]['target_index']

	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
	ind_target_states = np.ravel(np.nonzero(state == 'target'))
	ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
	num_successful_trials = ind_check_reward_states.size
	target_times = state_time[ind_target_states]
	# creates vector same size of state vectors for comparison
	instructed_or_freechoice = trial_type[state_time[ind_target_states]]  

	counter = 1
	time_counter = 1
	running_avg_length = 50

	target_all = np.zeros(len(ind_check_reward_states))
	reward_all = np.zeros(len(ind_check_reward_states))
	target_freechoice = np.zeros(np.sum(np.equal(instructed_or_freechoice,2)))
	reward_freechoice = np.zeros(len(target_freechoice))

	'''
	Target choices for all (instructed and free-choice) and free-choice only trials
	'''
	for i in range(0,ind_check_reward_states.size):
		target_state = state[ind_check_reward_states[i] - 2]
		trial = trial_type[state_time[ind_check_reward_states[i] -2]]
		if target_state == 'hold_targetL':
			target_all[i] = 1
		else:
			target_all[i] = 2

		reward_all[i] = state[ind_check_reward_states[i]+1] == 'reward'

		if instructed_or_freechoice[i] == 2:
			target_freechoice[counter] = target_all[i]
			reward_freechoice[counter] = reward_all[i]

	'''
	Probability of target selection, reward, and switching for free-choice trials
	'''
	prob_choose_high_freechioce = np.zeros(target_freechoice.size)
	prob_choose_low_freechoice = np.zeros(target_freechoice.size)
	prob_reward_high_frechoice = np.zeros(target_freechoice.size)
	prob_reward_low_freechoice = np.zeros(target_freechoice.size)
	for i in range(0,target_freechoice.size):
		chosen_high_freechoice = target_freechoice[range(np.maximum(1,i+1 - running_avg_length),i)] == 2
		chosen_low_freechoice = target_freechoice[range(np.maximum(1,i+1 - running_avg_length),i)] == 1
		reward_high_freechoice = chosen_high_freechoice and reward_freechoice[range(np.maximum(1,i+1 - running_avg_length),i)]
		reward_low_freechoice = chosen_low_freechoice and reward_freechoice[range(np.maximum(1,i+1 - running_avg_length),i)]

		prob_choose_high_freechioce[i] = sum(chosen_high_freechoice)/np.minimum(i+1,running_avg_length)
		prob_choose_low_freechoice[i] = sum(chosen_low_freechoice)/np.minimum(i+1,running_avg_length)
		prob_reward_high_frechoice[i] = sum(reward_high_freechoice)/sum(chosen_high_freechoice)
		prob_reward_low_freechoice[i] = sum(reward_low_freechoice)/sum(chosen_low_freechoice)

		#lowR_switchH = chosen_high_freechoice[1:] and reward_low_freechoice[:-1]
		#lowNR_switchH = chosen_high_freechoice[1:] and chosen_low_freechoice[:-1] and not reward_low_freechoice[:-1]
		#highR_switchL = chosen_low_freechoice[1:] and reward_high_freechoice[:-1]
		#highNR_switchL = chosen_low_freechoice[1:] and chosen_high_freechoice[:-1] and not reward_high_freechoice[:-1]

		#prob_lowR_switchH[i] = sum(lowR_switchH)/sum(reward_low_freechoice[:-1])
		#prob_lowNR_switchH[i] = sum(lowNR_switchH)/sum(chosen_low_freechoice[:-1] and not reward_low_freechoice[:-1])
		#prob_highR_switchL[i] = sum(highR_switchL)/sum(reward_high_freechoice[:-1])
		#prob_highNR_switchL[i] = sum(highNR_switchL)/sum(chosen_high_freechoice[:-1] and not reward_high_freechoice[:-1])

	'''
	Probability of target select and reward for all trials
	'''
	prob_choose_high_all = np.zeros(target_all.size)
	prob_choose_low_all = np.zeros(target_all.size)
	prob_reward_high_all = np.zeros(target_all.size)
	prob_reward_low_all = np.zeros(target_all.size)
	for i in range(0,target_all.size):
		chosen_high_all = target_all[range(np.maximum(1,i+1 - running_avg_length),i)] == 2
		chosen_low_all = target_all[range(np.maximum(1,i+1 - running_avg_length),i)] == 1
		reward_high_all = chosen_high_all and reward_all[range(np.maximum(1,i+1 - running_avg_length),i)]
		reward_low_all = chosen_low_all and reward_all[range(np.maximum(1,i+1 - running_avg_length),i)]

		prob_choose_high_all[i] = sum(chosen_high_all)/np.minimum(i+1,running_avg_length)
		prob_choose_low_all[i] = sum(chosen_low_all)/np.minimum(i+1,running_avg_lengh)
		prob_reward_high_all[i] = sum(reward_high_all)/sum(chosen_high_all)
		prob_reward_low_all[i] = sum(reward_low_all)/sum(chosen_low_all)

	'''
	Plot results
	'''
	plt.plot(range(1,target_all.size+1),prob_choose_high_all,'b',range(1,target_all.size+1),prob_choose_low_all,'r')
	plt.ylabel('Probability of Target Selection - All')
	plt.xlabel('Trial')
	plt.show()

	plt.plot(range(1,target_all.size+1),prob_reward_high_all,'b',range(1,target_all.size+1),prob_reward_low_all,'r')
	plt.ylabel('Probability of Reward')
	plt.xlabel('Trial')
	plt.show()

	plt.plot(range(1,target_freechoice.size+1),prob_choose_high_freechioce,'b',range(1,target_freechoice.size+1),prob_choose_low_freechoice,'r')
	plt.ylabel('Probability of Target Selection - Free Choice')
	plt.xlabel('Trial')
	plt.show()

	plt.plot(range(1,target_freechoice.size+1),prob_reward_high_frechoice,'b',range(1,target_freechoice.size+1),prob_reward_low_freechoice,'r')
	plt.ylabel('Probability of Reward')
	plt.xlabel('Trial')
	plt.show()

	'''
	Save data
	'''
