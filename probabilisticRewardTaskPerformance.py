from numpy import sin, linspace, pi
import matplotlib
import numpy as np
import tables
from matplotlib import pyplot as plt
from basicAnalysis import computeCursorPathLength


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

	targetH_side = targetH_info[:,2]
	#print len(targetH_side)
	#print len(ind_target_states)

	# Initialize variables use for in performance computation
	counter = 0	# counter for counting number of free-choice trials
	time_counter = 0
	running_avg_length = 20
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
	chose_high = np.ravel(np.nonzero(np.equal(target_freechoice,2)))
	chose_high_reward = reward_freechoice[chose_high]
	chose_low = np.ravel(np.nonzero(np.equal(target_freechoice,1)))
	chose_low_reward = reward_freechoice[chose_low]
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
	"""
	#plt.subplot(122)
	plt.figure()
	plt.plot(range(1,target_freechoice.size+1),prob_choose_high_freechoice,'b',label='High-value target')
	plt.plot(range(1,target_freechoice.size+1),prob_choose_low_freechoice,'r',label='Low-value target')
	plt.axis([1,target_freechoice.size,0,1])
	plt.axis([1,target_freechoice.size, -0.5, 1.5])
	plt.plot(chose_high,0.2*chose_high_reward+1.1,'b|')
	plt.plot(chose_low,-0.2*chose_low_reward-0.1,'r|')
	plt.xlabel('Trials')
	plt.ylabel('Probability of Target Selection')
	plt.title('Free-Choice Trials')
	plt.legend()
	#plt.savefig('C:/Users/Samantha Summerson/Documents/GitHub/analysis/Papa_Performance_figs/FCPerformance_targets_%s.svg' % hdf_file[:-4])    # save this filetype for AI editing
	#plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_targets_%s.png' % hdf_file[:-4])    # save this filetype for easy viewing
	#plt.close()
	plt.show()
	
	plt.figure()
	'''
	plt.subplot(121)
	plt.plot(range(1,target_all.size+1),prob_reward_high_all,'b',label='High-value target')
	plt.plot(range(1,target_all.size+1),prob_reward_low_all,'r',label='Low-value target')
	plt.axis([1,target_all.size,0, 1])
	plt.xlabel('Trials')
	plt.ylabel('Probability of Reward')
	plt.legend()
	plt.title('All Trials')

	plt.subplot(122)
	'''
	plt.plot(range(1,target_freechoice.size+1),prob_reward_high_freechoice,'b',label='High-value target')
	plt.plot(range(1,target_freechoice.size+1),prob_reward_low_freechoice,'r',label='Low-value target')
	plt.axis([1,target_freechoice.size, 0, 1])
	plt.xlabel('Trials')
	plt.ylabel('Probability of Reward')
	plt.legend()
	plt.title('Free-Choice Trials')

	#plt.savefig('C:/Users/Samantha Summerson/Documents/GitHub/analysis/Papa_Performance_figs/FCPerformance_rewards_%s.svg' % hdf_file[:-4])    # save this filetype for AI editing
	#plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_rewards_%s.png' % hdf_file[:-4])    # save this filetype for easy viewing
	#plt.close()
	plt.show()
	
	hdf.close()

	return

def FreeChoicePilotTaskPerformance(hdf_file):
    hdf = tables.openFile(hdf_file)
    counter_block1 = 0
    counter_block3 = 0
    running_avg_length = 10

    state = hdf.root.task_msgs[:]['msg']
    state_time = hdf.root.task_msgs[:]['time']
    trial_type = hdf.root.task[:]['target_index']
    # reward schedules
    reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
    reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
      
    ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
    ind_target_states = np.ravel(np.nonzero(state == 'target'))
    ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
    num_successful_trials = ind_check_reward_states.size
    instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]
    rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
    rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]

    target1 = np.zeros(100)
    reward1 = np.zeros(target1.size)
    target3 = np.zeros(ind_check_reward_states.size-200)
    #target3 = np.zeros(np.min([num_successful_trials-200,100]))
    reward3 = np.zeros(target3.size)
    target_freechoice_block1 = np.zeros(70)
    reward_freechoice_block1 = np.zeros(70)
    target_freechoice_block3 = []
    reward_freechoice_block3 = []
    #reward_freechoice_block3 = np.zeros(np.min([num_successful_trials-200,100]))
    trial1 = np.zeros(target1.size)
    trial3 = np.zeros(target3.size)
    stim_trials = np.zeros(target3.size)

    '''
    Target choices for all (free-choice only) and associated reward assignments
    '''
    for i in range(0,100):
        target_state1 = state[ind_check_reward_states[i] - 2]
        trial1[i] = instructed_or_freechoice[i]
        if target_state1 == 'hold_targetL':
            target1[i] = 1
            reward1[i] = rewarded_reward_scheduleL[i]
        else:
            target1[i] = 2
            reward1[i] = rewarded_reward_scheduleH[i]
        if trial1[i] == 2:
            target_freechoice_block1[counter_block1] = target1[i]
            reward_freechoice_block1[counter_block1] = reward1[i]
            counter_block1 += 1
    for i in range(200,num_successful_trials):
    #for i in range(200,np.min([num_successful_trials,300])):
        target_state3 = state[ind_check_reward_states[i] - 2]
        trial3[i-200] = instructed_or_freechoice[i]
        if target_state3 == 'hold_targetL':
            target3[i-200] = 1
            reward3[i-200] = rewarded_reward_scheduleL[i]
            if trial3[i-200]==1:   # instructed trial to low-value targer paired with stim
                stim_trials[i-200] = 1
            else:
                stim_trials[i-200] = 0
        else:
            target3[i-200] = 2
            reward3[i-200] = rewarded_reward_scheduleH[i]
            stim_trials[i-200] = 0
        if trial3[i-200] == 2:
            target_freechoice_block3.append(target3[i-200])
            reward_freechoice_block3.append(reward3[i-200])
            counter_block3 += 1
    target_freechoice_block3 = np.array(target_freechoice_block3)
    reward_freechoice_block3 = np.array(reward_freechoice_block3)
    prob_choose_high_freechoice_block1 = np.zeros(len(target_freechoice_block1))
    prob_choose_low_freechoice_block1 = np.zeros(len(target_freechoice_block1))
    prob_reward_high_freechoice_block1 = np.zeros(len(target_freechoice_block1))
    prob_reward_low_freechoice_block1 = np.zeros(len(target_freechoice_block1))
    prob_choose_high_freechoice_block3 = np.zeros(len(target_freechoice_block3))
    prob_choose_low_freechoice_block3 = np.zeros(len(target_freechoice_block3))
    prob_reward_high_freechoice_block3 = np.zeros(len(target_freechoice_block3))
    prob_reward_low_freechoice_block3 = np.zeros(len(target_freechoice_block3))

    for i in range(0,len(target_freechoice_block1)):
        chosen_high_freechoice = target_freechoice_block1[range(np.maximum(0,i - running_avg_length),i+1)] == 2
        chosen_low_freechoice = target_freechoice_block1[range(np.maximum(0,i - running_avg_length),i+1)] == 1
        reward_high_freechoice = np.logical_and(chosen_high_freechoice,reward_freechoice_block1[range(np.maximum(0,i - running_avg_length),i+1)])
        reward_low_freechoice = np.logical_and(chosen_low_freechoice,reward_freechoice_block1[range(np.maximum(0,i - running_avg_length),i+1)])
    
        #prob_choose_high_freechoice[i] = float(sum(chosen_high_freechoice))/np.minimum(i+1,running_avg_length)
        #prob_choose_low_freechoice[i] = float(sum(chosen_low_freechoice))/np.minimum(i+1,running_avg_length)
        prob_choose_high_freechoice_block1[i] = float(sum(chosen_high_freechoice))/chosen_high_freechoice.size
        prob_choose_low_freechoice_block1[i] = float(sum(chosen_low_freechoice))/chosen_low_freechoice.size
        prob_reward_high_freechoice_block1[i] = float(sum(reward_high_freechoice))/(sum(chosen_high_freechoice) + (sum(chosen_high_freechoice)==0))  # add logic statment to denominator so we never divide by 0
        prob_reward_low_freechoice_block1[i] = float(sum(reward_low_freechoice))/(sum(chosen_low_freechoice) + (sum(chosen_low_freechoice)==0))

    for i in range(0,len(target_freechoice_block3)):
        chosen_high_freechoice = target_freechoice_block3[range(np.maximum(0,i - running_avg_length),i+1)] == 2
        chosen_low_freechoice = target_freechoice_block3[range(np.maximum(0,i - running_avg_length),i+1)] == 1
        reward_high_freechoice = np.logical_and(chosen_high_freechoice,reward_freechoice_block3[range(np.maximum(0,i - running_avg_length),i+1)])
        reward_low_freechoice = np.logical_and(chosen_low_freechoice,reward_freechoice_block3[range(np.maximum(0,i - running_avg_length),i+1)])
    
        #prob_choose_high_freechoice[i] = float(sum(chosen_high_freechoice))/np.minimum(i+1,running_avg_length)
        #prob_choose_low_freechoice[i] = float(sum(chosen_low_freechoice))/np.minimum(i+1,running_avg_length)
        prob_choose_high_freechoice_block3[i] = float(sum(chosen_high_freechoice))/chosen_high_freechoice.size
        prob_choose_low_freechoice_block3[i] = float(sum(chosen_low_freechoice))/chosen_low_freechoice.size
        prob_reward_high_freechoice_block3[i] = float(sum(reward_high_freechoice))/(sum(chosen_high_freechoice) + (sum(chosen_high_freechoice)==0))  # add logic statment to denominator so we never divide by 0
        prob_reward_low_freechoice_block3[i] = float(sum(reward_low_freechoice))/(sum(chosen_low_freechoice) + (sum(chosen_low_freechoice)==0))

    chose_high_block1 = np.ravel(np.nonzero(np.equal(target_freechoice_block1,2)))
    chose_high_reward_block1 = reward_freechoice_block1[chose_high_block1]
    chose_low_block1 = np.ravel(np.nonzero(np.equal(target_freechoice_block1,1)))
    chose_low_reward_block1 = reward_freechoice_block1[chose_low_block1]

    chose_high_block3 = np.ravel(np.nonzero(np.equal(target_freechoice_block3,2)))
    chose_high_reward_block3 = reward_freechoice_block3[chose_high_block3]
    chose_low_block3 = np.ravel(np.nonzero(np.equal(target_freechoice_block3,1)))
    chose_low_reward_block3 = reward_freechoice_block3[chose_low_block3]

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(range(1,len(target_freechoice_block1)+1),prob_choose_high_freechoice_block1,'b',label='High-value target')
    plt.plot(range(1,len(target_freechoice_block1)+1),prob_choose_low_freechoice_block1,'r',label='Low-value target')
    plt.axis([1,target_freechoice_block1.size,0,1])
    plt.axis([1,target_freechoice_block1.size, -0.5,1.5])
    plt.plot(chose_high_block1,0.2*chose_high_reward_block1+1.1,'b|')
    plt.plot(chose_low_block1,-0.2*chose_low_reward_block1-0.1,'r|')
    plt.xlabel('Trials')
    plt.ylabel('Probability of Target Selection')
    plt.title('Block A: Free-Choice Trials')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(range(1,len(target_freechoice_block3)+1),prob_choose_high_freechoice_block3,'b',label='High-value target')
    plt.plot(range(1,len(target_freechoice_block3)+1),prob_choose_low_freechoice_block3,'r',label='Low-value target')
    plt.axis([1,target_freechoice_block3.size,0,1])
    plt.axis([1,target_freechoice_block3.size, -0.5,1.5])
    plt.plot(chose_high_block3,0.2*chose_high_reward_block3+1.1,'b|')
    plt.plot(chose_low_block3,-0.2*chose_low_reward_block3-0.1,'r|')
    plt.xlabel('Trials')
    plt.ylabel('Probability of Target Selection')
    plt.title("Block A': Free-Choice Trials")
    plt.legend()
    
    plt.subplot(2,2,3)
    plt.plot(range(1,len(target_freechoice_block1)+1),prob_reward_high_freechoice_block1,'b',label='High-value target')
    plt.plot(range(1,len(target_freechoice_block1)+1),prob_reward_low_freechoice_block1,'r',label='Low-value target')
    plt.axis([1,target_freechoice_block1.size,0,1])
    plt.axis([1,target_freechoice_block1.size, 0,1])
    plt.xlabel('Trials')
    plt.ylabel('Probability of Reward')
    plt.title('Block A: Free-Choice Trials')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(range(1,len(target_freechoice_block3)+1),prob_reward_high_freechoice_block3,'b',label='High-value target')
    plt.plot(range(1,len(target_freechoice_block3)+1),prob_reward_low_freechoice_block3,'r',label='Low-value target')
    plt.axis([1,target_freechoice_block3.size,0,1])
    plt.axis([1,target_freechoice_block3.size, 0,1])
    plt.xlabel('Trials')
    plt.ylabel('Probability of Reward')
    plt.title("Block A': Free-Choice Trials")
    plt.legend()
    

    #plt.savefig('C:/Users/Samantha Summerson/Documents/GitHub/analysis/Papa_Performance_figs/FCPerformance_targets_%s.svg' % hdf_file[:-4])    # save this filetype for AI editing
    plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/FCPerformance_targets_%s.svg' % hdf_file[:-4])    # save this filetype for easy viewing

    #plt.savefig('C:/Users/Samantha Summerson/Documents/GitHub/analysis/Papa_Performance_figs/FCPerformance_targets_%s.svg' % hdf_file[:-4])    # save this filetype for AI editing
    #plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_targets_%s.svg' % hdf_file[:-4])    # save this filetype for easy viewing

    #plt.savefig('/home/srsummerson/code/analysis/Luigi_Performance_figs/FCPerformance_targets_%s.png' % hdf_file[:-4])    # save this filetype for easy viewing
    #plt.close()
    hdf.close()
    return 

def FreeChoicePilotTask_LowValueChoiceProb(hdf_file):
    hdf = tables.openFile(hdf_file)
    counter_block1 = 0
    counter_block3 = 0
    running_avg_length = 10

    state = hdf.root.task_msgs[:]['msg']
    state_time = hdf.root.task_msgs[:]['time']
    trial_type = hdf.root.task[:]['target_index']
    # reward schedules
    reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
    reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
      
    ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
    ind_target_states = np.ravel(np.nonzero(state == 'target'))
    ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
    num_successful_trials = ind_check_reward_states.size
    instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]
    rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
    rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]

    target1 = np.zeros(100)
    reward1 = np.zeros(target1.size)
    target3 = np.zeros(ind_check_reward_states.size-200)
    #target3 = np.zeros(np.min([num_successful_trials-200,100]))
    reward3 = np.zeros(target3.size)
    target_freechoice_block1 = np.zeros(70)
    reward_freechoice_block1 = np.zeros(70)
    target_freechoice_block3 = []
    reward_freechoice_block3 = []
    #reward_freechoice_block3 = np.zeros(np.min([num_successful_trials-200,100]))
    trial1 = np.zeros(target1.size)
    trial3 = np.zeros(target3.size)
    stim_trials = np.zeros(target3.size)

    '''
    Target choices for all (free-choice only) and associated reward assignments
    '''
    for i in range(0,100):
        target_state1 = state[ind_check_reward_states[i] - 2]
        trial1[i] = instructed_or_freechoice[i]
        if target_state1 == 'hold_targetL':
            target1[i] = 1
            reward1[i] = rewarded_reward_scheduleL[i]
        else:
            target1[i] = 2
            reward1[i] = rewarded_reward_scheduleH[i]
        if trial1[i] == 2:
            target_freechoice_block1[counter_block1] = target1[i]
            reward_freechoice_block1[counter_block1] = reward1[i]
            counter_block1 += 1
    for i in range(200,num_successful_trials):
    #for i in range(200,np.min([num_successful_trials,300])):
        target_state3 = state[ind_check_reward_states[i] - 2]
        trial3[i-200] = instructed_or_freechoice[i]
        if target_state3 == 'hold_targetL':
            target3[i-200] = 1
            reward3[i-200] = rewarded_reward_scheduleL[i]
            if trial3[i-200]==1:   # instructed trial to low-value targer paired with stim
                stim_trials[i-200] = 1
            else:
                stim_trials[i-200] = 0
        else:
            target3[i-200] = 2
            reward3[i-200] = rewarded_reward_scheduleH[i]
            stim_trials[i-200] = 0
        if trial3[i-200] == 2:
            target_freechoice_block3.append(target3[i-200])
            reward_freechoice_block3.append(reward3[i-200])
            counter_block3 += 1
    

    block1_prob_choose_low = np.sum(target_freechoice_block1==1)/float(len(target_freechoice_block1))
    block3_prob_choose_low = np.sum(np.equal(target_freechoice_block3,1))/float(len(target_freechoice_block3))

    hdf.close()

    return	block1_prob_choose_low,block3_prob_choose_low

def FreeChoiceTask_Behavior(hdf_file):
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
	running_avg_length = 20
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
	
	hdf.close()

	return reward_all, target_all, instructed_or_freechoice

def FreeChoicePilotTask_Behavior(hdf_file):
    hdf = tables.openFile(hdf_file)
    counter_block1 = 0
    counter_block3 = 0
    running_avg_length = 20

    state = hdf.root.task_msgs[:]['msg']
    state_time = hdf.root.task_msgs[:]['time']
    trial_type = hdf.root.task[:]['target_index']
    # reward schedules
    reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
    reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
    # Target information: high-value target= targetH, low-value target= targetL
    targetH = hdf.root.task[:]['targetH']
    targetL = hdf.root.task[:]['targetL']
      
    ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
    ind_target_states = np.ravel(np.nonzero(state == 'target'))
    ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
    num_successful_trials = ind_check_reward_states.size
    instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]
    rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
    rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]

    target1 = np.zeros(100)
    reward1 = np.zeros(target1.size)
    target3 = np.zeros(ind_check_reward_states.size-200)
    #target3 = np.zeros(np.min([num_successful_trials-200,100]))
    reward3 = np.zeros(target3.size)
    target_freechoice_block1 = np.zeros(70)
    reward_freechoice_block1 = np.zeros(70)
    target_freechoice_block3 = []
    reward_freechoice_block3 = []
    #reward_freechoice_block3 = np.zeros(np.min([num_successful_trials-200,100]))
    trial1 = np.zeros(target1.size)
    target_side1 = np.zeros(target1.size)
    trial3 = np.zeros(target3.size)
    target_side3 = np.zeros(target3.size)
    stim_trials = np.zeros(target3.size)

    targetH_info = targetH[state_time[ind_target_states]]
    targetL_info = targetL[state_time[ind_target_states]]

    targetH_side = targetH_info[:,2]


    '''
    Target choices for all (free-choice only) and associated reward assignments
    '''
    for i in range(0,100):
        target_state1 = state[ind_check_reward_states[i] - 2]
        trial1[i] = instructed_or_freechoice[i]
        target_side1[i] = targetH_side[i]
        if target_state1 == 'hold_targetL':
            target1[i] = 1
            reward1[i] = rewarded_reward_scheduleL[i]
        else:
            target1[i] = 2
            reward1[i] = rewarded_reward_scheduleH[i]
        if trial1[i] == 2:
            target_freechoice_block1[counter_block1] = target1[i]
            reward_freechoice_block1[counter_block1] = reward1[i]
            counter_block1 += 1
    for i in range(200,num_successful_trials):
    #for i in range(200,np.min([num_successful_trials,300])):
        target_state3 = state[ind_check_reward_states[i] - 2]
        trial3[i-200] = instructed_or_freechoice[i]
        target_side3[i-200] = targetH_side[i]
        if target_state3 == 'hold_targetL':
            target3[i-200] = 1
            reward3[i-200] = rewarded_reward_scheduleL[i]
        else:
            target3[i-200] = 2
            reward3[i-200] = rewarded_reward_scheduleH[i]
            stim_trials[i-200] = 0
        if trial3[i-200]==1:   # instructed trial paired with stim
            stim_trials[i-200] = 1
        else:
            stim_trials[i-200] = 0
        if trial3[i-200] == 2:
            target_freechoice_block3.append(target3[i-200])
            reward_freechoice_block3.append(reward3[i-200])
            counter_block3 += 1
    target_freechoice_block3 = np.array(target_freechoice_block3)
    reward_freechoice_block3 = np.array(reward_freechoice_block3)

    instructed_or_freechoice_block1 = instructed_or_freechoice[0:100]
    instructed_or_freechoice_block3 = instructed_or_freechoice[200:num_successful_trials]
    
    hdf.close()
    return reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials

def FreeChoicePilotTask_Behavior_ProbChooseLow(hdf_file):
	running_avg_length = 20
	reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_file)

	free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
	target_freechoice_block3 = target3[free_choice_ind]
	reward_freechoice_block3 = reward3[free_choice_ind]
	
	prob_choose_low_freechoice_block3 = np.zeros(len(target_freechoice_block3))
	prob_reward_low_freechoice_block3 = np.zeros(len(target_freechoice_block3))


	for i in range(0,len(target_freechoice_block3)):
		chosen_low_freechoice = target_freechoice_block3[range(np.maximum(0,i - running_avg_length),i+1)] == 1
		reward_low_freechoice = np.logical_and(chosen_low_freechoice,reward_freechoice_block3[range(np.maximum(0,i - running_avg_length),i+1)])
		prob_choose_low_freechoice_block3[i] = float(np.sum(chosen_low_freechoice))/len(chosen_low_freechoice)
		prob_reward_low_freechoice_block3[i] = float(np.sum(reward_low_freechoice))/(np.sum(chosen_low_freechoice) + (np.sum(chosen_low_freechoice)==0))
	
	return prob_choose_low_freechoice_block3, prob_reward_low_freechoice_block3


def PeriStimulusFreeChoiceBehavior(hdf_file):
	reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_file)

	stim_ind = np.ravel(np.nonzero(stim_trials))
	first_100 = np.less(stim_ind,100)
	stim_trial_ind = stim_ind[0:np.sum(first_100)]

	num_stim_trials = len(stim_trial_ind)
	aligned_lv_choices = np.zeros((num_stim_trials,5))  # look at most five free-choice trials out of from stim trial
	aligned_lv_choices_rewarded = np.zeros((num_stim_trials,5))   # look at five free-choice trials out from stim trial when stim trial was rewarded
	aligned_lv_choices_unrewarded = np.zeros((num_stim_trials,5))
	number_aligned_choices = np.zeros(5)
	number_aligned_choices_rewarded = np.zeros(5)
	number_aligned_choices_unrewarded = np.zeros(5)
	counter_rewarded = 0
	counter_unrewarded = 0
	counter_stimtrials_used = 0

	for i in range(0,num_stim_trials-1):
		ind_stim = stim_trial_ind[i]
		max_ind_out = np.min([5,stim_trial_ind[i+1]-stim_trial_ind[i]-1])
		if max_ind_out > 0:
			aligned_lv_choices[i,0:max_ind_out] = (2 - target3[ind_stim+1:ind_stim+max_ind_out+1])
			number_aligned_choices[0:max_ind_out] += np.ones(max_ind_out)

			if (reward3[ind_stim]==1):  # reward paired with stim
				aligned_lv_choices_rewarded[counter_rewarded,0:max_ind_out] = aligned_lv_choices[i,0:max_ind_out]
				number_aligned_choices_rewarded[0:max_ind_out] += np.ones(max_ind_out)
				counter_rewarded += 1
			else:
				aligned_lv_choices_unrewarded[counter_unrewarded,0:max_ind_out] = aligned_lv_choices[i,0:max_ind_out]
				number_aligned_choices_unrewarded[0:max_ind_out] += np.ones(max_ind_out)
				counter_unrewarded += 1 
			counter_stimtrials_used += 1
		else:
			aligned_lv_choices[i,:] = np.zeros(5)

	prob_choose_low_aligned = np.sum(aligned_lv_choices,axis=0)
	prob_choose_low_aligned = prob_choose_low_aligned/number_aligned_choices

	prob_stim_rewarded = float(counter_rewarded)/(counter_stimtrials_used)  # didn't include data from last stim trial
	prob_stim_unrewarded = float(counter_unrewarded)/(counter_stimtrials_used)

	prob_choose_low_aligned_rewarded = np.sum(aligned_lv_choices_rewarded[0:counter_rewarded,:],axis=0)
	prob_choose_low_aligned_rewarded = prob_choose_low_aligned_rewarded/np.max([np.ones(5), number_aligned_choices_rewarded],axis=0) # conditional probability: p(choose low | rewarded)
	#print 'Condition prob - reward:', prob_choose_low_aligned_rewarded
	prob_choose_low_aligned_rewarded = prob_stim_rewarded*prob_choose_low_aligned_rewarded  # joint probability: p(choose low, rewarded)
	#print 'Joint prob - reward:', prob_choose_low_aligned_rewarded
	prob_choose_low_aligned_unrewarded = np.sum(aligned_lv_choices_unrewarded[0:counter_unrewarded,:],axis=0)
	prob_choose_low_aligned_unrewarded = prob_choose_low_aligned_unrewarded/number_aligned_choices_unrewarded  # conditional probability
	#print 'Condition prob - unrewarded', prob_choose_low_aligned_unrewarded
	prob_choose_low_aligned_unrewarded = prob_stim_unrewarded*prob_choose_low_aligned_unrewarded
	#print 'Joint prob - unrewarded', prob_choose_low_aligned_unrewarded


	return prob_choose_low_aligned, prob_choose_low_aligned_rewarded, prob_choose_low_aligned_unrewarded


def FreeChoiceBehaviorConditionalProbabilities(hdf_file):

	reward1, target_block1, trial_block1, target_side1, reward3, target_block3, trial_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_file)

	'''
	Target_side1 is the side that the HV target is on in Block 1.
	Target_side3 is the side that the HV target is on in Block 3
	'''

	fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
	fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))

	prob_hv_on_left_block1 = 1 - np.sum(target_side1[fc_trial_ind_block1])/len(target_side1[fc_trial_ind_block1])  # prob(hv target is on left )
	choose_left_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],1))  
	prob_choose_left_and_low_block1 = float(np.sum(choose_left_and_low_block1))/len(choose_left_and_low_block1)   # joint prob: prob(choose left and low-value)
	choose_left_and_high_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],2))
	prob_choose_left_and_high_block1 = float(np.sum(choose_left_and_high_block1))/len(choose_left_and_high_block1)  # joint prob: prob(choose left and high-value)
	#prob_choose_left_block1 = prob_choose_left_and_low_block1 + prob_choose_left_and_high_block1  # prob(choosing left  target)

	prob_hv_on_right_block1 = np.sum(target_side1[fc_trial_ind_block1])/len(target_side1[fc_trial_ind_block1])  # prob(hv target is on right )
	choose_right_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],1)&np.equal(target_block1[fc_trial_ind_block1],1))  
	prob_choose_right_and_low_block1 = float(np.sum(choose_right_and_low_block1))/len(choose_right_and_low_block1)   # joint prob: prob(choose left and low-value)
	choose_right_and_high_block1 = (np.equal(target_side1[fc_trial_ind_block1],1)&np.equal(target_block1[fc_trial_ind_block1],2))
	prob_choose_right_and_high_block1 = float(np.sum(choose_right_and_high_block1))/len(choose_right_and_high_block1)  # joint prob: prob(choose left and high-value)
	#prob_choose_right_block1 = prob_choose_right_and_low_block1 + prob_choose_right_and_high_block1
	
	prob_high_given_rhs_block1 = prob_choose_right_and_high_block1/prob_hv_on_right_block1  # conditional prob: prob(high-value | high value on right)
	prob_high_given_lhs_block1 = prob_choose_left_and_high_block1/prob_hv_on_left_block1  # conditional prob: prob(high-vale | high value on left)
	prob_low_given_rhs_block1 = prob_choose_right_and_low_block1/prob_hv_on_left_block1  # conditional prob: prob(low-value | low value on right) = prob(low value | high value on left)
	prob_low_given_lhs_block1 = prob_choose_left_and_low_block1/prob_hv_on_right_block1 # conditional prob: prob(low-value | low value on left) = prob(low value | high value on right)
	'''
	prob_high_given_rhs_block1 = prob_choose_right_and_high_block1  # conditional prob: prob(high-value | choose right)
	prob_high_given_lhs_block1 = prob_choose_left_and_high_block1  # conditional prob: prob(high-vale | choose left)
	prob_low_given_rhs_block1 = prob_choose_right_and_low_block1  # conditional prob: prob(low-value | choose right)
	prob_low_given_lhs_block1 = prob_choose_left_and_low_block1 # conditional prob: prob(low-value | choose left)
	'''
	prob_hv_on_left_block3 = 1 - np.sum(target_side3[fc_trial_ind_block3])/len(target_side3[fc_trial_ind_block3])  # prob(hv on left )
	choose_left_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],1))  
	prob_choose_left_and_low_block3 = float(np.sum(choose_left_and_low_block3))/len(choose_left_and_low_block3)   # joint prob: prob(choose left and low-value)
	choose_left_and_high_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],2))
	prob_choose_left_and_high_block3 = float(np.sum(choose_left_and_high_block3))/len(choose_left_and_high_block3)  # joint prob: prob(choose left and high-value)
	#prob_choose_left_block3 = prob_choose_left_and_low_block3 + prob_choose_left_and_high_block3

	prob_hv_on_right_block3 = np.sum(target_side3[fc_trial_ind_block3])/len(target_side3[fc_trial_ind_block3])  # prob(hv on right )
	choose_right_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],1)&np.equal(target_block3[fc_trial_ind_block3],1))  
	prob_choose_right_and_low_block3 = float(np.sum(choose_right_and_low_block3))/len(choose_right_and_low_block3)   # joint prob: prob(choose left and low-value)
	choose_right_and_high_block3 = (np.equal(target_side3[fc_trial_ind_block3],1)&np.equal(target_block3[fc_trial_ind_block3],2))
	prob_choose_right_and_high_block3 = float(np.sum(choose_right_and_high_block3))/len(choose_right_and_high_block3)  # joint prob: prob(choose left and high-value)
	#prob_choose_right_block3 = prob_choose_right_and_low_block3 + prob_choose_right_and_high_block3
	
	prob_high_given_rhs_block3 = prob_choose_right_and_high_block3/prob_hv_on_right_block3  # conditional prob: prob(high-value | hv on right)
	prob_high_given_lhs_block3 = prob_choose_left_and_high_block3/prob_hv_on_left_block3  # conditional prob: prob(high-vale | hv on left)
	prob_low_given_rhs_block3 = prob_choose_right_and_low_block3/prob_hv_on_left_block3  # conditional prob: prob(low-value | lv on right)
	prob_low_given_lhs_block3 = prob_choose_left_and_low_block3/prob_hv_on_right_block3 # conditional prob: prob(low-value | lv on left)
	'''
	prob_high_given_rhs_block3 = prob_choose_right_and_high_block3  # conditional prob: prob(high-value | choose right)
	prob_high_given_lhs_block3 = prob_choose_left_and_high_block3  # conditional prob: prob(high-vale | choose left)
	prob_low_given_rhs_block3 = prob_choose_right_and_low_block3  # conditional prob: prob(low-value | choose right)
	prob_low_given_lhs_block3 = prob_choose_left_and_low_block3 # conditional prob: prob(low-value | choose left)
	'''
	return prob_high_given_rhs_block1, prob_high_given_lhs_block1, prob_high_given_rhs_block3,prob_high_given_lhs_block3, prob_low_given_rhs_block1,prob_low_given_lhs_block1,prob_low_given_rhs_block3,prob_low_given_lhs_block3

def FreeChoiceBehaviorConditionalProbabilitiesAllBlocks(hdf_file):

	reward1, target_block1, trial_block1, target_side1, reward3, target_block3, trial_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_file)

	'''
	Target_side1 is the side that the HV target is on in Block 1.
	Target_side3 is the side that the HV target is on in Block 3
	'''

	fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
	fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))

	left1 = (np.equal(target_side1[fc_trial_ind_block1],0))
	choose_left_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],1))
	left3 = (np.equal(target_side3[fc_trial_ind_block3],0))  
	choose_left_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],1)) 
	prob_choose_left_and_low = float(np.sum(choose_left_and_low_block1) + np.sum(choose_left_and_low_block3))/(len(choose_left_and_low_block1) + len(choose_left_and_low_block3))   # joint prob: prob(choose left and low-value)
	
	right1 = (np.equal(target_side1[fc_trial_ind_block1],1))
	choose_right_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],1)&np.equal(target_block1[fc_trial_ind_block1],1))
	right3 = (np.equal(target_side3[fc_trial_ind_block3],1))  
	choose_right_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],1)&np.equal(target_block3[fc_trial_ind_block3],1)) 
	prob_choose_right_and_low = float(np.sum(choose_right_and_low_block1) + np.sum(choose_right_and_low_block3))/(len(choose_right_and_low_block1) + len(choose_right_and_low_block3))   # joint prob: prob(choose left and low-value)
	
	return prob_choose_left_and_low, prob_choose_right_and_low

def FreeChoiceBehavior_withStressTrials(hdf_file):
	hdf = tables.openFile(hdf_file)

	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	trial_type = hdf.root.task[:]['target_index']
	stress_type = hdf.root.task[:]['stress_trial']
	# reward schedules
	reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
	reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
	  
	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))   # total number of unique trials
	ind_center_states = np.ravel(np.nonzero(state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
	ind_target_states = np.ravel(np.nonzero(state == 'target'))
	ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
	instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]	# free choice trial = 2, instructed = 1
	all_instructed_or_freechoice = trial_type[state_time[ind_center_states]]
	successful_stress_or_not = np.ravel(stress_type[state_time[ind_check_reward_states]])
	all_stress_or_not = np.ravel(stress_type[state_time[ind_center_states]])
	rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_check_reward_states]]
	rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_check_reward_states]]

	num_trials = ind_center_states.size
	total_states = state.size

	trial_success = np.zeros(num_trials)
	target = np.zeros(num_trials)
	reward = np.zeros(num_trials)
	counter = 0 	# counter increments for all successful trials

	for i in range(0,num_trials):
		if (state[np.minimum(ind_center_states[i]+5,total_states-1)] == 'check_reward'):	 
			trial_success[i] = 1
			target_state = state[ind_center_states[i] + 3]
			if (target_state == 'hold_targetL'):
				target[i] = 1
				reward[i] = rewarded_reward_scheduleL[counter]
			else:
				target[i] = 2
				reward[i] = rewarded_reward_scheduleH[counter]
			counter += 1
		else:
			trial_success[i] = 0
			target[i] = 0 	# no target selected
			reward[i] = 0 	# no reward givens

	hdf.close()
	return state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward

def FreeChoicePilotTask_ChoiceAfterStim(reward3, target3, instructed_or_freechoice_block3,stim_trials):
	'''
	This method computes the likelihood of selecting the LV target on the free-choice trial following a trial with stimulation.
	This likelihood is divided into two cases: when the stimulation trial is rewarded and unrewarded.
	'''

	# Find rewarded and unrewarded stim trial indices
	stim_reward = np.ravel(np.nonzero(np.logical_and(reward3,stim_trials)))
	stim_noreward = np.ravel(np.nonzero(np.logical_and(np.logical_not(reward3),stim_trials)))

	# Delete index from arrays if it corresponds to last trial
	if (stim_reward[-1]==(len(stim_trials)-1)):
		stim_reward = np.delete(stim_reward,len(stim_reward)-1)
	if (stim_noreward[-1]==(len(stim_trials)-1)):
		stim_noreward = np.delete(stim_noreward,len(stim_noreward)-1)

	# Look at free-choice trial target selections following these stimulation trials
	choice_reward = [target3[stim_reward[i]+1] for i in range(len(stim_reward)) if (instructed_or_freechoice_block3[stim_reward[i]+1]==2)]
	choice_noreward = [target3[stim_noreward[i]+1] for i in range(len(stim_noreward)) if (instructed_or_freechoice_block3[stim_noreward[i]+1]==2)]

	# Calculate the likelihood of selecting the low-value target
	prob_lv_rewarded = float(np.sum(np.equal(choice_reward,1)))/len(choice_reward)
	prob_lv_unrewarded = float(np.sum(np.equal(choice_noreward,1)))/len(choice_noreward)


	return prob_lv_rewarded, prob_lv_unrewarded

def FreeChoiceTask_PathLengths(hdf_file):
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

	target1 = np.zeros(100)
	target3 = np.zeros(ind_check_reward_states.size-200)
	trial1 = np.zeros(target1.size)
	trial3 = np.zeros(target3.size)
	stim_trials = np.zeros(target3.size)

	# Initialize variables use for in performance computation

	LV_block1 = []
	HV_block1 = []
	LV_block3 = []
	HV_block3 = []

	"""
	Find start and stop state times instructed and free-choice trials only.
	"""
	for i in range(0,100):
		end_time = state_time[ind_check_reward_states[i]]
		start_time = state_time[ind_check_reward_states[i]-3]
		target_state1 = state[ind_check_reward_states[i] - 2]
		trial1[i] = instructed_or_freechoice[i]
		if target_state1 == 'hold_targetL':
			LV_block1.append([start_time, end_time])
		else:
			HV_block1.append([start_time, end_time])
	for i in range(200,num_successful_trials):
		end_time = state_time[ind_check_reward_states[i]]
		start_time = state_time[ind_check_reward_states[i]-3]
		target_state3 = state[ind_check_reward_states[i] - 2]
		trial3[i-200] = instructed_or_freechoice[i]
		if target_state3 == 'hold_targetL':
			LV_block3.append([start_time, end_time])
			if trial3[i-200]==1:   # instructed trial to low-value targer paired with stim
				stim_trials[i-200] = 1
			else:
				stim_trials[i-200] = 0
		else:
			HV_block3.append([start_time, end_time])
			stim_trials[i-200] = 0

	LV_block1 = np.array(LV_block1)
	HV_block1 = np.array(HV_block1)
	LV_block3 = np.array(LV_block3)
	HV_block3 = np.array(HV_block3)    
	
	hdf.close()

	return LV_block1, LV_block3, HV_block1, HV_block3