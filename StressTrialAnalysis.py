'''
Stress Trial Days:
*11/24/2015 - Blocks 1, 2 (DIO data didn't save to recording)
12/4/2015 - Blocks 1 (luig20151204_05.hdf), 2 (luig20151204_07.hdf), 3 (luig20151204_08.hdf)
12/6/2015 - Block 1 (luig20151206_04.hdf)
12/7/2015 - Block 1 (luig20151207_03.hdf)
12/17/2015 - Block 1 (luig20151217_05.hdf; reversal, not stress)
12/22/2015 - Block 1 (luig20151222_05.hdf)
12/23/2015 - Blocks 1 (luig20151223_03.hdf), 2 (luig20151223_05.hdf)
12/28/2015 - Blocks 1 (luig20151228_09.hdf), 2 (luig20151228_10.hdf)
12/29/2015 - Blocks 1 (luig20151229_02.hdf)
1/5/2016 - Block 1 ()
1/6/2016 - Block 1 (luig20160106_02.hdf)
1/11/2016 - Block 1 (luig20160111_06.hdf)

Trial types:
1. Regular (before stress) and rewarded
2. Regular (before stress) and unrewarded
3. Regular (after stress) and rewarded
4. Regular (after stress) and unrewarded
5. Stress and rewarded
6. Stress and unrewarded
7. Stress and unsuccessful

Behavior:
- Fraction of selecting low vs high value target (reg vs stress trials)
- Rewards received for each selection
- Number of successful trials for each trial type (reg vs stress)
- Response time for each trial type, i.e time to successfully complete trial
Physiological:
- Distribution of IBIs for stress trials, regular trials before stress trials are introduced, and regular trials after stress 
  trials are introduced -- divide up between successful and unsuccessful trials
- Average pupil diameter during stress trials, regular trials before stress trials are introduced, and regular trials after stress 
  is introduced -- divide up between successful and unsuccessful trials
Neurological:
- Average power in different bands for all channels, divided up by target
'''
import numpy as np 
import scipy as sp
import tables
from neo import io
import matplotlib.pyplot as plt

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 

# Set up code for particular day and block
hdf_filename = 'luig20151228_09.hdf'
filename = 'Luigi20151204_HDEEG'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
#hdf_location = '/storage/rawdata/hdf/'+hdf_filename
hdf_location = hdf_filename
block_num = 3

num_avg = 10 	# number of trials to compute running average of trial statistics over

# Load behavior data
## self.target_index = 1 for instructed, 2 for free choice
## self.stress_trial =1 for stress trial, 0 for regular trial
hdf = tables.openFile(hdf_location)

state = hdf.root.task_msgs[:]['msg']
state_time = hdf.root.task_msgs[:]['time']
trial_type = hdf.root.task[:]['target_index']
stress_type = hdf.root.task[:]['stress_trial']
# reward schedules
reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
  
ind_wait_states = np.ravel(np.nonzero(state == 'wait'))   # total number of unique trials
ind_center_states = np.ravel(no.nonzero(state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
ind_target_states = np.ravel(np.nonzero(state == 'target'))
ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]	# free choice trial = 2, instructed = 1
all_instructed_or_freechoice = trial_type[state_time[ind_center_states]]
successful_stress_or_not = np.ravel(stress_type[state_time[ind_check_reward_states]])
all_stress_or_not = np.ravel(stress_type[state_time[ind_center_states]])
rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_check_reward_states]]
rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_check_reward_states]]

num_trials = ind_center_states.size
num_successful_trials = ind_check_reward_states.size
total_states = state.size

trial_success = np.zeros(num_trials)
target = np.zeros(num_trials)
reward = np.zeros(num_trials)
ind_successful_center_states = []
counter = 0 	# counter increments for all successful trials

for i in range(0,num_trials):
	# Want to know: successful or not, stress or not (can just use all_stress_or_not?), target selection, instructed or not 
	# (can use all_instructed_or_freechoice), rewarded or not
	if (state[np.minimum(ind_center_states[i]+5,total_states-1)] == 'check_reward'):	 
		trial_success[i] = 1
		ind_successful_center_states.append(ind_center_states[i])  # save row for successful trials
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

# Number of successful stress trials
tot_successful_stress = np.logical_and(trial_success,all_stress_or_not)
success_stress_trials = float(np.sum(tot_successful_stress))/np.sum(all_stress_or_not)

# Number of successful non-stress trials
tot_successful_reg = np.logical_and(trial_success,np.logical_not(all_stress_or_not))
successful_reg_trials = float(np.sum(tot_successful_reg))/(num_trials - np.sum(all_stress_or_not))

# Response times for successful stress trials
ind_successful_stress = np.ravel(np.nonzero(tot_successful_stress))   	# gives trial index, not row index
row_ind_successful_stress = ind_center_states[ind_successful_stress]		# gives row index
ind_successful_stress_reward = np.ravel(np.nonzero(successful_stress_or_not))
row_ind_successful_stress_reward = ind_check_reward_states[ind_successful_stress_reward]
response_time_successful_stress = (state_time[row_ind_successful_stress_reward] - state_time[row_ind_successful_stress])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response times for successful regular trials
ind_successful_reg = np.ravel(np.nonzero(tot_successful_reg))
row_ind_successful_reg = ind_center_states[ind_successful_reg]
ind_successful_reg_reward = np.ravel(np.nonzero(np.logical_not(successful_stress_or_not)))
row_ind_successful_reg_reward = ind_check_reward_states[ind_successful_reg_reward]
response_time_successful_reg = (state_time[row_ind_successful_reg_reward] - state_time[row_ind_successful_reg])/float(60)

# Target choice for successful stress trials - look at free-choice trials only
tot_successful_fc_stress = np.logical_and(tot_successful_stress,np.ravel(np.equal(all_instructed_or_freechoice,2)))
ind_successful_fc_stress = np.ravel(np.nonzero(tot_successful_fc_stress))
target_choice_successful_stress = target[ind_successful_fc_stress]
prob_choose_low_successful_stress = running_mean(np.equal(target_choice_successful_stress,1),num_avg)
prob_choose_high_successful_stress = running_mean(np,equal(target_choice_successful_stress,2),num_avg)
reward_successful_stress = reward[ind_successful_fc_stress]
prob_reward_low_successful_stress = running_mean(np.equal(reward_successful_stress,1),num_avg)
prob_reward_high_successful_stress = running_mean(np.equal(reward_successful_stress,1),num_avg)

# Target choice for successful regular trials - look at free-choice trials only
tot_successful_fc_reg = np.logical_and(tot_successful_reg,np.ravel(np.equal(all_instructed_or_freechoice,2)))
ind_successful_fc_reg = np.ravel(np.nonzero(tot_successful_fc_reg))
target_choice_successful_reg = target[ind_successful_fc_reg]
prob_choose_low_successful_reg = running_mean(np.equal(target_choice_successful_reg,1),num_avg)
prob_choose_high_successful_reg = running_mean(np.equal(target_choice_successful_reg,1),num_avg)
reward_successful_reg = reward[ind_successful_fc_reg]
prob_reward_low_successful_reg = running_mean(np.equal(reward_successful_reg,1),num_avg)
prob_reward_high_successful_reg = running_mean(np.equal(reward_successful_reg,1),num_avg)



# Load syncing data for hdf file and TDT recording
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)




