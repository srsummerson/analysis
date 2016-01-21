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
from neo import io

# Set up code for particular day and block
hdf_filename = 'luig20151204_05.hdf'
filename = 'Luigi20151204_HDEEG'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
block_num = 3

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
  
ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
ind_center_states = np.ravel(no.nonzero(state == 'center'))
ind_target_states = np.ravel(np.nonzero(state == 'target'))
ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
num_successful_trials = ind_check_reward_states.size
instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]
all_instructed_or_freechoice = trial_type[state_time[ind_center_states]]
successful_stress_or_not = stress_type[state_time[ind_check_reward_states]]
all_stress_or_not = stress_type[state_time[ind_wait_states]]
rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]

num_trials = ind_center_states.size
num_successful_trials = ind_check_reward_states.size

for i in range(0,num_trials):
	# Want to know: successful or not, stress or not (can just use all_stress_or_not?), target selection, instructed or not 
	# (can use all_instructed_or_freechoice), rewarded or not
	if (state[ind_wait_states[i]] == reward): 
		trial_success[i] = 1
		ind_successful_center_states.append(ind_wait_states[i])  # save row for successful trials
		target_state = state[ind_center_states[i] + 3]
		if target_state == 'hold_targetL':
        	target[i] = 1
        	reward[i] = rewarded_reward_scheduleL[i]
    	else:
        	target[i] = 2
        	reward[i] = rewarded_reward_scheduleH[i]
	else:
		trial_success[i] = 0


# Load syncing data for hdf file and TDT recording
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)




