from numpy import sin, linspace, pi
import matplotlib
import numpy as np
import tables
from matplotlib import pyplot as plt
from basicAnalysis import computePSTH
from plexon import plexfile


def probabilisticRewardTask_PSTH(hdf_filename, filename, block_num):

	# Define file paths and names 
	plx_filename1 = 'Offline_eNe1.plx'
	plx_filename2 = 'Offline_eNe2.plx'
	TDT_tank = '/home/srsummerson/storage/tdt/'+filename
	hdf_location = '/storage/rawdata/hdf/'+hdf_filename

	plx_location1 = '/home/srsummerson/storage/tdt/'+filename+'/'+'Block-'+ str(block_num) + '/'+plx_filename1
	plx_location2 = '/home/srsummerson/storage/tdt/'+filename+'/'+'Block-'+ str(block_num) + '/'+plx_filename2

	# Get spike data
	plx1 = plexfile.openFile(plx_location1)
	spike_file1 = plx1.spikes[:].data
	plx2 = plexfile.openFile(plx_location2)
	spike_file2 = plx2.spikes[:].data

	# Unpack behavioral data
	hdf = tables.openFile(hdf_filename)

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
	cursor = hdf.root.task[:]['cursor']

	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
	ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
	ind_target_states = ind_check_reward_states - 3 # only look at targets when the trial was successful (2 states before reward state)
	ind_hold_center_states = ind_check_reward_states - 4 # only look at center holds for successful trials
	num_successful_trials = ind_check_reward_states.size
	target_times = state_time[ind_target_states]
	center_hold_times = state_time[ind_hold_center_states]
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

	neural_data_center_hold_times = np.zeros(len(center_hold_times))

	# Load syncing data for hdf file and TDT recording
	hdf_times = dict()
	mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
	sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

	hdf_rows = np.ravel(hdf_times['row_number'])
	hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
	dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
	dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])
	dio_recording_start = hdf_times['tdt_recording_start']  # starting sample value
	dio_tstart = dio_recording_start/dio_freq # starting time in seconds

	# Find corresponding timestamps for neural data from behavioral time points

	for i, time in enumerate(center_hold_times):
		hdf_index = np.argmin(np.abs(hdf_rows - time))
		neural_data_center_hold_times[i] = dio_tdt_sample[hdf_index]/dio_freq

	"""
	Find target choices and trial type across the blocks.
	"""
	for i in range(0,100):
		target_state1 = state[ind_check_reward_states[i] - 2]
		trial1[i] = instructed_or_freechoice[i]
		if target_state1 == 'hold_targetL':
			target1.append(1)
		else:
			target1.append(2)
	for i in range(200,num_successful_trials):
		target_state3 = state[ind_check_reward_states[i] - 2]
		trial3[i-200] = instructed_or_freechoice[i]
		if target_state3 == 'hold_targetL':
			target3.append(1)
		else:
			target3.append(2)
	
	# Compute PSTH for units over all trials
	window_before = 2  # PSTH time window before alignment point in seconds
	window_after = 3  # PSTH time window after alignment point in seconds
	binsize = 100 # spike bin size in ms		
	psth_all_trials, smooth_psth_all_trials, labels_all_trials = computePSTH(spike_file1,spike_file2,neural_data_center_hold_times,window_before,window_after, binsize)
	psth_time_window = np.arange(-window_before,window_after-float(binsize)/1000,float(binsize)/1000)

	# Plot PSTHs all together
	cmap_all = mpl.cm.brg
	plt.figure()
	for i in range(len(psth_all_trials)):
		unit_name = psth_all_trials.keys()[i]
		plt.plot(psth_time_window,psth_all_trials[unit_name],color=cmap_all(i/float(len(psth_all_trials))),label=unit_name)
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('PSTH')
	plt.show()

	plt.figure()
	for i in range(len(psth_all_trials)):
		unit_name = psth_all_trials.keys()[i]
		plt.plot(psth_time_window,smooth_psth_all_trials[unit_name],color=cmap_all(i/float(len(psth_all_trials))),label=unit_name)
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH')
	plt.show()
	
	hdf.close()
	return

# Set up code for particular day and block
hdf_filename = 'mari20160418_04_te2002.hdf'
filename = 'Mario20160418'
block_num = 1

probabilisticRewardTask_PSTH(hdf_filename, filename, block_num)