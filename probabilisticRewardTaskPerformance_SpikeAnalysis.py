import scipy as sp
import tables
from neo import io
from numpy import sin, linspace, pi
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from basicAnalysis import computePSTH, computePSTH_SingleChannel
from plexon import plexfile
import matplotlib as mpl
from matplotlib import mlab
import glob

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

	print "Loaded spike data."

	# Unpack behavioral data
	hdf = tables.openFile(hdf_location)

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

	print "Loaded sync data."

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
			target1[i] = 1
		else:
			target1[i] = 2
	for i in range(200,num_successful_trials):
		target_state3 = state[ind_check_reward_states[i] - 2]
		trial3[i-200] = instructed_or_freechoice[i]
		if target_state3 == 'hold_targetL':
			target3[i-200] = 1
		else:
			target3[i-200] = 2
	
	# Compute PSTH for units over all trials
	window_before = 2  # PSTH time window before alignment point in seconds
	window_after = 3  # PSTH time window after alignment point in seconds
	binsize = 100 # spike bin size in ms		
	psth_all_trials, smooth_psth_all_trials, labels_all_trials = computePSTH(spike_file1,spike_file2,neural_data_center_hold_times,window_before,window_after, binsize)
	psth_time_window = np.arange(-window_before,window_after-float(binsize)/1000,float(binsize)/1000)


	# Compute PSTH for units over trials (free-choice and instructed) where the LV target was selected
	target_state = state[ind_check_reward_states - 2]
	
	choose_lv = np.ravel(np.nonzero(target_state == 'hold_targetL'))
	neural_choose_lv = neural_data_center_hold_times[choose_lv]
	
	psth_lv_trials, smooth_psth_lv_trials, labels_lv_trials = computePSTH(spike_file1,spike_file2,neural_data_center_hold_times[choose_lv],window_before,window_after, binsize)
	
	# Compute PSTH for units over trials (free-choice and instructed) where the HV target was selected
	choose_hv = np.ravel(np.nonzero(target_state == 'hold_targetH'))
	psth_hv_trials, smooth_psth_hv_trials, labels_hv_trials = computePSTH(spike_file1,spike_file2,neural_data_center_hold_times[choose_hv],window_before,window_after, binsize)

	print "Plotting results."
	
	# Plot PSTHs all together
	cmap_all = mpl.cm.brg
	plt.figure()
	for i in range(len(psth_all_trials)):
		unit_name = psth_all_trials.keys()[i]
		plt.plot(psth_time_window,psth_all_trials[unit_name],color=cmap_all(i/float(len(psth_all_trials))),label=unit_name)
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('PSTH')
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_PSTH-CenterHold.svg')

	plt.figure()
	for i in range(len(psth_all_trials)):
		unit_name = psth_all_trials.keys()[i]
		if np.max(smooth_psth_all_trials[unit_name]) > 10:
			plt.plot(psth_time_window,smooth_psth_all_trials[unit_name],color=cmap_all(i/float(len(psth_all_trials))),label=unit_name)
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_SmoothPSTH-CenterHold.svg')

	plt.figure()
	for i in range(len(psth_lv_trials)):
		unit_name = psth_lv_trials.keys()[i]
		if np.max(smooth_psth_lv_trials[unit_name]) > 20:
			plt.plot(psth_time_window,smooth_psth_lv_trials[unit_name],color=cmap_all(i/float(len(psth_lv_trials))),label=unit_name)
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH for Trials with LV Target Selection')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_SmoothPSTH-CenterHold-LV.svg')

	plt.figure()
	for i in range(len(psth_hv_trials)):
		unit_name = psth_hv_trials.keys()[i]
		if np.max(smooth_psth_hv_trials[unit_name]) > 20:
			plt.plot(psth_time_window,smooth_psth_hv_trials[unit_name],color=cmap_all(i/float(len(psth_hv_trials))),label=unit_name)
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH for Trials with HV Target Selection')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_SmoothPSTH-CenterHold-HV.svg')
	
	plt.close()
	hdf.close()
	return

def probabilisticRewardTask_PSTH_SepSpikeFiles(hdf_filename, filename, block_num):
	'''
	This method computes the PSTH for all sorted single-unit/multi-unit data using separately generated
	plx files for all channels. PSTHs are aligned to the center hold.
	'''
	# Define file paths and names 
	plx_filename1_prefix = 'Offline_eNe1'
	plx_filename2_prefix = 'Offline_eNe2'
	#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
	TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
	hdf_location = '/storage/rawdata/hdf/'+hdf_filename

	# Unpack behavioral data
	hdf = tables.openFile(hdf_location)

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

	print "Loaded sync data."

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
			target1[i] = 1
		else:
			target1[i] = 2
	for i in range(200,num_successful_trials):
		target_state3 = state[ind_check_reward_states[i] - 2]
		trial3[i-200] = instructed_or_freechoice[i]
		if target_state3 == 'hold_targetL':
			target3[i-200] = 1
		else:
			target3[i-200] = 2
	
	# Compute PSTH for units over all trials
	window_before = 2  # PSTH time window before alignment point in seconds
	window_after = 3  # PSTH time window after alignment point in seconds
	binsize = 100 # spike bin size in ms	

	# Get behavior data for computing PSTH for units over trials (free-choice and instructed) where the LV target was selected
	target_state = state[ind_check_reward_states - 2]
	choose_lv = np.ravel(np.nonzero(target_state == 'hold_targetL'))
	neural_choose_lv = neural_data_center_hold_times[choose_lv]
	# Get behavior data for computing PSTH for units over trials (free-choice and instructed) where the HV target was selected
	choose_hv = np.ravel(np.nonzero(target_state == 'hold_targetH'))
	neural_choose_hv = neural_data_center_hold_times[choose_hv]

	psth_all_trials = dict()
	psth_lv_trials = dict()
	psth_hv_trials = dict()
	smooth_psth_all_trials = dict()
	smooth_psth_lv_trials = dict()
	smooth_psth_hv_trials = dict()

	total_units = 0
	
	print "Getting spike data."
	plx_location1 = TDT_tank + '/'+'Block-'+ str(block_num) + '/'
	plx_location2 = TDT_tank + '/'+'Block-'+ str(block_num) + '/'
	print plx_location1
	eNe1_channs = glob.glob(plx_location1+'Offline_eNe1_*.plx')
	eNe2_channs = glob.glob(plx_location1+'Offline_eNe2_*.plx')
	print len(eNe1_channs)
	print len(eNe2_channs)
	all_channs = []
	for plx_data in eNe1_channs:
		chann = int(plx_data[len(plx_location1)+len('Offline_eNe1_CH'):-len('.plx')])
		all_channs.append(chann)
		# Get spike data
		plx1 = plexfile.openFile(plx_data)
		spike_file = plx1.spikes[:].data
		psth_all_trials[str(chann)], smooth_psth_all_trials[str(chann)], labels_all_trials = computePSTH_SingleChannel(spike_file,plx_filename1,neural_data_center_hold_times,window_before,window_after, binsize)
		#psth_lv_trials[str(chann)], smooth_psth_lv_trials[str(chann)], labels_lv_trials = computePSTH_SingleChannel(spike_file,plx_filename1,neural_data_center_hold_times[choose_lv],window_before,window_after, binsize)
		#psth_hv_trials[str(chann)], smooth_psth_hv_trials[str(chann)], labels_hv_trials = computePSTH_SingleChannel(spike_file,plx_filename1,neural_data_center_hold_times[choose_hv],window_before,window_after, binsize)

		#total_units += len(labels_all_trials)
		print total_units
	for plx_data in eNe2_channs:
		chann = int(plx_data[len(plx_location1)+len('Offline_eNe2_CH'):-len('.plx')])+96
		all_channs.append(chann)
		# Get spike data
		plx2 = plexfile.openFile(plx_data)
		spike_file = plx2.spikes[:].data
		psth_all_trials[str(chann)+96], smooth_psth_all_trials[str(chann)+96], labels_all_trials = computePSTH_SingleChannel(spike_file,plx_filename2,neural_data_center_hold_times,window_before,window_after, binsize)
		#psth_lv_trials[str(chann)+96], smooth_psth_lv_trials[str(chann)+96], labels_lv_trials = computePSTH_SingleChannel(spike_file,plx_filename2,neural_data_center_hold_times[choose_lv],window_before,window_after, binsize)
		#psth_hv_trials[str(chann)+96], smooth_psth_hv_trials[str(chann)+96], labels_hv_trials = computePSTH_SingleChannel(spike_file,plx_filename2,neural_data_center_hold_times[choose_hv],window_before,window_after, binsize)
		#total_units += len(labels_all_trials)
		print total_units
	
	psth_time_window = np.arange(-window_before,window_after-float(binsize)/1000,float(binsize)/1000)
	print len(all_channs)
	
	print "Plotting results."
	
	# Plot PSTHs all together
	cmap_all = mpl.cm.brg
	unit_counter = 1.
	plt.figure()
	for chann in all_channs:
		print len(psth_all_trials[str(chann)])
		for i in range(len(psth_all_trials[str(chann)])):
			unit_name = psth_all_trials[str(chann)].keys()[i]
			print unit_name
			print len(psth_all_trials[str(chann)][unit_name])
			plt.plot(psth_time_window,psth_all_trials[str(chann)][unit_name],color=cmap_all(unit_counter/total_units),label=unit_name)
			unit_counter += 1.
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('PSTH')
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_PSTH-CenterHold.svg')

	plt.figure()
	unit_counter = 1.
	for chann in all_channs:
		for i in range(len(psth_all_trials[str(chann)])):
			unit_name = psth_all_trials[str(chann)].keys()[i]
			if np.max(smooth_psth_all_trials[str(chann)][unit_name]) > 10:
				plt.plot(psth_time_window,smooth_psth_all_trials[str(chann)][unit_name],color=cmap_all(unit_counter/total_units),label=unit_name)
				unit_counter += 1.
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_SmoothPSTH-CenterHold.svg')
	'''
	plt.figure()
	unit_counter = 1.
	for chann in all_channs:
		for i in range(len(psth_lv_trials[str(chann)])):
			unit_name = psth_lv_trials[str(chann)].keys()[i]
			if np.max(smooth_psth_lv_trials[str(chann)][unit_name]) > 20:
				plt.plot(psth_time_window,smooth_psth_lv_trials[str(chann)][unit_name],color=cmap_all(unit_counter/total_units),label=unit_name)
				unit_counter += 1.
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH for Trials with LV Target Selection')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_SmoothPSTH-CenterHold-LV.svg')

	plt.figure()
	unit_counter = 1.
	for chann in all_channs:
		unit_name = psth_hv_trials[str(chann)].keys()[i]
		if np.max(smooth_psth_hv_trials[str(chann)][unit_name]) > 20:
			plt.plot(psth_time_window,smooth_psth_hv_trials[str(chann)][unit_name],color=cmap_all(unit_counter/total_units),label=unit_name)
			unit_counter += 1.
	plt.xlabel('Time (s)')
	plt.ylabel('spks/s')
	plt.title('Smooth PSTH for Trials with HV Target Selection')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+filename+'_b'+str(block_num)+'_SmoothPSTH-CenterHold-HV.svg')
	'''
	plt.close("all")
	hdf.close()
	return 

# Set up code for particular day and block
#hdf_filename = 'mari20160524_11_te2135.hdf'
#filename = 'Mario20160524'
#block_num = 1

#probabilisticRewardTask_PSTH_SepSpikeFiles(hdf_filename, filename, block_num)