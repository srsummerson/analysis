import numpy as np 
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from neo import io
from PulseMonitorData import findIBIs
from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile



def trial_sliding_avg(trial_array, num_trials_slide):

	num_trials = len(trial_array)
	slide_avg = np.zeros(num_trials)

	for i in range(num_trials):
		if i < num_trials_slide:
			slide_avg[i] = np.sum(trial_array[:i+1])/float(i+1)
		else:
			slide_avg[i] = np.sum(trial_array[i-num_trials_slide+1:i+1])/float(num_trials_slide)

	return slide_avg


class ChoiceBehavior_ThreeTargets():

	def __init__(self, hdf_file):
		self.filename =  hdf_file
		self.table = tables.openFile(self.filename)

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
		self.trial_type = self.table.root.task[:]['target_index']
		self.targets_on = self.table.root.task[:]['LHM_target_on']
	  
		self.ind_wait_states = np.ravel(np.nonzero(self.state == 'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == 'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == 'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == 'check_reward'))
		#self.trial_type = np.ravel(trial_type[state_time[ind_center_states]])
		#self.stress_type = np.ravel(stress_type[state_time[ind_center_states]])
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_check_reward_states.size

		self.table.close()


	def get_state_TDT_LFPvalues(self,ind_state,syncHDF_file):
		'''
		This method finds the TDT sample numbers that correspond to indicated task state using the syncHDF.mat file.

		Inputs:
			- ind_state: array with state numbers corresponding to which state we're interested in finding TDT sample numbers for, e.g. self.ind_hold_center_states
			- syncHDF_file: syncHDF.mat file path, e.g. '/home/srsummerson/storage/syncHDF/Mario20161104_b1_syncHDF.mat'
		Output:
			- lfp_state_row_ind: array of tdt sample numbers that correspond the the task state events in ind_state array
		'''
		# Load syncing data
		hdf_times = dict()
		sp.io.loadmat(syncHDF_file, hdf_times)
		hdf_rows = np.ravel(hdf_times['row_number'])
		hdf_rows = [val for val in hdf_rows]
		dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
		dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

		lfp_dio_sample_num = dio_tdt_sample  # assumes DIOx and LFPx are saved using the same sampling rate

		state_row_ind = self.state_time[ind_state]		# gives the hdf row number sampled at 60 Hz
		lfp_state_row_ind = np.zeros(state_row_ind.size)

		for i in range(len(state_row_ind)):
			hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind[i]))
			if np.abs(hdf_rows[hdf_index] - state_row_ind[i])==0:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
			elif hdf_rows[hdf_index] > state_row_ind[i]:
				hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
				m = (lfp_dio_sample_num[hdf_index]-lfp_dio_sample_num[hdf_index - 1])/hdf_row_diff
				b = lfp_dio_sample_num[hdf_index-1] - m*hdf_rows[hdf_index-1]
				lfp_state_row_ind[i] = np.rint(m*state_row_ind[i] + b)
			elif (hdf_rows[hdf_index] < state_row_ind[i])&(hdf_index + 1 < len(hdf_rows)):
				hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
				if (hdf_row_diff > 0):
					m = (lfp_dio_sample_num[hdf_index + 1] - lfp_dio_sample_num[hdf_index])/hdf_row_diff
					b = lfp_dio_sample_num[hdf_index] - m*hdf_rows[hdf_index]
					lfp_state_row_ind[i] = np.rint(m*state_row_ind[i] + b)
				else:
					lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
			else:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]

		return lfp_state_row_ind, dio_freq

	def TrialChoices(self, num_trials_slide, plot_results = False):
		'''
		This method computes the sliding average over num_trials_slide trials of the number of choices for the 
		optimal target choice. It looks at overall the liklihood of selecting the better choice, as well as the 
		choice behavior for the three different scenarios: L-H targets shown, L-M targets shown, and M-H targets
		shown.
		'''
		freechoice_trial = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]]) - 1
		freechoice_trial_ind = np.ravel(np.nonzero(freechoice_trial))
		target_choices = self.state[self.ind_check_reward_states - 2]
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM

		num_trials = len(target_choices)
		all_choices = np.zeros(np.sum(freechoice_trial))
		LM_choices = []
		LH_choices = []
		MH_choices = []
		

		cmap = mpl.cm.hsv

		for i, choice in enumerate(target_choices[freechoice_trial_ind]):
			# only look at freechoice trials
			targ_presented = targets_on[freechoice_trial_ind[i]]
			# L-M targets presented
			if (targ_presented[0]==1)&(targ_presented[2]==1):
				if choice=='hold_targetM':
					all_choices[i] = 1		# optimal choice was made
					LM_choices = np.append(LM_choices, 1)
				else:
					LM_choices = np.append(LM_choices, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice=='hold_targetH':
					all_choices[i] = 1
					LH_choices = np.append(LH_choices, 1)
				else:
					LH_choices = np.append(LH_choices, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice=='hold_targetH':
					all_choices[i] = 1
					MH_choices = np.append(MH_choices, 1)
				else:
					MH_choices = np.append(MH_choices, 0)


		sliding_avg_all_choices = trial_sliding_avg(all_choices, num_trials_slide)
		sliding_avg_LM_choices = trial_sliding_avg(LM_choices, num_trials_slide)
		sliding_avg_LH_choices = trial_sliding_avg(LH_choices, num_trials_slide)
		sliding_avg_MH_choices = trial_sliding_avg(MH_choices, num_trials_slide)

		if plot_results:
			fig = plt.figure()
			ax11 = plt.subplot(221)
			plt.plot(sliding_avg_LM_choices, c = 'b', label = 'Mid')
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Low vs. Mid')
			ax11.get_yaxis().set_tick_params(direction='out')
			ax11.get_xaxis().set_tick_params(direction='out')
			ax11.get_xaxis().tick_bottom()
			ax11.get_yaxis().tick_left()
			plt.legend()

			ax12 = plt.subplot(222)
			plt.plot(sliding_avg_LH_choices, c = 'b', label = 'High')
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Low vs. High')
			ax12.get_yaxis().set_tick_params(direction='out')
			ax12.get_xaxis().set_tick_params(direction='out')
			ax12.get_xaxis().tick_bottom()
			ax12.get_yaxis().tick_left()
			plt.legend()

			ax21 = plt.subplot(223)
			plt.plot(sliding_avg_MH_choices, c = 'b', label = 'High')
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Mid vs. High')
			ax21.get_yaxis().set_tick_params(direction='out')
			ax21.get_xaxis().set_tick_params(direction='out')
			ax21.get_xaxis().tick_bottom()
			ax21.get_yaxis().tick_left()
			plt.legend()

			ax22 = plt.subplot(224)
			plt.plot(sliding_avg_all_choices, c = 'b', label = 'Mid/High')
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('All Choices')
			ax22.get_yaxis().set_tick_params(direction='out')
			ax22.get_xaxis().set_tick_params(direction='out')
			ax22.get_xaxis().tick_bottom()
			ax22.get_yaxis().tick_left()
			plt.legend()


		return all_choices, LM_choices, LH_choices, MH_choices

	def TrialOptionsAndChoice(self):
		'''
		This method extracts for each trial which targets were on options and what the ultimate target
		choice was. It also gives whether or not that choice was rewarded.

		Output:
		- target_options: N x 3 array, where N is the number of trials (instructed + freechoice), which contains 0 or 1s
							to indicate on a given trial which of the 3 targets was shown. The order is Low - High - Middle.
							For example, if on the ith trial the Low and High value targets are shown, then 
							target_options[i,:] = [1, 1, 0].
		- target_chosen: N x 3 array, which contains 0 or 1s to indicate on a given trial which of the 3 targets was chosen.
							The order is Low - High - Middle. Note that only one entry per row may be non-zero (1). For 
							example, if on the ith trial the High value target was selected, then 
							target_chosen[i,:] = [0, 1, 0].
		- reward_chosen: length N array, which contains 0 or 1s to indicate whether a reward was received at the end
							of the ith trial. 
		'''
		target_choices = self.state[self.ind_check_reward_states - 2]
		target_options = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM
		target_chosen = np.zeros((len(target_choices), 3)) 								 # placeholder for array with boolen values indicating choice: LHM
		rewarded_choice = np.array(self.state[self.ind_check_reward_states + 1] == 'reward', dtype = float)
		num_trials = len(target_choices)

		for i, choice in enumerate(target_choices):
			if choice == 'hold_targetM':
				target_chosen[i,2] = 1
			if choice == 'hold_targetL':
				target_chosen[i,0] = 1
			if choice == 'hold_targetH':
				target_chosen[i,1] = 1

		return target_options, target_chosen, rewarded_choice


class ChoiceBehavior_ThreeTargets_Stimulation():
	'''
	Class for behavior taken from ABA' task, where there are three targets of different probabilities of reward
	and stimulation is paired with the middle-value target during the hold-period of instructed trials during
	blocks B and A'. Can pass in a list of hdf files when initially instantiated in the case that behavioral data
	is split across multiple hdf files. In this case, the files should be listed in the order in which they were saved.
	'''

	def __init__(self, hdf_files, num_trials_A, num_trials_B):
		for i, hdf_file in enumerate(hdf_files): 
			filename =  hdf_file
			table = tables.openFile(filename)
			if i == 0:
				self.state = table.root.task_msgs[:]['msg']
				self.state_time = table.root.task_msgs[:]['time']
				self.trial_type = table.root.task[:]['target_index']
				self.targets_on = table.root.task[:]['LHM_target_on']
			else:
				self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
				self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
				self.trial_type = np.append(self.trial_type, table.root.task[:]['target_index'])
				self.targets_on = np.append(self.targets_on, table.root.task[:]['LHM_target_on'])
		
		if len(hdf_files) > 1:
			self.targets_on = np.reshape(self.targets_on, (len(self.targets_on)/3,3))  				# this should contain triples indicating targets
		self.ind_wait_states = np.ravel(np.nonzero(self.state == 'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == 'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == 'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == 'check_reward'))
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_check_reward_states.size
		self.num_trials_A = num_trials_A
		self.num_trials_B = num_trials_B


	def get_state_TDT_LFPvalues(self,ind_state,syncHDF_file):
		'''
		This method finds the TDT sample numbers that correspond to indicated task state using the syncHDF.mat file.

		Inputs:
			- ind_state: array with state numbers corresponding to which state we're interested in finding TDT sample numbers for, e.g. self.ind_hold_center_states
			- syncHDF_file: syncHDF.mat file path, e.g. '/home/srsummerson/storage/syncHDF/Mario20161104_b1_syncHDF.mat'
		Output:
			- lfp_state_row_ind: array of tdt sample numbers that correspond the the task state events in ind_state array
		'''
		# Load syncing data
		hdf_times = dict()
		sp.io.loadmat(syncHDF_file, hdf_times)
		hdf_rows = np.ravel(hdf_times['row_number'])
		hdf_rows = [val for val in hdf_rows]
		dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
		dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

		lfp_dio_sample_num = dio_tdt_sample  # assumes DIOx and LFPx are saved using the same sampling rate

		state_row_ind = self.state_time[ind_state]		# gives the hdf row number sampled at 60 Hz
		lfp_state_row_ind = np.zeros(state_row_ind.size)

		for i in range(len(state_row_ind)):
			hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind[i]))
			if np.abs(hdf_rows[hdf_index] - state_row_ind[i])==0:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
			elif hdf_rows[hdf_index] > state_row_ind[i]:
				hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
				m = (lfp_dio_sample_num[hdf_index]-lfp_dio_sample_num[hdf_index - 1])/hdf_row_diff
				b = lfp_dio_sample_num[hdf_index-1] - m*hdf_rows[hdf_index-1]
				lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
			elif (hdf_rows[hdf_index] < state_row_ind[i])&(hdf_index + 1 < len(hdf_rows)):
				hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
				if (hdf_row_diff > 0):
					m = (lfp_dio_sample_num[hdf_index + 1] - lfp_dio_sample_num[hdf_index])/hdf_row_diff
					b = lfp_dio_sample_num[hdf_index] - m*hdf_rows[hdf_index]
					lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
				else:
					lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
			else:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]

		return lfp_state_row_ind, dio_freq


	def TrialChoices(self, num_trials_slide, plot_results = False):
		'''
		This method computes the sliding average over num_trials_slide trials of the number of choices for the 
		optimal target choice. It looks at overall the liklihood of selecting the better choice, as well as the 
		choice behavior for the three different scenarios: L-H targets shown, L-M targets shown, and M-H targets
		shown. Choice behavior is split across the three blocks.
		'''

		# Get indices of free-choice trials for Blocks A and A', as well as the corresponding target selections.
		freechoice_trial = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]]) - 1
		freechoice_trial_ind_A = np.ravel(np.nonzero(freechoice_trial[:self.num_trials_A]))
		freechoice_trial_ind_Aprime = np.ravel(np.nonzero(freechoice_trial[self.num_trials_A+self.num_trials_B:])) + self.num_trials_A+self.num_trials_B
		
		target_choices_A = self.state[self.ind_check_reward_states - 2][freechoice_trial_ind_A]
		target_choices_Aprime = self.state[self.ind_check_reward_states - 2][freechoice_trial_ind_Aprime]
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM

		# Initialize variables
		num_FC_trials_A = len(freechoice_trial_ind_A)
		num_FC_trials_Aprime = len(freechoice_trial_ind_Aprime)
		all_choices_A = np.zeros(num_FC_trials_A)
		all_choices_Aprime = np.zeros(num_FC_trials_Aprime)
		LM_choices_A = []
		LH_choices_A = []
		MH_choices_A = []
		LM_choices_Aprime = []
		LH_choices_Aprime = []
		MH_choices_Aprime = []
		

		cmap = mpl.cm.hsv

		for i, choice in enumerate(target_choices_A):
			# only look at freechoice trials
			targ_presented = targets_on[freechoice_trial_ind_A[i]]
			# L-M targets presented
			if (targ_presented[0]==1)&(targ_presented[2]==1):
				if choice=='hold_targetM':
					all_choices_A[i] = 1		# optimal choice was made
					LM_choices_A = np.append(LM_choices_A, 1)
				else:
					LM_choices_A = np.append(LM_choices_A, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice=='hold_targetH':
					all_choices_A[i] = 1
					LH_choices_A = np.append(LH_choices_A, 1)
				else:
					LH_choices_A = np.append(LH_choices_A, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice=='hold_targetH':
					all_choices_A[i] = 1
					MH_choices_A = np.append(MH_choices_A, 1)
				else:
					MH_choices_A = np.append(MH_choices_A, 0)

		for i, choice in enumerate(target_choices_Aprime):
			# only look at freechoice trials
			targ_presented = targets_on[freechoice_trial_ind_Aprime[i]]
			# L-M targets presented
			if (targ_presented[0]==1)&(targ_presented[2]==1):
				if choice=='hold_targetM':
					all_choices_Aprime[i] = 1		# optimal choice was made
					LM_choices_Aprime = np.append(LM_choices_Aprime, 1)
				else:
					LM_choices_Aprime = np.append(LM_choices_Aprime, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice=='hold_targetH':
					all_choices_Aprime[i] = 1
					LH_choices_Aprime = np.append(LH_choices_Aprime, 1)
				else:
					LH_choices_Aprime = np.append(LH_choices_Aprime, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice=='hold_targetH':
					all_choices_Aprime[i] = 1
					MH_choices_Aprime = np.append(MH_choices_Aprime, 1)
				else:
					MH_choices_Aprime = np.append(MH_choices_Aprime, 0)

		sliding_avg_all_choices_A = trial_sliding_avg(all_choices_A, num_trials_slide)
		sliding_avg_LM_choices_A = trial_sliding_avg(LM_choices_A, num_trials_slide)
		sliding_avg_LH_choices_A = trial_sliding_avg(LH_choices_A, num_trials_slide)
		sliding_avg_MH_choices_A = trial_sliding_avg(MH_choices_A, num_trials_slide)

		sliding_avg_all_choices_Aprime = trial_sliding_avg(all_choices_Aprime, num_trials_slide)
		sliding_avg_LM_choices_Aprime = trial_sliding_avg(LM_choices_Aprime, num_trials_slide)
		sliding_avg_LH_choices_Aprime = trial_sliding_avg(LH_choices_Aprime, num_trials_slide)
		sliding_avg_MH_choices_Aprime = trial_sliding_avg(MH_choices_Aprime, num_trials_slide)

		if plot_results:
			fig = plt.figure()
			ax11 = plt.subplot(221)
			plt.plot(sliding_avg_LM_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_LM_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Low vs. Mid')
			ax11.get_yaxis().set_tick_params(direction='out')
			ax11.get_xaxis().set_tick_params(direction='out')
			ax11.get_xaxis().tick_bottom()
			ax11.get_yaxis().tick_left()
			plt.legend()

			ax12 = plt.subplot(222)
			plt.plot(sliding_avg_LH_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_LH_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Low vs. High')
			ax12.get_yaxis().set_tick_params(direction='out')
			ax12.get_xaxis().set_tick_params(direction='out')
			ax12.get_xaxis().tick_bottom()
			ax12.get_yaxis().tick_left()
			plt.legend()

			ax21 = plt.subplot(223)
			plt.plot(sliding_avg_MH_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_MH_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Mid vs. High')
			ax21.get_yaxis().set_tick_params(direction='out')
			ax21.get_xaxis().set_tick_params(direction='out')
			ax21.get_xaxis().tick_bottom()
			ax21.get_yaxis().tick_left()
			plt.legend()

			ax22 = plt.subplot(224)
			plt.plot(sliding_avg_all_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_all_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('All Choices')
			ax22.get_yaxis().set_tick_params(direction='out')
			ax22.get_xaxis().set_tick_params(direction='out')
			ax22.get_xaxis().tick_bottom()
			ax22.get_yaxis().tick_left()
			plt.legend()

		return all_choices_A, LM_choices_A, LH_choices_A, MH_choices_A, all_choices_Aprime, LM_choices_Aprime, LH_choices_Aprime, MH_choices_Aprime


def ThreeTargetTask_SpikeAnalysis(hdf_files, syncHDF_files, spike_files):
	'''
	This method aligns spiking data to behavioral choices for all different target presentation combinations
	in the Three Target Task, where there is a low-value, middle-value and high-value target. This version does not 
	differentiate between choices in different blocks.

	Inputs:
	- hdf_files: list of N hdf_files corresponding to the behavior in the three target task
	- syncHDF_files: list of N syncHDF_files that containes the syncing DIO data for the corresponding hdf_file and it's
					TDT recording. If TDT data does not exist, an empty entry should strill be entered. I.e. if there is data for the first
					epoch of recording but not the second, syncHDF_files should have the form [syncHDF_file1.mat, '']
	- spike_files: list of N tuples of spike_files, where each entry is a list of 2 spike files, one corresponding to spike
					data from the first 96 channels and the other corresponding to the spike data from the last 64 channels.
					If spike data does not exist, an empty entry should strill be entered. I.e. if there is data for the first
					epoch of recording but not the second, the hdf_files and syncHDF_files will both have 2 file names, and the 
					spike_files entry should be of the form [[spike_file1.csv, spike_file2.csv], ''].

	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)
	'''
	Get data for each set of files
	'''
	for i in range(num_files):
		# Load behavior data
		cb = ChoiceBehavior_ThreeTargets(hdf_files[i])
		num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_picture_onset = cb.ind_check_reward_states - 5
		

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			spike1 = OfflineSorted_CSVFile(spike_files[i][0])
			spike2 = OfflineSorted_CSVFile(spike_files[i][1])
			# Find all sort codes associated with good channels
			all_units1, total_units1 = spike1.find_unit_sc(spike1.good_channels)
			all_units2, total_units2 = spike2.find_unit_sc(spike2.good_channels)

			print "Total number of units: ", total_units1 + total_units2

			cd_units = [1, 3, 4, 17, 18, 20, 40, 41, 54, 56, 57, 63, 64, 72, 75, 81, 83, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
			spike1_good_channels = np.array([unit for unit in cd_units if unit in spike1.good_channels])
			spike2_good_channels = np.array([unit for unit in cd_units if unit in spike2.good_channels])

			# Plot average rate for all neurons divided in six cases of targets on option
			plt.figure()
			t_before = 1			# 1 s
			t_after = 3				# 3 s
			t_resolution = 0.1 		# 100 ms time bins

			# 1. LH presented
			LH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,1,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list1 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list2 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			
			plt.subplot(3,2,1)
			plt.title('Low-High Presented')
			plt.plot(smooth_avg_psth1.T)
			plt.plot(smooth_avg_psth2.T)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)
			
			# 2. LM presented
			LM_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,0,1]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list1 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[LM_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list2 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[LM_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,2)
			plt.title('Low-Middle Presented')
			plt.plot(smooth_avg_psth1.T)
			plt.plot(smooth_avg_psth2.T)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 3. MH presented
			MH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,1,1]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list1 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[MH_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list2 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[MH_ind],t_before,t_after,t_resolution)
			print unit_list1
			plt.subplot(3,2,3)
			plt.title('Middle-High Presented')
			plt.plot(smooth_avg_psth1.T)
			plt.plot(smooth_avg_psth2.T)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 4. L presented
			L_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,0,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list1 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[L_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list2 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[L_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,4)
			plt.title('Low Presented')
			plt.plot(smooth_avg_psth1.T)
			plt.plot(smooth_avg_psth2.T)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 5. H presented
			H_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,1,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list1 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[H_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list2 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[H_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,5)
			plt.title('High Presented')
			plt.plot(smooth_avg_psth1.T)
			plt.plot(smooth_avg_psth2.T)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 6. M presented
			M_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,0,1]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list1 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[M_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list2 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[M_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,6)
			plt.title('Middle Presented')
			plt.plot(smooth_avg_psth1.T)
			plt.plot(smooth_avg_psth2.T)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			plt_name = syncHDF_files[i][34:-12]
			plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+'_PSTH.svg')

	return

def ThreeTargetTask_SpikeAnalysis_SingleChannel(hdf_files, syncHDF_files, spike_files, chann, sc):
	'''
	This method aligns spiking data to behavioral choices for all different target presentation combinations
	in the Three Target Task, where there is a low-value, middle-value and high-value target. This version does not 
	differentiate between choices in different blocks.

	Inputs:
	- hdf_files: list of N hdf_files corresponding to the behavior in the three target task
	- syncHDF_files: list of N syncHDF_files that containes the syncing DIO data for the corresponding hdf_file and it's
					TDT recording. If TDT data does not exist, an empty entry should strill be entered. I.e. if there is data for the first
					epoch of recording but not the second, syncHDF_files should have the form [syncHDF_file1.mat, '']
	- spike_files: list of N tuples of spike_files, where each entry is a list of 2 spike files, one corresponding to spike
					data from the first 96 channels and the other corresponding to the spike data from the last 64 channels.
					If spike data does not exist, an empty entry should strill be entered. I.e. if there is data for the first
					epoch of recording but not the second, the hdf_files and syncHDF_files will both have 2 file names, and the 
					spike_files entry should be of the form [[spike_file1.csv, spike_file2.csv], ''].
	- chann: integer representing a channel
	
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)
	'''
	Get data for each set of files
	'''
	for i in range(num_files):
		# Load behavior data
		cb = ChoiceBehavior_ThreeTargets(hdf_files[i])
		num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_picture_onset = cb.ind_check_reward_states - 5
		

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			spike1 = OfflineSorted_CSVFile(spike_files[i][0])
			spike2 = OfflineSorted_CSVFile(spike_files[i][1])
			# Find all sort codes associated with good channels
			all_units1, total_units1 = spike1.find_unit_sc(spike1.good_channels)
			all_units2, total_units2 = spike2.find_unit_sc(spike2.good_channels)

			print "Total number of units: ", total_units1 + total_units2

			cd_units = [chann]
			spike1_good_channels = np.array([unit for unit in cd_units if unit in spike1.good_channels])
			spike2_good_channels = np.array([unit for unit in cd_units if unit in spike2.good_channels])

			# Plot average rate for all neurons divided in six cases of targets on option
			plt.figure()
			t_before = 1			# 1 s
			t_after = 3				# 3 s
			t_resolution = 0.1 		# 100 ms time bins

			# 1. LH presented
			LH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,1,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1 = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2 = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			
			plt.subplot(3,2,1)
			plt.title('Low-High Presented')
			plt.plot(smooth_avg_psth1)
			plt.plot(smooth_avg_psth2)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)
			
			# 2. LM presented
			LM_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,0,1]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1 = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[LM_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2 = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[LM_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,2)
			plt.title('Low-Middle Presented')
			plt.plot(smooth_avg_psth1)
			plt.plot(smooth_avg_psth2)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 3. MH presented
			MH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,1,1]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1 = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[MH_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2 = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[MH_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,3)
			plt.title('Middle-High Presented')
			plt.plot(smooth_avg_psth1)
			plt.plot(smooth_avg_psth2)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 4. L presented
			L_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,0,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1 = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2 = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,4)
			plt.title('Low Presented')
			plt.plot(smooth_avg_psth1)
			plt.plot(smooth_avg_psth2)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 5. H presented
			H_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,1,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1 = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2 = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,5)
			plt.title('High Presented')
			plt.plot(smooth_avg_psth1)
			plt.plot(smooth_avg_psth2)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			# 6. M presented
			M_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,0,1]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1 = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[M_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2 = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[M_ind],t_before,t_after,t_resolution)

			plt.subplot(3,2,6)
			plt.title('Middle Presented')
			plt.plot(smooth_avg_psth1)
			plt.plot(smooth_avg_psth2)
			xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
			xticks = np.arange(0, len(xticklabels), 10)
			xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
			plt.xticks(xticks, xticklabels)

			plt_name = syncHDF_files[i][34:-12]
			plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+'_PSTH_Chan'+str(chann)+'.svg')

	return