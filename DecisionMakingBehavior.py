import numpy as np 
import scipy as sp
import scipy.optimize as op
import statsmodels.api as sm
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

	def GetChoicesAndRewards(self):
		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st=='reward') for st in self.state[ind_rewards]])
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.zeros(len(ind_holds))
		rewards = np.zeros(len(ind_holds))

		for i, ind in enumerate(ind_holds):
			if self.state[ind] == 'hold_targetM':
				chosen_target[i] = 1
			elif self.state[ind] == 'hold_targetH':
				chosen_target[i] = 2

		return targets_on, chosen_target, rewards, instructed_or_freechoice


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

	def GetChoicesAndRewards(self):
		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st=='reward') for st in self.state[ind_rewards]])
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.zeros(len(ind_holds))
		rewards = np.zeros(len(ind_holds))

		for i, ind in enumerate(ind_holds):
			if self.state[ind] == 'hold_targetM':
				chosen_target[i] = 1
			elif self.state[ind] == 'hold_targetH':
				chosen_target[i] = 2

		return targets_on, chosen_target, rewards, instructed_or_freechoice

def loglikelihood_ThreeTargetTask_Qlearning(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. 

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_low = np.zeros(len(chosen_target))
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_low = Q_initial[0]
	Q_mid = Q_initial[1]
	Q_high = Q_initial[2]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	prob_choice_low[0] = 0.5
	prob_choice_mid[0] = 0.5
	prob_choice_high[0] = 0.5

	log_prob_total = 0.

	for i in range(0,len(chosen_target)-1):
		# Update Q values with temporal difference error
		delta_low = float(rewards[i]) - Q_low[i]
		delta_mid = float(rewards[i]) - Q_mid[i]
        delta_high = float(rewards[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(chosen_target[i]==0)*(delta_low)
        Q_mid[i+1] = Q_mid[i] + alpha*(chosen_target[i]==1)*(delta_mid)
        Q_high[i+1] = Q_high[i] + alpha*(chosen_target[i]==2)*(delta_high)

        # Update probabilities with new Q-values
        if instructed_or_freechoice[i+1] == 2:
        	if np.array_equal(targets_on[i+1],[1,1,0]):
        		Q_opt = Q_high[i+1]
        		Q_nonopt = Q_low[i+1]

        		prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_low[i+1])))
        		prob_choice_high[i+1] = 1. - prob_choice_low[i+1]
        		prob_choice_mid[i+1] = prob_choice_mid[i]

        		prob_choice_opt = prob_choice_high[i+1]
        		prob_choice_nonopt = prob_choice_low[i+1]

        		# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
        		choice = 0.5*chosen_target[i+1]+1

        	elif np.array_equal(targets_on[i+1],[1,0,1]):
        		Q_opt = Q_mid[i+1]
        		Q_nonopt = Q_low[i+1]

        		prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
        		prob_choice_high[i+1] = prob_choice_high[i]
        		prob_choice_mid[i+1] = 1. - prob_choice_low[i+1]

        		prob_choice_opt = prob_choice_mid[i+1]
        		prob_choice_nonopt = prob_choice_low[i+1]

        		# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
        		choice = chosen_target[i+1]+1
        	else:
        		Q_opt = Q_high[i+1]
        		Q_nonopt = Q_mid[i+1]

        		prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
        		prob_choice_low[i+1] = prob_choice_low[i]
        		prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]
        		
        		prob_choice_opt = prob_choice_high[i+1]
        		prob_choice_nonopt = prob_choice_mid[i+1]

        		# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
        		choice = chosen_target[i+1]

        	log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))
        else:
        	prob_choice_low[i+1] = prob_choice_low[i]
        	prob_choice_mid[i+1] = prob_choice_mid[i]
        	prob_choice_high[i+1] = prob_choice_high[i]

	return log_prob_total

def loglikelihood_ThreeTargetTask_Qlearning(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. 

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_low = np.zeros(len(chosen_target))
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_low = Q_initial[0]
	Q_mid = Q_initial[1]
	Q_high = Q_initial[2]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	prob_choice_low[0] = 0.5
	prob_choice_mid[0] = 0.5
	prob_choice_high[0] = 0.5

	log_prob_total = 0.

	for i in range(0,len(chosen_target)-1):
		# Update Q values with temporal difference error
		delta_low = float(rewards[i]) - Q_low[i]
		delta_mid = float(rewards[i]) - Q_mid[i]
        delta_high = float(rewards[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(chosen_target[i]==0)*(delta_low)
        Q_mid[i+1] = Q_mid[i] + alpha*(chosen_target[i]==1)*(delta_mid)
        Q_high[i+1] = Q_high[i] + alpha*(chosen_target[i]==2)*(delta_high)

        # Update probabilities with new Q-values
        if instructed_or_freechoice[i+1] == 2:
        	if np.array_equal(targets_on[i+1], [1,1,0]):
        		Q_opt = Q_high[i+1]
        		Q_nonopt = Q_low[i+1]

        		prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_low[i+1])))
        		prob_choice_high[i+1] = 1. - prob_choice_low[i+1]
        		prob_choice_mid[i+1] = prob_choice_mid[i]

        		prob_choice_opt = prob_choice_high[i+1]
        		prob_choice_nonopt = prob_choice_low[i+1]

        		# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
        		choice = 0.5*chosen_target[i+1]+1

        	elif np.array_equal(targets_on[i+1],[1,0,1]):
        		Q_opt = Q_mid[i+1]
        		Q_nonopt = Q_low[i+1]

        		prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
        		prob_choice_high[i+1] = prob_choice_high[i]
        		prob_choice_mid[i+1] = 1. - prob_choice_low[i+1]

        		prob_choice_opt = prob_choice_mid[i+1]
        		prob_choice_nonopt = prob_choice_low[i+1]

        		# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
        		choice = chosen_target[i+1]+1
        	else:
        		Q_opt = Q_high[i+1]
        		Q_nonopt = Q_mid[i+1]

        		prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
        		prob_choice_low[i+1] = prob_choice_low[i]
        		prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]
        		
        		prob_choice_opt = prob_choice_high[i+1]
        		prob_choice_nonopt = prob_choice_mid[i+1]

        		# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
        		choice = chosen_target[i+1]

        	log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

        else:
        	prob_choice_low[i+1] = prob_choice_low[i]
        	prob_choice_mid[i+1] = prob_choice_mid[i]
        	prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high

def ThreeTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
	'''
	This method returns the average firing rate of all units on the indicated channel during picture onset.

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
	- channel: integer value indicating what channel will be used to regress activity
	- t_before: time before (s) the picture onset that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of onset forward when considering the window of activity.
	- t_after: time after (s) the picture onset that should be included when computing the firing rate.

	Output:
	- num_trials: array of length N with an element corresponding to the number of successful trials in each of the
					hdf_files
	- window_fr: dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
					dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
					to the average firing rate over the window indicated.
	'''
	num_trials = np.zeros(len(hdf_files))
	num_units = np.zeros(len(hdf_files))
	window_fr = dict()
	for i, hdf_file in enumerate(hdf_files):
		cb_block = ChoiceBehavior_ThreeTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_picture_onset = cb_block.ind_check_reward_states - 5
		
		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data and find all sort codes associated with good channels
			if channel < 97:
				spike = OfflineSorted_CSVFile(spike_files[i][0])
			else:
				spike = OfflineSorted_CSVFile(spike_files[i][1])

			# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
			# designated window.
			sc_chan = spike.find_chan_sc(channel)
			num_units[i] = len(sc_chan)
			for sc in sc_chan:
				sc_fr = spike.compute_window_fr(channel,sc,times_row_ind,t_before,t_after)
				if i == 0:
					all_fr = sc_fr
				else:
					all_fr = np.vstack([all_fr, sc_fr])

			# Save matrix of firing rates for units on channel from trials during hdf_file as dictionary element
			window_fr[i] = all_fr

	return num_trials, num_units, window_fr


def ThreeTargetTask_RegressFiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, trial_case, var_value, channel, t_before, t_after):
	'''
	This method regresses the firing rate of all units as a function of value. Only trials from the specified
	trial_case are considered in the regression. There are six trial cases: (1) instructed to low-value [1,0,0] (2) instructed
	to middle-value [0,0,1] (3) instructed to high-value [0,1,0] (4) free-choice with low and middle values [1,0,1] (5) free-choice with low and
	high values [1,1,0] (6) free-choice to middle and high values [0,1,1].

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
	- trial_case: 3-tuple that indicates the trial type considered in the regression. Trial cases are 
					enumerated as in the above description, which matches the notation used for the 
					targets_on record of the targets presented.
	- var_value: boolenan indicating whether the Q value(s) used in the regression should be fixed (var_value == False) 
				and defined based on their reward probabilties, or whether the Q value(s) should be varying trial-by-trial
				(var_value == True) based on the Q-learning model fit
	- channel: integer value indicating what channel will be used to regress activity
	- t_before: time before (s) the picture onset that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of onset forward when considering the window of activity.
	- t_after: time after (s) the picture onset that should be included when computing the firing rate.

	'''
	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]
	ind_trial_case = np.array([ind for ind in range(cb.num_successful_trials) if np.array_equal(targets_on[ind],trial_case)])
	
	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr = ThreeTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials)

	# 3. Get Q-values
	if var_value:
	# Varying Q-values
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards
		# Find ML fit of alpha and beta
		Q_initial = 0.5*np.ones(3)
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
		result = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,None)])
		alpha_ml, beta_ml = result["x"]
		# RL model fit for Q values
		Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high = loglikelihood_ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)
	else:
	# Fixed Q-values
		Q_low = 0.35*np.ones(total_trials)
		Q_mid = 0.6*np.ones(total_trials)
		Q_high = 0.85*np.ones(total_trials)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.empty([max_num_units,total_trials])
	fr_mat[:] = np.NAN
	trial_counter = 0
	for j in window_fr.keys():
		block_fr = window_fr[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 
		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	# 5. Do regression for each unit only on trials of correct trial type with spike data saved.
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=np.NAN], dtype = int)
		
		x = np.vstack((Q_low[trial_inds], Q_mid[trial_inds], Q_high[trial_inds]))
		x = np.transpose(x)
		x = sm.add_constant(x,prepend='False')
		print x.shape
		y = unit_data[trial_inds]
		print y.shape

		print "Regression for unit ", k
		model_glm = sm.OLS(y,x)
		fit_glm = model_glm.fit()
		print fit_glm.summary()

	return window_fr, fr_mat


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
			plt.close()

	return

def ThreeTargetTask_SpikeAnalysis_SingleChannel(hdf_files, syncHDF_files, spike_files, chann, sc, plot_output):
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
	- sc: integer representing unit sort code
	- plot_output: binary indicating whether output should be plotted + saved or not
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)

	# Define timing parameters for PSTHs
	t_before = 1			# 1 s
	t_after = 3				# 3 s
	t_resolution = 0.1 		# 100 ms time bins
	num_bins = len(np.arange(-t_before, t_after, t_resolution)) - 1

	# Define arrays to save psth for each trial
	smooth_psth_lm = np.array([])
	smooth_psth_lh = np.array([])
	smooth_psth_mh = np.array([])
	smooth_psth_l = np.array([])
	smooth_psth_m = np.array([])
	smooth_psth_h = np.array([])
	psth_lm = np.array([])
	psth_lh = np.array([])
	psth_mh = np.array([])
	psth_l = np.array([])
	psth_m = np.array([])
	psth_h = np.array([])

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
			spike1_good_channels = [unit for unit in cd_units if unit in spike1.good_channels]
			spike2_good_channels = [unit for unit in cd_units if unit in spike2.good_channels]

			# 1. LH presented
			LH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,1,0]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_lh, smooth_avg_psth_lh = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			else:
				avg_psth_lh, smooth_avg_psth_lh = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			if i == 0:
				psth_lh = avg_psth_lh
				smooth_psth_lh = smooth_avg_psth_lh
			else:
				psth_lh = np.vstack([psth_lh, avg_psth_lh])
				smooth_psth_lh = np.vstack([smooth_psth_lh, smooth_avg_psth_lh])

			# 2. LM presented
			LM_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,0,1]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_lm, smooth_avg_psth_lm = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[LM_ind],t_before,t_after,t_resolution)
			else:
				avg_psth_lm, smooth_avg_psth_lm = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[LM_ind],t_before,t_after,t_resolution)
			if i == 0:
				psth_lm = avg_psth_lm
				smooth_psth_lm = smooth_avg_psth_lm
			else:
				psth_lm = np.vstack([psth_lm, avg_psth_lm])
				smooth_psth_lm = np.vstack([smooth_psth_lm, smooth_avg_psth_lm])

			# 3. MH presented
			MH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,1,1]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_mh, smooth_avg_psth_mh = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[MH_ind],t_before,t_after,t_resolution)
			else:
				avg_psth_mh, smooth_avg_psth_mh = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[MH_ind],t_before,t_after,t_resolution)
			if i == 0:
				psth_mh = avg_psth_mh
				smooth_psth_mh = smooth_avg_psth_mh
			else:
				psth_mh = np.vstack([psth_mh, avg_psth_mh])
				smooth_psth_mh = np.vstack([smooth_psth_mh, smooth_avg_psth_mh])

			# 4. L presented
			L_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,0,0]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_l, smooth_avg_psth_l = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
			else:
				avg_psth_l, smooth_avg_psth_l = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
			if i == 0:
				psth_l = avg_psth_l
				smooth_psth_l = smooth_avg_psth_l
			else:
				psth_l = np.vstack([psth_l, avg_psth_l])
				smooth_psth_l = np.vstack([smooth_psth_l, smooth_avg_psth_l])

			# 5. H presented
			H_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,1,0]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_h, smooth_avg_psth_h = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
			else:
				avg_psth_h, smooth_avg_psth_h = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
			if i == 0:
				psth_h = avg_psth_h
				smooth_psth_h = smooth_avg_psth_h
			else:
				psth_h = np.vstack([psth_h, avg_psth_h])
				smooth_psth_h = np.vstack([smooth_psth_h, smooth_avg_psth_h])

			# 6. M presented
			M_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [0,0,1]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_m, smooth_avg_psth_m = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[M_ind],t_before,t_after,t_resolution)
			else:
				avg_psth_m, smooth_avg_psth_m = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[M_ind],t_before,t_after,t_resolution)
			if i == 0:
				psth_m = avg_psth_m
				smooth_psth_m = smooth_avg_psth_m
			else:
				psth_m = np.vstack([psth_m, avg_psth_m])
				smooth_psth_m = np.vstack([smooth_psth_m, smooth_avg_psth_m])

			# Plot average rate for all neurons divided in six cases of targets on option
			if plot_output:
				plt.figure()

				avg_psth_lh = np.nanmean(avg_psth_lh, axis = 0)
				smooth_avg_psth_lh = np.nanmean(smooth_avg_psth_lh, axis = 0)
				plt.subplot(3,2,1)
				plt.title('Low-High Presented')
				plt.plot(smooth_avg_psth_lh)
				xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
				xticks = np.arange(0, len(xticklabels), 10)
				xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
				plt.xticks(xticks, xticklabels)

				avg_psth_lm = np.nanmean(avg_psth_lm, axis = 0)
				smooth_avg_psth_lm = np.nanmean(smooth_avg_psth_lm, axis = 0)
				plt.subplot(3,2,2)
				plt.title('Low-Middle Presented')
				plt.plot(smooth_avg_psth_lm)
				xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
				xticks = np.arange(0, len(xticklabels), 10)
				xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
				plt.xticks(xticks, xticklabels)

				avg_psth_mh = np.nanmean(avg_psth_mh, axis = 0)
				smooth_avg_psth_mh = np.nanmean(smooth_avg_psth_mh, axis = 0)
				plt.subplot(3,2,3)
				plt.title('Middle-High Presented')
				plt.plot(smooth_avg_psth_mh)
				xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
				xticks = np.arange(0, len(xticklabels), 10)
				xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
				plt.xticks(xticks, xticklabels)

				avg_psth_l = np.nanmean(avg_psth_l, axis = 0)
				smooth_avg_psth_l = np.nanmean(smooth_avg_psth_l, axis = 0)
				plt.subplot(3,2,4)
				plt.title('Low Presented')
				plt.plot(smooth_avg_psth_l)
				xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
				xticks = np.arange(0, len(xticklabels), 10)
				xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
				plt.xticks(xticks, xticklabels)

				avg_psth_h = np.nanmean(avg_psth_h, axis = 0)
				smooth_avg_psth_h = np.nanmean(smooth_avg_psth_h, axis = 0)
				plt.subplot(3,2,5)
				plt.title('High Presented')
				plt.plot(smooth_avg_psth_h)
				xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
				xticks = np.arange(0, len(xticklabels), 10)
				xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
				plt.xticks(xticks, xticklabels)

				avg_psth_m = np.nanmean(avg_psth_m, axis = 0)
				smooth_avg_psth_m = np.nanmean(smooth_avg_psth_m, axis = 0)
				plt.subplot(3,2,6)
				plt.title('Middle Presented')
				plt.plot(smooth_avg_psth_m)
				xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
				xticks = np.arange(0, len(xticklabels), 10)
				xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
				plt.xticks(xticks, xticklabels)

				plt_name = syncHDF_files[i][34:-12]
				plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+'_PSTH_Chan'+str(chann)+'.svg')
				plt.close()

	reg_psth = [psth_lm, psth_lh, psth_mh, psth_l, psth_h, psth_m]
	smooth_psth = [smooth_psth_lm, smooth_psth_lh, smooth_psth_mh, smooth_psth_l, smooth_psth_h, smooth_psth_m]
	return reg_psth, smooth_psth