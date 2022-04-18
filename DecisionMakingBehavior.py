import numpy as np 
import scipy as sp
import tdt
from scipy import stats
from statsmodels.formula.api import ols
from scipy.interpolate import spline
from scipy import signal
from scipy.ndimage import filters
import scipy.optimize as op
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
from scipy import io
from scipy import stats
import matplotlib as mpl
from matplotlib import mlab
import matplotlib.pyplot as plt
import tables
from rt_calc import compute_rt_per_trial_FreeChoiceTask
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from neo import io
from PulseMonitorData import findIBIs
from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile
from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance
from spectralAnalysis import computePowerFeatures



cd_units = [1, 3, 4, 17, 18, 20, 40, 41, 54, 56, 57, 63, 64, 72, 75, 81, 83, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
acc_units = [5, 6, 19, 22, 30, 39, 42, 43, 55, 58, 59, 69, 74, 77, 85, 90, 91, 102, 105, 121, 128]	
dir = "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Mario/spike_data/"
dir_luigi = "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Luigi/spike_data/"
dir_figs = "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Paper/Figures/"			

def trial_sliding_avg(trial_array, num_trials_slide):

	num_trials = len(trial_array)
	slide_avg = np.zeros(num_trials)

	for i in range(num_trials):
		if i < num_trials_slide:
			slide_avg[i] = np.sum(trial_array[:i+1])/float(i+1)
		else:
			slide_avg[i] = np.sum(trial_array[i-num_trials_slide+1:i+1])/float(num_trials_slide)

	return slide_avg

def Value_from_reward_history_TwoTargetTask(hdf_files):

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 2. Get chosen targets and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	chose_hv_inds = np.ravel(np.nonzero(chosen_target[:-1]-1))
	chose_lv_inds = np.ravel(np.nonzero(2-chosen_target[:-1]))

	# 3. Extract reward information for each target
	q_hv = np.zeros(len(chosen_target))
	q_lv = np.zeros(len(chosen_target))
	q_hv[0] = 0.5
	q_lv[0] = 0.5

	reward_hv = rewards[chose_hv_inds]
	reward_lv = rewards[chose_lv_inds]

	# 4. Determine values as sliding average of reward history
	reward_hv_avg = trial_sliding_avg(reward_hv,10)
	reward_lv_avg = trial_sliding_avg(reward_lv,10)

	q_hv[chose_hv_inds+1] = reward_hv_avg
	q_lv[chose_lv_inds+1] = reward_lv_avg

	# 4a. Fill in values for trials where target was not selected
	for i in range(len(q_hv)-1):
		if i not in chose_hv_inds:
			q_hv[i+1] = q_hv[i]

	for i in range(len(q_lv)-1):
		if i not in chose_lv_inds:
			q_lv[i+1] = q_lv[i]

	# 5. Smooth value information
	b = signal.gaussian(20, 1)
	q_hv_smooth = filters.convolve1d(q_hv, b/b.sum())
	q_lv_smooth = filters.convolve1d(q_lv, b/b.sum())

	'''
	plt.figure()
	plt.plot(q_hv,'r')
	plt.plot(q_hv_smooth,'m')
	plt.plot(q_lv,'b')
	plt.plot(q_lv_smooth,'c')
	plt.show()
	'''

	return q_hv_smooth, q_lv_smooth, q_hv, q_lv

def Value_from_reward_history_ThreeTargetTask(hdf_files):

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	
	# 2. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	chose_hv_inds = np.ravel(np.nonzero(np.floor(0.5*chosen_target[:-1])))
	chose_mv_inds = np.ravel(np.nonzero(1 - np.abs(chosen_target[:-1] -1)))
	chose_lv_inds = np.ravel(np.nonzero(np.floor(1 - 0.5*chosen_target[:-1])))

	# 3. Extract reward information for each target
	q_hv = np.zeros(len(chosen_target))
	q_lv = np.zeros(len(chosen_target))
	q_mv = np.zeros(len(chosen_target))
	q_hv[0] = 0.5
	q_lv[0] = 0.5
	q_mv[0] = 0.5

	reward_hv = rewards[chose_hv_inds]
	reward_lv = rewards[chose_lv_inds]
	reward_mv = rewards[chose_mv_inds]

	# 4. Determine values as sliding average of reward history
	reward_hv_avg = trial_sliding_avg(reward_hv,10)
	reward_lv_avg = trial_sliding_avg(reward_lv,10)
	reward_mv_avg = trial_sliding_avg(reward_mv,10)

	q_hv[chose_hv_inds+1] = reward_hv_avg
	q_lv[chose_lv_inds+1] = reward_lv_avg
	q_mv[chose_mv_inds+1] = reward_mv_avg

	# 4a. Fill in values for trials where target was not selected
	for i in range(len(q_hv)-1):
		if i not in chose_hv_inds:
			q_hv[i+1] = q_hv[i]

	for i in range(len(q_lv)-1):
		if i not in chose_lv_inds:
			q_lv[i+1] = q_lv[i]

	for i in range(len(q_mv)-1):
		if i not in chose_mv_inds:
			q_mv[i+1] = q_mv[i]

	# 5. Smooth value information
	b = signal.gaussian(20, 1)
	q_hv_smooth = filters.convolve1d(q_hv, b/b.sum())
	q_lv_smooth = filters.convolve1d(q_lv, b/b.sum())
	q_mv_smooth = filters.convolve1d(q_mv, b/b.sum())

	'''
	plt.figure()
	plt.plot(q_hv,'r', label = 'HV')
	plt.plot(q_hv_smooth,'m')
	plt.plot(q_lv,'b', label = 'LV')
	plt.plot(q_lv_smooth,'c')
	plt.plot(q_mv,'g', label = 'MV')
	plt.plot(q_mv_smooth,'y')
	plt.plot(0.5*chosen_target,'k')
	plt.show()
	'''

	return q_hv_smooth, q_mv_smooth, q_lv_smooth


class ChoiceBehavior_ThreeTargets():

	def __init__(self, hdf_file):
		self.filename =  hdf_file
		self.table = tables.open_file(self.filename)

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
		self.trial_type = self.table.root.task[:]['target_index']
		self.targets_on = self.table.root.task[:]['LHM_target_on']
	  
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
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

	def TrialChoices(self, num_trials_slide, plot_results):
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
				if choice==b'hold_targetM':
					all_choices[i] = 1		# optimal choice was made
					LM_choices = np.append(LM_choices, 1)
				else:
					LM_choices = np.append(LM_choices, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice==b'hold_targetH':
					all_choices[i] = 1
					LH_choices = np.append(LH_choices, 1)
				else:
					LH_choices = np.append(LH_choices, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice==b'hold_targetH':
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
		rewarded_choice = np.array(self.state[self.ind_check_reward_states + 1] == b'reward', dtype = float)
		num_trials = len(target_choices)

		for i, choice in enumerate(target_choices):
			if choice == b'hold_targetM':
				target_chosen[i,2] = 1
			if choice == b'hold_targetL':
				target_chosen[i,0] = 1
			if choice == b'hold_targetH':
				target_chosen[i,1] = 1

		return target_options, target_chosen, rewarded_choice

	def GetChoicesAndRewards(self):
		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.zeros(len(ind_holds))
		
		for i, ind in enumerate(ind_holds):
			if self.state[ind] == b'hold_targetM':
				chosen_target[i] = 1
			elif self.state[ind] == b'hold_targetH':
				chosen_target[i] = 2

		return targets_on, chosen_target, rewards, instructed_or_freechoice

class ChoiceBehavior_TwoTargets():

	def __init__(self, hdf_file):
		self.filename =  hdf_file
		self.table = tables.open_file(self.filename)

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
		self.trial_type = self.table.root.task[:]['target_index']
	  
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
		
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
		print(syncHDF_file)
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
		optimal target choice (high-value target). 

		Input:
		- num_trials_slide: integer, indicates the number of points to perform sliding average over
		- plot_results: Boolean, indicates if the output should be plotted and shown (=True) or not (=False)

		Output:
		- all_choices: array of length N, where N is the number of free-choice trials, that indicates whether an
					optimal choice was made or not. Low-value choices are indicated with 0s and high-value choices
					are indicated with 1s. 
		'''
		freechoice_trial = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]]) - 1
		freechoice_trial_ind = np.ravel(np.nonzero(freechoice_trial))
		target_choices = self.state[self.ind_check_reward_states - 2]
		num_trials = len(target_choices)
		all_choices = np.array([int(choice==b'hold_targetH') for choice in target_choices[freechoice_trial_ind]])
		cmap = mpl.cm.hsv

		sliding_avg_all_choices = trial_sliding_avg(all_choices, num_trials_slide)

		if plot_results:
			fig = plt.figure()
			ax = plt.subplot(111)
			plt.plot(sliding_avg_all_choices, c = 'b', label = 'HV')
			plt.plot(1 - sliding_avg_all_choices, c = 'r', label = 'LV')
			plt.xlabel('Free-choice Trials')
			plt.ylabel('Probability Target Choice')
			ax.get_yaxis().set_tick_params(direction='out')
			ax.get_xaxis().set_tick_params(direction='out')
			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()
			plt.legend()
			plt.show()

		return all_choices

	def TrialOptionsAndChoice(self):
		'''
		This method extracts for each trial which targets were on options and what the ultimate target
		choice was. It also gives whether or not that choice was rewarded.

		Output:
		- target_options: N x 2 array, where N is the number of trials (instructed + freechoice), which contains 0 or 1s
							to indicate on a given trial which of the 2 targets was shown. The order is Low - High.
							For example, if on the ith trial the Low and High value targets are shown, then 
							target_options[i,:] = [1, 1]. If on the ith trial, only the Low value target is shown, then 
							target_options[i,:] = [1, 0].
		- target_chosen: N x 2 array, which contains 0 or 1s to indicate on a given trial which of the 2 targets was chosen.
							The order is Low - High. Note that only one entry per row may be non-zero (1). For 
							example, if on the ith trial the High value target was selected, then 
							target_chosen[i,:] = [0, 1].
		- reward_chosen: length N array, which contains 0s or 1s to indicate whether a reward was received at the end
							of the ith trial. 
		'''
		target_choices = self.state[self.ind_check_reward_states - 2]
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])
		target_options = np.zeros((len(target_choices), 2))  	# array of two boolean values: LH
		target_chosen = np.zeros((len(target_choices), 2)) 		# placeholder for array with boolen values indicating choice: LH
		rewarded_choice = np.array(self.state[self.ind_check_reward_states + 1] == b'reward', dtype = float)
		num_trials = len(target_choices)

		for i, choice in enumerate(target_choices):
			if choice == b'hold_targetL':
				target_chosen[i,0] = 1
				target_options[i,:] = [1,instructed_or_freechoice[i] - 1]
			if choice == b'hold_targetH':
				target_chosen[i,1] = 1
				target_options[i,:] = [instructed_or_freechoice[i] - 1, 1]

		return target_options, target_chosen, rewarded_choice

	def GetChoicesAndRewards(self):
		'''
		This method extracts for each trial which target was chosen, whether or not a reward was given, and if
		the trial was instructed or free-choice. These arrays are needed for the RLPerformance methods in 
		logLikelihoodRLPerformance.py.

		Output:
		- chosen_target: array of length N, where N is the number of trials (instructed + freechoice), which contains 
						contains values indicating a low-value target choice was made (=1) or if a high-value target
						choice was made (=2)
		- rewards: array of length N, which contains 0s or 1s to indicate whether a reward was received at the end
							of the ith trial. 
		- instructed_or_freechoice: array of length N, which indicates whether a trial was instructed (=1) or free-choice (=2)
		'''

		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.zeros(len(ind_holds))
		chosen_target = np.array([(int(self.state[ind]==b'hold_targetH') + 1) for ind in ind_holds])

		'''
		for i, ind in enumerate(ind_holds):
			if self.state[ind] == 'hold_targetL':
				chosen_target[i] = 1
			elif self.state[ind] == 'hold_targetH':
				chosen_target[i] = 2
		'''
		return chosen_target, rewards, instructed_or_freechoice


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
			table = tables.open_file(filename)
			if i == 0:
				self.state = table.root.task_msgs[:]['msg']
				self.state_time = table.root.task_msgs[:]['time']
				self.trial_type = table.root.task[:]['target_index']
				self.targets_on = table.root.task[:]['LHM_target_on']
				self.targetL = table.root.task[:]['targetL']
				self.targetH = table.root.task[:]['targetH']
				self.targetM = table.root.task[:]['targetM']
			else:
				self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
				self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
				self.trial_type = np.append(self.trial_type, table.root.task[:]['target_index'])
				self.targets_on = np.append(self.targets_on, table.root.task[:]['LHM_target_on'])
				self.targetL = np.vstack([self.targetL, table.root.task[:]['targetL']])
				self.targetH = np.vstack([self.targetH, table.root.task[:]['targetH']])
				self.targetM = np.vstack([self.targetM, table.root.task[:]['targetM']])
		
		if len(hdf_files) > 1:
			self.targets_on = np.reshape(self.targets_on, (int(len(self.targets_on)/3),3))  				# this should contain triples indicating targets
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
		
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
				if choice==b'hold_targetM':
					all_choices_A[i] = 1		# optimal choice was made
					LM_choices_A = np.append(LM_choices_A, 1)
				else:
					LM_choices_A = np.append(LM_choices_A, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice==b'hold_targetH':
					all_choices_A[i] = 1
					LH_choices_A = np.append(LH_choices_A, 1)
				else:
					LH_choices_A = np.append(LH_choices_A, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice==b'hold_targetH':
					all_choices_A[i] = 1
					MH_choices_A = np.append(MH_choices_A, 1)
				else:
					MH_choices_A = np.append(MH_choices_A, 0)

		for i, choice in enumerate(target_choices_Aprime):
			# only look at freechoice trials
			targ_presented = targets_on[freechoice_trial_ind_Aprime[i]]
			# L-M targets presented
			if (targ_presented[0]==1)&(targ_presented[2]==1):
				if choice==b'hold_targetM':
					all_choices_Aprime[i] = 1		# optimal choice was made
					LM_choices_Aprime = np.append(LM_choices_Aprime, 1)
				else:
					LM_choices_Aprime = np.append(LM_choices_Aprime, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice==b'hold_targetH':
					all_choices_Aprime[i] = 1
					LH_choices_Aprime = np.append(LH_choices_Aprime, 1)
				else:
					LH_choices_Aprime = np.append(LH_choices_Aprime, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice==b'hold_targetH':
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
		rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.zeros(len(ind_holds))

		for i, ind in enumerate(ind_holds):
			if self.state[ind] == b'hold_targetM':
				chosen_target[i] = 1
			elif self.state[ind] == b'hold_targetH':
				chosen_target[i] = 2

		return targets_on, chosen_target, rewards, instructed_or_freechoice


	def TargetSideSelection_Block3(self):
		# Get target selection information
		targets_on, chosen_target, rewards, instructed_or_freechoice = self.GetChoicesAndRewards()

		# Get target side information
		ind_targets = self.ind_check_reward_states - 3
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		targetM_side = self.targetM[self.state_time[ind_targets]][:,2]

		# Compute probabilities of target selection given target is on left vs right
		# 1. Compute choosing HV target when presented with LV target
		counter_left = 0
		counter_right = 0
		prob_chooseHV_left = 0
		prob_chooseHV_right = 0
		for i in range(self.num_trials_A + self.num_trials_B,len(chosen_target)):  				# only consider trials in Block A'
			if np.array_equal(targets_on[i],[0,1,1]):  		# only consider when H and M are shown
				choice = chosen_target[i]						# choice = 1 if MV, choice = 2 if HV
				sideH = targetH_side[i]							# side of HV target
				prob_chooseHV_left += float((choice==2)*(sideH==1))
				prob_chooseHV_right += float((choice==2)*(sideH==-1))
				counter_left += float(sideH==1)
				counter_right += float(sideH==-1)

		prob_chooseHV_left = prob_chooseHV_left/float(counter_left + counter_right)  	# joint prob: P(choose HV, HV is on left)
		prob_chooseHV_right = prob_chooseHV_right/float(counter_left + counter_right) 	# joint prob: P(choose HV, HV is on right)
		prob_HV_left = counter_left/float(counter_left + counter_right) 		# prob: P(HV on left)
		prob_HV_right = counter_right/float(counter_left + counter_right)		# prob: P(HV on right)
		prob_chooseHV_given_left = prob_chooseHV_left/prob_HV_left 				# conditional prob: P(choose HV|HV on left)
		prob_chooseHV_given_right = prob_chooseHV_right/prob_HV_right 			# conditional prob: P(choose HV|HV on right)

		return prob_chooseHV_given_left, prob_chooseHV_given_right

	def ChoicesAfterStimulation(self):
		'''
		Method to extract information about the stimulation trials and the trial immediately following
		stimulation. 

		Output: 
		- targets_on_after: n x 3 array, where n is the number of stimulation trials, containing indicators
			of whether the LV, HV, or MV targets are shown, respectively. E.g. if after the first stimulation
			trial the LV and MV targets are shown, then targets_on_after[0] = [1,0,1].
		- choice: length n array with values indicating which target was selecting in the trial following
			stimulation (=0 if LV, = 1 if MV, = 2 if HV)
		- stim_reward: length n array with Boolean values indicating whether reward was given (True) during
			the stimulation trial or not (False)
		- target_reward: length n array with Boolean values indicating whether reward was given (True) in
			the trial following the stimulation trial or not (False)
		- stim side: length n array with values indicating what side the MV target was on during the 
			stimulation trial (= 1 for left, -1 for right)
		- stim_trial_ind: length n array containing trial numbers during which stimulation was performed. 
			Since stimulation is during Block A' only, the minimum value should be num_trials_A + num_trials_B
			and the maximum value should be the total number of trials.
		'''

		# Get target selection information
		targets_on, chosen_target, rewards, instructed_or_freechoice = self.GetChoicesAndRewards()
		ind_targets = self.ind_check_reward_states - 3
		targetM_side = self.targetM[self.state_time[ind_targets]][:,2]
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice

		targets_on_after = []	# what targets are presented after stim trial
		choice = []				# what target was selected in trial following stim trial
		choice_side = []		# what side the selected target was on following the stim trial
		stim_reward = []		# was the stim trial rewarded
		target_reward = []		# was the target selected in trial after stimulation rewarded
		stim_side = [] 			# side the MV target was on during stimulation
		stim_trial_ind = []		# index of trial with stimulation

		counter = 0

		# Find trials only following a trial with stimulation
		for i in range(self.num_trials_A + self.num_trials_B,len(chosen_target)-1):  					# only consider trials in Block A'
			# only consider M is shown (stim trial) and next trial is not also a stim trial
			if np.array_equal(targets_on[i],[0,0,1])&(~np.array_equal(targets_on[i+1],[0,0,1])):  		
				if counter==0:
					targets_on_after = targets_on[i+1]
				else:
					targets_on_after = np.vstack([targets_on_after,targets_on[i+1]])
				
				choice = np.append(choice,chosen_target[i+1])					# choice = 0 if LV, choice = 1 if MV, choice = 2 if HV
				if chosen_target[i+1]==0:
					side = targetL_side[i+1]*(targetL_side[i+1]!=0) + (2*np.random.randint(2) - 1)*(targetL_side[i+1]==0)
					choice_side = np.append(choice_side, side)
				elif chosen_target[i+1]==1:
					side = targetM_side[i+1]*(targetM_side[i+1]!=0) + (2*np.random.randint(2) - 1)*(targetM_side[i+1]==0)
					choice_side = np.append(choice_side, side)
				else:
					side = targetH_side[i+1]*(targetH_side[i+1]!=0) + (2*np.random.randint(2) - 1)*(targetH_side[i+1]==0)
					choice_side = np.append(choice_side, side)

				stim_side_find = targetM_side[i]*(targetM_side[i]!=0) + (2*np.random.randint(2) - 1)*(targetM_side[i]==0)
				#stim_side_find = targetM_side[i]
				stim_reward = np.append(stim_reward, rewards[i])
				target_reward = np.append(target_reward, rewards[i+1])
				stim_side = np.append(stim_side, stim_side_find)
				stim_trial_ind = np.append(stim_trial_ind, i)
				counter += 1

		return targets_on_after, choice, choice_side, stim_reward, target_reward, stim_side, stim_trial_ind

class ChoiceBehavior_TwoTargets_Stimulation():
	'''
	Class for behavior taken from ABA' task, where there are two targets of different probabilities of reward
	and stimulation is paired with the middle-value target during the hold-period of instructed trials during
	blocks B and A'. Can pass in a list of hdf files when initially instantiated in the case that behavioral data
	is split across multiple hdf files. In this case, the files should be listed in the order in which they were saved.
	'''

	def __init__(self, hdf_files, num_trials_A, num_trials_B):
		for i, hdf_file in enumerate(hdf_files): 
			filename =  hdf_file
			table = tables.open_file(filename)
			if i == 0:
				self.state = table.root.task_msgs[:]['msg']
				self.state_time = table.root.task_msgs[:]['time']
				self.trial_type = table.root.task[:]['target_index']
				self.targetL = table.root.task[:]['targetL']
				self.targetH = table.root.task[:]['targetH']
			else:
				self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
				self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
				self.trial_type = np.append(self.trial_type, table.root.task[:]['target_index'])
				self.targetL = np.vstack([self.targetL, table.root.task[:]['targetL']])
				self.targetH = np.vstack([self.targetH, table.root.task[:]['targetH']])
				
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
		
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
		optimal target choice. It looks at overall the liklihood of selecting the better choice in free-choice
		trials. Choice behavior is split across the three blocks.

		Input:
		- num_trials_slide: integer, indicates the number of points to perform sliding average over
		- plot_results: Boolean, indicates if the output should be plotted and shown (=True) or not (=False)

		Output:
		- all_choices_A: array of length num_trials_A that indicates whether an
					optimal choice was made or not. Low-value choices are indicated with 0s and high-value choices
					are indicated with 1s. 
		- all_choices_Aprime: array of length num_successful_trials - num_trials_A - num_trials_B (equal to length
					of the final block of behavior, A'), that indicates whether an optimal choice was made or not.
					Low-value choices are indicated with 0s and high-value choices are indicated with 1s.
		'''

		# Get indices of free-choice trials for Blocks A and A', as well as the corresponding target selections.
		freechoice_trial = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]]) - 1
		freechoice_trial_ind_A = np.ravel(np.nonzero(freechoice_trial[:self.num_trials_A]))
		freechoice_trial_ind_Aprime = np.ravel(np.nonzero(freechoice_trial[self.num_trials_A+self.num_trials_B:])) + self.num_trials_A+self.num_trials_B
		
		target_choices_A = self.state[self.ind_check_reward_states - 2][freechoice_trial_ind_A]
		target_choices_Aprime = self.state[self.ind_check_reward_states - 2][freechoice_trial_ind_Aprime]
		
		# Initialize variables
		num_FC_trials_A = len(freechoice_trial_ind_A)
		num_FC_trials_Aprime = len(freechoice_trial_ind_Aprime)
		all_choices_A = np.array([int(choice==b'hold_targetH') for choice in target_choices_A])
		all_choices_Aprime = np.array([int(choice==b'hold_targetH') for choice in target_choices_Aprime])

		sliding_avg_all_choices_A = trial_sliding_avg(all_choices_A, num_trials_slide)
		sliding_avg_all_choices_Aprime = trial_sliding_avg(all_choices_Aprime, num_trials_slide)
		
		if plot_results:
			fig = plt.figure()
			ax = plt.subplot(121)
			plt.plot(sliding_avg_all_choices_A, c = 'b', label = 'Block A')
			plt.xlabel('Free-choice Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Block A')
			plt.ylim((0,1))
			ax.get_yaxis().set_tick_params(direction='out')
			ax.get_xaxis().set_tick_params(direction='out')
			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()
			ax = plt.subplot(122)
			plt.plot(sliding_avg_all_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Free-choice Trials')
			plt.ylabel('Probability Best Choice')
			plt.title("Block A'")
			plt.ylim((0,1))
			ax.get_yaxis().set_tick_params(direction='out')
			ax.get_xaxis().set_tick_params(direction='out')
			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()
			plt.show()

		return all_choices_A, all_choices_Aprime

	def GetChoicesAndRewards(self):
		'''
		This method extracts for each trial which target was chosen, whether or not a reward was given, and if
		the trial was instructed or free-choice. These arrays are needed for the RLPerformance methods in 
		logLikelihoodRLPerformance.py.

		Output:
		- chosen_target: array of length N, where N is the number of trials (instructed + freechoice), which contains 
						contains values indicating a low-value target choice was made (=1) or if a high-value target
						choice was made (=2)
		- rewards: array of length N, which contains 0s or 1s to indicate whether a reward was received at the end
							of the ith trial. 
		- instructed_or_freechoice: array of length N, which indicates whether a trial was instructed (=1) or free-choice (=2)
		'''

		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.array([(int(self.state[ind]==b'hold_targetH') + 1) for ind in ind_holds])

		return chosen_target, rewards, instructed_or_freechoice


	def ChoicesAfterStimulation(self):
		'''
		Method to extract information about the stimulation trials and the trial immediately following
		stimulation. 

		Output: 
		- choice: length n array with values indicating which target was selecting in the trial following
			stimulation (= 1 if LV, = 2 if HV)
		- stim_reward: length n array with Boolean values indicating whether reward was given (True) during
			the stimulation trial or not (False)
		- target_reward: length n array with Boolean values indicating whether reward was given (True) in
			the trial following the stimulation trial or not (False)
		- stim side: length n array with values indicating what side the MV target was on during the 
			stimulation trial (= 1 for left, -1 for right)
		- stim_trial_ind: length n array containing trial numbers during which stimulation was performed. 
			Since stimulation is during Block A' only, the minimum value should be num_trials_A + num_trials_B
			and the maximum value should be the total number of trials.
		'''

		# Get target selection information
		chosen_target, rewards, instructed_or_freechoice = self.GetChoicesAndRewards()
		ind_targets = self.ind_check_reward_states - 3
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		instructed_trials_inds = np.ravel(np.nonzero(2 - instructed_or_freechoice))
		stim_trial_inds = instructed_trials_inds[np.ravel(np.nonzero(np.greater(instructed_trials_inds, self.num_trials_A + self.num_trials_B)))]
		stim_trial_inds = np.array([ind for ind in stim_trial_inds if ((ind+1) not in stim_trial_inds)&(ind < (self.num_successful_trials-1))])
		fc_trial_inds = stim_trial_inds + 1

		choice = chosen_target[fc_trial_inds]		# what target was selected in trial following stim trial
		stim_reward = rewards[stim_trial_inds]		# was the stim trial rewarded
		target_reward = rewards[fc_trial_inds]		# was the target selected in trial after stimulation rewarded
		stim_side = targetL_side[stim_trial_inds] 		# side the MV target was on during stimulation
		choice_side = np.array([(targetH_side[ind]*(choice[i]==2) + targetL_side[ind]*(choice[i]==1)) for i,ind in enumerate(fc_trial_inds)])		# what side the selected target was on following the stim trial
		
		return choice, choice_side, stim_reward, target_reward, stim_side, stim_trial_inds



def ThreeTargetTask_Qlearning(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
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
	Q_low[0] = Q_initial[0]
	Q_mid[0] = Q_initial[1]
	Q_high[0] = Q_initial[2]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	prob_choice_low[0] = 0.5
	prob_choice_mid[0] = 0.5
	prob_choice_high[0] = 0.5

	prob_choice_opt_lvhv = np.array([])
	prob_choice_opt_mvhv = np.array([])
	prob_choice_opt_lvmv = np.array([])

	log_prob_total = 0.
	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# Update Q values with temporal difference error
		delta_low = float(rewards[i]) - Q_low[i]
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_low[i+1] = Q_low[i] + alpha*delta_low*float(chosen_target[i]==0)
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)

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
				prob_choice_opt_lvhv = np.append(prob_choice_opt_lvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = 0.5*chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==0))

			elif np.array_equal(targets_on[i+1],[1,0,1]):
				Q_opt = Q_mid[i+1]
				Q_nonopt = Q_low[i+1]

				prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
				prob_choice_high[i+1] = prob_choice_high[i]
				prob_choice_mid[i+1] = 1. - prob_choice_low[i+1]

				prob_choice_opt = prob_choice_mid[i+1]
				prob_choice_nonopt = prob_choice_low[i+1]
				prob_choice_opt_lvmv = np.append(prob_choice_opt_lvmv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_mid[i+1] >= 0.5)&(chosen_target[i+1]==1) or (prob_choice_mid[i+1] < 0.5)&(chosen_target[i+1]==0))


			else:
				Q_opt = Q_high[i+1]
				Q_nonopt = Q_mid[i+1]

				prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
				prob_choice_low[i+1] = prob_choice_low[i]
				prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

				prob_choice_opt = prob_choice_high[i+1]
				prob_choice_nonopt = prob_choice_mid[i+1]
				prob_choice_opt_mvhv = np.append(prob_choice_opt_mvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]

				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))


			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total

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
	Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total = ThreeTargetTask_Qlearning(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	return log_prob_total

def ThreeTargetTask_Qlearning_sep_parameters(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Separate learning rates, alpha, and inverse temperatures, beta,
	are used for each contingency (low vs high, medium vs high, low vs medium).

	Note: do not update values during instructed trials in this version because that requires additional parameters.

	Inputs:
	- parameters: length 6 array, where first three elements correspond to the learning rates for the three contingencies
					and the last three elements correspond to the inverse temperature values for the three contingencies. 
					Order is low vs high, medium vs high, low vs medium.
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = medium-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[:3]
	beta = parameters[3:]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_low = np.zeros(len(chosen_target))
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_low[0] = Q_initial[0]
	Q_mid[0] = Q_initial[1]
	Q_high[0] = Q_initial[2]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	prob_choice_low[0] = 0.5
	prob_choice_mid[0] = 0.5
	prob_choice_high[0] = 0.5

	prob_choice_opt_lvhv = np.array([])
	prob_choice_opt_mvhv = np.array([])
	prob_choice_opt_lvmv = np.array([])

	log_prob_total = 0.

	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# Update Q values with temporal difference error
		delta_low = float(rewards[i]) - Q_low[i]
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_low[i+1] = Q_low[i] + alpha[0]*delta_low*float(chosen_target[i]==0)*np.array_equal(targets_on[i],[1,1,0]) \
						+ alpha[2]*delta_low*float(chosen_target[i]==0)*np.array_equal(targets_on[i],[1,0,1])
		Q_mid[i+1] = Q_mid[i] + alpha[1]*delta_mid*float(chosen_target[i]==1)*np.array_equal(targets_on[i],[0,1,1]) \
						+ alpha[2]*delta_mid*float(chosen_target[i]==1)*np.array_equal(targets_on[i],[1,0,1])
		Q_high[i+1] = Q_high[i] + alpha[0]*delta_high*float(chosen_target[i]==2)*np.array_equal(targets_on[i],[1,1,0]) \
						+ alpha[1]*delta_high*float(chosen_target[i]==2)*np.array_equal(targets_on[i],[0,1,1])

		# Update probabilities with new Q-values
		if instructed_or_freechoice[i+1] == 2:
			if np.array_equal(targets_on[i+1], [1,1,0]):
				Q_opt = Q_high[i+1]
				Q_nonopt = Q_low[i+1]

				prob_choice_low[i+1] = 1./(1 + np.exp(beta[0]*(Q_high[i+1] - Q_low[i+1])))
				prob_choice_high[i+1] = 1. - prob_choice_low[i+1]
				prob_choice_mid[i+1] = prob_choice_mid[i]

				prob_choice_opt = prob_choice_high[i+1]
				prob_choice_nonopt = prob_choice_low[i+1]
				prob_choice_opt_lvhv = np.append(prob_choice_opt_lvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = 0.5*chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==0))


			elif np.array_equal(targets_on[i+1],[1,0,1]):
				Q_opt = Q_mid[i+1]
				Q_nonopt = Q_low[i+1]

				prob_choice_low[i+1] = 1./(1 + np.exp(beta[2]*(Q_mid[i+1] - Q_low[i+1])))
				prob_choice_high[i+1] = prob_choice_high[i]
				prob_choice_mid[i+1] = 1. - prob_choice_low[i+1]

				prob_choice_opt = prob_choice_mid[i+1]
				prob_choice_nonopt = prob_choice_low[i+1]
				prob_choice_opt_lvmv = np.append(prob_choice_opt_lvmv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_mid[i+1] >= 0.5)&(chosen_target[i+1]==1) or (prob_choice_mid[i+1] < 0.5)&(chosen_target[i+1]==0))


			else:
				Q_opt = Q_high[i+1]
				Q_nonopt = Q_mid[i+1]

				prob_choice_mid[i+1] = 1./(1 + np.exp(beta[1]*(Q_high[i+1] - Q_mid[i+1])))
				prob_choice_low[i+1] = prob_choice_low[i]
				prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

				prob_choice_opt = prob_choice_high[i+1]
				prob_choice_nonopt = prob_choice_mid[i+1]
				prob_choice_opt_mvhv = np.append(prob_choice_opt_mvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))


			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total

def loglikelihood_ThreeTargetTask_Qlearning_sep_parameters(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Separate learning rates, alpha, and inverse temperatures, beta,
	are used for each contingency (low vs high, medium vs high, low vs medium).

	Note: do not update values during instructed trials in this version because that requires additional parameters.

	Inputs:
	- parameters: length 6 array, where first three elements correspond to the learning rates for the three contingencies
					and the last three elements correspond to the inverse temperature values for the three contingencies. 
					Order is low vs high, medium vs high, low vs medium.
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = medium-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''

	Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total = ThreeTargetTask_Qlearning_sep_parameters(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	return log_prob_total

def ThreeTargetTask_Qlearning_ind_parameters(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Separate learning rates, alpha, 
	are used for each stimulus (low value, medium value, high value). A single beta parameter is used for all 
	contingencies presented.

	Inputs:
	- parameters: length 4 array, where first three elements correspond to the learning rates for the three value
					and the last element corresponds to the inverse temperature. 
					Order is low, medium, and high for alpha parameters.
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = medium-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[:3]
	beta = parameters[3]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_low = np.zeros(len(chosen_target))
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_low[0] = Q_initial[0]
	Q_mid[0] = Q_initial[1]
	Q_high[0] = Q_initial[2]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	prob_choice_low[0] = 0.5
	prob_choice_mid[0] = 0.5
	prob_choice_high[0] = 0.5

	prob_choice_opt_lvhv = np.array([])
	prob_choice_opt_mvhv = np.array([])
	prob_choice_opt_lvmv = np.array([])

	log_prob_total = 0.
	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# Update Q values with temporal difference error
		delta_low = float(rewards[i]) - Q_low[i]
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_low[i+1] = Q_low[i] + alpha[0]*delta_low*float(chosen_target[i]==0)
		Q_mid[i+1] = Q_mid[i] + alpha[1]*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha[2]*delta_high*float(chosen_target[i]==2)

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
				prob_choice_opt_lvhv = np.append(prob_choice_opt_lvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = 0.5*chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==0))


			elif np.array_equal(targets_on[i+1],[1,0,1]):
				Q_opt = Q_mid[i+1]
				Q_nonopt = Q_low[i+1]

				prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
				prob_choice_high[i+1] = prob_choice_high[i]
				prob_choice_mid[i+1] = 1. - prob_choice_low[i+1]

				prob_choice_opt = prob_choice_mid[i+1]
				prob_choice_nonopt = prob_choice_low[i+1]
				prob_choice_opt_lvmv = np.append(prob_choice_opt_lvmv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_mid[i+1] >= 0.5)&(chosen_target[i+1]==1) or (prob_choice_mid[i+1] < 0.5)&(chosen_target[i+1]==0))


			else:
				Q_opt = Q_high[i+1]
				Q_nonopt = Q_mid[i+1]

				prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
				prob_choice_low[i+1] = prob_choice_low[i]
				prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

				prob_choice_opt = prob_choice_high[i+1]
				prob_choice_nonopt = prob_choice_mid[i+1]
				prob_choice_opt_mvhv = np.append(prob_choice_opt_mvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))


			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total

def loglikelihood_ThreeTargetTask_Qlearning_ind_parameters(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Separate learning rates, alpha, 
	are used for each stimulus (low value, medium value, high value). A single beta parameter is used for all 
	contingencies presented.

	Inputs:
	- parameters: length 4 array, where first three elements correspond to the learning rates for the three value
					and the last element corresponds to the inverse temperature. 
					Order is low, medium, and high for alpha parameters.
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = medium-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''

	Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total = ThreeTargetTask_Qlearning_ind_parameters(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	return log_prob_total



def ThreeTargetTask_Qlearning_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and HV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
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
	
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))
	
	# Set values for first trial (indexed as trial 0)
	
	Q_mid[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):
		
		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)
		
		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [0,1,1]):
			
			Q_opt = Q_high[i+1]
			Q_nonopt = Q_mid[i+1]
			
			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]
		
			prob_choice_opt = prob_choice_high[i+1]
			prob_choice_nonopt = prob_choice_mid[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood
			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]
	
	return Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter


def loglikelihood_ThreeTargetTask_Qlearning_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter = ThreeTargetTask_Qlearning_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total

def ThreeTargetTask_Qlearning_MVLV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and LV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
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
	Q_mid = np.zeros(len(chosen_target))
	Q_low = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_mid[0] = Q_initial[0]
	Q_low[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_low = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):

		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_low = float(rewards[i]) - Q_low[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_low[i+1] = Q_low[i] + alpha*delta_low*float(chosen_target[i]==0)

		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [1,0,1]):
			
			Q_opt = Q_mid[i+1]
			Q_nonopt = Q_low[i+1]

			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
			prob_choice_low[i+1] = 1. - prob_choice_mid[i+1]

			prob_choice_opt = prob_choice_mid[i+1]
			prob_choice_nonopt = prob_choice_low[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood

			log_prob_total += np.log(prob_choice_nonopt*(choice==0) + prob_choice_opt*(choice==1))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_low[i+1] = prob_choice_low[i]

	return Q_mid, Q_low, prob_choice_mid, prob_choice_low, log_prob_total, counter

def loglikelihood_ThreeTargetTask_Qlearning_MVLV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter = ThreeTargetTask_Qlearning_MVLV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total

def ThreeTargetTask_Qlearning_QAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and HV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_mid[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):

		stim_trial = np.array_equal(targets_on[i], [0,0,1])

		Q_mid[i] = Q_mid[i] + gamma*(stim_trial==1)
		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)

		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [0,1,1]):
			
			Q_opt = Q_high[i+1]
			Q_nonopt = Q_mid[i+1]

			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

			prob_choice_opt = prob_choice_high[i+1]
			prob_choice_nonopt = prob_choice_mid[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood

			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter

def loglikelihood_ThreeTargetTask_Qlearning_QAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter = ThreeTargetTask_Qlearning_QAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total

def ThreeTargetTask_Qlearning_QMultiplicative_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and HV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_mid[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):

		stim_trial = np.array_equal(targets_on[i], [0,0,1])

		Q_mid[i] = Q_mid[i] + gamma*Q_mid[i]*(stim_trial==1)
		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)

		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [0,1,1]):
			
			Q_opt = Q_high[i+1]
			Q_nonopt = Q_mid[i+1]

			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

			prob_choice_opt = prob_choice_high[i+1]
			prob_choice_nonopt = prob_choice_mid[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood

			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter

def loglikelihood_ThreeTargetTask_Qlearning_QMultiplicative_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter = ThreeTargetTask_Qlearning_QAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total

def ThreeTargetTask_Qlearning_PAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and HV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_mid[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):

		stim_trial = np.array_equal(targets_on[i], [0,0,1])

		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)

		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [0,1,1]):
			
			Q_opt = Q_high[i+1]
			Q_nonopt = Q_mid[i+1]

			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

			prob_choice_opt = prob_choice_high[i+1]
			prob_choice_nonopt = prob_choice_mid[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood

			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i] + gamma*(stim_trial==1)
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

	return Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter

def loglikelihood_ThreeTargetTask_Qlearning_PAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter = ThreeTargetTask_Qlearning_QAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total

def ThreeTargetTask_Qlearning_PMultiplicative_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and HV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_mid[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):

		stim_trial = np.array_equal(targets_on[i], [0,0,1])

		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)

		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [0,1,1]):
			
			Q_opt = Q_high[i+1]
			Q_nonopt = Q_mid[i+1]

			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

			prob_choice_opt = prob_choice_high[i+1]
			prob_choice_nonopt = prob_choice_mid[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood

			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i]*((stim_trial==0) + gamma*(stim_trial==1))
			prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

	return Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter

def loglikelihood_ThreeTargetTask_Qlearning_PMultiplicative_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_high, prob_choice_mid, prob_choice_high, log_prob_total, counter = ThreeTargetTask_Qlearning_QAdditive_MVHV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total


def ThreeTargetTask_Qlearning_QMultiplicative_MVLV(parameters, Q_initial, chosen_target, rewards, targets_on):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. Only freechoice trials with the MV and HV targets are 
	considered, although the Q values advance for all trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 2 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_mid = np.zeros(len(chosen_target))
	Q_low = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_mid[0] = Q_initial[0]
	Q_low[0] = Q_initial[1]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_low = np.zeros(len(chosen_target))

	log_prob_total = 0.
	counter = 0

	for i in range(len(chosen_target)-1):

		stim_trial = np.array_equal(targets_on[i], [0,0,1])

		Q_mid[i] = Q_mid[i] + gamma*Q_mid[i]*(stim_trial==1)
		# Update Q values with temporal difference error
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_low = float(rewards[i]) - Q_low[i]
		Q_mid[i+1] = Q_mid[i] + alpha*delta_mid*float(chosen_target[i]==1)
		Q_low[i+1] = Q_low[i] + alpha*delta_low*float(chosen_target[i]==0)

		# Update probabilities with new Q-values
		if np.array_equal(targets_on[i+1], [1,0,1]):
			
			Q_opt = Q_mid[i+1]
			Q_nonopt = Q_low[i+1]

			prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
			prob_choice_low[i+1] = 1. - prob_choice_mid[i+1]

			prob_choice_opt = prob_choice_mid[i+1]
			prob_choice_nonopt = prob_choice_low[i+1]

			# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
			choice = chosen_target[i+1]

			counter += 1 		# count number of trials used to compute the log likelihood

			log_prob_total += np.log(prob_choice_nonopt*(choice==0) + prob_choice_opt*(choice==1))

		else:
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_low[i+1] = prob_choice_low[i]

	return Q_mid, Q_low, prob_choice_mid, prob_choice_low, log_prob_total, counter

def loglikelihood_ThreeTargetTask_Qlearning_QMultiplicative_MVLV(parameters, Q_initial, chosen_target, rewards, targets_on):
	
	Q_mid, Q_low, prob_choice_mid, prob_choice_low, log_prob_total, counter = ThreeTargetTask_Qlearning_QMultiplicative_MVLV(parameters, Q_initial, chosen_target, rewards, targets_on)

	return log_prob_total

def ThreeTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
	'''
	11/20/19 - This method was updated to use the hold center as the behavioral time
	point to align neural activity to (rather than picture onset, before hold begins) (2152-2153)

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
	window_fr_smooth = dict()
	for i, hdf_file in enumerate(hdf_files):
		cb_block = ChoiceBehavior_ThreeTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_picture_onset = cb_block.ind_check_reward_states - 5
		
		# Load spike data:
		if (spike_files[i] == ['']):
			print('no data')
		elif ((channel < 97) and spike_files[i][0] != '') or ((channel > 96) and (spike_files[i][1] != '')):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data

			#lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_hold_center, syncHDF_files[i])

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
			all_fr = np.array([])
			all_fr_smooth = np.array([])
			for j, sc in enumerate(sc_chan):
				sc_fr = spike.compute_window_fr(channel,sc,times_row_ind,t_before,t_after)
				sc_fr_smooth = spike.compute_window_fr_smooth(channel,sc,times_row_ind,t_before,t_after)
				if j == 0:
					all_fr = sc_fr
					all_fr_smooth = sc_fr_smooth
				else:
					all_fr = np.vstack([all_fr, sc_fr])
					all_fr_smooth = np.vstack([all_fr_smooth, sc_fr_smooth])

			# Save matrix of firing rates for units on channel from trials during hdf_file as dictionary element
			window_fr[i] = all_fr
			window_fr_smooth[i] = all_fr_smooth

	return num_trials, num_units, window_fr, window_fr_smooth

def ThreeTargetTask_FiringRates_RewardOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
	'''
	This method returns the average firing rate of all units on the indicated channel during check reward onset.

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
	window_fr_smooth = dict()
	for i, hdf_file in enumerate(hdf_files):
		cb_block = ChoiceBehavior_ThreeTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		ind_check_reward = cb_block.ind_check_reward_states
		
		# Load spike data:
		if (spike_files[i] == ['']):
			print('no data')
		elif ((channel < 97) and spike_files[i][0] != '') or ((channel > 96) and (spike_files[i][1] != '')):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data

			#lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_check_reward, syncHDF_files[i])

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
			all_fr = np.array([])
			all_fr_smooth = np.array([])
			for j, sc in enumerate(sc_chan):
				sc_fr = spike.compute_window_fr(channel,sc,times_row_ind,t_before,t_after)
				sc_fr_smooth = spike.compute_window_fr_smooth(channel,sc,times_row_ind,t_before,t_after)
				if j == 0:
					all_fr = sc_fr
					all_fr_smooth = sc_fr_smooth
				else:
					all_fr = np.vstack([all_fr, sc_fr])
					all_fr_smooth = np.vstack([all_fr_smooth, sc_fr_smooth])

			# Save matrix of firing rates for units on channel from trials during hdf_file as dictionary element
			window_fr[i] = all_fr
			window_fr_smooth[i] = all_fr_smooth

	return num_trials, num_units, window_fr, window_fr_smooth

def MultiTargetTask_FiringRates_DifferenceBetweenBlocks(hdf_files, syncHDF_files, spike_files, num_targets, channel):
	'''
	This method returns the baseline firing rate difference between Blocks A' and A for the given indicated channel, 
	as well as the baseline firing rate for Block A in case this data is used for normalization. Firing rates are returned
	for all units on the indicated channel. This method can be used for both the 2-target task and the 3-target task,
	with the number of target indicated as an input.

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
	- num_targets: integer, either 2 or 3, indicating whether this analysis is for the 2-target task or 3-target task
	- channel: integer value indicating what channel will be used to compute activity
	
	Output:
	- avg_fr_diff: array, size (num units)x(1) with elements corresponding
					to the average firing rate difference over the two blocks.
	- avg_fr_blockA: array, size (num units)x(1) with elements corresponding
					to the average firing rate in the first block, in case it's used for normalization
	- avg_fr_blockAprime: array, size (num_units)x(1) with elements corresponding
					to the average firing rate in Block A' block
	'''
	num_trials = np.zeros(len(hdf_files))
	num_units = np.zeros(len(hdf_files))
	window_fr = dict()
	window_fr_blockA = dict()
	window_fr_blockAprime = dict()
	for i, hdf_file in enumerate(hdf_files):
		if num_targets==3:
			cb_block = ChoiceBehavior_ThreeTargets(hdf_file)
		elif num_targets==2:
			cb_block = ChoiceBehavior_TwoTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		total_num_trials = np.sum(num_trials)
		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_picture_onset = cb_block.ind_check_reward_states - 5
		
		# Load spike data:
		if (spike_files[i] == ['']):
			print('no data')
		elif ((channel < 97) and spike_files[i][0] != '') or ((channel > 96) and (spike_files[i][1] != '')):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data and find all sort codes associated with good channels
			if channel < 97:
				print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][0])
			else:
				print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][1])

			# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
			# designated window.

			sc_chan = spike.find_chan_sc(channel)
			num_units = len(sc_chan)
			
			if num_targets==3:
				trial_targ1 = 149
				trial_targ2 = 249
			else:
				trial_targ1 = 99
				trial_targ2 = 199
			
			sc_fr_blockA = np.full(num_units, np.nan)
			sc_fr_blockAprime = np.full(num_units, np.nan)

			if i==0&(num_trials[i]>trial_targ2):
				sc_fr_blockA = spike.get_avg_firing_rates_range([channel], times_row_ind[0], times_row_ind[trial_targ1])
				sc_fr_blockAprime = spike.get_avg_firing_rates_range([channel], times_row_ind[trial_targ2], times_row_ind[-1])
			if i==0&(num_trials[i]<=trial_targ2):
				sc_fr_blockA = spike.get_avg_firing_rates_range([channel], times_row_ind[0], times_row_ind[np.min([trial_targ1,num_trials[i]])])
				#sc_fr_blockAprime = spike.get_avg_firing_rates_range(channel, times_row_ind[249], times_row_ind[-1])
			if i>0&(total_num_trials>trial_targ2)&(total_num_trials - num_trials[i] <trial_targ1):
				sc_fr_blockA = spike.get_avg_firing_rates_range([channel], times_row_ind[0], times_row_ind[np.min([num_trials[i], trial_targ1 - total_num_trials + num_trials[i]])])
				sc_fr_blockAprime = spike.get_avg_firing_rates_range([channel], times_row_ind[trial_targ2 - total_num_trials + num_trials[i]], times_row_ind[-1])
			if i>0&(total_num_trials>trial_targ2)&(total_num_trials - num_trials[i] > trial_targ1):
				sc_fr_blockAprime = spike.get_avg_firing_rates_range([channel], times_row_ind[trial_targ2 - total_num_trials + num_trials[i]], times_row_ind[-1])
			if i>0&(total_num_trials < trial_targ2)&(total_num_trials - num_trials[i] <trial_targ1):
				sc_fr_blockA = spike.get_avg_firing_rates_range([channel], times_row_ind[0], times_row_ind[np.min([num_trials[i], trial_targ1 - total_num_trials + num_trials[i]])])

			# Save dictionary containing lists of firing rates for units on channel from trials during hdf_file
			if i==0:
				window_fr_blockA = np.array(sc_fr_blockA)
				window_fr_blockAprime = np.array(sc_fr_blockAprime)
			else:
				window_fr_blockA = np.vstack([window_fr_blockA, np.array(sc_fr_blockA)])
				window_fr_blockAprime = np.vstack([window_fr_blockAprime, np.array(sc_fr_blockAprime)])

	avg_fr_blockA = np.nanmean(window_fr_blockA,axis = 0)
	avg_fr_blockAprime = np.nanmean(window_fr_blockAprime,axis=0)
	avg_fr_diff = avg_fr_blockAprime - avg_fr_blockA

	return avg_fr_diff, avg_fr_blockA, avg_fr_blockAprime

def MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, channels):
	'''
	SAME AS ABOVE METHOD BUT FOR MUTLIPLE CHANNELS AT THE SAME TIME.
	This method returns the baseline firing rate difference between Blocks A' and A for the given indicated channel, 
	as well as the baseline firing rate for Block A in case this data is used for normalization. Firing rates are returned
	for all units on the indicated channel. This method can be used for both the 2-target task and the 3-target task,
	with the number of target indicated as an input.

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
	- num_targets: integer, either 2 or 3, indicating whether this analysis is for the 2-target task or 3-target task
	- channels: list or array, integer values indicating what channels will be used to compute activity
	
	Output:
	- avg_fr_diff: array, size (num units)x(1) with elements corresponding
					to the average firing rate difference over the two blocks.
	- avg_fr_blockA: array, size (num units)x(1) with elements corresponding
					to the average firing rate in the first block, in case it's used for normalization
	- avg_fr_blockAprime: array, size (num_units)x(1) with elements corresponding
					to the average firing rate in Block A' block
	'''
	num_trials = np.zeros(len(hdf_files))
	num_units = np.zeros(len(hdf_files))
	window_fr_blockA_lowchannels = dict()
	window_fr_blockAprime_lowchannels = dict()
	window_fr_blockA_highchannels = dict()
	window_fr_blockAprime_highchannels = dict()
	avg_fr_blockA = dict()
	avg_fr_blockAprime = dict()
	avg_fr_diff = dict()

	low_channels = [chann for chann in channels if chann < 97]
	high_channels = [chann for chann in channels if chann > 96]

	for i, hdf_file in enumerate(hdf_files):
		if num_targets==3:
			cb_block = ChoiceBehavior_ThreeTargets(hdf_file)
		elif num_targets==2:
			cb_block = ChoiceBehavior_TwoTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		total_num_trials = np.sum(num_trials)
		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_picture_onset = cb_block.ind_check_reward_states - 5

		print("Num trials in this file: %i" % (num_trials[i]))
		print("Num trials all together: %i" % (total_num_trials))
		
		# Load spike data:
		if (spike_files[i] == ['']):
			print('no data')
		else:
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)
			# Define how trials should be split across blocks
			if num_targets==3:
				trial_targ1 = 149
				trial_targ2 = 249
			else:
				trial_targ1 = 99
				trial_targ2 = 199

			print("Trials needed in Block A: %i" % (trial_targ1))
			print("Trials needed in Block A prime: %i" % (trial_targ2))

			# Load spike data for first 96 channels and find average firing rates
			if spike_files[i][0] != '':
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				sc, total_units_low = spike1.find_unit_sc(low_channels)

				# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
				# designated window.
				sc_fr_blockA = np.full(total_units_low, np.nan)
				sc_fr_blockAprime = np.full(total_units_low, np.nan)

				# Compute firing rates for each block. Cases divide up computation over the files the data is spread across.
				# Output of get_avg_firing_rates_range method is a dictionary with keys equal to the channel numbers and 
				# entries as an array of firing rate for each unit on the channel.
				if i==0 and (num_trials[i]>trial_targ2):
					print("Case 1 for file %i" % (i))
					sc_fr_blockA = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[0], times_row_ind[trial_targ1])
					sc_fr_blockAprime = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[trial_targ2], times_row_ind[-1])
				if i==0 and (num_trials[i]<=trial_targ2):
					print("Case 2 for file %i" % (i))
					sc_fr_blockA = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[0], times_row_ind[np.min([trial_targ1,num_trials[i]-1])])
					#sc_fr_blockAprime = spike.get_avg_firing_rates_range(channel, times_row_ind[249], times_row_ind[-1])
				if i>0 and (total_num_trials>trial_targ2) and ((total_num_trials - num_trials[i]) < trial_targ1):
					print(total_num_trials - num_trials[i], trial_targ1)
					print("Case 3 for file %i" % (i))
					sc_fr_blockA = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[0], times_row_ind[np.min([num_trials[i], trial_targ1 - total_num_trials + num_trials[i]])])
					sc_fr_blockAprime = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[trial_targ2 - total_num_trials + num_trials[i]], times_row_ind[-1])
				if i>0 and (total_num_trials>trial_targ2) and ((total_num_trials - num_trials[i]) > trial_targ1):
					print("Case 4 for file %i" % (i))
					print(np.max([0, trial_targ2 - total_num_trials + num_trials[i]]))
					sc_fr_blockAprime = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[np.max([0, trial_targ2 - total_num_trials + num_trials[i]])], times_row_ind[-1])
				if i>0 and (total_num_trials < trial_targ2) and ((total_num_trials - num_trials[i]) <trial_targ1):
					print("Case 5 for file %i" % (i))
					sc_fr_blockA = spike1.get_avg_firing_rates_range(low_channels, times_row_ind[0], times_row_ind[np.min([num_trials[i], trial_targ1 - total_num_trials + num_trials[i]])])

				# Combine lists of spike rates across files. Change everything to long arrays.
				if i==0:
					print(sc_fr_blockA)
					window_fr_blockA_lowchannels = sc_fr_blockA
					window_fr_blockAprime_lowchannels = sc_fr_blockAprime
				else:
					window_fr_blockA_lowchannels = np.concatenate([window_fr_blockA_lowchannels, sc_fr_blockA])
					window_fr_blockAprime_lowchannels = np.concatenate([window_fr_blockAprime_lowchannels, sc_fr_blockAprime])
			else:
				window_fr_blockA_lowchannels = np.array([])
				window_fr_blockAprime_lowchannels = np.array([])

			# Load spike data for last 60 channels and find average firing rates
			if spike_files[i][1] != '':
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				sc, total_units_high = spike2.find_unit_sc(high_channels)
				# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
				# designated window.
				sc_fr_blockA = np.full(total_units_high, np.nan)
				sc_fr_blockAprime = np.full(total_units_high, np.nan)

				# Compute firing rates for each block. Cases divide up computation over the files the data is spread across.
				# Output of get_avg_firing_rates_range method is a list containing firing rates for all units on all 
				# indicated channels. Entries in list are arrays.
				if i==0 and (num_trials[i]>trial_targ2):
					sc_fr_blockA = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[0], times_row_ind[trial_targ1])
					sc_fr_blockAprime = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[trial_targ2], times_row_ind[-1])
				if i==0 and (num_trials[i]<=trial_targ2):
					sc_fr_blockA = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[0], times_row_ind[np.min([trial_targ1,num_trials[i]-1])])
					#sc_fr_blockAprime = spike.get_avg_firing_rates_range(channel, times_row_ind[249], times_row_ind[-1])
				if i>0 and (total_num_trials>trial_targ2) and ((total_num_trials - num_trials[i]) <trial_targ1):
					sc_fr_blockA = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[0], times_row_ind[np.min([num_trials[i], trial_targ1 - total_num_trials + num_trials[i]])])
					sc_fr_blockAprime = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[trial_targ2 - total_num_trials + num_trials[i]], times_row_ind[-1])
				if i>0 and (total_num_trials>trial_targ2) and ((total_num_trials - num_trials[i]) > trial_targ1):
					sc_fr_blockAprime = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[np.max([0, trial_targ2 - total_num_trials + num_trials[i]])], times_row_ind[-1])
				if i>0 and (total_num_trials < trial_targ2) and ((total_num_trials - num_trials[i]) <trial_targ1):
					sc_fr_blockA = spike2.get_avg_firing_rates_range(high_channels, times_row_ind[0], times_row_ind[np.min([num_trials[i], trial_targ1 - total_num_trials + num_trials[i]])])

				# Combine lists of spike rates across files. Change everything to long arrays.
				if i==0:
					print(sc_fr_blockA)
					window_fr_blockA_highchannels = sc_fr_blockA
					window_fr_blockAprime_highchannels = sc_fr_blockAprime
				else:
					window_fr_blockA_highchannels = np.concatenate([window_fr_blockA_highchannels, sc_fr_blockA])
					window_fr_blockAprime_highchannels = np.concatenate([window_fr_blockAprime_highchannels, sc_fr_blockAprime])
			else:
				window_fr_blockA_highchannels = np.array([])
				window_fr_blockAprime_highchannels = np.array([])
	
	avg_fr_blockA = np.concatenate([window_fr_blockA_lowchannels, window_fr_blockA_highchannels])
	avg_fr_blockAprime = np.concatenate([window_fr_blockAprime_lowchannels, window_fr_blockAprime_highchannels])
	avg_fr_diff = avg_fr_blockAprime - avg_fr_blockA

	return avg_fr_diff, avg_fr_blockA, avg_fr_blockAprime

def TwoTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
	'''
	11/20/2019 - this method updated to use the hold_center behavioral event rather than picture onset
	which occurs before hold begins (2499-2500)

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
	window_fr_smooth = dict()
	for i, hdf_file in enumerate(hdf_files):
		cb_block = ChoiceBehavior_TwoTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_picture_onset = cb_block.ind_check_reward_states - 5
		
		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data

			#lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_hold_center, syncHDF_files[i])

			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data and find all sort codes associated with good channels
			if channel < 97:
				#print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][0])
			else:
				#print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][1])

			avg_firing_rates = spike.get_avg_firing_rates(np.array([channel]))[channel]		# array of average firing rates, length is equal to the number of units on the channel
			unit_class = np.array(avg_firing_rates > 4, dtype = int)						# array that will be used to identify the unit class: 0 = slow, 1 = fast

			# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
			# designated window.
			sc_chan = spike.find_chan_sc(channel)
			num_units[i] = len(sc_chan)
			
			if (num_units[i] != 0):
				for j, sc in enumerate(sc_chan):
					
					sc_fr = spike.compute_window_fr(channel,sc,times_row_ind,t_before,t_after)
					sc_fr_smooth = spike.compute_window_fr_smooth(channel,sc,times_row_ind,t_before,t_after)
					if j == 0:
						all_fr = sc_fr
						all_fr_smooth = sc_fr_smooth
					else:
						all_fr = np.vstack([all_fr, sc_fr])
						all_fr_smooth = np.vstack([all_fr_smooth, sc_fr_smooth])

				# Save matrix of firing rates for units on channel from trials during hdf_file as dictionary element
				window_fr[i] = all_fr
				window_fr_smooth[i] = all_fr_smooth
	

	return num_trials, num_units, window_fr, window_fr_smooth, unit_class

def TwoTargetTask_FiringRates_RewardOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
	'''
	11/20/2019 - this method updated to use the hold_center behavioral event rather than picture onset
	which occurs before hold begins (2499-2500)

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
	window_fr_smooth = dict()
	for i, hdf_file in enumerate(hdf_files):
		cb_block = ChoiceBehavior_TwoTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		ind_check_reward = cb_block.ind_check_reward_states

		
		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data

			#lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_check_reward, syncHDF_files[i])

			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data and find all sort codes associated with good channels
			if channel < 97:
				#print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][0])
			else:
				#print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][1])

			avg_firing_rates = spike.get_avg_firing_rates(np.array([channel]))[channel]		# array of average firing rates, length is equal to the number of units on the channel
			unit_class = np.array(avg_firing_rates > 4, dtype = int)						# array that will be used to identify the unit class: 0 = slow, 1 = fast

			# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
			# designated window.
			sc_chan = spike.find_chan_sc(channel)
			num_units[i] = len(sc_chan)
			
			if (num_units[i] != 0):
				for j, sc in enumerate(sc_chan):
					
					sc_fr = spike.compute_window_fr(channel,sc,times_row_ind,t_before,t_after)
					sc_fr_smooth = spike.compute_window_fr_smooth(channel,sc,times_row_ind,t_before,t_after)
					if j == 0:
						all_fr = sc_fr
						all_fr_smooth = sc_fr_smooth
					else:
						all_fr = np.vstack([all_fr, sc_fr])
						all_fr_smooth = np.vstack([all_fr_smooth, sc_fr_smooth])

				# Save matrix of firing rates for units on channel from trials during hdf_file as dictionary element
				window_fr[i] = all_fr
				window_fr_smooth[i] = all_fr_smooth
	

	return num_trials, num_units, window_fr, window_fr_smooth, unit_class

def TwoTargetTask_LFPPower_PictureOnset_Multichannels(tdt_data_dir, hdf_files, syncHDF_files, channels, t_before, t_after):
	'''
	This method computes to z-scored power across all trials and channels for four biologically relevant
	frequency bands: alpha, beta, gamma, and high-gamma. 

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
	- channels: integer values indicating what channels will be used to regress activity, enumerate starts at 1 (i.e. true channel numbers) and will need to be adjusted by -1 for indexing purposes
	- t_before: time before (s) the picture onset that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of onset forward when considering the window of activity.
	- t_after: time after (s) the picture onset that should be included when computing the firing rate.

	Output:
	- all_powers: 3D-array; array with z-scored power features, arranged as (trials)x(channels)x(power features)
	'''
	num_trials = np.zeros(len(hdf_files))
	power_features = dict()
	for i, hdf_file in enumerate(hdf_files):
		cb_block = ChoiceBehavior_TwoTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_picture_onset = cb_block.ind_check_reward_states - 5
		
		# 1. Load the TDT LFP data
		tdt_data_location = tdt_data_dir + syncHDF_files[i][76:-15] + '/Block-' + syncHDF_files[i][-13]
		# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data

		lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_picture_onset, syncHDF_files[i])
		#lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(ind_hold_center, syncHDF_files[i])

		# load TDT data
		bl = tdt.read_block(tdt_data_location)
		lfp1 = bl.streams.LFP1.data
		lfp2 = bl.streams.LFP2.data
		lfp_dur = np.min((lfp1.shape[1], lfp2.shape[1]))
		lfp = np.concatenate((lfp1[:,:lfp_dur], lfp2[:,:lfp_dur]))
		lfp = lfp[channels-1,:]
		Fs = bl.streams.LFP1.fs

		C = lfp.shape[0]

		power_bands = [[4,8],[8,13],[13,30],[30,60],[70,200]]
		K = len(power_bands)

		# make lfp dictionary for computePowerFeatures method
		lfp_data = dict()
		for j in range(C):
			lfp_data[j] = lfp[j,:]

		# compute power features
		lfp_state_row_ind = lfp_state_row_ind.reshape(len(lfp_state_row_ind),1)
		features = computePowerFeatures(lfp_data, Fs, power_bands, lfp_state_row_ind, [t_after])  	# features: dictionary with N entries (one per trial), with a C x K matric which C is the number of channels and K is the number of features (number of power bands times M)
		
		# arrange into dictionary that builds up across trials
		for k in range(int(num_trials[i])):
			if i==0:
				power_features[k] = features[str(k)]
			else:
				power_features[np.sum(num_trials[:i-1]) + k] = features[k]

	# After extracting all power information, arrange into matrix and z-score per channel and per frequency band
	# Matrix should be trials x channels x bands
	N = int(np.sum(num_trials))		# total number of trials
	print(N, C, K)
	all_powers = np.zeros((N, C, K))
	for k in range(N):
		all_powers[k,:,:] = power_features[k]

	all_powers = stats.zscore(all_powers, axis = 0)


	return all_powers, num_trials

def TwoTargetTask_RegressLFPPower_PictureOnset_Multichannels(tdt_data_dir, hdf_files, syncHDF_files, channels, trial_start, trial_end):
	'''
	This method regresses z-scored power across all trials for four biologically relevant
	frequency bands: alpha, beta, gamma, and high-gamma, against the target values

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
	- channels: integer values indicating what channels will be used to regress activity

	Output:
	'''

	# 0. Get session information for saving data
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Luigi' + hdf_files[0][str_ind:str_ind + 8]
	if syncHDF_files[0]!='':
		str_ind = syncHDF_files[0].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	elif syncHDF_files[1]!='':
		str_ind = syncHDF_files[1].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	else:
		session_name = 'Unknown'

	bands = ['theta', 'alpha', 'beta', 'gamma', 'high gamma']
	
	# 1. Load power data for around time point of picture onset
	t_before = 0
	t_after = 1
	all_powers, num_trials = TwoTargetTask_LFPPower_PictureOnset_Multichannels(tdt_data_dir, hdf_files, syncHDF_files, channels, t_before, t_after)	# matrix is trials x channels x power features
	cum_sum_trials = np.cumsum(num_trials)
	
	# 2. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	trial_end = trial_end*(trial_end < total_trials) + total_trials*(trial_end >= total_trials)
	#ind_trial_case = np.array([ind for ind in range(total_trials) if np.array_equal(targets_on[ind],trial_case)])
	

	# 2a. Get reaction time information
	rt = np.array([])
	for file in hdf_files:
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(file)
		rt = np.append(rt, reaction_time)

	# 2b. Get movementment time information
	mt = (cb.state_time[cb.ind_target_states + 1] - cb.state_time[cb.ind_target_states])/60.


	# 3. Get Q-values, chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	Q_high, Q_low = Value_from_reward_history_TwoTargetTask(hdf_files)

	# 4. Do regression separately for each channel and frequency band.
	#    Current regression uses Q-values and constant, as well as: 
	#    reaction time (rt), movement time (mt), choice (chosen_target), and reward (rewards)
	for k in range(all_powers.shape[1]):
		for m in range(all_powers.shape[2]):
			unit_data = all_powers[:,k,m]

			trial_inds = np.array([index for index in range(trial_start,trial_end)], dtype = int)
			x = np.vstack((Q_low[trial_inds], Q_high[trial_inds]))
			x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
			x = np.transpose(x)
			x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
			#x = sm.add_constant(x, prepend=False)

			y = unit_data[trial_inds]

			print("Regression for channel %i and band %i" % (channels[k],m+1))
			model_glm = sm.OLS(y,x)
			fit_glm = model_glm.fit()

			data_filename = session_name + ' - Channel %i - Band %s' %(channels[k], bands[m])
			data = dict()
			data['regression_labels'] = ['Q_low', 'Q_high','RT', 'MT', 'Choice', 'Reward']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['Q_low'] = Q_low[trial_inds]
			data['Power'] = y
			sp.io.savemat( dir_luigi + 'picture_onset_lfp/' + data_filename + '.mat', data)

	return

def ThreeTargetTask_MaxFiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
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
					to the peak firing rate in the window indicated.
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
				print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][0])
			else:
				print(channel)
				spike = OfflineSorted_CSVFile(spike_files[i][1])

			# Get matrix that is (Num units on channel)x(num trials in hdf_file) containing the firing rates during the
			# designated window.
			sc_chan = spike.find_chan_sc(channel)
			num_units[i] = len(sc_chan)
			for j, sc in enumerate(sc_chan):
				sc_fr, smooth_sc_fr, psth = spike.compute_window_peak_fr(channel,sc,times_row_ind, t_after)
				if j == 0:
					#all_fr = sc_fr
					all_fr = smooth_sc_fr
				else:
					#all_fr = np.vstack([all_fr, sc_fr])
					all_fr = np.vstack([all_fr, smooth_sc_fr])
					
			# Save matrix of firing rates for units on channel from trials during hdf_file as dictionary element
			window_fr[i] = all_fr
			

	return num_trials, num_units, window_fr


def ThreeTargetTask_RegressFiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, trial_case, var_value, channel, t_before, t_after, smoothed, trial_start, trial_end):
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
	- var_value: boolean indicating whether the Q value(s) used in the regression should be fixed (var_value == False) 
				and defined based on their reward probabilties, or whether the Q value(s) should be varying trial-by-trial
				(var_value == True) based on the Q-learning model fit
	- channel: integer value indicating what channel will be used to regress activity
	- t_before: time before (s) the picture onset that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of onset forward when considering the window of activity.
	- t_after: time after (s) the picture onset that should be included when computing the firing rate.
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)
	- trial_start: integer indicating at which trial to start for the analysis
	- trial_end: integer indicating at which trial to end for the analysis

	'''
	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	trial_end = trial_end*(trial_end < total_trials) + total_trials*(trial_end >= total_trials)
	
	targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]
	ind_trial_case = np.array([ind for ind in range(total_trials) if np.array_equal(targets_on[ind],trial_case)])
	

	# 1a. Get reaction time information
	rt = np.array([])
	for file in hdf_files:
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(file)
		rt = np.append(rt, reaction_time)

	# 1b. Get movementment time information
	mt = (cb.state_time[cb.ind_target_states + 1] - cb.state_time[cb.ind_target_states])/60.


	print("Total trials: ", total_trials)
	print("Length reaction time vec: ", len(rt))

	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr, window_fr_smooth = ThreeTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials)

	# 3. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	if var_value:
		# Varying Q-values
		# Find ML fit of alpha and beta
		Q_initial = 0.5*np.ones(3)
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
		result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,None)])
		alpha_ml, beta_ml = result["x"]
		print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
		# RL model fit for Q values
		Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, log_likelihood = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)
	else:
		# Fixed Q-values
		Q_low = 0.35*np.ones(total_trials)
		Q_mid = 0.6*np.ones(total_trials)
		Q_high = 0.85*np.ones(total_trials)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	#fr_mat = np.empty([max_num_units,total_trials])
	#fr_mat[:] = np.NAN
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 
		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	# 5. Do regression for each unit only on trials of correct trial type with spike data saved.
	#    Current regression is just using Q-values and constant. Should be updated to consider: 
	#    reaction time (rt), movement time (mt), choice (chosen_target), and reward (rewards)
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A and B
		#trial_inds = np.array([index for index in range(50,250) if unit_data[index]!=0], dtype = int)
		trial_inds = np.array([index for index in range(trial_start,trial_end) if unit_data[index]!=0], dtype = int)
		x = np.vstack((Q_low[trial_inds], Q_mid[trial_inds], Q_high[trial_inds]))
		x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		# include which targets were shown
		x = np.vstack((x, targets_on[:,0][trial_inds], targets_on[:,2][trial_inds], targets_on[:,1][trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		print(x.shape)
		y = unit_data[trial_inds]
		print(y.shape)
		#y = y/np.max(y)  # normalize y

		print("Regression for unit ", k)
		model_glm = sm.OLS(y,x)
		fit_glm = model_glm.fit()
		print(fit_glm.summary())

	return window_fr, window_fr_smooth, fr_mat, x, y, Q_low, Q_mid, Q_high

def TwoTargetTask_RegressFiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, var_value, channel, t_before, t_after, smoothed, trial_start,trial_end):
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
	- var_value: boolean indicating whether the Q value(s) used in the regression should be fixed (var_value == False) 
				and defined based on their reward probabilties, or whether the Q value(s) should be varying trial-by-trial
				(var_value == True) based on the Q-learning model fit
	- channel: integer value indicating what channel will be used to regress activity
	- t_before: time before (s) the picture onset that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of onset forward when considering the window of activity.
	- t_after: time after (s) the picture onset that should be included when computing the firing rate.
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)
	- trial_start: integer indicating at which trial to start for the analysis
	- trial_end: integer indicating at which trial to end for the analysis

	'''
	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	trial_end = trial_end*(trial_end < total_trials) + total_trials*(trial_end >= total_trials)
	#ind_trial_case = np.array([ind for ind in range(total_trials) if np.array_equal(targets_on[ind],trial_case)])
	

	# 1a. Get reaction time information
	rt = np.array([])
	for file in hdf_files:
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(file)
		rt = np.append(rt, reaction_time)

	# 1b. Get movementment time information
	mt = (cb.state_time[cb.ind_target_states + 1] - cb.state_time[cb.ind_target_states])/60.


	print("Total trials: ", total_trials)
	print("Length reaction time vec: ", len(rt))

	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr, window_fr_smooth, unit_class = TwoTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials)

	# 3. Get Q-values, chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	if var_value:
		# Varying Q-values
		# Find ML fit of alpha and beta
		Q_initial = 0.5*np.ones(3)
		nll = lambda *args: -logLikelihoodRLPerformance(*args)
		result = op.minimize(nll, [0.2, 1], args=(Q_initial, rewards, chosen_target, instructed_or_freechoice), bounds=[(0,1),(0,None)])
		alpha_ml, beta_ml = result["x"]
		print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
		# RL model fit for Q values
		Q_low, Q_high, prob_choice_low, log_likelihood = RLPerformance([alpha_ml, beta_ml], Q_initial, rewards,  chosen_target, instructed_or_freechoice)
	else:
		# Fixed Q-values
		Q_low = 0.4*np.ones(total_trials)
		Q_high = 0.8*np.ones(total_trials)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	#fr_mat = np.empty([max_num_units,total_trials])
	#fr_mat[:] = np.NAN
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 
		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	# 5. Do regression for each unit with spike data saved.
	#    Current regression uses Q-values and constant, as well as: 
	#    reaction time (rt), movement time (mt), choice (chosen_target), and reward (rewards)
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A and B
		trial_inds = np.array([index for index in range(trial_start,trial_end) if unit_data[index]!=0], dtype = int)
		x = np.vstack((Q_low[trial_inds], Q_high[trial_inds]))
		x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		print(x.shape)
		y = unit_data[trial_inds]
		print(y.shape)
		#y = y/np.max(y)  # normalize y

		print("Regression for unit ", k)
		model_glm = sm.OLS(y,x)
		fit_glm = model_glm.fit()
		print(fit_glm.summary())

	return window_fr, window_fr_smooth, fr_mat, x, y, Q_low, Q_high

def ThreeTargetTask_RegressedFiringRatesWithValue_PictureOnset(dir1, hdf_files, syncHDF_files, spike_files, channel, t_before, t_after, smoothed):
	'''
	8/15/19 - Updated to use vales determined from smoothed reward history (2915-2925) and to use the hold center as the behavioral time
	point to align neural activity to (rather than picture onset, before hold begins) (2908), and to save to the right folder (3174)

	This method regresses the firing rate of all units as a function of value. It then plots the firing rate as a function of 
	the modeled value and uses the regression coefficient to plot a linear fit of the relationship between value and 
	firing rate.

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
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Mario' + hdf_files[0][str_ind:str_ind + 8]
	if syncHDF_files[0]!='':
		str_ind = syncHDF_files[0].index('201')
		session_name = 'Mario' + syncHDF_files[0][str_ind:str_ind + 11]
	elif syncHDF_files[1]!='':
		str_ind = syncHDF_files[1].index('201')
		session_name = 'Mario' + syncHDF_files[0][str_ind:str_ind + 11]
	else:
		session_name = 'Unknown'
	

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 1a. Get reaction time information
	rt = np.array([])
	for file in hdf_files:
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(file)
		rt = np.append(rt, reaction_time)

	# 1b. Get movementment time information
	mt = (cb.state_time[cb.ind_target_states + 1] - cb.state_time[cb.ind_target_states])/60.

	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr, window_fr_smooth = ThreeTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials).astype(int)
	
	# 3. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
	# Varying Q-values
	"""
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, log_likelihood = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)
	"""
	Q_high, Q_mid, Q_low = Value_from_reward_history_ThreeTargetTask(hdf_files)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 
		
		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	# 5. Do regression for each unit only on trials in Blocks A and B with spike data saved.
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A 
		trial_inds = np.array([index for index in range(50,150) if unit_data[index]!=0], dtype = int)
		#trial_inds = np.array([index for index in range(50,150)], dtype = int)
		x = np.vstack((Q_low[trial_inds], Q_mid[trial_inds], Q_high[trial_inds]))
		x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		# include which targets were shown
		x = np.vstack((x, targets_on[:,0][trial_inds], targets_on[:,2][trial_inds], targets_on[:,1][trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)

		#y = y/np.max(y)  # normalize y

		try:
			#print("Regression for unit ", k)
			model_glm = sm.OLS(y_zscore,x)
			fit_glm = model_glm.fit()
			#print(fit_glm.summary())

			regress_coef = fit_glm.params[1] 		# The regression coefficient for Qmid is the second parameter
			regress_intercept = y[0] - regress_coef*Q_mid[trial_inds[0]]

			# Get linear regression fit for just Q_mid
			Q_mid_min = np.amin(Q_mid[trial_inds])
			Q_mid_max = np.amax(Q_mid[trial_inds])
			x_lin = np.linspace(Q_mid_min, Q_mid_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_mid[trial_inds],y, c= 'k', marker = 'o', label ='Learning Trials')
			plt.plot(x_lin, m*x_lin + b, c = 'k')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'y')
			plt.xlabel('Q_mid')
			plt.ylabel('Firing Rate (spk/s)')
			plt.title(sess_name + ' - Channel %i - Unit %i' %(channel, k))
			'''

			# save Q and firing rate data
			Q_learning = Q_mid[trial_inds]
			FR_learning = y
			Q_mid_BlockA = Q_mid[trial_inds]

			max_fr = np.amax(y)
			xlim_min = np.amin(Q_mid[trial_inds])
			xlim_max = np.amax(Q_mid[trial_inds])

			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data = dict()
			data['regression_labels'] = ['Q_low', 'Q_mid', 'Q_high','RT', 'MT', 'Choice', 'Reward', 'Q_low_on', 'Q_mid_on', 'Q_high_on']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['Q_mid_early'] = Q_mid_BlockA
			data['FR_early'] = FR_learning
			sp.io.savemat( dir1 + 'hold_center_fr/' + data_filename + '.mat', data)


			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			num_bins = 5
			sorted_Qvals_inds = np.argsort(Q_mid[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_mid[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			reorg_FR_BlockA = reorg_FR
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			# Save data for binning by bins of fixed size (rather than equally populated)
			Q_range_min = np.min(Q_mid[trial_inds])
			Q_range_max = np.max(Q_mid[trial_inds])
			FR_BlockA = y
			
			'''
			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.legend()
			'''
		except:
			pass
	
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A and B
		trial_inds = np.array([index for index in range(250,len(unit_data)) if unit_data[index]!=0], dtype = int)
		#trial_inds = np.array([index for index in range(250,len(unit_data))], dtype = int)
		x = np.vstack((Q_low[trial_inds], Q_mid[trial_inds], Q_high[trial_inds]))
		x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		# include which targets were shown
		x = np.vstack((x, targets_on[:,0][trial_inds], targets_on[:,2][trial_inds], targets_on[:,1][trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)

		#y = y/np.max(y)  # normalize y

		try:
			#print("Regression for unit ", k)
			model_glm_late = sm.OLS(y_zscore,x)
			fit_glm_late = model_glm_late.fit()
			#print(fit_glm_late.summary())

			regress_coef = fit_glm_late.params[1] 		# The regression coefficient for Qmid is the second parameter
			regress_intercept = y[0] - regress_coef*Q_mid[trial_inds[0]]

			# Get linear regression fit for just Q_mid
			Q_mid_min = np.amin(Q_mid[trial_inds])
			Q_mid_max = np.amax(Q_mid[trial_inds])
			x_lin = np.linspace(Q_mid_min, Q_mid_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			max_fr_stim = np.amax(y)
			fr_lim = np.maximum(max_fr, max_fr_stim)

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_mid[trial_inds],y, c= 'c', label = 'Stimulation trials')
			plt.plot(x_lin, m*x_lin + b, c = 'c')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'g')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*xlim_min, 1.1*xlim_max))
			plt.legend()
			'''

			# save Q and firing rate data
			Q_late = Q_mid[trial_inds]
			FR_late = y

			# Get binned firing rates: bins of fixed size
			Q_range_min = np.min(np.min(Q_late), Q_range_min)
			Q_range_max = np.max(np.max(Q_late), Q_range_max)
			bins = np.arange(Q_range_min, Q_range_max + 0.5*(Q_range_max - Q_range_min)/5., (Q_range_max - Q_range_min)/5.)
			hist_BlockA, bins = np.histogram(Q_mid_BlockA, bins)
			hist_late, bins = np.histogram(Q_late, bins)

			sorted_Qvals_inds_BlockA = np.argsort(Q_mid_BlockA)
			sorted_Qvals_inds_late = np.argsort(Q_late)

			begin_BlockA = 0
			begin_late = 0
			dta_all = []
			avg_FR_BlockA = np.zeros(5)
			avg_FR_late = np.zeros(5)
			sem_FR_BlockA = np.zeros(5)
			sem_FR_late = np.zeros(5)
			for j in range(len(hist_BlockA)):
				data_BlockA = FR_BlockA[sorted_Qvals_inds_BlockA[begin_BlockA:begin_BlockA+hist_BlockA[j]]]
				data_late = FR_late[sorted_Qvals_inds_late[begin_late:begin_late+hist_late[j]]]
				begin_BlockA += hist_BlockA[j]
				begin_late += hist_late[j]

				avg_FR_BlockA[j] = np.nanmean(data_BlockA)
				sem_FR_BlockA[j] = np.nanstd(data_BlockA)/np.sqrt(len(data_BlockA))
				avg_FR_late[j] = np.nanmean(data_late)
				sem_FR_late[j] = np.nanstd(data_late)/np.sqrt(len(data_late))

				for item in data_BlockA:
					dta_all += [(j,0,item)]

				for item in data_late:
					dta_all += [(j,1,item)]

			dta_all = pd.DataFrame(dta_all, columns = ['Bin', 'Condition', 'FR'])
			bin_centers = (bins[1:] + bins[:-1])/2.
			
			'''
			formula = 'FR ~ C(Bin) + C(Condition) + C(Bin):C(Condition)'
			model = ols(formula, dta_all).fit()
			aov_table = anova_lm(model, typ=2)

			print "Two-way ANOVA analysis"
			print(aov_table)
			'''

			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			sorted_Qvals_inds = np.argsort(Q_mid[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_mid[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			'''
			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR/2., fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()

			plt.figure(k)
			plt.subplot(1,3,3)
			plt.errorbar(bin_centers, avg_FR_BlockA, yerr = sem_FR_BlockA, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.errorbar(bin_centers, avg_FR_late, yerr = sem_FR_late, fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()
			'''

			# Save data
			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data = dict()
			data['regression_labels'] = ['Q_low', 'Q_mid', 'Q_high','RT', 'MT', 'Choice', 'Reward', 'Q_low_on', 'Q_mid_on', 'Q_high_on']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['beta_values_blocksAB'] = fit_glm_late.params
			data['pvalues_blocksAB'] = fit_glm_late.pvalues
			data['rsquared_blocksAB'] = fit_glm_late.rsquared
			data['Q_mid_early'] = Q_mid_BlockA
			data['Q_mid_late'] = Q_late
			data['FR_early'] = FR_BlockA
			data['FR_late'] = FR_late
			sp.io.savemat( dir1 + 'hold_center_fr/' + data_filename + '.mat', data)
		except:
			pass

	#plt.show()


	
	#return window_fr, window_fr_smooth, fr_mat, x, y, Q_low, Q_mid, Q_high, Q_learning, Q_late, FR_learning, FR_late, fit_glm
	return Q_low, Q_mid, Q_high

def ThreeTargetTask_RegressedFiringRatesWithRPE_RewardOnset(dir1, hdf_files, syncHDF_files, spike_files, channel, t_before, t_after, smoothed):
	'''
	This method regresses the firing rate of all units as a function of positive reward prediction error and negative reward prediction error. It then plots the firing rate as a function of 
	the modeled RPEs and uses the regression coefficient to plot a linear fit of the relationship between RPE and firing rate.

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
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Mario' + hdf_files[0][str_ind:str_ind + 8]
	if syncHDF_files[0]!='':
		str_ind = syncHDF_files[0].index('201')
		session_name = 'Mario' + syncHDF_files[0][str_ind:str_ind + 11]
	elif syncHDF_files[1]!='':
		str_ind = syncHDF_files[1].index('201')
		session_name = 'Mario' + syncHDF_files[0][str_ind:str_ind + 11]
	else:
		session_name = 'Unknown'
	

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]


	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr, window_fr_smooth = ThreeTargetTask_FiringRates_RewardOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials).astype(int)
	
	# 3. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()		# chosen target is 0 (L), 1 (M), or 2 (H)
	
	# Varying Q-values
	"""
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, log_likelihood = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)
	"""
	Q_high, Q_mid, Q_low = Value_from_reward_history_ThreeTargetTask(hdf_files)

	# 3a. Compute Positive and Negative RPEs
	Q_mat = np.vstack([Q_low, Q_mid, Q_high])
	chosen_target = np.array(chosen_target,dtype = int)
	chosen_target_value = np.array([Q_mat[chosen_target[i],i] for i in range(len(chosen_target))])

	RPE = rewards - chosen_target_value
	RPE_positive = RPE*(RPE > 0)
	RPE_negative = RPE*(RPE < 0)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 
		
		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	# 5. Do regression for each unit only on trials in Blocks A and B with spike data saved.
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A 
		trial_inds = np.array([index for index in range(50,150) if unit_data[index]!=0], dtype = int)
		#trial_inds = np.array([index for index in range(50,150)], dtype = int)
		x = np.vstack((RPE_positive[trial_inds], RPE_negative[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)

		#y = y/np.max(y)  # normalize y

		try:
			#print("Regression for unit ", k)
			model_glm = sm.OLS(y_zscore,x)
			fit_glm = model_glm.fit()
			#print(fit_glm.summary())

			regress_coef = fit_glm.params[1] 		# The regression coefficient for Qmid is the second parameter
			regress_intercept = y[0] - regress_coef*Q_mid[trial_inds[0]]

			# Get linear regression fit for just Q_mid
			Q_mid_min = np.amin(Q_mid[trial_inds])
			Q_mid_max = np.amax(Q_mid[trial_inds])
			x_lin = np.linspace(Q_mid_min, Q_mid_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_mid[trial_inds],y, c= 'k', marker = 'o', label ='Learning Trials')
			plt.plot(x_lin, m*x_lin + b, c = 'k')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'y')
			plt.xlabel('Q_mid')
			plt.ylabel('Firing Rate (spk/s)')
			plt.title(sess_name + ' - Channel %i - Unit %i' %(channel, k))
			'''

			# save Q and firing rate data
			Q_learning = Q_mid[trial_inds]
			FR_learning = y
			Q_mid_BlockA = Q_mid[trial_inds]

			RPE_early = RPE[trial_inds]

			max_fr = np.amax(y)
			xlim_min = np.amin(Q_mid[trial_inds])
			xlim_max = np.amax(Q_mid[trial_inds])

			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data = dict()
			data['regression_labels'] = ['RPE_positive', 'RPE_negative', 'Choice', 'Reward']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['RPE_early'] = RPE_early
			data['FR_early'] = FR_learning
			sp.io.savemat( dir1 + 'check_reward_fr/' + data_filename + '.mat', data)


			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			num_bins = 5
			sorted_Qvals_inds = np.argsort(Q_mid[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_mid[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			reorg_FR_BlockA = reorg_FR
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			# Save data for binning by bins of fixed size (rather than equally populated)
			Q_range_min = np.min(Q_mid[trial_inds])
			Q_range_max = np.max(Q_mid[trial_inds])
			FR_BlockA = y
			
			'''
			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.legend()
			'''
		except:
			pass
	
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A and B
		trial_inds = np.array([index for index in range(250,len(unit_data)) if unit_data[index]!=0], dtype = int)
		#trial_inds = np.array([index for index in range(250,len(unit_data))], dtype = int)
		x = np.vstack((RPE_positive[trial_inds], RPE_negative[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)

		#y = y/np.max(y)  # normalize y

		try:
			#print("Regression for unit ", k)
			model_glm_late = sm.OLS(y_zscore,x)
			fit_glm_late = model_glm_late.fit()
			#print(fit_glm_late.summary())

			regress_coef = fit_glm_late.params[1] 		# The regression coefficient for Qmid is the second parameter
			regress_intercept = y[0] - regress_coef*Q_mid[trial_inds[0]]

			# Get linear regression fit for just Q_mid
			Q_mid_min = np.amin(Q_mid[trial_inds])
			Q_mid_max = np.amax(Q_mid[trial_inds])
			x_lin = np.linspace(Q_mid_min, Q_mid_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			max_fr_stim = np.amax(y)
			fr_lim = np.maximum(max_fr, max_fr_stim)

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_mid[trial_inds],y, c= 'c', label = 'Stimulation trials')
			plt.plot(x_lin, m*x_lin + b, c = 'c')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'g')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*xlim_min, 1.1*xlim_max))
			plt.legend()
			'''

			# save Q and firing rate data
			Q_late = Q_mid[trial_inds]
			FR_late = y

			RPE_late = RPE[trial_inds]

			# Get binned firing rates: bins of fixed size
			Q_range_min = np.min(np.min(Q_late), Q_range_min)
			Q_range_max = np.max(np.max(Q_late), Q_range_max)
			bins = np.arange(Q_range_min, Q_range_max + 0.5*(Q_range_max - Q_range_min)/5., (Q_range_max - Q_range_min)/5.)
			hist_BlockA, bins = np.histogram(Q_mid_BlockA, bins)
			hist_late, bins = np.histogram(Q_late, bins)

			sorted_Qvals_inds_BlockA = np.argsort(Q_mid_BlockA)
			sorted_Qvals_inds_late = np.argsort(Q_late)

			begin_BlockA = 0
			begin_late = 0
			dta_all = []
			avg_FR_BlockA = np.zeros(5)
			avg_FR_late = np.zeros(5)
			sem_FR_BlockA = np.zeros(5)
			sem_FR_late = np.zeros(5)
			for j in range(len(hist_BlockA)):
				data_BlockA = FR_BlockA[sorted_Qvals_inds_BlockA[begin_BlockA:begin_BlockA+hist_BlockA[j]]]
				data_late = FR_late[sorted_Qvals_inds_late[begin_late:begin_late+hist_late[j]]]
				begin_BlockA += hist_BlockA[j]
				begin_late += hist_late[j]

				avg_FR_BlockA[j] = np.nanmean(data_BlockA)
				sem_FR_BlockA[j] = np.nanstd(data_BlockA)/np.sqrt(len(data_BlockA))
				avg_FR_late[j] = np.nanmean(data_late)
				sem_FR_late[j] = np.nanstd(data_late)/np.sqrt(len(data_late))

				for item in data_BlockA:
					dta_all += [(j,0,item)]

				for item in data_late:
					dta_all += [(j,1,item)]

			dta_all = pd.DataFrame(dta_all, columns = ['Bin', 'Condition', 'FR'])
			bin_centers = (bins[1:] + bins[:-1])/2.
			
			'''
			formula = 'FR ~ C(Bin) + C(Condition) + C(Bin):C(Condition)'
			model = ols(formula, dta_all).fit()
			aov_table = anova_lm(model, typ=2)

			print "Two-way ANOVA analysis"
			print(aov_table)
			'''

			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			sorted_Qvals_inds = np.argsort(Q_mid[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_mid[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			'''
			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR/2., fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()

			plt.figure(k)
			plt.subplot(1,3,3)
			plt.errorbar(bin_centers, avg_FR_BlockA, yerr = sem_FR_BlockA, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.errorbar(bin_centers, avg_FR_late, yerr = sem_FR_late, fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()
			'''

			# Save data
			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data = dict()

			data['regression_labels'] = ['RPE_positive', 'RPE_negative', 'Choice', 'Reward']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['RPE_early'] = RPE_early
			data['beta_values_blocksAB'] = fit_glm_late.params
			data['pvalues_blocksAB'] = fit_glm_late.pvalues
			data['rsquared_blocksAB'] = fit_glm_late.rsquared
			data['RPE_late'] = RPE_late
			data['FR_early'] = FR_BlockA
			data['FR_late'] = FR_late
			sp.io.savemat( dir1 + 'check_reward_fr/' + data_filename + '.mat', data)
		except:
			pass

	#plt.show()


	
	#return window_fr, window_fr_smooth, fr_mat, x, y, Q_low, Q_mid, Q_high, Q_learning, Q_late, FR_learning, FR_late, fit_glm
	return 

def ThreeTargetTask_PeakFiringRatesWithValue_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after):
	'''
	This method regresses the firing rate of all units as a function of value. It then plots the firing rate as a function of 
	the modeled value and uses the regression coefficient to plot a linear fit of the relationship between value and 
	firing rate.

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
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Mario' + hdf_files[0][str_ind:str_ind + 8]

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]


	# 2. Get peak firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr = ThreeTargetTask_MaxFiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials)

	# 3. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
	# Varying Q-values
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, log_likelihood = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		block_fr = window_fr[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 
		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	# 5. Do binning for each unit only on trials in Blocks A and C with spike data saved.
	Qbins = np.linspace(0,1,21) 	# bin widths of 0.05
	FRbinned_late = dict()
	FRbinned_early = dict()

	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A
		trial_inds = np.array([index for index in range(50,150) if unit_data[index]!=0], dtype = int)
		x = Q_mid[trial_inds]
		y = unit_data[trial_inds]

		# find sorted Q values
		sorted_Qvals_inds = np.argsort(Q_mid[trial_inds])
		# find number of Q values in each Q value bin
		Qhist, Qbin_edges = np.histogram(Q_mid[trial_inds], Qbins)
		cumnum_per_bin = np.cumsum(Qhist)
		for j in range(len(Qhist)):
			if j==0:
				FRbinned_early[j] = y[sorted_Qvals_inds][0:cumnum_per_bin[j]]
			else:
				FRbinned_early[j] = y[sorted_Qvals_inds][cumnum_per_bin[j-1]:cumnum_per_bin[j]]
	
		# look at all trial types within Blocks A'
		trial_inds = np.array([index for index in range(250,len(unit_data)) if unit_data[index]!=0], dtype = int)
		x = Q_mid[trial_inds]
		y = unit_data[trial_inds]

		# find sorted Q values
		sorted_Qvals_inds = np.argsort(Q_mid[trial_inds])
		# find number of Q values in each Q value bin
		Qhist, Qbin_edges = np.histogram(Q_mid[trial_inds], Qbins)
		cumnum_per_bin = np.cumsum(Qhist)
		for j in range(len(Qhist)):
			if j==0:
				FRbinned_late[j] = y[sorted_Qvals_inds][0:cumnum_per_bin[j]]
			else:
				FRbinned_late[j] = y[sorted_Qvals_inds][cumnum_per_bin[j-1]:cumnum_per_bin[j]]



		## plot (a) peak firing rate for Blocks A vs C (b) average change in firing rates between blocks per bins
		plt.figure(k)
		#plt.subplot(1,2,1)
		for j in range(len(Qhist)):
			plt.scatter(Qbins[j]*np.ones(len(FRbinned_early[j])), FRbinned_early[j], color = 'k')
			plt.scatter(Qbins[j]*np.ones(len(FRbinned_late[j])), FRbinned_late[j], color = 'c')
		plt.ylabel('Peak Firing Rate (spks/s)')
		plt.xlabel('Q_mid')

		#plt.subplot(1,2,2)
		
		avg_FRbinned_early = np.zeros(len(Qhist))
		avg_FRbinned_late = np.zeros(len(Qhist))
		sem_FRbinned_early = np.zeros(len(Qhist))
		sem_FRbinned_late = np.zeros(len(Qhist))

		for j in range(len(Qhist)):
			avg_FRbinned_early[j] = np.nanmean(FRbinned_early[j])
			avg_FRbinned_late[j] = np.nanmean(FRbinned_late[j])
			sem_FRbinned_early[j] = np.nanstd(FRbinned_early[j])/np.sqrt(len(FRbinned_early[j]))
			sem_FRbinned_late[j] = np.nanstd(FRbinned_late[j])/np.sqrt(len(FRbinned_late[j]))

		plt.errorbar(Qbins[:-1], avg_FRbinned_early, yerr = sem_FRbinned_early/2.,color = 'k', ecolor = 'k', label = 'Block A')
		plt.errorbar(Qbins[:-1], avg_FRbinned_late, yerr = sem_FRbinned_late/2.,color = 'c', ecolor = 'c', label = 'Block C')
		plt.xlabel('Q_mid')
		plt.ylabel('Avg Peak Firing Rate (spks/s)')
		plt.ylim((0,65))

		plt.figure(max_num_units+k)
		plt.plot(Qbins[:-1], avg_FRbinned_late - avg_FRbinned_early, 'k', label = 'Diff: Block C- A')
		plt.xlabel('Q_mid')
		plt.ylabel('Delta Peak Firing Rate (spks/s)')

		## return average change in firing rates per bins

	plt.show()


	return avg_FRbinned_late, avg_FRbinned_early


def ThreeTargetTask_SpikeAnalysis(hdf_files, syncHDF_files, spike_files, cd_only,align_to):
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
	- cd_only: Boolean indicating if we only want to look at units in the Caudate (cd_only = 1) or if we want to look at
				all units (cd_only = 0)
	- align_to: integer in range [1,2] that indicates whether we align the (1) picture onset or (2) center-hold.

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

		if align_to == 1:
			inds = ind_picture_onset
		elif align_to == 2:
			inds = ind_hold_center
		
		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			spike1 = OfflineSorted_CSVFile(spike_files[i][0])
			spike2 = OfflineSorted_CSVFile(spike_files[i][1])
			# Find all sort codes associated with good channels
			all_units1, total_units1 = spike1.find_unit_sc(spike1.good_channels)
			all_units2, total_units2 = spike2.find_unit_sc(spike2.good_channels)

			print("Total number of units: ", total_units1 + total_units2)

			if cd_only:
				cd_units = [1, 3, 4, 17, 18, 20, 40, 41, 54, 56, 57, 63, 64, 72, 75, 81, 83, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
				spike1_good_channels = np.array([unit for unit in cd_units if unit in spike1.good_channels])
				spike2_good_channels = np.array([unit for unit in cd_units if unit in spike2.good_channels])
			else:
				spike1_good_channels = spike1.good_channels
				spike2_good_channels = spike2.good_channels

			# Plot average rate for all neurons divided in six cases of targets on option
			plt.figure()
			t_before = 1			# 1 s
			t_after = 3				# 3 s
			t_resolution = 0.1 		# 100 ms time bins

			cmap = mpl.cm.Blues

			# 1. LH presented
			LH_ind = np.ravel(np.nonzero([np.array_equal(target_options[j,:], [1,1,0]) for j in range(int(num_successful_trials[i]))]))
			avg_psth1, smooth_avg_psth1, unit_list11 = spike1.compute_multiple_channel_avg_psth(spike1_good_channels, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			avg_psth2, smooth_avg_psth2, unit_list21 = spike2.compute_multiple_channel_avg_psth(spike2_good_channels, times_row_ind[LH_ind],t_before,t_after,t_resolution)
			
			num_units1,num_samples = avg_psth1.shape
			num_units2,num_samples = avg_psth2.shape
			plt.subplot(3,2,1)
			plt.title('Low-High Presented')
			for k in range(num_units1):
				plt.plot(smooth_avg_psth1[k,:], color=cmap(k/float(num_units1 + num_units2)))
			for k in range(num_units2):
				plt.plot(smooth_avg_psth2[k,:], color=cmap((k + num_units1)/float(num_units1 + num_units2)))
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
			print(unit_list1)
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

	return unit_list11, unit_list21

def ThreeTargetTask_SpikeAnalysis_SingleChannel(hdf_files, syncHDF_files, spike_files, chann, sc, align_to, plot_output):
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
	- align_to: integer in range [1,2] that indicates whether we align the (1) picture onset or (2) center-hold.
	- plot_output: binary indicating whether output should be plotted + saved or not
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)

	# Define timing parameters for PSTHs
	t_before = 2			# 1 s
	t_after = 2				# 3 s
	t_resolution = 0.1 		# 100 ms time bins
	num_bins = len(np.arange(-t_before, t_after, t_resolution)) - 1

	# Define arrays to save psth for each trial
	smooth_psth_lm = np.array([])
	smooth_psth_lh = np.array([])
	smooth_psth_mh = np.array([])
	smooth_psth_l = np.array([])
	smooth_psth_m = np.array([])
	smooth_psth_h = np.array([])
	smooth_psth = np.array([])
	psth_lm = np.array([])
	psth_lh = np.array([])
	psth_mh = np.array([])
	psth_l = np.array([])
	psth_m = np.array([])
	psth_h = np.array([])
	psth = np.array([])

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
		if align_to == 1:
			inds = ind_picture_onset
		elif align_to == 2:
			inds = ind_hold_center

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			spike1 = OfflineSorted_CSVFile(spike_files[i][0])
			spike2 = OfflineSorted_CSVFile(spike_files[i][1])
			# Find all sort codes associated with good channels
			all_units1, total_units1 = spike1.find_unit_sc(spike1.good_channels)
			all_units2, total_units2 = spike2.find_unit_sc(spike2.good_channels)

			#print "Total number of units: ", total_units1 + total_units2

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

			# Plot Blocks A and A' separately
			BlockA_ind = np.array([j for j in range(0,int(num_successful_trials[i]))])

			if not spike2_good_channels:
				avg_psth, smooth_avg_psth = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[BlockA_ind],t_before,t_after,t_resolution)
			else:
				avg_psth, smooth_avg_psth = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[BlockA_ind],t_before,t_after,t_resolution)
			
			if i == 0:
				psth = avg_psth
				smooth_psth = smooth_avg_psth
			else:
				psth = np.vstack([psth, avg_psth])
				smooth_psth = np.vstack([smooth_psth, smooth_avg_psth])

	# Plot average rate for all neurons divided in six cases of targets on option
	if plot_output:
		plt.figure(0)

		avg_psth_lh = np.nanmean(psth_lh, axis = 0)
		smooth_avg_psth_lh = np.nanmean(smooth_psth_lh, axis = 0)
		plt.subplot(3,2,1)
		plt.title('Low-High Presented')
		plt.plot(smooth_avg_psth_lh)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		avg_psth_lm = np.nanmean(psth_lm, axis = 0)
		smooth_avg_psth_lm = np.nanmean(smooth_psth_lm, axis = 0)
		plt.subplot(3,2,2)
		plt.title('Low-Middle Presented')
		plt.plot(smooth_avg_psth_lm)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		avg_psth_mh = np.nanmean(psth_mh, axis = 0)
		smooth_avg_psth_mh = np.nanmean(smooth_psth_mh, axis = 0)

		avg_psth_mh_early = np.nanmean(psth_mh[:psth_mh.shape[0]/2,:], axis = 0)
		avg_psth_mh_late = np.nanmean(psth_mh[psth_mh.shape[0]/2:,:], axis = 0)
		plt.subplot(3,2,3)
		plt.title('Middle-High Presented')
		plt.plot(smooth_avg_psth_mh)
		plt.plot(avg_psth_mh_early, c = 'k')
		plt.plot(avg_psth_mh_late, c = 'c')
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		avg_psth_l = np.nanmean(psth_l, axis = 0)
		smooth_avg_psth_l = np.nanmean(smooth_psth_l, axis = 0)
		plt.subplot(3,2,4)
		plt.title('Low Presented')
		plt.plot(smooth_avg_psth_l)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		avg_psth_h = np.nanmean(psth_h, axis = 0)
		smooth_avg_psth_h = np.nanmean(smooth_psth_h, axis = 0)
		plt.subplot(3,2,5)
		plt.title('High Presented')
		plt.plot(smooth_avg_psth_h)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		avg_psth_m = np.nanmean(psth_m, axis = 0)
		smooth_avg_psth_m = np.nanmean(smooth_psth_m, axis = 0)
		plt.subplot(3,2,6)
		plt.title('Middle Presented')
		plt.plot(smooth_avg_psth_m)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		#plt_name = syncHDF_files[i][34:-15]
		#plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		plt_name = syncHDF_files[i][syncHDF_files[i].index('Mario201'):-15]
		plt.savefig('C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/Caudate Stim/'+plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')

		avg_psth = np.nanmean(psth[50:150,:], axis = 0)
		sem_avg_psth = np.nanstd(psth[50:150,:], axis = 0)/np.sqrt(100)
		avg_psth_p = np.nanmean(psth[250:,:], axis = 0)
		sem_avg_psth_p = np.nanstd(smooth_psth[250:,:], axis = 0)/np.sqrt(smooth_psth[250:,:].shape[0])

		plt.figure(1)
		plt.title('PSTH Aligned to Target Presentation')
		plt.plot(avg_psth, 'k', label = 'Block A')
		plt.plot(avg_psth_p, 'c', label = 'Block C')
		plt.fill_between(range(39), avg_psth - sem_avg_psth/2., avg_psth + sem_avg_psth/2., color = 'k', alpha = 0.5)
		plt.fill_between(range(39), avg_psth_p - sem_avg_psth_p/2., avg_psth_p + sem_avg_psth_p/2., color = 'c', alpha = 0.5)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		plt_name = syncHDF_files[i][syncHDF_files[i].index('Mario201'):-15]
		plt.savefig('C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/Caudate Stim/'+plt_name+ '_' + str(align_to) +'_PSTH_CompareBlocks_Chan'+str(chann)+'-'+str(sc)+'.svg')
		
		plt.figure(2)
		plt.title('PSTH Aligned to Target Onset - MH Trials Only')
		sem_psth_mh_early = np.nanstd(psth_mh[:psth_mh.shape[0]/2,:], axis = 0)/np.sqrt(psth_mh.shape[0]/2)
		sem_psth_mh_late = np.nanstd(psth_mh[psth_mh.shape[0]/2:,:], axis = 0)/np.sqrt(psth_mh.shape[0]/2)
		plt.plot(avg_psth_mh_early, c = 'k')
		plt.plot(avg_psth_mh_late, c = 'c')
		plt.fill_between(range(39), avg_psth_mh_early - sem_psth_mh_early/2., avg_psth_mh_early + sem_psth_mh_early/2., color = 'k', alpha = 0.5)
		plt.fill_between(range(39), avg_psth_mh_late - sem_psth_mh_late/2., avg_psth_mh_late + sem_psth_mh_late/2., color = 'c', alpha = 0.5)
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		xticks = np.arange(0, len(xticklabels), 10)
		xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		plt.xticks(xticks, xticklabels)

		plt.savefig('C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/Caudate Stim/'+plt_name+ '_' + str(align_to) +'_PSTH_CompareBlocksMH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		
		plt.close()

	reg_psth = [psth_lm, psth_lh, psth_mh, psth_l, psth_h, psth_m, psth]
	smooth_psth = [smooth_psth_lm, smooth_psth_lh, smooth_psth_mh, smooth_psth_l, smooth_psth_h, smooth_psth_m, smooth_psth]
	return reg_psth, smooth_psth

def TwoTargetTask_SpikeAnalysis_SingleChannel(hdf_files, syncHDF_files, spike_files, chann, sc, align_to, t_before, t_after, plot_output):
	'''

	This method aligns spiking data to behavioral choices 
	in the Two Target Task, where there is a low-value and high-value target. This version does not 
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
	- align_to: integer in range [1,2] that indicates whether we align the (1) picture onset, a.k. center-hold or (2) check_reward.
	- plot_output: binary indicating whether output should be plotted + saved or not

	Outputs:
	- reg_psth: a 2 x N array, where the first row corresponds to the psth values for trials where the low-value 
				target was selected and the second row corresponds to the psth values for trials where the high-value
				target was selected
	- smooth_psth: a 2 x N array of the same format as reg_psth, but with values taken from psths smoothed with a 
				Gaussian kernel
	- all_raster_l: a dictionary with number of elements equal to the number of trials where the low-value target was selected,
				where each element contains spike times for the designated unit within that trial
	- all_raster_h: a dictionary with number of elements equal to the number of trials where the high-value target was selected,
				where each element contains spike times for the designated unit within that trial
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)

	# Define timing parameters for PSTHs
	t_resolution = 0.1 		# 100 ms time bins
	num_bins = len(np.arange(-t_before, t_after, t_resolution)) - 1

	# Define arrays to save psth for each trial
	smooth_psth_l = np.array([])
	smooth_psth_h = np.array([])
	psth_l = np.array([])
	psth_h = np.array([])

	'''
	Get data for each set of files
	'''
	for i in range(num_files):
		# Load behavior data
		cb = ChoiceBehavior_TwoTargets(hdf_files[i])
		num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_check_reward = cb.ind_check_reward_states
		if align_to == 1:
			inds = ind_hold_center
		elif align_to == 2:
			inds = ind_check_reward

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			cd_units = [chann]
			if chann < 97:
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				all_units1, total_units1 = spike1.find_unit_sc(spike1.good_channels)
				spike1_good_channels = [unit for unit in cd_units if unit in spike1.good_channels]
				spike2_good_channels = []
			else:
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				all_units2, total_units2 = spike2.find_unit_sc(spike2.good_channels)
				spike2_good_channels = [unit for unit in cd_units if unit in spike2.good_channels]
				spike1_good_channels = []

			# 2. L chosen
			L_ind = np.ravel(np.nonzero([np.array_equal(target_chosen[j,:], [1,0]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_l, smooth_avg_psth_l = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			else:
				avg_psth_l, smooth_avg_psth_l = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike2.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			if i == 0:
				psth_l = avg_psth_l
				smooth_psth_l = smooth_avg_psth_l
				all_raster_l = raster_l
			else:
				psth_l = np.vstack([psth_l, avg_psth_l])
				smooth_psth_l = np.vstack([smooth_psth_l, smooth_avg_psth_l])
				all_raster_l.update(raster_l)

			# 5. H chosen
			H_ind = np.ravel(np.nonzero([np.array_equal(target_chosen[j,:], [0,1]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_h, smooth_avg_psth_h = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			else:
				avg_psth_h, smooth_avg_psth_h = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike2.compute_raster(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			if i == 0:
				psth_h = avg_psth_h
				smooth_psth_h = smooth_avg_psth_h
				all_raster_h = raster_h
			else:
				psth_h = np.vstack([psth_h, avg_psth_h])
				smooth_psth_h = np.vstack([smooth_psth_h, smooth_avg_psth_h])
				all_raster_h.update(raster_h)

	# Plot average rate for all neurons divided in six cases of targets on option
	if plot_output:
		plt.figure(0)
		b = signal.gaussian(39,0.6)
		avg_psth_l = np.nanmean(psth_l, axis = 0)
		sem_avg_psth_l = np.nanstd(psth_l, axis = 0)/np.sqrt(psth_l.shape[0])
		#smooth_avg_psth_l = np.nanmean(smooth_psth_l, axis = 0)
		smooth_avg_psth_l = filters.convolve1d(np.nanmean(psth_l,axis=0), b/b.sum())
		sem_smooth_avg_psth_l = np.nanstd(smooth_psth_l, axis = 0)/np.sqrt(smooth_psth_l.shape[0])

		avg_psth_h = np.nanmean(psth_h, axis = 0)
		sem_avg_psth_h = np.nanstd(psth_h, axis = 0)/np.sqrt(psth_h.shape[0])
		smooth_avg_psth_h = np.nanmean(smooth_psth_h, axis = 0)
		smooth_avg_psth_h = filters.convolve1d(np.nanmean(psth_h,axis=0), b/b.sum())
		sem_smooth_avg_psth_h = np.nanstd(smooth_psth_h, axis = 0)/np.sqrt(smooth_psth_h.shape[0])
		
		y_min_l = (smooth_avg_psth_l - sem_smooth_avg_psth_l).min()
		y_max_l = (smooth_avg_psth_l+ sem_smooth_avg_psth_l).max()
		y_min_h = (smooth_avg_psth_h - sem_smooth_avg_psth_h).min()
		y_max_h = (smooth_avg_psth_h+ sem_smooth_avg_psth_h).max()

		y_min = np.array([y_min_l, y_min_h]).min()
		y_max = np.array([y_max_l, y_max_h]).max()


		num_trials = len(all_raster_l.keys())

		linelengths = float((y_max - y_min))/num_trials
		lineoffsets = 1
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)

		ax1 = plt.subplot(1,2,1)
		plt.title('All Trials')
		plt.plot(xticklabels, smooth_avg_psth_l,'b', label = 'LV Chosen')
		plt.fill_between(xticklabels, smooth_avg_psth_l - sem_smooth_avg_psth_l, smooth_avg_psth_l + sem_smooth_avg_psth_l, facecolor = 'b', alpha = 0.2)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_l.keys())):
			plt.eventplot(all_raster_l[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))

		ax2 = plt.subplot(1,2,2)
		plt.plot(xticklabels, smooth_avg_psth_h, 'r', label = 'HV Chosen')
		plt.fill_between(xticklabels, smooth_avg_psth_h - sem_smooth_avg_psth_h, smooth_avg_psth_h + sem_smooth_avg_psth_h, facecolor = 'r', alpha = 0.2)
		#xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax2.get_yaxis().set_tick_params(direction='out')
		ax2.get_xaxis().set_tick_params(direction='out')
		ax2.get_xaxis().tick_bottom()
		ax2.get_yaxis().tick_left()

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_h.keys())):
			plt.eventplot(all_raster_h[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))


		#plt_name = syncHDF_files[i][34:-15]
		#plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		plt_name = syncHDF_files[i][syncHDF_files[i].index('Luigi201'):-15]
		plt.savefig(dir_figs +plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		#print('Figure saved:',dir_figs +plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		
		plt.close()

	reg_psth = [psth_l, psth_h]
	smooth_psth = [smooth_psth_l, smooth_psth_h]
	return reg_psth, smooth_psth, all_raster_l, all_raster_h

def TwoTargetTask_SpikeAnalysis_PSTH_AllSortedGoodChannels(hdf_files, syncHDF_files, spike_files, align_to, t_before, t_after, plot_output):
	'''

	This method aligns spiking data to behavioral choices 
	in the Two Target Task, where there is a low-value and high-value target. This version does not 
	differentiate between choices in different blocks. It generates PSTHs for all re-sorted good channels
	and plots them in a single figure.

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
	- align_to: integer in range [1,2] that indicates whether we align the (1) picture onset, a.k. center-hold or (2) check_reward.
	- plot_output: binary indicating whether output should be plotted + saved or not

	Outputs:
	- reg_psth: a 2 x N array, where the first row corresponds to the psth values for trials where the low-value 
				target was selected and the second row corresponds to the psth values for trials where the high-value
				target was selected
	- smooth_psth: a 2 x N array of the same format as reg_psth, but with values taken from psths smoothed with a 
				Gaussian kernel
	- all_raster_l: a dictionary with number of elements equal to the number of trials where the low-value target was selected,
				where each element contains spike times for the designated unit within that trial
	- all_raster_h: a dictionary with number of elements equal to the number of trials where the high-value target was selected,
				where each element contains spike times for the designated unit within that trial
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)

	# Define timing parameters for PSTHs
	t_resolution = 0.1 		# 100 ms time bins
	num_bins = len(np.arange(-t_before, t_after, t_resolution)) - 1

	# Define arrays to save psth for each trial
	smooth_psth_l = np.array([])
	smooth_psth_h = np.array([])
	psth_l = np.array([])
	psth_h = np.array([])

	'''
	Get data for each set of files
	'''
	for i in range(num_files):
		# Load behavior data
		cb = ChoiceBehavior_TwoTargets(hdf_files[i])
		num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_check_reward = cb.ind_check_reward_states
		if align_to == 1:
			inds = ind_hold_center
		elif align_to == 2:
			inds = ind_check_reward

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			if (spike_files[i][0] != ''):
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				spike1_good_channels = spike1.sorted_good_chans_sc
				spike2_good_channels = []
			elif (spike_files[i][1] != ''):
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				spike2_good_channels = spike2.sorted_good_chans_sc
				spike1_good_channels = []

			##############
			# STOPPED HERE
			##############

			# 2. L chosen
			L_ind = np.ravel(np.nonzero([np.array_equal(target_chosen[j,:], [1,0]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_l, smooth_avg_psth_l = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			else:
				avg_psth_l, smooth_avg_psth_l = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike2.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			if i == 0:
				psth_l = avg_psth_l
				smooth_psth_l = smooth_avg_psth_l
				all_raster_l = raster_l
			else:
				psth_l = np.vstack([psth_l, avg_psth_l])
				smooth_psth_l = np.vstack([smooth_psth_l, smooth_avg_psth_l])
				all_raster_l.update(raster_l)

			# 5. H chosen
			H_ind = np.ravel(np.nonzero([np.array_equal(target_chosen[j,:], [0,1]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_h, smooth_avg_psth_h = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			else:
				avg_psth_h, smooth_avg_psth_h = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike2.compute_raster(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			if i == 0:
				psth_h = avg_psth_h
				smooth_psth_h = smooth_avg_psth_h
				all_raster_h = raster_h
			else:
				psth_h = np.vstack([psth_h, avg_psth_h])
				smooth_psth_h = np.vstack([smooth_psth_h, smooth_avg_psth_h])
				all_raster_h.update(raster_h)

	# Plot average rate for all neurons divided in six cases of targets on option
	if plot_output:
		plt.figure(0)
		b = signal.gaussian(39,0.6)
		avg_psth_l = np.nanmean(psth_l, axis = 0)
		sem_avg_psth_l = np.nanstd(psth_l, axis = 0)/np.sqrt(psth_l.shape[0])
		#smooth_avg_psth_l = np.nanmean(smooth_psth_l, axis = 0)
		smooth_avg_psth_l = filters.convolve1d(np.nanmean(psth_l,axis=0), b/b.sum())
		sem_smooth_avg_psth_l = np.nanstd(smooth_psth_l, axis = 0)/np.sqrt(smooth_psth_l.shape[0])

		avg_psth_h = np.nanmean(psth_h, axis = 0)
		sem_avg_psth_h = np.nanstd(psth_h, axis = 0)/np.sqrt(psth_h.shape[0])
		smooth_avg_psth_h = np.nanmean(smooth_psth_h, axis = 0)
		smooth_avg_psth_h = filters.convolve1d(np.nanmean(psth_h,axis=0), b/b.sum())
		sem_smooth_avg_psth_h = np.nanstd(smooth_psth_h, axis = 0)/np.sqrt(smooth_psth_h.shape[0])
		
		y_min_l = (smooth_avg_psth_l - sem_smooth_avg_psth_l).min()
		y_max_l = (smooth_avg_psth_l+ sem_smooth_avg_psth_l).max()
		y_min_h = (smooth_avg_psth_h - sem_smooth_avg_psth_h).min()
		y_max_h = (smooth_avg_psth_h+ sem_smooth_avg_psth_h).max()

		y_min = np.array([y_min_l, y_min_h]).min()
		y_max = np.array([y_max_l, y_max_h]).max()


		num_trials = len(all_raster_l.keys())

		linelengths = float((y_max - y_min))/num_trials
		lineoffsets = 1
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)

		ax1 = plt.subplot(1,2,1)
		plt.title('All Trials')
		plt.plot(xticklabels, smooth_avg_psth_l,'b', label = 'LV Chosen')
		plt.fill_between(xticklabels, smooth_avg_psth_l - sem_smooth_avg_psth_l, smooth_avg_psth_l + sem_smooth_avg_psth_l, facecolor = 'b', alpha = 0.2)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_l.keys())):
			plt.eventplot(all_raster_l[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))

		ax2 = plt.subplot(1,2,2)
		plt.plot(xticklabels, smooth_avg_psth_h, 'r', label = 'HV Chosen')
		plt.fill_between(xticklabels, smooth_avg_psth_h - sem_smooth_avg_psth_h, smooth_avg_psth_h + sem_smooth_avg_psth_h, facecolor = 'r', alpha = 0.2)
		#xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax2.get_yaxis().set_tick_params(direction='out')
		ax2.get_xaxis().set_tick_params(direction='out')
		ax2.get_xaxis().tick_bottom()
		ax2.get_yaxis().tick_left()

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_h.keys())):
			plt.eventplot(all_raster_h[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))


		#plt_name = syncHDF_files[i][34:-15]
		#plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		plt_name = syncHDF_files[i][syncHDF_files[i].index('Luigi201'):-15]
		plt.savefig(dir_figs +plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		#print('Figure saved:',dir_figs +plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		
		plt.close()

	reg_psth = [psth_l, psth_h]
	smooth_psth = [smooth_psth_l, smooth_psth_h]
	return reg_psth, smooth_psth, all_raster_l, all_raster_h


def TwoTargetTask_SpikeAnalysis_SingleChannel_RPE(hdf_files, syncHDF_files, spike_files, chann, sc, align_to, t_before, t_after, plot_output):
	'''

	This method aligns spiking data to behavioral choices 
	in the Two Target Task, where there is a low-value and high-value target. This version does not 
	differentiate between choices in different blocks. Trials are separated by whether they are rewarded (+RPE) or
	not rewarded (-RPE).

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
	- align_to: integer in range [1,2] that indicates whether we align the (1) picture onset, a.k. center-hold or (2) check_reward.
	- plot_output: binary indicating whether output should be plotted + saved or not

	Outputs:
	- reg_psth: a 2 x N array, where the first row corresponds to the psth values for trials where the low-value 
				target was selected and the second row corresponds to the psth values for trials where the high-value
				target was selected
	- smooth_psth: a 2 x N array of the same format as reg_psth, but with values taken from psths smoothed with a 
				Gaussian kernel
	- all_raster_l: a dictionary with number of elements equal to the number of trials where the low-value target was selected,
				where each element contains spike times for the designated unit within that trial
	- all_raster_h: a dictionary with number of elements equal to the number of trials where the high-value target was selected,
				where each element contains spike times for the designated unit within that trial
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)

	# Define timing parameters for PSTHs
	t_resolution = 0.1 		# 100 ms time bins
	num_bins = len(np.arange(-t_before, t_after, t_resolution)) - 1

	# Define arrays to save psth for each trial
	smooth_psth_l = np.array([])
	smooth_psth_h = np.array([])
	psth_l = np.array([])
	psth_h = np.array([])

	'''
	Get data for each set of files
	'''
	for i in range(num_files):
		# Load behavior data
		cb = ChoiceBehavior_TwoTargets(hdf_files[i])
		num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_check_reward = cb.ind_check_reward_states
		if align_to == 1:
			inds = ind_hold_center
		elif align_to == 2:
			inds = ind_check_reward

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			cd_units = [chann]
			if chann < 97:
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				all_units1, total_units1 = spike1.find_unit_sc(spike1.good_channels)
				spike1_good_channels = [unit for unit in cd_units if unit in spike1.good_channels]
				spike2_good_channels = []
			else:
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				all_units2, total_units2 = spike2.find_unit_sc(spike2.good_channels)
				spike2_good_channels = [unit for unit in cd_units if unit in spike2.good_channels]
				spike1_good_channels = []

			# 2. Unrewarded trials
			L_ind = np.ravel(np.nonzero([np.array_equal(rewarded_choice[j], 0) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_l, smooth_avg_psth_l = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			else:
				avg_psth_l, smooth_avg_psth_l = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike2.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			if i == 0:
				psth_l = avg_psth_l
				smooth_psth_l = smooth_avg_psth_l
				all_raster_l = raster_l
			else:
				psth_l = np.vstack([psth_l, avg_psth_l])
				smooth_psth_l = np.vstack([smooth_psth_l, smooth_avg_psth_l])
				all_raster_l.update(raster_l)

			# 3. Rewarded trials
			H_ind = np.ravel(np.nonzero([np.array_equal(rewarded_choice[j], 1) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_h, smooth_avg_psth_h = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			else:
				avg_psth_h, smooth_avg_psth_h = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike2.compute_raster(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			if i == 0:
				psth_h = avg_psth_h
				smooth_psth_h = smooth_avg_psth_h
				all_raster_h = raster_h
			else:
				psth_h = np.vstack([psth_h, avg_psth_h])
				smooth_psth_h = np.vstack([smooth_psth_h, smooth_avg_psth_h])
				all_raster_h.update(raster_h)

	# Plot average rate for all neurons divided in six cases of targets on option
	if plot_output:
		plt.figure(0)
		b = signal.gaussian(39,0.6)
		avg_psth_l = np.nanmean(psth_l, axis = 0)
		sem_avg_psth_l = np.nanstd(psth_l, axis = 0)/np.sqrt(psth_l.shape[0])
		#smooth_avg_psth_l = np.nanmean(smooth_psth_l, axis = 0)
		smooth_avg_psth_l = filters.convolve1d(np.nanmean(psth_l,axis=0), b/b.sum())
		sem_smooth_avg_psth_l = np.nanstd(smooth_psth_l, axis = 0)/np.sqrt(smooth_psth_l.shape[0])

		avg_psth_h = np.nanmean(psth_h, axis = 0)
		sem_avg_psth_h = np.nanstd(psth_h, axis = 0)/np.sqrt(psth_h.shape[0])
		smooth_avg_psth_h = np.nanmean(smooth_psth_h, axis = 0)
		smooth_avg_psth_h = filters.convolve1d(np.nanmean(psth_h,axis=0), b/b.sum())
		sem_smooth_avg_psth_h = np.nanstd(smooth_psth_h, axis = 0)/np.sqrt(smooth_psth_h.shape[0])
		
		y_min_l = (smooth_avg_psth_l - sem_smooth_avg_psth_l).min()
		y_max_l = (smooth_avg_psth_l+ sem_smooth_avg_psth_l).max()
		y_min_h = (smooth_avg_psth_h - sem_smooth_avg_psth_h).min()
		y_max_h = (smooth_avg_psth_h+ sem_smooth_avg_psth_h).max()

		y_min = np.array([y_min_l, y_min_h]).min()
		y_max = np.array([y_max_l, y_max_h]).max()


		num_trials = len(all_raster_l.keys())

		linelengths = float((y_max - y_min))/num_trials
		lineoffsets = 1
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)

		ax1 = plt.subplot(1,2,1)
		plt.title('All Trials')
		plt.plot(xticklabels, smooth_avg_psth_l,'b', label = '-RPE')
		plt.fill_between(xticklabels, smooth_avg_psth_l - sem_smooth_avg_psth_l, smooth_avg_psth_l + sem_smooth_avg_psth_l, facecolor = 'b', alpha = 0.2)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_l.keys())):
			plt.eventplot(all_raster_l[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))

		ax2 = plt.subplot(1,2,2)
		plt.plot(xticklabels, smooth_avg_psth_h, 'r', label = '+RPE')
		plt.fill_between(xticklabels, smooth_avg_psth_h - sem_smooth_avg_psth_h, smooth_avg_psth_h + sem_smooth_avg_psth_h, facecolor = 'r', alpha = 0.2)
		#xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		if align_to == 1:
			plt.xlabel('Time from Center Hold (s)')
		else:
			plt.xlabel('Time from Check Reward (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax2.get_yaxis().set_tick_params(direction='out')
		ax2.get_xaxis().set_tick_params(direction='out')
		ax2.get_xaxis().tick_bottom()
		ax2.get_yaxis().tick_left()

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_h.keys())):
			plt.eventplot(all_raster_h[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))


		#plt_name = syncHDF_files[i][34:-15]
		#plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		plt_name = syncHDF_files[i][syncHDF_files[i].index('Luigi201'):-15]
		plt.savefig(dir_figs +plt_name+ '_' + str(align_to) +'_RPE_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		
		plt.close()

	reg_psth = [psth_l, psth_h]
	smooth_psth = [smooth_psth_l, smooth_psth_h]
	return reg_psth, smooth_psth, all_raster_l, all_raster_h

def TwoTargetTask_RegressFiringRates_Value(hdf_files, syncHDF_files, spike_files, t_before, t_after, t_resolution, t_overlap, smoothed, align_to, trial_first, trial_last):
	'''
	This method regresses the firing rate of all units as a function of value over time. 

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
	- t_resolution: float, time (s) for size of bins of firing rates
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)
	- align_to: indicates if aligning to hold_center or check_reward

	'''

	# 1. Get Q-values: taken as sliding average of reward history over 10 trials
	Smooth_Q_high, Smooth_Q_low, Q_high, Q_low = Value_from_reward_history_TwoTargetTask(hdf_files)

	# 2a. Load behavior data and pull out trial indices for the designated trial case.
	# 2b. Get firing rates from units on channel around time of target presentation on all trials.
	num_files = len(hdf_files)
	print("Number of files: ", num_files)
	'''
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	
	
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)
	'''
	# Define timing parameters for PSTHs
	num_timebins = int(np.rint((t_before + t_after - t_overlap)/(t_resolution - t_overlap)))

	# Define arrays to save psth for each trial
	smooth_avg_psth = dict()
	avg_psth = dict()
	

	'''
	Get data for each set of files
	'''
	keys = []

	for i in range(num_files):
		# Load behavior data
		cb = ChoiceBehavior_TwoTargets(hdf_files[i])
		#num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_check_reward = cb.ind_check_reward_states
		if align_to == 1:
			inds = ind_hold_center
		elif align_to == 2:
			inds = ind_check_reward

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data
			if (spike_files[i][0] != ''):
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				spike1_good_channels = spike1.sorted_good_chans_sc
				
				for chann in spike1_good_channels.keys():
					for sc in spike1_good_channels[chann]:
						psth, smooth_psth = spike1.compute_sliding_psth(chann, sc, times_row_ind,t_before,t_after,t_resolution,t_overlap)
						# Normalize firing rates by z-scoring
						psth = (psth - np.nanmean(psth))/np.nanstd(psth)
						smooth_psth = (smooth_psth - np.mean(smooth_psth))/np.std(smooth_psth)
						if i==0:
							avg_psth[str(chann)+'-'+str(sc)] = psth
							smooth_avg_psth[str(chann)+'-'+str(sc)] = smooth_psth
							keys += [str(chann)+'-'+str(sc)]
						elif (str(chann)+'-'+str(sc)) in keys:
							avg_psth[str(chann)+'-'+str(sc)] = np.vstack([avg_psth[str(chann)+'-'+str(sc)], psth])
							smooth_avg_psth[str(chann)+'-'+str(sc)] = np.vstack([smooth_avg_psth[str(chann)+'-'+str(sc)], smooth_psth])
			
			if (spike_files[i][1] != ''):
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				spike2_good_channels = spike2.sorted_good_chans_sc
				
				for chann in spike2_good_channels.keys():
					for sc in spike2_good_channels[chann]:
						psth, smooth_psth = spike2.compute_sliding_psth(chann, sc, times_row_ind,t_before,t_after,t_resolution,t_overlap)
						# Normalize firing rates by z-scoring
						psth = (psth - np.nanmean(psth))/np.nanstd(psth)
						smooth_psth = (smooth_psth - np.mean(smooth_psth))/np.std(smooth_psth)
						if i==0:
							avg_psth[str(chann)+'-'+str(sc)] = psth
							smooth_avg_psth[str(chann)+'-'+str(sc)] = smooth_psth
							keys += [str(chann)+'-'+str(sc)]
						elif (str(chann)+'-'+str(sc)) in keys:
							avg_psth[str(chann)+'-'+str(sc)] = np.vstack([avg_psth[str(chann)+'-'+str(sc)], psth])
							smooth_avg_psth[str(chann)+'-'+str(sc)] = np.vstack([smooth_avg_psth[str(chann)+'-'+str(sc)], smooth_psth])
			
			print('Number of units:',len(avg_psth))
			print('Size of psth:',avg_psth[str(chann)+'-'+str(sc)].shape)
			
		
	
	# 4. Do regression for each unit in each time bin.
	#    Current regression uses Q-values and constant.
	num_units = len(avg_psth.keys())
	beta_Q_low = np.zeros((num_units, num_timebins))
	beta_Q_high = np.zeros((num_units, num_timebins))
	p_Q_low = np.zeros((num_units, num_timebins))
	p_Q_high = np.zeros((num_units, num_timebins))
	
	for m, unit in enumerate(avg_psth.keys()):
		psth = avg_psth[unit]
		smooth_psth = smooth_avg_psth[unit]
		print(psth.shape)

		for k in range(num_timebins):
			unit_data = psth[:,k]
			#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)
			trial_start = trial_first
			trial_end = np.min([trial_last, psth.shape[0]])
			# look at all trial types within Blocks A and B
			trial_inds = np.array([index for index in range(trial_start,trial_end) if (unit_data[index]!=0)&(target_options[index,:].all())], dtype = int)
			x = np.vstack((Q_low[trial_inds], Q_high[trial_inds]))
			x = np.transpose(x)
			x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
			#x = sm.add_constant(x, prepend=False)
			y = unit_data[trial_inds]
			#y = y/np.max(y)  # normalize y

			#print("Regression for timebin ", k)
			model_glm = sm.OLS(y,x)
			fit_glm = model_glm.fit()
			beta_Q_low[m,k] = fit_glm.params[0]
			beta_Q_high[m,k] = fit_glm.params[1]
			p_Q_low[m,k] = fit_glm.pvalues[0]
			p_Q_high[m,k] = fit_glm.pvalues[1]
		#print fit_glm.summary()
	
	Q_low_sig = p_Q_low <= 0.05
	Q_high_sig = p_Q_high <= 0.05
	
	frac_Q_low = np.sum(Q_low_sig, axis = 0)/num_units
	frac_Q_high = np.sum(Q_high_sig, axis = 0)/num_units

	sem_denom_low = np.sum(Q_low_sig, axis = 0)
	sem_denom_high = np.sum(Q_high_sig, axis = 0)

	beta_Q_low_sig = beta_Q_low
	beta_Q_low_sig[np.nonzero(~Q_low_sig)] = np.nan # only keep significant values, replace non-significant value with nan
	mean_beta_low = np.nanmean(beta_Q_low_sig,axis = 0)
	sem_beta_low = np.nanstd(beta_Q_low_sig, axis = 0)/np.sqrt(sem_denom_low)

	beta_Q_high_sig = beta_Q_high
	beta_Q_high_sig[np.nonzero(~Q_high_sig)] = np.nan # only keep significant values, replace non-significant value with nan
	mean_beta_high = np.nanmean(beta_Q_high_sig,axis = 0)
	sem_beta_high = np.nanstd(beta_Q_high_sig, axis = 0)/np.sqrt(sem_denom_high)

	time_values = np.arange(-t_before,t_after,float(t_after + t_before)/num_timebins)
	print('block 1 trials')
	plt.figure()
	ax = plt.subplot(111)
	plt.plot(time_values, frac_Q_low, 'r', label = 'LV')
	plt.plot(time_values, frac_Q_high, 'b', label = 'HV')
	plt.xlabel('Time from Center Hold (s) ')
	plt.ylabel('Fraction of Neurons')
	plt.title('Encoding of Value over Time')
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.legend()
	ax = plt.subplot(121)
	plt.plot(time_values, mean_beta_low, 'r', label = 'LV')
	plt.fill_between(time_values,mean_beta_low - sem_beta_low, mean_beta_low + sem_beta_low, facecolor = 'r', alpha = 0.25)
	plt.plot(time_values, mean_beta_high, 'b', label = 'HV')
	plt.fill_between(time_values,mean_beta_high - sem_beta_high, mean_beta_high + sem_beta_high, facecolor = 'b', alpha = 0.25)
	plt.xlabel('Time from Center Hold (s)')
	plt.ylabel('Beta Coefficient')
	plt.title('Significant Betas over Time')
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.legend()
	plt_name = syncHDF_files[i][syncHDF_files[i].index('Luigi201'):-15]
	plt.savefig(dir_figs +plt_name+ '_FractionEncodingValOverTimee_trials%i-%i.png' % (trial_first, trial_last))
	plt.savefig(dir_figs +plt_name+ '_FractionEncodingValOverTimee_trials%i-%i.svg' % (trial_first, trial_last))
	plt.clf()
	'''
	plt.figure()
	ax = plt.subplot(111)
	plt.plot(time_values, beta_Q_low, 'r', alpha = 0.2)
	plt.plot(time_values, beta_Q_low_sig, 'r', label = 'LV')
	plt.plot(time_values, beta_Q_high, 'b', alpha = 0.2)
	plt.plot(time_values, beta_Q_high_sig, 'b', label = 'HV')
	plt.xlabel('Time from Center Hold (s) ')
	plt.ylabel('Beta coefficient')
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.legend()
	plt.show()
	
	return time_values, beta_Q_low, beta_Q_high, p_Q_low, p_Q_high, fit_glm
	'''
	return time_values, frac_Q_low, frac_Q_high, Q_low_sig, Q_high_sig

def TwoTargetTask_FiringRates_OverTime(hdf_files, syncHDF_files, spike_files, channel, sc, t_before, t_after, align_to, bin_size, bin_overlap):
	'''
	Method that gets firing rates from units on indicated channel around time of target presentation (hold_center) or reward (check_reward) 
	on all trials. Provides firing rates over time for the duration of the time window indicated. 

	Inputs:
	- hdf_files: list of N hdf_files corresponding to the behavior in the two target task
	- syncHDF_files: list of N syncHDF_files that containes the syncing DIO data for the corresponding hdf_file and it's
					TDT recording. If TDT data does not exist, an empty entry should strill be entered. I.e. if there is data for the first
					epoch of recording but not the second, syncHDF_files should have the form [syncHDF_file1.mat, '']
	- spike_files: list of N tuples of spike_files, where each entry is a list of 2 spike files, one corresponding to spike
					data from the first 96 channels and the other corresponding to the spike data from the last 64 channels.
					If spike data does not exist, an empty entry should strill be entered. I.e. if there is data for the first
					epoch of recording but not the second, the hdf_files and syncHDF_files will both have 2 file names, and the 
					spike_files entry should be of the form [[spike_file1.csv, spike_file2.csv], ''].
	- t_before: time before (s) the align_to point that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of align_to forward when considering the window of activity.
	- t_after: time after (s) the align_to point that should be included when computing the firing rate.
	- align_to: string, either 'hold_center' or 'check_reward', indicating what time point in the task we're aligned to
	- bin_size: float representing the size of the spike bins (s)
	- bin_overlap: float representing the amount that bins overlap (s)

	Output:
	- psth: 2D array of size T x M, where T is the number of trials and M is the number of bins over the time range [t_before, t_after]
	'''
	num_trials = np.zeros(len(hdf_files))

	for i, hdf_file in enumerate(hdf_files):
		print("Open HDF file number %f" % (i + 1))
		# 1. Load behavior data and pull out trial indices for the designated trial case
		print("Loading behavior")
		cb_block = ChoiceBehavior_TwoTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials

		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_check_reward = cb_block.ind_check_reward_states
		if align_to == b'hold_center':
			inds = ind_hold_center
		else:
			inds = ind_check_reward

		# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
		lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
		# Convert lfp sample numbers to times in seconds
		times_row_ind = lfp_state_row_ind/float(lfp_freq)

		print("Getting spike data")
		print("Number of trials:%i" % (len(inds)))
		# 2. Get spike data for channel indicated
		if channel < 97:
			if (spike_files[i] != ''):
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				psth = spike1.compute_sliding_psth(channel,sc,inds,t_before,t_after,bin_size, bin_overlap)
				#psth = spike1.compute_psth(channel,sc,inds,t_before,t_after,bin_size)
			else:
				psth = nd.array([])
		else:		
			if spike_files[i][1] != '':
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				psth = spike2.compute_sliding_psth(channel,sc,inds,t_before,t_after,bin_size, bin_overlap)
			else:
				psth = nd.array([])

		if i == 0:
			all_psth = psth
		else:
			all_psth = np.vstack([all_psth, psth])

	return psth, all_psth, num_trials

def ThreeTargetTask_DecodeChoice_LogisticRegression(hdf_files, syncHDF_files, spike_files, t_before, t_after, align_to, bin_size, bin_overlap):
	'''
	This method performs logistic regression of the subject's choice on free-choice trials where the LV and MV are presented
	together. Only free-choice trials, where there is a choice to decode, are considered. 

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
	- t_before: time before (s) the align_to point that should be included when computing the firing rate. t_before = 0 indicates
					that we only look from the time of align_to forward when considering the window of activity.
	- t_after: time after (s) the align_to point that should be included when computing the firing rate.
	- align_to: string, either 'hold_center' or 'check_reward', indicating what time point in the task we're aligned to
	- bin_size: float representing the size of the spike bins (s)
	- bin_overlap: float representing the amount that bins overlap (s)

	Output:

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Mario' + hdf_files[0][str_ind:str_ind + 8]
	print(sess_name)
	# Define variables
	choices = np.array([])
	num_trials = np.zeros(len(hdf_files))
	all_fr_smooth = dict()
 
	for i, hdf_file in enumerate(hdf_files):
		print("Open HDF file number %f" % (i + 1))
		# 1. Load behavior data and pull out trial indices for the designated trial case
		print("Loading behavior")
		cb_block = ChoiceBehavior_ThreeTargets(hdf_file)
		num_trials[i] = cb_block.num_successful_trials
		print("Getting choice information")
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb_block.GetChoicesAndRewards()
		ind_fc_LVMV = np.array([ind for ind in range(int(num_trials[i])) if np.array_equal(targets_on[ind],[1,1,0])])
		chosen_target_fc_LVMV = chosen_target[ind_fc_LVMV]
		choices = np.append(choices, chosen_target_fc_LVMV)

		ind_hold_center = cb_block.ind_check_reward_states - 4
		ind_check_reward = cb_block.ind_check_reward_states
		if align_to == b'hold_center':
			inds = ind_hold_center[ind_fc_LVMV]
		else:
			inds = ind_check_reward[ind_fc_LVMV]

		print("Getting spike data")
		# 2. Get all spike data
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb_block.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			# Load spike data and find all sort codes associated with good channels
			unit_counter = 0
			if spike_files[i][0] != '':
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				spike1_good_channels = spike1.good_channels
				for chann in spike1_good_channels:
					sc_chan = spike1.find_chan_sc(chann)
					for j, sc in enumerate(sc_chan):
						psth = spike1.compute_sliding_psth(chann,sc,inds,t_before,t_after,bin_size, bin_overlap)
						# unit keys, array is trials x timepoints
						unit_name = str(chann) + '_' + str(sc)
						if unit_name in all_fr_smooth.keys():
							all_fr_smooth[str(chann) + '_' + str(sc)] = np.vstack([all_fr_smooth[str(chann) + '_' + str(sc)], psth])
						else:
							all_fr_smooth[str(chann) + '_' + str(sc)] = psth
			
			if spike_files[i][1] != '':
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				spike2_good_channels = spike2.good_channels
				for chann in spike2_good_channels:
					sc_chan = spike2.find_chan_sc(chann)
					for j, sc in enumerate(sc_chan):
						psth = spike2.compute_sliding_psth(chann,sc,inds,t_before,t_after,bin_size, bin_overlap)
						# unit keys, array is trials x timepoints
						unit_name = str(chann+96) + '_' + str(sc)
						if unit_name in all_fr_smooth.keys():
							all_fr_smooth[str(chann+96) + '_' + str(sc)] = np.vstack([all_fr_smooth[str(chann+96) + '_' + str(sc)], psth])
						else:
							all_fr_smooth[str(chann+96) + '_' + str(sc)] = psth

	print("Reformatting spike data"		)
	# 3. Re-format spike data into 3-dim array of units x trials x timepoints
	num_keys = np.shape(all_fr_smooth.keys())[0]
	num_time_bins = psth.shape[1]
	all_trials = len(choices)
	fr_mat = np.zeros([num_keys, all_trials,num_time_bins])

	for k, entry in enumerate(all_fr_smooth.keys()):
		fr_mat[k,:,:] = all_fr_smooth[entry]
	
	print("Starting logistic regression")
	# 4. Do logistic regression for each timepoint
	for l in range(num_time_bins)[1:]:
		# 3. Create firing rate matrix with size units x trials for each time point
		x = fr_mat[:,:,l]
		# switch to trials x units
		x = np.transpose(x)
		x = np.hstack((x, np.ones([all_trials,1]))) 	
		
		# 4. Do regression for each bin 
		y = choices
		print(np.sum(x,1))

		print("Regression for bin ", l)
		#model_glm = sm.OLS(y,x, family = sm.families.Binomial())
		model_glm = sm.GLM(y,x, family = sm.families.Binomial())
		fit_glm = model_glm.fit()
		print(fit_glm.summary())

	return choices

def Compare_QValue_Models_ThreeTarget(hdf_files):
	'''
	This method models the Q-values derived from three different Q-learning models and compares the results for data
	from a three-target task. The first model assumes that a fixed alpha and beta parameter pair are used for all 
	contingencies, while the second model assumes that there are different learning rates and inverse temperatures
	depending on what two of the three targets are presented. One major difference is that for the latter method, instructed trials are not used to update values since this
	would require additional parameter choices. The third model assumes that there is a separate learning rate for 
	each stimulus, but the same inverse temperature.

	Inputs:
	- hdf_files: list of N hdf_files corresponding to the behavior in the three target task
	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Mario' + hdf_files[0][str_ind:str_ind + 8]

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	
	# 2. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
	# 3. Fit with first model where there is a single alpha and beta used for all contingencies
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150]), bounds=[(0.01,1),(0.01,None)])
	alpha_ml, beta_ml = result["x"]
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_likelihood = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150])
	

	# 4. Fit with second model where there are separate parameters used for each contingency.
	# Find ML fit of alpha and beta
	nll2 = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_sep_parameters(*args)
	result2 = op.minimize(nll2, [0.2,0.2,0.2,1,1,1], args=(Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150]), bounds=[(0.01,1),(0.01,1),(0.01,1),(0.01,None),(0.01,None),(0.01,None)])
	parameters_ml = result2["x"]
	# RL model fit for Q values
	Q_low2, Q_mid2, Q_high2, prob_choice_opt_lvmv2, prob_choice_opt_lvhv2, prob_choice_opt_mvhv2, accuracy2, log_likelihood2 = ThreeTargetTask_Qlearning_sep_parameters(parameters_ml, Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150])
	

	# 5. Fit with third model where there are separate alphas for each value.
	# Find ML fit of alpha and beta
	nll3 = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_ind_parameters(*args)
	result3 = op.minimize(nll3, [0.2,0.2,0.2,1], args=(Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150]), bounds=[(0.01,1),(0.01,1),(0.01,1),(0.01,None)])
	parameters_ml = result3["x"]
	# RL model fit for Q values
	Q_low3, Q_mid3, Q_high3, prob_choice_opt_lvmv3, prob_choice_opt_lvhv3, prob_choice_opt_mvhv3, accuracy3, log_likelihood3 = ThreeTargetTask_Qlearning_ind_parameters(parameters_ml, Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150])
	

	# 5. Compute BIC and accuracy.
	BIC = -2*log_likelihood + len(result["x"])*np.log(len(Q_mid))
	BIC2 = -2*log_likelihood2 + len(result2["x"])*np.log(len(Q_mid2))
	BIC3 = -2*log_likelihood3 + len(result3["x"])*np.log(len(Q_mid3))
	print("Accuracy Model 1 = %0.2f, BIC = %0.2f" % (np.mean(accuracy), BIC))
	print("Accuracy Model 2 = %0.2f, BIC = %0.2f" % (np.mean(accuracy2), BIC2))
	print("Accuracy Model 3 = %0.2f, BIC = %0.2f" % (np.mean(accuracy3), BIC3))

	# 6. Compare to real behavior
	num_trials_slide = 10
	plot_results = False
	all_choices_A, LM_choices_A, LH_choices_A, MH_choices_A, all_choices_Aprime, \
		LM_choices_Aprime, LH_choices_Aprime, MH_choices_Aprime = cb.TrialChoices(num_trials_slide, plot_results)


	LM_choices = trial_sliding_avg(np.append(LM_choices_A, LM_choices_Aprime), num_trials_slide)
	LH_choices = trial_sliding_avg(np.append(LH_choices_A, LH_choices_Aprime), num_trials_slide)
	MH_choices = trial_sliding_avg(np.append(MH_choices_A, MH_choices_Aprime), num_trials_slide)

	plt.figure()
	plt.subplot(2,3,1)
	plt.plot(Q_low,'r',label='Fixed parameters')
	plt.plot(Q_low2,'b',label='Separate parameters')
	plt.plot(Q_low3,'m', label = 'Individual parameters')
	plt.legend()
	plt.ylabel('Q_low')
	plt.xlabel('Trials')
	plt.subplot(2,3,2)
	plt.plot(Q_mid,'r',label='Fixed parameters')
	plt.plot(Q_mid2,'b',label='Separate parameters')
	plt.plot(Q_mid3,'m', label = 'Individual parameters')
	plt.legend()
	plt.ylabel('Q_med')
	plt.xlabel('Trials')
	plt.subplot(2,3,3)
	plt.plot(Q_high,'r',label='Fixed parameters')
	plt.plot(Q_high2,'b',label='Separate parameters')
	plt.plot(Q_high3,'m', label = 'Individual parameters')
	plt.legend()
	plt.ylabel('Q_high')
	plt.xlabel('Trials')

	plt.subplot(2,3,4)
	plt.plot(prob_choice_opt_lvhv,'r',label='Fixed parameters')
	plt.plot(prob_choice_opt_lvhv2,'b',label='Separate parameters')
	plt.plot(prob_choice_opt_lvhv3,'m', label = 'Individual parameters')
	plt.plot(LH_choices, 'k', label = 'Behavior')
	plt.legend()
	plt.ylabel('P(Choose HV over LV)')
	plt.xlabel('Trials')
	plt.subplot(2,3,5)
	plt.plot(prob_choice_opt_lvmv,'r',label='Fixed parameters')
	plt.plot(prob_choice_opt_lvmv2,'b',label='Separate parameters')
	plt.plot(prob_choice_opt_lvmv3,'m', label = 'Individual parameters')
	plt.plot(LM_choices, 'k', label = 'Behavior')
	plt.legend()
	plt.ylabel('P(Choose MV over LV)')
	plt.xlabel('Trials')
	plt.subplot(2,3,6)
	plt.plot(prob_choice_opt_mvhv,'r',label='Fixed parameters')
	plt.plot(prob_choice_opt_mvhv2,'b',label='Separate parameters')
	plt.plot(prob_choice_opt_mvhv3,'m', label = 'Individual parameters')
	plt.plot(MH_choices, 'k', label = 'Behavior')
	plt.legend()
	plt.ylabel('P(Choose HV over MV)')
	plt.xlabel('Trials')
	plt.show()

	return BIC, BIC2, BIC3, np.mean(accuracy), np.mean(accuracy2), np.mean(accuracy3)

def Compare_Qlearning_across_blocks(hdf_files):
	'''
	This method models the Q-values derived from the standard Q-learning model with a single alpha and
	beta parameter for data from a three-target task. The first model is fit from data in Block A,
	while the second model is fit with data from Block A', with initial Q-values taken from the end of Block A.
	
	Inputs:
	- hdf_files: list of N hdf_files corresponding to the behavior in the three target task
	'''

	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Mario' + hdf_files[0][str_ind:str_ind + 8]

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	total_trials = cb.num_successful_trials
	
	# 2. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
	# 3. Fit first model where there is a single alpha and beta used for all contingencies. Do for Block A.
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150]), bounds=[(0.01,1),(0.01,None)])
	alpha_ml, beta_ml = result["x"]
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_likelihood = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target[:150], rewards[:150], targets_on[:150], instructed_or_freechoice[:150])
	Q_chosen = Q_low*np.equal(chosen_target[:150],0) + Q_mid*np.equal(chosen_target[:150],1) + Q_high*np.equal(chosen_target[:150],2)
	RPE = rewards[:150] - Q_chosen


	#4. Re-fit model for Block A'.
	Q_initial = np.array([Q_low[-1], Q_mid[-1], Q_high[-1]])
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:], instructed_or_freechoice[250:]), bounds=[(0.01,1),(0.01,None)])
	alpha_ml, beta_ml = result["x"]
	# RL model fit for Q values
	Q_low2, Q_mid2, Q_high2, prob_choice_opt_lvmv2, prob_choice_opt_lvhv2, prob_choice_opt_mvhv2, accuracy2, log_likelihood2 = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:], instructed_or_freechoice[250:])
	Q_chosen = Q_low2*np.equal(chosen_target[250:],0) + Q_mid2*np.equal(chosen_target[250:],1) + Q_high2*np.equal(chosen_target[250:],2)
	RPE2 = rewards[250:] - Q_chosen

	return Q_mid, Q_mid2, RPE, RPE2

def TwoTargetTask_RegressedFiringRatesWithValue_PictureOnset(dir, hdf_files, syncHDF_files, spike_files, channel, t_before, t_after, smoothed):
	'''
	8/15/19 - Updated to use vales determined from smoothed reward history (4441-4451) and to use the picture onset as the behavioral time
	point to align neural activity to (rather than picture onset, before hold begins) (4433)

	This method regresses the firing rate of all units as a function of value. It then plots the firing rate as a function of 
	the modeled value and uses the regression coefficient to plot a linear fit of the relationship between value and 
	firing rate.

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
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Luigi' + hdf_files[0][str_ind:str_ind + 8]
	if syncHDF_files[0]!='':
		str_ind = syncHDF_files[0].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	elif syncHDF_files[1]!='':
		str_ind = syncHDF_files[1].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	else:
		session_name = 'Unknown'
	

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 1a. Get reaction time information
	rt = np.array([])
	for file in hdf_files:
		print(file)
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(file)
		rt = np.append(rt, reaction_time)

	# 1b. Get movementment time information
	mt = (cb.state_time[cb.ind_target_states + 1] - cb.state_time[cb.ind_target_states])/60.

	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr, window_fr_smooth, unit_class = TwoTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials).astype(int)
	print(window_fr)

	# 3. Get chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
	# Varying Q-values
	"""
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(2)
	nll = lambda *args: -logLikelihoodRLPerformance(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, rewards, chosen_target, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	Q_low, Q_high, prob_choice_low, log_likelihood = RLPerformance([alpha_ml, beta_ml], Q_initial, rewards, chosen_target, instructed_or_freechoice)
	"""
	Q_high, Q_low = Value_from_reward_history_TwoTargetTask(hdf_files)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 

		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	
	# 5. Do regression for each unit only on trials in Blocks A and B with spike data saved.
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A 
		trial_inds = np.array([index for index in range(33,100) if unit_data[index]!=0], dtype = int)
		
		x = np.vstack((Q_low[trial_inds], Q_high[trial_inds]))
		x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		# include which targets were shown
		#x = np.vstack((x, targets_on[:,0][trial_inds], targets_on[:,2][trial_inds], targets_on[:,1][trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)
		
		#y = y/np.max(y)  # normalize y

		# save Q and firing rate data
		Q_learning = Q_low[trial_inds]
		FR_learning = y
		Q_low_BlockA = Q_low[trial_inds]
		data = dict()
		
		# Use the below statement to only perform analysis if unit had non-zero firing on at least 10 trials
		#if len(trial_inds) > 9:
		try:
			#print("Regression for unit ", k)
			model_glm = sm.OLS(y_zscore,x)
			fit_glm = model_glm.fit()
			#print(fit_glm.summary())

			regress_coef = fit_glm.params[0] 		# The regression coefficient for Qlow is the first parameter
			regress_intercept = y[0] - regress_coef*Q_low[trial_inds[0]]

			# Get linear regression fit for just Q_low
			Q_low_min = np.amin(Q_low[trial_inds])
			Q_low_max = np.amax(Q_low[trial_inds])
			x_lin = np.linspace(Q_low_min, Q_low_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_low[trial_inds],y, c= 'k', marker = 'o', label ='Learning Trials')
			plt.plot(x_lin, m*x_lin + b, c = 'k')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'y')
			plt.xlabel('Q_low')
			plt.ylabel('Firing Rate (spk/s)')
			plt.title(sess_name + ' - Channel %i - Unit %i' %(channel, k))
			'''

			max_fr = np.amax(y)
			xlim_min = np.amin(Q_low[trial_inds])
			xlim_max = np.amax(Q_low[trial_inds])

			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data = dict()
			data['regression_labels'] = ['Q_low', 'Q_high','RT', 'MT', 'Choice', 'Reward']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['Q_low_early'] = Q_low_BlockA
			data['FR_early'] = FR_learning
			sp.io.savemat( dir + 'hold_center_fr/' + data_filename + '.mat', data)


			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			num_bins = 5
			sorted_Qvals_inds = np.argsort(Q_low[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_low[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			reorg_FR_BlockA = reorg_FR
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			# Save data for binning by bins of fixed size (rather than equally populated)
			Q_range_min = np.min(Q_low[trial_inds])
			Q_range_max = np.max(Q_low[trial_inds])
			FR_BlockA = y
			
			'''
			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.legend()
			'''
		except:
			pass
		
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A and B
		trial_inds = np.array([index for index in range(200,len(unit_data)) if unit_data[index]!=0], dtype = int)
		#trial_inds = np.array([index for index in range(250,len(unit_data))], dtype = int)
		x = np.vstack((Q_low[trial_inds], Q_high[trial_inds]))
		x = np.vstack((x, rt[trial_inds], mt[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		# include which targets were shown
		#x = np.vstack((x, targets_on[:,0][trial_inds], targets_on[:,2][trial_inds], targets_on[:,1][trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)
		
		#y = y/np.max(y)  # normalize y

		try:
			#print("Regression for unit ", k)
			model_glm_late = sm.OLS(y_zscore,x)
			fit_glm_late = model_glm_late.fit()
			#print(fit_glm_late.summary())

			regress_coef = fit_glm_late.params[0] 		# The regression coefficient for Qlow is the first parameter
			regress_intercept = y[0] - regress_coef*Q_low[trial_inds[0]]

			# Get linear regression fit for just Q_mid
			Q_low_min = np.amin(Q_low[trial_inds])
			Q_low_max = np.amax(Q_low[trial_inds])
			x_lin = np.linspace(Q_low_min, Q_low_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			max_fr_stim = np.amax(y)
			#fr_lim = np.maximum(max_fr, max_fr_stim)
			fr_lim = max_fr_stim

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_low[trial_inds],y, c= 'c', label = 'Stimulation trials')
			plt.plot(x_lin, m*x_lin + b, c = 'c')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'g')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_low_min, 1.1*Q_low_max))
			plt.legend()
			'''
			# save Q and firing rate data
			Q_late = Q_low[trial_inds]
			FR_late = y
			"""
			# Get binned firing rates: bins of fixed size
			#Q_range_min = np.min(np.min(Q_late), Q_range_min)
			#Q_range_max = np.max(np.max(Q_late), Q_range_max)
			Q_range_min = np.min(Q_low[trial_inds])
			Q_range_max = np.max(Q_low[trial_inds])
			bins = np.arange(Q_range_min, Q_range_max + 0.5*(Q_range_max - Q_range_min)/5., (Q_range_max - Q_range_min)/5.)
			hist_BlockA, bins = np.histogram(Q_low_BlockA, bins)
			hist_late, bins = np.histogram(Q_late, bins)

			sorted_Qvals_inds_BlockA = np.argsort(Q_low_BlockA)
			sorted_Qvals_inds_late = np.argsort(Q_late)

			begin_BlockA = 0
			begin_late = 0
			dta_all = []
			avg_FR_BlockA = np.zeros(5)
			avg_FR_late = np.zeros(5)
			sem_FR_BlockA = np.zeros(5)
			sem_FR_late = np.zeros(5)
			for j in range(len(hist_BlockA)):
				data_BlockA = FR_BlockA[sorted_Qvals_inds_BlockA[begin_BlockA:begin_BlockA+hist_BlockA[j]]]
				data_late = FR_late[sorted_Qvals_inds_late[begin_late:begin_late+hist_late[j]]]
				begin_BlockA += hist_BlockA[j]
				begin_late += hist_late[j]

				avg_FR_BlockA[j] = np.nanmean(data_BlockA)
				sem_FR_BlockA[j] = np.nanstd(data_BlockA)/np.sqrt(len(data_BlockA))
				avg_FR_late[j] = np.nanmean(data_late)
				sem_FR_late[j] = np.nanstd(data_late)/np.sqrt(len(data_late))

				for item in data_BlockA:
					dta_all += [(j,0,item)]

				for item in data_late:
					dta_all += [(j,1,item)]

			dta_all = pd.DataFrame(dta_all, columns = ['Bin', 'Condition', 'FR'])
			bin_centers = (bins[1:] + bins[:-1])/2.
			print(len(bin_centers))
			print(len(avg_FR_late))

			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			sorted_Qvals_inds = np.argsort(Q_low[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_low[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR/2., fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()

			plt.figure(k)
			plt.subplot(1,3,3)
			plt.errorbar(bin_centers, avg_FR_BlockA, yerr = sem_FR_BlockA, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.errorbar(bin_centers, avg_FR_late, yerr = sem_FR_late, fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()
			"""

			# Save data
			#print('Saving data')
			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data['regression_labels'] = ['Q_low', 'Q_high','RT', 'MT', 'Choice', 'Reward']
			data['beta_values_blocksAB'] = fit_glm_late.params
			data['pvalues_blocksAB'] = fit_glm_late.pvalues
			data['rsquared_blocksAB'] = fit_glm_late.rsquared
			data['Q_low_late'] = Q_late
			data['FR_late'] = FR_late
			#sp.io.savemat( dir + 'picture_onset_fr/' + data_filename + '.mat', data)
			sp.io.savemat( dir + 'hold_center_fr/' + data_filename + '.mat', data)
		except:
			pass

		
	#return window_fr, window_fr_smooth, fr_mat, x, y, Q_low, Q_mid, Q_high, Q_learning, Q_late, FR_learning, FR_late, fit_glm
	return Q_low, Q_high, fit_glm_late

def TwoTargetTask_RegressedFiringRatesWithRPE_RewardOnset(dir, hdf_files, syncHDF_files, spike_files, channel, t_before, t_after, smoothed):
	'''
	This method regresses the firing rate of all units as a function of RPE. It then plots the firing rate as a function of 
	the modeled RPE and uses the regression coefficient to plot a linear fit of the relationship between RPE and 
	firing rate.

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
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Luigi' + hdf_files[0][str_ind:str_ind + 8]
	if syncHDF_files[0]!='':
		str_ind = syncHDF_files[0].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	elif syncHDF_files[1]!='':
		str_ind = syncHDF_files[1].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	else:
		session_name = 'Unknown'
	

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]


	# 2. Get firing rates from units on indicated channel around time of target presentation on all trials. Note that
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding
	#	to the average firing rate over the window indicated.
	num_trials, num_units, window_fr, window_fr_smooth, unit_class = TwoTargetTask_FiringRates_RewardOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials).astype(int)
	print(window_fr)

	# 3. Get chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
	# Varying Q-values
	"""
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(2)
	nll = lambda *args: -logLikelihoodRLPerformance(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, rewards, chosen_target, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	Q_low, Q_high, prob_choice_low, log_likelihood = RLPerformance([alpha_ml, beta_ml], Q_initial, rewards, chosen_target, instructed_or_freechoice)
	"""
	Q_high, Q_low = Value_from_reward_history_TwoTargetTask(hdf_files)

	# 3a. Compute Positive and Negative RPEs
	Q_mat = np.vstack([Q_low, Q_high])
	chosen_target = np.array(chosen_target,dtype = int)   	# = 1 for LV, = 2 for HV
	chosen_target_value = np.array([Q_mat[chosen_target[i]-1,i] for i in range(len(chosen_target))])

	RPE = rewards - chosen_target_value
	RPE_positive = RPE*(RPE > 0)
	RPE_negative = RPE*(RPE < 0)

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 

		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	
	# 5. Do regression for each unit only on trials in Blocks A and B with spike data saved.
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A 
		trial_inds = np.array([index for index in range(33,100) if unit_data[index]!=0], dtype = int)
		
		x = np.vstack((RPE_positive[trial_inds], RPE_negative[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)
		
		#y = y/np.max(y)  # normalize y

		# save Q and firing rate data
		Q_learning = Q_low[trial_inds]
		FR_learning = y
		Q_low_BlockA = Q_low[trial_inds]
		data = dict()

		RPE_early = RPE[trial_inds]
		
		# Use the below statement to only perform analysis if unit had non-zero firing on at least 10 trials
		#if len(trial_inds) > 9:
		try:
			#print("Regression for unit ", k)
			model_glm = sm.OLS(y_zscore,x)
			fit_glm = model_glm.fit()
			#print(fit_glm.summary())

			regress_coef = fit_glm.params[0] 		# The regression coefficient for Qlow is the first parameter
			regress_intercept = y[0] - regress_coef*Q_low[trial_inds[0]]

			# Get linear regression fit for just Q_low
			Q_low_min = np.amin(Q_low[trial_inds])
			Q_low_max = np.amax(Q_low[trial_inds])
			x_lin = np.linspace(Q_low_min, Q_low_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_low[trial_inds],y, c= 'k', marker = 'o', label ='Learning Trials')
			plt.plot(x_lin, m*x_lin + b, c = 'k')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'y')
			plt.xlabel('Q_low')
			plt.ylabel('Firing Rate (spk/s)')
			plt.title(sess_name + ' - Channel %i - Unit %i' %(channel, k))
			'''

			max_fr = np.amax(y)
			xlim_min = np.amin(Q_low[trial_inds])
			xlim_max = np.amax(Q_low[trial_inds])

			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data = dict()
			data['regression_labels'] = ['RPE_positive', 'RPE_negative', 'Choice', 'Reward']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['RPE_early'] = RPE_early
			data['FR_early'] = FR_learning
			sp.io.savemat( dir + 'check_reward_fr/' + data_filename + '.mat', data)
			


			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			num_bins = 5
			sorted_Qvals_inds = np.argsort(Q_low[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_low[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			reorg_FR_BlockA = reorg_FR
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			# Save data for binning by bins of fixed size (rather than equally populated)
			Q_range_min = np.min(Q_low[trial_inds])
			Q_range_max = np.max(Q_low[trial_inds])
			FR_BlockA = y
			
			'''
			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.legend()
			'''
		except:
			pass
		
		unit_data = fr_mat[k,:]
		#trial_inds = np.array([index for index in ind_trial_case if unit_data[index]!=0], dtype = int)

		# look at all trial types within Blocks A and B
		trial_inds = np.array([index for index in range(200,len(unit_data)) if unit_data[index]!=0], dtype = int)
		#trial_inds = np.array([index for index in range(250,len(unit_data))], dtype = int)
		x = np.vstack((RPE_positive[trial_inds], RPE_negative[trial_inds], chosen_target[trial_inds], rewards[trial_inds]))
		x = np.transpose(x)
		x = np.hstack((x, np.ones([len(trial_inds),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used
		#x = sm.add_constant(x, prepend=False)
		
		y = unit_data[trial_inds]
		# z-score y
		y_zscore = stats.zscore(y)
		
		#y = y/np.max(y)  # normalize y

		try:
			#print("Regression for unit ", k)
			model_glm_late = sm.OLS(y_zscore,x)
			fit_glm_late = model_glm_late.fit()
			#print(fit_glm_late.summary())

			regress_coef = fit_glm_late.params[0] 		# The regression coefficient for Qlow is the first parameter
			regress_intercept = y[0] - regress_coef*Q_low[trial_inds[0]]

			# Get linear regression fit for just Q_mid
			Q_low_min = np.amin(Q_low[trial_inds])
			Q_low_max = np.amax(Q_low[trial_inds])
			x_lin = np.linspace(Q_low_min, Q_low_max, num = len(trial_inds), endpoint = True)

			m,b = np.polyfit(x_lin, y, 1)

			max_fr_stim = np.amax(y)
			#fr_lim = np.maximum(max_fr, max_fr_stim)
			fr_lim = max_fr_stim

			'''
			plt.figure(k)
			plt.subplot(1,3,1)
			plt.scatter(Q_low[trial_inds],y, c= 'c', label = 'Stimulation trials')
			plt.plot(x_lin, m*x_lin + b, c = 'c')
			#plt.plot(Q_mid[trial_inds], regress_coef*Q_mid[trial_inds] + regress_intercept, c = 'g')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_low_min, 1.1*Q_low_max))
			plt.legend()
			'''
			# save Q and firing rate data
			Q_late = Q_low[trial_inds]
			FR_late = y
			RPE_late = RPE[trial_inds]
			"""
			# Get binned firing rates: bins of fixed size
			#Q_range_min = np.min(np.min(Q_late), Q_range_min)
			#Q_range_max = np.max(np.max(Q_late), Q_range_max)
			Q_range_min = np.min(Q_low[trial_inds])
			Q_range_max = np.max(Q_low[trial_inds])
			bins = np.arange(Q_range_min, Q_range_max + 0.5*(Q_range_max - Q_range_min)/5., (Q_range_max - Q_range_min)/5.)
			hist_BlockA, bins = np.histogram(Q_low_BlockA, bins)
			hist_late, bins = np.histogram(Q_late, bins)

			sorted_Qvals_inds_BlockA = np.argsort(Q_low_BlockA)
			sorted_Qvals_inds_late = np.argsort(Q_late)

			begin_BlockA = 0
			begin_late = 0
			dta_all = []
			avg_FR_BlockA = np.zeros(5)
			avg_FR_late = np.zeros(5)
			sem_FR_BlockA = np.zeros(5)
			sem_FR_late = np.zeros(5)
			for j in range(len(hist_BlockA)):
				data_BlockA = FR_BlockA[sorted_Qvals_inds_BlockA[begin_BlockA:begin_BlockA+hist_BlockA[j]]]
				data_late = FR_late[sorted_Qvals_inds_late[begin_late:begin_late+hist_late[j]]]
				begin_BlockA += hist_BlockA[j]
				begin_late += hist_late[j]

				avg_FR_BlockA[j] = np.nanmean(data_BlockA)
				sem_FR_BlockA[j] = np.nanstd(data_BlockA)/np.sqrt(len(data_BlockA))
				avg_FR_late[j] = np.nanmean(data_late)
				sem_FR_late[j] = np.nanstd(data_late)/np.sqrt(len(data_late))

				for item in data_BlockA:
					dta_all += [(j,0,item)]

				for item in data_late:
					dta_all += [(j,1,item)]

			dta_all = pd.DataFrame(dta_all, columns = ['Bin', 'Condition', 'FR'])
			bin_centers = (bins[1:] + bins[:-1])/2.
			print(len(bin_centers))
			print(len(avg_FR_late))

			# Get binned firing rates: average firing rate for each of num_bins equally populated action value bins
			sorted_Qvals_inds = np.argsort(Q_low[trial_inds])
			pts_per_bin = len(trial_inds)/num_bins
			reorg_Qvals = np.reshape(Q_low[trial_inds][sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_Qvals = np.nanmean(reorg_Qvals, axis = 0)

			reorg_FR = np.reshape(y[sorted_Qvals_inds[:pts_per_bin*num_bins]], (pts_per_bin, num_bins), order = 'F')
			avg_FR = np.nanmean(reorg_FR, axis = 0)
			sem_FR = np.nanstd(reorg_FR, axis = 0)/np.sqrt(pts_per_bin)

			plt.figure(k)
			plt.subplot(1,3,2)
			plt.errorbar(avg_Qvals, avg_FR, yerr = sem_FR/2., fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()

			plt.figure(k)
			plt.subplot(1,3,3)
			plt.errorbar(bin_centers, avg_FR_BlockA, yerr = sem_FR_BlockA, fmt = '--o', color = 'k', ecolor = 'k', label = 'Learning - Avg FR')
			plt.errorbar(bin_centers, avg_FR_late, yerr = sem_FR_late, fmt = '--o', color = 'c', ecolor = 'c', label = 'Stim - Avg FR')
			plt.ylim((0,1.1*fr_lim))
			plt.xlim((0.9*Q_range_min, 1.1*Q_range_max))
			plt.legend()
			"""

			# Save data
			#print('Saving data')
			data_filename = session_name + ' - Channel %i - Unit %i' %(channel, k)
			data['regression_labels'] = ['RPE_positive', 'RPE_negative', 'Choice', 'Reward']
			data['beta_values_blockA'] = fit_glm.params
			data['pvalues_blockA'] = fit_glm.pvalues
			data['rsquared_blockA'] = fit_glm.rsquared
			data['RPE_early'] = RPE_early
			data['beta_values_blocksAB'] = fit_glm_late.params
			data['pvalues_blocksAB'] = fit_glm_late.pvalues
			data['rsquared_blocksAB'] = fit_glm_late.rsquared
			data['RPE_late'] = RPE_late
			data['FR_early'] = FR_BlockA
			data['FR_late'] = FR_late
			sp.io.savemat( dir1 + 'check_reward_fr/' + data_filename + '.mat', data)
			
		except:
			pass

		
	#return window_fr, window_fr_smooth, fr_mat, x, y, Q_low, Q_mid, Q_high, Q_learning, Q_late, FR_learning, FR_late, fit_glm
	return Q_low, Q_high

def TwoTargetTask_FiringRateChanges_FastVsSlow(dir, hdf_files, syncHDF_files, spike_files, channel, t_before, t_after, smoothed):
	'''
	This method categorizes neurons as either fast-firing or slow-firing

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
	- smoothed: boolean indicating whether to use smoothed firing rates (True) or not (False)

	'''
	# Get session information for plot
	str_ind = hdf_files[0].index('201')  	# search for beginning of year in string (used 201 to accomodate both 2016 and 2017)
	sess_name = 'Luigi' + hdf_files[0][str_ind:str_ind + 8]
	if syncHDF_files[0]!='':
		str_ind = syncHDF_files[0].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	elif syncHDF_files[1]!='':
		str_ind = syncHDF_files[1].index('201')
		session_name = 'Luigi' + syncHDF_files[0][str_ind:str_ind + 11]
	else:
		session_name = 'Unknown'
	

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 2. Get firing rates from units on indicated channel, classify as either fast or slow spiking, and then find average
	#	firing rate in first vs last block, and firing rates around picture onset for first vs last blocks.
	# 	window_fr is a dictionary with elements indexed such that the index matches the corresponding set of hdf_files. Each
	#	dictionary element contains a matrix of size (num units)x(num trials) with elements corresponding


	num_trials, num_units, window_fr, window_fr_smooth, unit_class = TwoTargetTask_FiringRates_PictureOnset(hdf_files, syncHDF_files, spike_files, channel, t_before, t_after)
	cum_sum_trials = np.cumsum(num_trials).astype(int)

	# 3. Get chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()

	# 4. Create firing rate matrix with size (max_num_units)x(total_trials)
	max_num_units = int(np.max(num_units))
	fr_mat = np.zeros([max_num_units, total_trials])
	trial_counter = 0
	for j in window_fr.keys():
		if not smoothed:
			block_fr = window_fr[j]
		else:
			block_fr = window_fr_smooth[j]
		if len(block_fr.shape) == 1:
			num_units = 1
			num_trials = len(block_fr)
		else:
			num_units,num_trials = block_fr.shape 

		fr_mat[:num_units,cum_sum_trials[j] - num_trials:cum_sum_trials[j]] = block_fr

	
	# 5. Do regression for each unit only on trials in Blocks A and B with spike data saved.
	avg_fr_modulation_index = np.zeros(max_num_units)
	for k in range(max_num_units):
		unit_data = fr_mat[k,:]
		trial_inds_early = np.array([index for index in range(100) if unit_data[index]!=0], dtype = int)
		trial_inds_late = np.array([index for index in range(200,len(unit_data)) if unit_data[index]!=0], dtype = int)

		avg_fr_modulation_index[k] = (np.nanmean(unit_data[trial_inds_late]) - np.nanmean(unit_data[trial_inds_early]))/(np.nanmean(unit_data[trial_inds_late]) + np.nanmean(unit_data[trial_inds_early]))
		
	return avg_fr_modulation_index, unit_class

def TwoTargetTask_SpikeAnalysis_PSTH_FactorAnalysis(hdf_files, syncHDF_files, spike_files, align_to, t_before, t_after, plot_output):
	'''

	This method aligns factor loadings from spike data to behavioral choices 
	in the Two Target Task, where there is a low-value and high-value target. This version does not 
	differentiate between choices in different blocks. It generates PSTHs using all re-sorted good channels.

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
	- align_to: integer in range [1,2] that indicates whether we align the (1) picture onset, a.k. center-hold or (2) check_reward.
	- plot_output: binary indicating whether output should be plotted + saved or not

	Outputs:
	- reg_psth: a 2 x N array, where the first row corresponds to the psth values for trials where the low-value 
				target was selected and the second row corresponds to the psth values for trials where the high-value
				target was selected
	- smooth_psth: a 2 x N array of the same format as reg_psth, but with values taken from psths smoothed with a 
				Gaussian kernel
	- all_raster_l: a dictionary with number of elements equal to the number of trials where the low-value target was selected,
				where each element contains spike times for the designated unit within that trial
	- all_raster_h: a dictionary with number of elements equal to the number of trials where the high-value target was selected,
				where each element contains spike times for the designated unit within that trial
	'''
	num_files = len(hdf_files)
	trials_per_file = np.zeros(num_files)
	num_successful_trials = np.zeros(num_files)

	# Define timing parameters for PSTHs
	t_resolution = 0.1 		# 100 ms time bins
	num_bins = len(np.arange(-t_before, t_after, t_resolution)) - 1

	# Define arrays to save psth for each trial
	smooth_psth_l = np.array([])
	smooth_psth_h = np.array([])
	psth_l = np.array([])
	psth_h = np.array([])

	'''
	Get data for each set of files
	'''
	for i in range(num_files):
		# Load behavior data
		print(hdf_files[i])
		cb = ChoiceBehavior_TwoTargets(hdf_files[i])
		num_successful_trials[i] = len(cb.ind_check_reward_states)
		target_options, target_chosen, rewarded_choice = cb.TrialOptionsAndChoice()

		# Find times corresponding to center holds of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4
		ind_check_reward = cb.ind_check_reward_states
		if align_to == 1:
			inds = ind_hold_center
		elif align_to == 2:
			inds = ind_check_reward

		# Load spike data: 
		if (spike_files[i] != ''):
			# Find lfp sample numbers corresponding to these times and the sampling frequency of the lfp data
			lfp_state_row_ind, lfp_freq = cb.get_state_TDT_LFPvalues(inds, syncHDF_files[i])
			# Convert lfp sample numbers to times in seconds
			times_row_ind = lfp_state_row_ind/float(lfp_freq)

			X1 = np.array([]) 		# placeholder
			X2 = np.array([])		# placeholder

			# Load spike data in 2D arrays
			if (spike_files[i][0] != ''):
				spike1 = OfflineSorted_CSVFile(spike_files[i][0])
				spike1_good_channels = spike1.sorted_good_chans_sc
				spike2_good_channels = []
				X1, unit_labels1 = spike1.bin_data(0.1)
				print(X1.shape)
			elif (spike_files[i][1] != ''):
				spike2 = OfflineSorted_CSVFile(spike_files[i][1])
				spike2_good_channels = spike2.sorted_good_chans_sc
				spike1_good_channels = []
				X2, unit_labels2 = spike2.bin_data(0.1)
				print(X2.shape)

			# Combine arrays if necessary
			if (np.any(X1) & np.any(X2)):
				X = np.vstack([X1, X2])
			elif np.any(X1):
				X = X1
			elif np.any(X2):
				X = X2

			print(X.shape)

			####################
			# Do factor analysis
			####################
			if not spike2_good_channels:
				fa = FactorAnalyzer(n_factors = len(spike1_good_channels)-1, rotation = None, method = 'ml')
				fa.fit(spike1)
			else:
				fa = FactorAnalyzer(n_factors = len(spike2_good_channels)-1, rotation = None, method = 'ml')
				fa.fit(spike2)
			
			ev, v = fa.get_eigenvalues()
			variance, proportionalVariance, cumulativeVariance = fa.get_factor_variance()

			"""
			# 2. L chosen
			L_ind = np.ravel(np.nonzero([np.array_equal(target_chosen[j,:], [1,0]) for j in range(int(num_successful_trials[i]))]))
			

			if not spike2_good_channels:
				avg_psth_l, smooth_avg_psth_l = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			else:
				avg_psth_l, smooth_avg_psth_l = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[L_ind],t_before,t_after,t_resolution)
				raster_l = spike2.compute_raster(spike1_good_channels, sc, times_row_ind[L_ind],t_before,t_after)
			if i == 0:
				psth_l = avg_psth_l
				smooth_psth_l = smooth_avg_psth_l
				all_raster_l = raster_l
			else:
				psth_l = np.vstack([psth_l, avg_psth_l])
				smooth_psth_l = np.vstack([smooth_psth_l, smooth_avg_psth_l])
				all_raster_l.update(raster_l)

			# 5. H chosen
			H_ind = np.ravel(np.nonzero([np.array_equal(target_chosen[j,:], [0,1]) for j in range(int(num_successful_trials[i]))]))
			if not spike2_good_channels:
				avg_psth_h, smooth_avg_psth_h = spike1.compute_psth(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike1.compute_raster(spike1_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			else:
				avg_psth_h, smooth_avg_psth_h = spike2.compute_psth(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after,t_resolution)
				raster_h = spike2.compute_raster(spike2_good_channels, sc, times_row_ind[H_ind],t_before,t_after)
			if i == 0:
				psth_h = avg_psth_h
				smooth_psth_h = smooth_avg_psth_h
				all_raster_h = raster_h
			else:
				psth_h = np.vstack([psth_h, avg_psth_h])
				smooth_psth_h = np.vstack([smooth_psth_h, smooth_avg_psth_h])
				all_raster_h.update(raster_h)

	# Plot average rate for all neurons divided in six cases of targets on option
	if plot_output:
		plt.figure(0)
		b = signal.gaussian(39,0.6)
		avg_psth_l = np.nanmean(psth_l, axis = 0)
		sem_avg_psth_l = np.nanstd(psth_l, axis = 0)/np.sqrt(psth_l.shape[0])
		#smooth_avg_psth_l = np.nanmean(smooth_psth_l, axis = 0)
		smooth_avg_psth_l = filters.convolve1d(np.nanmean(psth_l,axis=0), b/b.sum())
		sem_smooth_avg_psth_l = np.nanstd(smooth_psth_l, axis = 0)/np.sqrt(smooth_psth_l.shape[0])

		avg_psth_h = np.nanmean(psth_h, axis = 0)
		sem_avg_psth_h = np.nanstd(psth_h, axis = 0)/np.sqrt(psth_h.shape[0])
		smooth_avg_psth_h = np.nanmean(smooth_psth_h, axis = 0)
		smooth_avg_psth_h = filters.convolve1d(np.nanmean(psth_h,axis=0), b/b.sum())
		sem_smooth_avg_psth_h = np.nanstd(smooth_psth_h, axis = 0)/np.sqrt(smooth_psth_h.shape[0])
		
		y_min_l = (smooth_avg_psth_l - sem_smooth_avg_psth_l).min()
		y_max_l = (smooth_avg_psth_l+ sem_smooth_avg_psth_l).max()
		y_min_h = (smooth_avg_psth_h - sem_smooth_avg_psth_h).min()
		y_max_h = (smooth_avg_psth_h+ sem_smooth_avg_psth_h).max()

		y_min = np.array([y_min_l, y_min_h]).min()
		y_max = np.array([y_max_l, y_max_h]).max()


		num_trials = len(all_raster_l.keys())

		linelengths = float((y_max - y_min))/num_trials
		lineoffsets = 1
		xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)

		ax1 = plt.subplot(1,2,1)
		plt.title('All Trials')
		plt.plot(xticklabels, smooth_avg_psth_l,'b', label = 'LV Chosen')
		plt.fill_between(xticklabels, smooth_avg_psth_l - sem_smooth_avg_psth_l, smooth_avg_psth_l + sem_smooth_avg_psth_l, facecolor = 'b', alpha = 0.2)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_l.keys())):
			plt.eventplot(all_raster_l[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))

		ax2 = plt.subplot(1,2,2)
		plt.plot(xticklabels, smooth_avg_psth_h, 'r', label = 'HV Chosen')
		plt.fill_between(xticklabels, smooth_avg_psth_h - sem_smooth_avg_psth_h, smooth_avg_psth_h + sem_smooth_avg_psth_h, facecolor = 'r', alpha = 0.2)
		#xticklabels = np.arange(-t_before,t_after-t_resolution,t_resolution)
		#xticks = np.arange(0, len(xticklabels), 10)
		#xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
		#plt.xticks(xticks, xticklabels)
		plt.xlabel('Time from Center Hold (s)')
		plt.ylabel('Firing Rate (spk/s)')
		ax2.get_yaxis().set_tick_params(direction='out')
		ax2.get_xaxis().set_tick_params(direction='out')
		ax2.get_xaxis().tick_bottom()
		ax2.get_yaxis().tick_left()

		# DEFINE LINEOFFSETS AND LINELENGTHS BY Y-RANGE OF PSTH
		for k in range(len(all_raster_h.keys())):
			plt.eventplot(all_raster_h[k], colors=[[0,0,0]], lineoffsets= y_min + k*linelengths,linelengths=linelengths)
		plt.legend()
		plt.ylim((y_min - 1, y_max + 1))


		#plt_name = syncHDF_files[i][34:-15]
		#plt.savefig('/home/srsummerson/code/analysis/Mario_Performance_figs/'+plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		plt_name = syncHDF_files[i][syncHDF_files[i].index('Luigi201'):-15]
		plt.savefig(dir_figs +plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		#print('Figure saved:',dir_figs +plt_name+ '_' + str(align_to) +'_PSTH_Chan'+str(chann)+'-'+str(sc)+'.svg')
		
		plt.close()

	reg_psth = [psth_l, psth_h]
	smooth_psth = [smooth_psth_l, smooth_psth_h]
	"""
	return spike1