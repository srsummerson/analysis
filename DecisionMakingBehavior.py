import numpy as np 
import scipy as sp
from scipy import io
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from neo import io
from PulseMonitorData import findIBIs



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

		return lfp_state_row_ind

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
			plt.ylim((-0.05,1.05))
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
			plt.ylim((-0.05,1.05))
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
			plt.ylim((-0.05,1.05))
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
			plt.ylim((-0.05,1.05))
			ax22.get_yaxis().set_tick_params(direction='out')
			ax22.get_xaxis().set_tick_params(direction='out')
			ax22.get_xaxis().tick_bottom()
			ax22.get_yaxis().tick_left()
			plt.legend()


		return all_choices, LM_choices, LH_choices, MH_choices


class ChoiceBehavior_ThreeTargets_Stimulation():
	'''
	Class for behavior taken from ABA' task, where there are three targets of different probabilities of reward
	and stimulation is paired with the middle-value target during the hold-period of instructed trials during
	blocks B and A'. 
	'''

	def __init__(self, hdf_file, num_trials_A, num_trials_B):
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

		return lfp_state_row_ind


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
			plt.ylim((-0.05,1.05))
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
			plt.ylim((-0.05,1.05))
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
			plt.ylim((-0.05,1.05))
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
			plt.ylim((-0.05,1.05))
			ax22.get_yaxis().set_tick_params(direction='out')
			ax22.get_xaxis().set_tick_params(direction='out')
			ax22.get_xaxis().tick_bottom()
			ax22.get_yaxis().tick_left()
			plt.legend()

		return all_choices_A, LM_choices_A, LH_choices_A, MH_choices_A, all_choices_Aprime, LM_choices_Aprime, LH_choices_Aprime, MH_choices_Aprime