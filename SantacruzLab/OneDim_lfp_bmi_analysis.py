import numpy as np 
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import tables
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

# plot for time per trial over trials, separately for each target and all together

def trial_sliding_avg(trial_array, num_trials_slide):

	num_trials = len(trial_array)
	slide_avg = np.zeros(num_trials)

	for i in range(num_trials):
		if i < num_trials_slide:
			slide_avg[i] = np.sum(trial_array[:i+1])/float(i+1)
		else:
			slide_avg[i] = np.sum(trial_array[i-num_trials_slide+1:i+1])/float(num_trials_slide)

	return slide_avg

class OneDimLFPBMI_Behavior():

	def __init__(self, hdf_files):

		for i, hdf_file in enumerate(hdf_files): 
			filename =  hdf_file
			table = tables.open_file(filename)
			if i == 0:
				self.lfp_cursor = table.root.task[:]['lfp_cursor'][:,2]		# first two components correspond to dimensions that are not used for cursor movement
				self.lfp_target = table.root.task[:]['lfp_target'][:,2]		# first two components correspond to dimensions that are not used for cursor movement
				self.lfp_target[0] = self.lfp_target[1]						# first entry is -100, needs to be replaced with real value
				# target_index is -1 or 0, not sure what this is
				self.powercap_flag = table.root.task[:]['powercap_flag']
				self.lfp_power = table.root.task[:]['lfp_power']

				self.state = table.root.task_msgs[:]['msg']
				self.state_time = table.root.task_msgs[:]['time']
			else:
				self.lfp_cursor = np.append(self.lfp_cursor, table.root.task[:]['lfp_cursor'][:,2])
				lfptarget = table.root.task[:]['lfp_target'][:,2]
				lfptarget[0] = lfptarget[1]
				self.lfp_target = np.append(self.lfp_target, lfptarget)
				self.powercap_flag = np.append(self.powercap_flag, table.root.task[:]['powercap_flag'])
				self.lfp_power = np.append(self.lfp_power, table.root.task[:]['lfp_power'])

				self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
				self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])

		'''
		self.filename =  hdf_file
		self.table = tables.open_file(self.filename)
		
		self.lfp_cursor = self.table.root.task[:]['lfp_cursor'][:,2]		# first two components correspond to dimensions that are not used for cursor movement
		self.lfp_target = self.table.root.task[:]['lfp_target'][:,2]		# first two components correspond to dimensions that are not used for cursor movement
		self.lfp_target[0] = self.lfp_target[1]						# first entry is -100, needs to be replaced with real value
		# target_index is -1 or 0, not sure what this is
		self.powercap_flag = self.table.root.task[:]['powercap_flag']
		self.lfp_power = self.table.root.task[:]['lfp_power']

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
	  	'''
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_lfp_target_states = np.ravel(np.nonzero(self.state == b'lfp_target'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_lfp_hold_states = np.ravel(np.nonzero(self.state == b'lfp_hold'))
		self.ind_reward_states = np.ravel(np.nonzero(self.state == b'reward'))
		
		self.num_trials = self.ind_wait_states.size
		self.num_successful_trials = self.ind_reward_states.size

		#self.table.close()

	def plot_power_for_targets(self, t_before,t_after):
		'''
		Plot the cursor leading up to the successful targets. Data is aligned to reward.

		Input:
				- t_before: float, number of seconds to include before reward
				- t_after: float, number of seconds to include after reward
		'''
		targs = np.unique(self.lfp_target)
		num_targs = len(targs)
		color = iter(cm.rainbow(np.linspace(0, 1, num_targs)))
		time_samples = int(60*t_before) + int(60*t_after)		# number of time samples, sample rate is 60 samples/sec
		time = np.linspace(-t_before, t_after, time_samples)

		targ_per_trial = self.lfp_target[self.state_time[self.ind_reward_states]]

		for k,targ in enumerate(targs):
			trial_inds = [i for i in range(self.num_successful_trials) if targ_per_trial[i]==targ]

			reward_times = self.state_time[self.ind_reward_states][trial_inds]
			targ_data = np.empty([len(reward_times), time_samples])

			for j in range(len(reward_times)):
	
				#print(self.lfp_cursor[reward_times[j] - int(60*t_before):reward_times[j] + int(60*t_after)])
				if (reward_times[j] - int(60*t_before) < 0) | (reward_times[j] + int(60*t_after) > self.state_time[-1]):
					targ_data[j,:] = np.nan
				else:
					targ_data[j,:] = self.lfp_cursor[reward_times[j] - int(60*t_before):reward_times[j] + int(60*t_after)]

			targ_avg = np.nanmean(targ_data, axis = 0)
			targ_sem = np.nanstd(targ_data, axis = 0)/np.sqrt(len(reward_times)-1)

			c = next(color)

			plt.figure(1)
			plt.plot(time, targ_avg, c = c, label = ('Target %i' % k))
			plt.autoscale(enable=True, axis='x', tight=True)
			plt.fill_between(time, targ_avg	-targ_sem,targ_avg+targ_sem, facecolor = c, alpha = 0.5)
			plt.xlabel('Time (s)')
			plt.ylabel('Norm. LFP Power (aligned to reward begin)')
			plt.legend()

		return

	def plot_power_for_targets_cue(self, t_before,t_after):
		'''
		Plot the cursor leading up to the successful targets. Data is aligned to end of center hold.
		Only successful trials are included.
		
		Input:
				- t_before: float, number of seconds to include before end of center hold
				- t_after: float, number of seconds to include after end of center hold
		'''
		targs = np.unique(self.lfp_target)
		num_targs = len(targs)
		color = iter(cm.rainbow(np.linspace(0, 1, num_targs)))
		time_samples = int(60*t_before) + int(60*t_after)		# number of time samples, sample rate is 60 samples/sec
		time = np.linspace(-t_before, t_after, time_samples)

		targ_per_trial = self.lfp_target[self.state_time[self.ind_reward_states]]

		for k,targ in enumerate(targs):
			trial_inds = [i for i in range(self.num_successful_trials) if targ_per_trial[i]==targ]

			cue_times = self.state_time[self.ind_reward_states-2][trial_inds]
			targ_data = np.empty([len(cue_times), time_samples])

			for j in range(len(cue_times)):
	
				#print(self.lfp_cursor[reward_times[j] - int(60*t_before):reward_times[j] + int(60*t_after)])
				if (cue_times[j] - int(60*t_before) < 0) | (cue_times[j] + int(60*t_after) > self.state_time[-1]):
					targ_data[j,:] = np.nan
				else:
					targ_data[j,:] = self.lfp_cursor[cue_times[j] - int(60*t_before):cue_times[j] + int(60*t_after)]

			targ_avg = np.nanmean(targ_data, axis = 0)
			targ_sem = np.nanstd(targ_data, axis = 0)/np.sqrt(len(cue_times)-1)

			c = next(color)

			plt.figure(1)
			plt.plot(time, targ_avg, c = c, label = ('Target %i' % k))
			plt.autoscale(enable=True, axis='x', tight=True)
			plt.fill_between(time, targ_avg	-targ_sem,targ_avg+targ_sem, facecolor = c, alpha = 0.5)
			plt.xlabel('Time (s)')
			plt.ylabel('Norm. LFP Power (aligned to modulation begin cue)')
			plt.legend()

		return

	def plot_time_per_trial(self):
		'''
		Plot the amount of time per trial leading up to the successful targets

		'''
		targs = np.unique(self.lfp_target)
		num_targs = len(targs)
		color = iter(cm.rainbow(np.linspace(0, 1, num_targs)))

		targ_per_trial = self.lfp_target[self.state_time[self.ind_reward_states]]

		avg_trial_times = np.zeros(4)
		sem_trial_times = np.zeros(4)

		for k,targ in enumerate(targs):
			trial_inds = [i for i in range(self.num_successful_trials) if targ_per_trial[i]==targ]
			reward_times = self.state_time[self.ind_reward_states][trial_inds]
			start_times = self.state_time[self.ind_wait_states][trial_inds[:len(reward_times)]]

			trial_times = (reward_times - start_times)/60.
			#trial_times = np.flip(trial_times)

			avg_trial_times[k] = np.nanmean(trial_times)
			sem_trial_times[k] = np.nanstd(trial_times)/np.sqrt(len(trial_times)-1)

			trial_times_sliding_avg = trial_sliding_avg(trial_times, 20)
			
			c = next(color)
			
			plt.figure(1)
			plt.plot(np.arange(1,len(reward_times)+1), trial_times, c = c, label = ('Target %i' % k))
			plt.autoscale(enable=True, axis='x', tight=True)
			plt.ylabel('Trial duration (s)')
			plt.xlabel('Trial Number')
			plt.legend()

			plt.figure(2)
			plt.plot(np.arange(1,len(reward_times)+1), trial_times_sliding_avg, c = c, label = ('Target %i' % k))
			plt.autoscale(enable=True, axis='x', tight=True)
			plt.ylabel('Trial duration (s)')
			plt.xlabel('Trial Number')
			plt.legend()
			
		# plot for all targets together	
		trial_inds = [i for i in range(self.num_successful_trials)]
		reward_times = self.state_time[self.ind_reward_states][trial_inds]
		start_times = self.state_time[self.ind_wait_states][trial_inds[:len(reward_times)]]

		trial_times = (reward_times - start_times)/60.
		#trial_times = np.flip(trial_times)
		trial_times_sliding_avg = trial_sliding_avg(trial_times, 20)

		plt.figure(3)
		plt.plot(np.arange(1, len(trial_inds)+1), trial_times, c = 'k', label = ('All targets'))
		plt.autoscale(enable=True, axis='x', tight=True)
		plt.ylabel('Trial duration (s)')
		plt.xlabel('Trial Number')
		plt.legend()
		#plt.show()

		plt.figure(4)
		plt.plot(np.arange(1, len(trial_inds)+1), trial_times_sliding_avg, c = 'k', label = ('All targets'))
		plt.autoscale(enable=True, axis='x', tight=True)
		plt.ylabel('Trial duration (s)')
		plt.xlabel('Trial Number')
		plt.legend()

		ind = np.arange(4)
		width = 0.35
		plt.figure(5)
		plt.bar(ind, avg_trial_times, width, color = 'b', yerr = sem_trial_times)
		plt.ylim((0,10))
		plt.show()

		return