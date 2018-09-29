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


class StressBehavior():

	def __init__(self, hdf_file):
		self.filename =  hdf_file
		self.table = tables.openFile(self.filename)

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
		self.trial_type = self.table.root.task[:]['target_index']
		self.stress_type = self.table.root.task[:]['stress_trial']
	  
		self.ind_wait_states = np.ravel(np.nonzero(self.state == 'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == 'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == 'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == 'check_reward'))
		self.ind_reward_states = np.ravel(np.nonzero(self.state == 'reward'))
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

		return lfp_state_row_ind, dio_freq

	'''
	Sliding average of number of successful trials/min, sliding average of probability of selecting better option, raster plots showing each of these (do for all blocks of task)
	'''

	def PlotTrialsCompleted(self, num_trials_slide):
		'''
		This method computes the sliding average over num_trials_slide bins of the total number of trials successfully completed in each block.
		'''
		trial_times = self.state_time[self.ind_check_reward_states]/(60.**2)							# compute the times of the check reward states
		trials = np.ravel(self.stress_type[self.state_time[self.ind_check_reward_states]])				# check what trial type is for these successful trials
		transition_trials = np.ravel(np.nonzero(trials[1:] - trials[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials))									# add end of final block
		num_blocks = len(transition_trials)

		block_length = np.zeros(num_blocks)  															# holder for length of blocks in minutes

		trial_times_percentage_session = np.array([])
		num_times_per_session = np.array([])

		cmap = mpl.cm.hsv
		

		for i, trial_end in enumerate(transition_trials):
			if i==0:
				block_trial_times = trial_times[0:trial_end+1]
				end_prev_block = 0
			else:
				block_trial_times = trial_times[transition_trials[i-1]+1:trial_end+1]
				end_prev_block = trial_times[transition_trials[i-1]]

			block_trial_times = block_trial_times - end_prev_block
			block_length[i] = block_trial_times[-1]

			block_min_per_trial = np.zeros(len(block_trial_times))
			block_min_per_trial[0] = block_trial_times[0]
			block_min_per_trial[1:] = block_trial_times[1:] - block_trial_times[:-1]  					# gives the number of mins/trial
			block_trials_per_min = 1./block_min_per_trial												# gives the number of trials/min

			# compute average mins/trial
			sliding_avg_trial_rate = trial_sliding_avg(block_trials_per_min, num_trials_slide)   	# gives the average num of trials/min
			
			'''
			plt.figure(1)
			plt.subplot(211)
			plt.plot(block_trial_times + end_prev_block, sliding_avg_trial_rate,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			plt.plot(block_trial_times + end_prev_block,np.ones(len(block_trial_times)),'|', color=cmap(i/float(num_blocks)))
    		

			plt.figure(1)
			plt.subplot(212)
			plt.plot(block_trial_times/block_trial_times[-1] + i, sliding_avg_trial_rate,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			plt.plot(block_trial_times/block_trial_times[-1] + i,np.ones(len(block_trial_times)),'|', color=cmap(i/float(num_blocks)))
			'''
			trial_times_percentage_session = np.append(trial_times_percentage_session, block_trial_times + end_prev_block)
			num_times_per_session = np.append(num_times_per_session, len(block_trial_times))

		# Figure showing the average rate of trials/min as a function of minutes in the task
		'''
		plt.figure(1)
		ax1 = plt.subplot(211)
		plt.xlabel('Time (min)')
		plt.ylabel('Avg Trials/min')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		# Figure showing the average rate of trials/min as a function of minutes in the task
		plt.figure(1)
		ax2 = plt.subplot(212)
		plt.xlabel('Block (%)')
		plt.ylabel('Avg Trials/min')
		ax2.get_yaxis().set_tick_params(direction='out')
		ax2.get_xaxis().set_tick_params(direction='out')
		ax2.get_xaxis().tick_bottom()
		ax2.get_yaxis().tick_left()
		plt.legend()
		#plt.show()
		plt.close()
		'''
		return trial_times_percentage_session, num_times_per_session


	def PlotTrialChoices(self, num_trials_slide, plot_results = False):
		'''
		This method computes the sliding average over num_trials_slide trials of the number of choices for the HV target.
		'''
		trial_times = self.state_time[self.ind_check_reward_states]/(60.**2)
		trials = np.ravel(self.stress_type[self.state_time[self.ind_check_reward_states]])
		transition_trials = np.ravel(np.nonzero(trials[1:] - trials[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials))									# add end of final block
		num_blocks = len(transition_trials)

		target_choices = self.state[self.ind_check_reward_states - 2]

		block_length = np.zeros(num_blocks)											# holder for length of blocks in minutes
		avg_trials_per_min = np.zeros(num_blocks)
		end_block_avg_hv_choices = np.zeros(num_blocks)

		cmap = mpl.cm.hsv

		for i, trial_end in enumerate(transition_trials):
			if i==0:
				block_trial_times = trial_times[0:trial_end+1]
				choices = target_choices[0:trial_end+1]
				end_prev_block = 0
			else:
				block_trial_times = trial_times[transition_trials[i-1]+1:trial_end+1]
				choices = target_choices[transition_trials[i-1]+1:trial_end+1]
				end_prev_block = trial_times[transition_trials[i-1]]

			block_trial_times = block_trial_times - end_prev_block
			block_length[i] = block_trial_times[-1]

			block_min_per_trial = np.zeros(len(block_trial_times))
			block_min_per_trial[0] = block_trial_times[0]
			block_min_per_trial[1:] = block_trial_times[1:] - block_trial_times[:-1]  					# gives the number of mins/trial
			block_trials_per_min = 1./block_min_per_trial
			avg_trials_per_min[i] = np.nanmean(block_trials_per_min)

			hv_choices = np.zeros(len(block_trial_times))
			hv_choices[choices=='hold_targetH'] = 1

			sliding_avg_hv_choices = trial_sliding_avg(hv_choices, num_trials_slide)   	# gives the average num of mins/trial
			end_block_avg_hv_choices[i]= sliding_avg_hv_choices[-1]

			if plot_results:
				plt.figure(1)
				plt.subplot(211)
				plt.plot(block_trial_times + end_prev_block, sliding_avg_hv_choices,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
				plt.plot(block_trial_times + end_prev_block,0.1*hv_choices - 0.2,'|', color=cmap(i/float(num_blocks)))
	    		

				plt.figure(1)
				plt.subplot(212)
				plt.plot(block_trial_times/block_trial_times[-1] + i, sliding_avg_hv_choices,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
				plt.plot(block_trial_times/block_trial_times[-1] + i,0.1*hv_choices - 0.2,'|', color=cmap(i/float(num_blocks)))

		if plot_results:
			# Figure showing the average rate of trials/min as a function of minutes in the task
			plt.figure(1)
			ax1 = plt.subplot(211)
			plt.xlabel('Time (min)')
			plt.ylabel('Probability Best Choice')
			ax1.get_yaxis().set_tick_params(direction='out')
			ax1.get_xaxis().set_tick_params(direction='out')
			ax1.get_xaxis().tick_bottom()
			ax1.get_yaxis().tick_left()
			plt.ylim((-0.25,1))
			plt.legend()

			# Figure showing the average rate of trials/min as a function of minutes in the task
			plt.figure(1)
			ax2 = plt.subplot(212)
			plt.xlabel('Block (%)')
			plt.ylabel('Probability Best Choice')
			ax2.get_yaxis().set_tick_params(direction='out')
			ax2.get_xaxis().set_tick_params(direction='out')
			ax2.get_xaxis().tick_bottom()
			ax2.get_yaxis().tick_left()
			plt.ylim((-0.25,1))
			plt.legend()
			plt.show()

		return avg_trials_per_min, end_block_avg_hv_choices

	def get_cursor_velocity(self, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
	    '''
	    hdf file -- task file generated from bmi3d
	    go_cue_ix -- list of go cue indices (units of hdf file row numbers)
	    before_cue_time -- time before go cue to inclue in trial (units of sec)
	    after_cue_time -- time after go cue to include in trial (units of sec)

	    returns a time x (x,y) x trials filtered velocity array
	    '''

	    ix = np.arange(-1*before_cue_time*fs, after_cue_time*fs).astype(int)


	    # Get trial trajectory: 
	    cursor = []
	    for g in go_cue_ix:
	        try:
	            #Get cursor
	            cursor.append(self.table.root.task[ix+g]['cursor'][:, [0, 2]])

	        except:
	            print 'skipping index: ', g, ' -- too close to beginning or end of file'
	    cursor = np.dstack((cursor))    # time x (x,y) x trial
	    
	    dt = 1./fs
	    vel = np.diff(cursor,axis=0)/dt

	    #Filter velocity: 
	    if use_filt_vel:
	        filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
	    else:
	        filt_vel = vel
	    total_vel = np.zeros((int(filt_vel.shape[0]),int(filt_vel.shape[2])))
	    for n in range(int(filt_vel.shape[2])):
	        total_vel[:,n] = np.sqrt(filt_vel[:,0,n]**2 + filt_vel[:,1,n]**2)

	    vel_bins = np.linspace(-1*before_cue_time, after_cue_time, vel.shape[0])

	    return filt_vel, total_vel, vel_bins
	
	def compute_rt_per_trial_FreeChoiceTask(self, trials): 
		'''
		Compute the reaction time for all trials in array trials.
		'''

		#Extract go_cue_indices in units of hdf file row number
		go_cue_ix = np.array([self.table.root.task_msgs[j-3]['time'] for j, i in enumerate(self.table.root.task_msgs) if i['msg']=='check_reward'])
		go_cue_ix = go_cue_ix[trials]

		# Calculate filtered velocity and 'velocity mag. in target direction'
		filt_vel, total_vel, vel_bins = self.get_cursor_velocity(go_cue_ix, 0., 2., use_filt_vel=False)
		## Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
		kin_feat = get_rt_change_deriv(total_vel.T, vel_bins, d_vel_thres = 0.3, fs=60)
		return kin_feat[:,1], total_vel

	def PlotReactionAndTrialTimes(self, num_trials_slide):
		'''
		This method computes the sliding average over num_trials_slide bins of the reaction times and trial times, as well as the histograms.

		NOTE: Need to add reaction time computation and plotting.
		'''

		trial_times = (self.state_time[self.ind_check_reward_states] - self.state_time[self.ind_check_reward_states-5])/60.		# gives trial lengths in seconds (calculated as diff between check_reward state and center state)
		trials = np.ravel(self.stress_type[self.state_time[self.ind_check_reward_states]])
		transition_trials = np.ravel(np.nonzero(trials[1:] - trials[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials)-1)									# add end of final block
		num_blocks = len(transition_trials)

		block_length = np.zeros(num_blocks)  															# holder for length of blocks in minutes

		cmap = mpl.cm.hsv
		plt.figure()

		for i, trial_end in enumerate(transition_trials):
			if i==0:
				block_trial_times = trial_times[0:trial_end+1]
				num_trials_prev_block = 0
				trial_ind = np.arange(0,trial_end+1)
			else:
				block_trial_times = trial_times[transition_trials[i-1]+1:trial_end+1]
				num_trials_prev_block = transition_trials[i-1]
				trial_ind = np.arange(transition_trials[i-1]+1,trial_end+1)

			
			# compute average mins/trial
			sliding_avg_trial_length = trial_sliding_avg(block_trial_times, num_trials_slide)   	# gives the average num of sec/trial

			max_trial_length = np.max(block_trial_times)
			bins_trial_time = np.linspace(0,max_trial_length+1,10)

			# compute reaction times
			rt_per_trial, total_vel = self.compute_rt_per_trial_FreeChoiceTask(trial_ind)
			# compute sliding average of reaction times
			sliding_avg_rt = trial_sliding_avg(rt_per_trial, num_trials_slide)
			#max_rt = np.max(rt_per_trial)
			max_rt = 0.5
			bins_rt = np.linspace(0, max_rt+.1,10)
			
			
			plt.figure(1)
			plt.subplot(211)
			plt.plot(np.arange(len(sliding_avg_trial_length)) + num_trials_prev_block, sliding_avg_trial_length,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			
			plt.subplot(212)
			plt.hist(block_trial_times, bins_trial_time, alpha=0.5, label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))

			plt.figure(2)
			plt.subplot(211)
			plt.plot(np.arange(len(sliding_avg_rt)) + num_trials_prev_block, sliding_avg_rt,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			
			plt.subplot(212)
			plt.hist(rt_per_trial, bins_rt, alpha=0.5, label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))

		# Figure showing the average trial length in secs as a function of trials
		plt.figure(1)
		ax1 = plt.subplot(211)
		plt.xlabel('Trials')
		plt.ylabel('Avg Trial Length (s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		ax1 = plt.subplot(212)
		plt.xlabel('Trial Lengths (s)')
		plt.ylabel('Frequency')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		plt.figure(2)
		ax1 = plt.subplot(211)
		plt.xlabel('Trials')
		plt.ylabel('Avg Reaction Time (s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		ax1 = plt.subplot(212)
		plt.xlabel('Reaction Time (s)')
		plt.ylabel('Frequency')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		plt.show()

		return 

class StressBehavior_withSpikes(StressBehavior):

	def compute_value_PSTH(self, syncHDF_file, spike_file, chann, sc):
		'''
		Computes the raw and smooth PSTH of spike data around two times: (1) center hold and (2) reward.
		Average PSTHs are then plotted with error bars.

		Inputs:
		- syncHDF_file: str, file address of .mat sync file to synchronize TDT and hdf files
		- spike_file: str, file address of .csv file with offline-sorted spike data
		- chann: int, value of channel of interest
		- sc: int, sort code on the indicated channel

		'''
		# get indices of relevant events
		ind_picture_onset = self.ind_check_reward_states - 4  # -4: 'hold_center', -2: 'hold_targetH/L'
		lfp_state_row_ind, lfp_freq = self.get_state_TDT_LFPvalues(ind_picture_onset,syncHDF_file)
		times_align = lfp_state_row_ind/float(lfp_freq)

		ind_reward = self.ind_check_reward_states  # -4: 'hold_center', -2: 'hold_targetH/L'
		lfp_state_row_ind_reward, lfp_freq = self.get_state_TDT_LFPvalues(ind_reward,syncHDF_file)
		times_align_reward = lfp_state_row_ind_reward/float(lfp_freq)

		center_hold_dur = (self.state_time[self.ind_check_reward_states[0]-3] - self.state_time[self.ind_check_reward_states[0]-4])/60.
		reward_dur = (self.state_time[self.ind_reward_states[0]+1] - self.state_time[self.ind_reward_states[0]])/60.

		# divide trials up by choices for LV and HV options
		chooseH = np.ravel(np.nonzero(self.state[self.ind_check_reward_states-2] == 'hold_targetH'))
		chooseL = np.ravel(np.nonzero(self.state[self.ind_check_reward_states-2] == 'hold_targetL'))

		# set basic parameters for computing PSTH
		t_before = 2.0 		# seconds
		t_after = 2.0		# seconds
		t_resolution = 0.05	# seconds
		t = np.arange(-t_before,t_after-t_resolution,t_resolution)
		spk = OfflineSorted_CSVFile(spike_file)

		# compute PSTHs for time around center hold
		psth, smooth_psth = spk.compute_psth(chann,sc,times_align,t_before,t_after,t_resolution)

		psth_H, smooth_psth_H = spk.compute_psth(chann,sc,times_align[chooseH],t_before,t_after,t_resolution)
		psth_L, smooth_psth_L = spk.compute_psth(chann,sc,times_align[chooseL],t_before,t_after,t_resolution)

		avg_psth = np.nanmean(psth, axis=0)
		sem_psth = np.nanstd(psth,axis=0)/np.sqrt(psth.shape[0])
		avg_smooth_psth = np.nanmean(smooth_psth, axis=0)
		sem_smooth_psth = np.nanstd(smooth_psth,axis=0)/np.sqrt(smooth_psth.shape[0])

		avg_psth_H = np.nanmean(psth_H, axis=0)
		sem_psth_H = np.nanstd(psth_H,axis=0)/np.sqrt(psth_H.shape[0])
		avg_smooth_psth_H = np.nanmean(smooth_psth_H, axis=0)
		sem_smooth_psth_H = np.nanstd(smooth_psth_H,axis=0)/np.sqrt(smooth_psth_H.shape[0])

		avg_psth_L = np.nanmean(psth_L, axis=0)
		sem_psth_L = np.nanstd(psth_L,axis=0)/np.sqrt(psth_L.shape[0])
		avg_smooth_psth_L = np.nanmean(smooth_psth_L, axis=0)
		sem_smooth_psth_L = np.nanstd(smooth_psth_L,axis=0)/np.sqrt(smooth_psth_L.shape[0])

		# compute PSTHs for time around reward
		psth_reward, smooth_psth_reward = spk.compute_psth(chann,sc,times_align_reward,t_before,t_after,t_resolution)

		psth_H_reward, smooth_psth_H_reward = spk.compute_psth(chann,sc,times_align_reward[chooseH],t_before,t_after,t_resolution)
		psth_L_reward, smooth_psth_L_reward = spk.compute_psth(chann,sc,times_align_reward[chooseL],t_before,t_after,t_resolution)

		avg_psth_reward = np.nanmean(psth_reward, axis=0)
		sem_psth_reward = np.nanstd(psth_reward,axis=0)/np.sqrt(psth_reward.shape[0])
		avg_smooth_psth_reward = np.nanmean(smooth_psth_reward, axis=0)
		sem_smooth_psth_reward = np.nanstd(smooth_psth_reward,axis=0)/np.sqrt(smooth_psth_reward.shape[0])

		avg_psth_H_reward = np.nanmean(psth_H_reward, axis=0)
		sem_psth_H_reward = np.nanstd(psth_H_reward,axis=0)/np.sqrt(psth_H_reward.shape[0])
		avg_smooth_psth_H_reward = np.nanmean(smooth_psth_H_reward, axis=0)
		sem_smooth_psth_H_reward = np.nanstd(smooth_psth_H_reward,axis=0)/np.sqrt(smooth_psth_H_reward.shape[0])

		avg_psth_L_reward = np.nanmean(psth_L_reward, axis=0)
		sem_psth_L_reward = np.nanstd(psth_L_reward,axis=0)/np.sqrt(psth_L_reward.shape[0])
		avg_smooth_psth_L_reward = np.nanmean(smooth_psth_L_reward, axis=0)
		sem_smooth_psth_L_reward = np.nanstd(smooth_psth_L_reward,axis=0)/np.sqrt(smooth_psth_L_reward.shape[0])


		# make plots for around time of center hold
		top = 6
		bottom = 0
		plt.figure(1)
		plt.subplot(131)
		plt.plot(t,avg_psth,'b',label='Raw PSTH')
		plt.plot(t,avg_smooth_psth,'r',label='Smooth PSTH')
		plt.fill_between(t, avg_psth - sem_psth, avg_psth + sem_psth, color = 'b', alpha = 0.5)
		plt.fill_between(t, avg_smooth_psth - sem_smooth_psth, avg_smooth_psth + sem_smooth_psth, color = 'r', alpha = 0.5)
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		plt.title('PSTH for Channel %f - Unit %f' % (chann, sc))
		plt.text(0,top - (top*0.05),'Center\n Hold',horizontalalignment='center')
		plt.text(center_hold_dur,top - (top*0.05), 'End\n Hold',horizontalalignment='center')
		plt.ylim((bottom,top))

		plt.subplot(132)
		plt.plot(t,avg_psth_H,'b',label='Raw PSTH')
		plt.plot(t,avg_smooth_psth_H,'r',label='Smooth PSTH')
		plt.fill_between(t, avg_psth_H - sem_psth_H, avg_psth_H + sem_psth_H, color = 'b', alpha = 0.5)
		plt.fill_between(t, avg_smooth_psth_H - sem_smooth_psth_H, avg_smooth_psth_H + sem_smooth_psth_H, color = 'r', alpha = 0.5)
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		plt.title('HV Choices: PSTH for Channel %f - Unit %f' % (chann, sc))
		plt.text(0,top - (top*0.05),'Center\n Hold',horizontalalignment='center')
		plt.text(center_hold_dur,top - (top*0.05), 'End\n Hold',horizontalalignment='center')
		plt.ylim((bottom,top))

		plt.subplot(133)
		plt.plot(t,avg_psth_L,'b',label='Raw PSTH')
		plt.plot(t,avg_smooth_psth_L,'r',label='Smooth PSTH')
		plt.fill_between(t, avg_psth_L - sem_psth_L, avg_psth_L + sem_psth_L, color = 'b', alpha = 0.5)
		plt.fill_between(t, avg_smooth_psth_L - sem_smooth_psth_L, avg_smooth_psth_L + sem_smooth_psth_L, color = 'r', alpha = 0.5)
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		plt.title('LV Choices: PSTH for Channel %f - Unit %f' % (chann, sc))
		plt.text(0,top - (top*0.05),'Center\n Hold',horizontalalignment='center')
		plt.text(center_hold_dur,top - (top*0.05), 'End\n Hold',horizontalalignment='center')
		plt.ylim((bottom,top))

		# make plots for PSTH around reward
		plt.figure(2)
		plt.subplot(131)
		plt.plot(t,avg_psth_reward,'b',label='Raw PSTH')
		plt.plot(t,avg_smooth_psth_reward,'r',label='Smooth PSTH')
		plt.fill_between(t, avg_psth_reward - sem_psth_reward, avg_psth_reward + sem_psth_reward, color = 'b', alpha = 0.5)
		plt.fill_between(t, avg_smooth_psth_reward - sem_smooth_psth_reward, avg_smooth_psth_reward + sem_smooth_psth_reward, color = 'r', alpha = 0.5)
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		plt.title('PSTH for Channel %f - Unit %f' % (chann, sc))
		plt.text(0,top - (top*0.05),'Reward',horizontalalignment='center')
		plt.text(reward_dur,top - (top*0.05), 'End\n Reward',horizontalalignment='center')
		plt.ylim((bottom,top))

		plt.subplot(132)
		plt.plot(t,avg_psth_H_reward,'b',label='Raw PSTH')
		plt.plot(t,avg_smooth_psth_H_reward,'r',label='Smooth PSTH')
		plt.fill_between(t, avg_psth_H_reward - sem_psth_H_reward, avg_psth_H_reward + sem_psth_H_reward, color = 'b', alpha = 0.5)
		plt.fill_between(t, avg_smooth_psth_H_reward - sem_smooth_psth_H_reward, avg_smooth_psth_H_reward + sem_smooth_psth_H_reward, color = 'r', alpha = 0.5)
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		plt.title('HV Choices: PSTH for Channel %f - Unit %f' % (chann, sc))
		plt.text(0,top - (top*0.05),'Reward',horizontalalignment='center')
		plt.text(reward_dur,top - (top*0.05), 'End\n Reward',horizontalalignment='center')
		plt.ylim((bottom,top))

		plt.subplot(133)
		plt.plot(t,avg_psth_L_reward,'b',label='Raw PSTH')
		plt.plot(t,avg_smooth_psth_L_reward,'r',label='Smooth PSTH')
		plt.fill_between(t, avg_psth_L_reward - sem_psth_L_reward, avg_psth_L_reward + sem_psth_L_reward, color = 'b', alpha = 0.5)
		plt.fill_between(t, avg_smooth_psth_L_reward - sem_smooth_psth_L_reward, avg_smooth_psth_L_reward + sem_smooth_psth_L_reward, color = 'r', alpha = 0.5)
		plt.xlabel('Time (s)')
		plt.ylabel('Firing Rate (Hz)')
		plt.title('LV Choices: PSTH for Channel %f - Unit %f' % (chann, sc))
		plt.text(0,top - (top*0.05),'Reward',horizontalalignment='center')
		plt.text(reward_dur,top - (top*0.05), 'End\n Reward',horizontalalignment='center')
		plt.ylim((bottom,top))
		plt.show()
		
		return psth, smooth_psth, psth_reward, smooth_psth_reward, t

class TDTNeuralData():
	'''
	This class gives a structure for neural data, pupil data (if any), and pulse data (if any) recording during a recording using the TDT neurophysiology system.
	It also loads the synchronizing data from the _syncHDF.mat file so that the data may easily be analyzed in conjuction with behavioral data. 

	'''

	def __init__(self, TDT_directory, block_num):
		# load syncing data: hdf timestamps matching with TDT sample numbers
		tank_name = TDT_directory[-14:]   # assumes TDT directory has format ".../MarioYYYYMMDD/"
		print tank_name
		#mat_filename = TDT_directory[:-17] + 'syncHDF/' + tank_name+ '_b'+str(block_num)+'_syncHDF.mat'
		mat_filename = TDT_directory[:-13] + '/' + tank_name+ '_b'+str(block_num)+'_syncHDF.mat'
		print TDT_directory[:-13]
		self.hdf_times = dict()
		sp.io.loadmat(mat_filename,self.hdf_times)

		'''
		Convert DIO TDT samples for pupil and pulse data for regular and stress trials
		'''
		# divide up analysis for regular trials before stress trials, stress trials, and regular trials after stress trials are introduced
		hdf_rows = np.ravel(self.hdf_times['row_number'])
		self.hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
		dio_tdt_sample = np.ravel(self.hdf_times['tdt_samplenumber'])
		dio_freq = np.ravel(self.hdf_times['tdt_dio_samplerate'])

		r = io.TdtIO(TDT_directory)
		bl = r.read_block(lazy=False,cascade=True)
		print "File read."
		self.lfp = dict()
		# Get Pulse and Pupil Data
		for sig in bl.segments[block_num-1].analogsignals:
			if (sig.name == 'PupD 1'):
				self.pupil_data = np.ravel(sig)
				self.pupil_samprate = sig.sampling_rate.item()
				# Convert DIO TDT sample numbers to for pupil and pulse data:
				# if dio sample num is x, then data sample number is R*(x-1) + 1 where
				# R = data_sample_rate/dio_sample_rate
				self.pupil_dio_sample_num = (float(self.pupil_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
			if (sig.name == 'HrtR 1'):
				self.pulse_data = np.ravel(sig)
				self.pulse_samprate = sig.sampling_rate.item()
				# Convert DIO TDT sample numbers to for pupil and pulse data:
				# if dio sample num is x, then data sample number is R*(x-1) + 1 where
				# R = data_sample_rate/dio_sample_rate
				self.pulse_dio_sample_num = (float(self.pulse_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
			if (sig.name[0:4] == 'LFP1'):
				channel = sig.channel_index
				lfp_samprate = sig.sampling_rate.item()
				self.lfp[channel] = np.ravel(sig)
			if (sig.name[0:4] == 'LFP2'):
				channel = sig.channel_index + 96
				self.lfp[channel] = np.ravel(sig)
				self.lfp_samprate = sig.sampling_rate.item()

		# Convert DIO TDT sample numbers for LFP data:
		# if dio sample num is x, then data sample number is R*(x-1) + 1 where
		# R = data_sample_rate/dio_sample_rate
		self.lfp_dio_sample_num = (float(lfp_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1

	def get_indices_for_behavior_events(self, hdf_row_ind):
		'''
		Input:
			- hdf_row_ind: row indices sampled at rate 60 Hz corresponding to behavioral events of interest (e.g. center hold), these should come from the self.state_time array.
		Output
			- pusle_ind: corresponding indices for pulse signal
			- pupil_ind: corresponding indices for pupil signal
			- lfp_ind: corresponding indices for lfp signals
		'''
		pulse_ind = np.zeros(len(hdf_row_ind))
		pupil_ind = np.zeros(len(hdf_row_ind))
		lfp_ind = np.zeros(len(hdf_row_ind))
		for i, ind in enumerate(hdf_row_ind):
			hdf_index = np.argmin(np.abs(self.hdf_rows - ind))
			pulse_ind[i] = self.pulse_dio_sample_num[hdf_index]
			pupil_ind[i] = self.pupil_dio_sample_num[hdf_index]
			lfp_ind[i] = self.lfp_dio_sample_num[hdf_index]

		return pulse_ind, pupil_ind, lfp_ind

	def getIBIandPuilDilation(self, pulse_ind, samples_pulse, pupil_ind,samples_pupil):
		'''
		This method computes statistics on the IBI and pupil diameter per trial, as well as aggregating data
		from all the trials indicated by the row_ind input to compute histograms of the data. Additionally it 
		computes the heart rate variability (HRV, standard deviation of the IBIs) per trial.

		Inputs:
			- pulse_ind: N x 1 array containing indices of the pulse signal that correspond to particular events, e.g. center hold, where there are N events of interest
			- samples_pulse: N x 1 array containing the number of samples to compute the data from the pulse signal for the event of interest
			- pupil_ind: N x 1 array containing indices of the pupil signal that correspond to particular events, e.g. center hold, where there are N events of interest
			- samples_pupil: N x 1 array containing the number of samples to compute the data from the pupil signal for the event of interest. Note: the pulse data is often
							computed over the duration of the trial but pupil data in some cases may be restricted to short epochs within the trial.
		Outputs:
			- ibi_mean: list of length N containing the average IBI for each event
			- hrv: list of length N containing the standard deviation of the IBIs (HRV) for each event 
			- all_ibi: list of variable length containing all IBI values measured across all events
			- pupil_mean: list of length N containing the average pupil diameter for each event
			- all_pupil: list of variable length containing all pupil diameter values (measured while eyes are detected to be open) across all events

		'''
		ibi_mean = []
		hrv = []
		pupil_mean = []
		all_ibi = []
		all_pupil = []

		for i in range(0,len(pulse_ind)):
			pulse_snippet = self.pulse_data[pulse_ind[i]:pulse_ind[i]+samples_pulse[i]]
			ibi_snippet = findIBIs(pulse_snippet,self.pulse_samprate)
			all_ibi += ibi_snippet.tolist()
			if np.isnan(np.nanmean(ibi_snippet)):
				ibi_mean.append(ibi_mean[-1])   # repeat last measurement
				hrv.append(hrv[-1])
			else:
				ibi_mean.append(np.nanmean(ibi_snippet))
				hrv.append(np.nanstd(ibi_snippet))
			ibi_std.append(np.nanstd(ibi_snippet))
			
			pupil_snippet = self.pupil_data[pupil_ind[i]:pupil_ind[i]+samples_pupil[i]]
			pupil_snippet_range = range(0,len(pupil_snippet))
			eyes_closed = np.nonzero(np.less(pupil_snippet,-3.3))
			eyes_closed = np.ravel(eyes_closed)
			if len(eyes_closed) > 1:
				find_blinks = eyes_closed[1:] - eyes_closed[:-1]
				blink_inds = np.ravel(np.nonzero(np.not_equal(find_blinks,1)))
				eyes_closed_ind = [eyes_closed[0]]
				eyes_closed_ind += eyes_closed[blink_inds].tolist()
				eyes_closed_ind += eyes_closed[blink_inds+1].tolist()
				eyes_closed_ind += [eyes_closed[-1]]
				eyes_closed_ind.sort()
				
				for i in np.arange(1,len(eyes_closed_ind),2):
					rm_range = range(np.nanmax(eyes_closed_ind[i-1]-20,0),np.minimum(eyes_closed_ind[i] + 20,len(pupil_snippet)-1))
					rm_indices = [pupil_snippet_range.index(rm_range[ind]) for ind in range(0,len(rm_range)) if (rm_range[ind] in pupil_snippet_range)]
					pupil_snippet_range = np.delete(pupil_snippet_range,rm_indices)
					pupil_snippet_range = pupil_snippet_range.tolist()
			pupil_snippet = pupil_snippet[pupil_snippet_range]
			pupil_snippet_mean = np.nanmean(pupil_snippet)
			pupil_snippet_std = np.nanstd(pupil_snippet)
			window = np.floor(pupil_samprate/10) # sample window equal to ~100 ms
			#pupil_snippet = (pupil_snippet[0:window]- pupil_snippet_mean)/float(pupil_snippet_std)
			pupil_snippet = pupil_snippet[0:window]
			all_pupil += pupil_snippet.tolist()
			if np.isnan(np.nanmean(pupil_snippet)):
				pupil_mean.append(pupil_mean[-1])
			else:
				pupil_mean.append(np.nanmean(pupil_snippet))
			pupil_std.append(np.nanstd(pupil_snippet))

		return ibi_mean, hrv, all_ibi, pupil_mean, all_pupil

	def relate_RT_IBI_PD_stress_task(self, hdf_file): 
		'''
		This method extracts the regular and stress trials from the hdf_file and computes the average values for each trial of the IBI, HRV, and pupil diameter.
		Subsequently it computes whether each metric is significantly different across the blocks and if they're correlated with each other. 
		'''

		stress_data = StressBehavior(hdf_file)
		# find which trials are stress trials
		trial_type = np.ravel(stress_data.stress_type[stress_data.state_time[stress_data.ind_check_reward_states]])
		reg_trial_inds = np.array([ind for ind in range(len(trial_type)) if trial_type[ind] == 0])
		stress_trial_inds = np.array([ind for ind in range(len(trial_type)) if trial_type[ind] == 1])

		rt_reg, total_vel = stress_data.compute_rt_per_trial_FreeChoiceTask(reg_trial_inds)
		rt_stress, total_vel = stress_data.compute_rt_per_trial_FreeChoiceTask(stress_trial_inds)

		hdf_row_reg_trials = stress_data.state_time[stress_data.ind_check_reward_states[reg_trial_inds] - 4]  # times of hold center state
		hdf_row_stress_trials = stress_data.state_time[stress_data.ind_check_reward_states[stress_trial_inds] - 4]  # times of hold center state

		pulse_ind_reg, pupil_ind_reg, lfp_ind_reg = get_indices_for_behavior_events(self, hdf_row_reg_trials)
		pulse_ind_stress, pupil_ind_stress, lfp_ind_stress = get_indices_for_behavior_events(self, hdf_row_stress_trials)

		trial_length_reg = (stress_data.state_time[stress_data.ind_check_reward_states[reg_trial_inds]] - stress_data.state_time[stress_data.ind_check_reward_states[reg_trial_inds] - 4])/60.  # gives length in seconds
		trial_length_reg = np.array([int(length*self.pulse_samprate) for length in trial_length_reg])  # gives length in terms of number of pulse signal samples
		trial_length_stress = (stress_data.state_time[stress_data.ind_check_reward_states[stress_trial_inds]] - stress_data.state_time[stress_data.ind_check_reward_states[stress_trial_inds] - 4])/60.  # gives length in seconds
		trial_length_stress = np.array([int(length*self.pulse_samprate) for length in trial_length_stress])  # gives length in terms of number of pulse signal samples
		
		# compute IBI and HRV over lengths of trials, compute PD over the first 100 ms of the hold period
		ibi_mean_reg, hrv_reg, all_ibi_reg, pupil_mean_reg, all_pupil_reg = getIBIandPuilDilation(self, pulse_ind_reg, trial_length_reg, pupil_ind_reg,int(0.1*self.pupil_samprate))
		ibi_mean_stress, hrv_stress, all_ibi_stress, pupil_mean_stress, all_pupil_stress = getIBIandPuilDilation(self, pulse_ind_stress, trial_length_stress, pupil_ind_stress,int(0.1*self.pupil_samprate))

		# compute if metrics are significantly different between the two blocks:

		t_ibi, p_ibi = stats.ttest_ind(ibi_mean_reg, ibi_mean_stress)
		t_hrv, p_hrv = stats.ttest_ind(hrv_reg, hrv_stress)
		t_pupil, p_pupil = stats.ttest_ind(pupil_mean_reg, pupil_mean_stress)

		ibi_pupil_reg = stats.pearsonr(ibi_mean_reg, pupil_mean_reg)
		ibi_pupil_stress = stats.pearsonr(ibi_mean_stress, pupil_mean_stress)
		ibi_rt_reg = stats.pearsonr(ibi_mean_reg, rt_reg)
		ibi_rt_stress = stats.pearsonr(ibi_mean_stress, rt_stress)
		pupil_rt_reg = stats.pearsonr(pupil_mean_reg, rt_reg)
		pupil_rt_stress = stats.pearsonr(pupil_mean_stress, rt_stress)

		metrics_sig_diff = [[t_ibi, p_ibi], [t_hrv, p_hrv], [t_pupil, p_pupil]]
		metrics_corr = [ibi_pupil_reg, ibi_pupil_stress, ibi_rt_reg, ibi_rt_stress, pupil_rt_reg, pupil_rt_stress]		

		return metrics_sig_diff, metrics_corr

class StressBehavior_CenterOut():

	def __init__(self, hdf_file):
		self.filename =  hdf_file
		self.table = tables.openFile(self.filename)

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
		self.stress_type = self.table.root.task[:]['stress_trial']
	  
		self.ind_wait_states = np.ravel(np.nonzero(self.state == 'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == 'hold_center'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == 'target'))
		self.ind_reward_states = np.ravel(np.nonzero(self.state == 'reward'))
		self.stress_trial = np.ravel(self.stress_type[self.state_time[self.ind_reward_states-4]])
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_reward_states.size
		self.trial_times = (self.state_time[self.ind_reward_states] - self.state_time[self.ind_reward_states-4])/60.		# gives trial lengths in seconds (calculated as diff between reward state and hold center state)
		

	
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


	def get_cursor_velocity(self, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
	    '''
	    hdf file -- task file generated from bmi3d
	    go_cue_ix -- list of go cue indices (units of hdf file row numbers)
	    before_cue_time -- time before go cue to inclue in trial (units of sec)
	    after_cue_time -- time after go cue to include in trial (units of sec)

	    returns a time x (x,y) x trials filtered velocity array
	    '''

	    ix = np.arange(-1*before_cue_time*fs, after_cue_time*fs).astype(int)


	    # Get trial trajectory: 
	    cursor = []
	    for g in go_cue_ix:
	        try:
	            #Get cursor
	            cursor.append(self.table.root.task[ix+g]['cursor'][:, [0, 2]])

	        except:
	            print 'skipping index: ', g, ' -- too close to beginning or end of file'
	    cursor = np.dstack((cursor))    # time x (x,y) x trial
	    
	    dt = 1./fs
	    vel = np.diff(cursor,axis=0)/dt

	    #Filter velocity: 
	    if use_filt_vel:
	        filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
	    else:
	        filt_vel = vel
	    total_vel = np.zeros((int(filt_vel.shape[0]),int(filt_vel.shape[2])))
	    for n in range(int(filt_vel.shape[2])):
	        total_vel[:,n] = np.sqrt(filt_vel[:,0,n]**2 + filt_vel[:,1,n]**2)

	    vel_bins = np.linspace(-1*before_cue_time, after_cue_time, vel.shape[0])

	    return filt_vel, total_vel, vel_bins
	
	def compute_rt_per_trial(self, trials): 
		'''
		Compute the reaction time for all trials in array trials.
		'''

		#Extract go_cue_indices in units of hdf file row number
		go_cue_ix = np.array([self.table.root.task_msgs[j-3]['time'] for j, i in enumerate(self.table.root.task_msgs) if i['msg']=='reward'])
		go_cue_ix = go_cue_ix[trials]

		# Calculate filtered velocity and 'velocity mag. in target direction'
		filt_vel, total_vel, vel_bins = self.get_cursor_velocity(go_cue_ix, 0., 2., use_filt_vel=False)
		## Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
		kin_feat = get_rt_change_deriv(total_vel.T, vel_bins, d_vel_thres = 0.3, fs=60)
		return kin_feat[:,1], total_vel

	def PlotReactionAndTrialTimes(self, num_trials_slide):
		'''
		This method computes the sliding average over num_trials_slide bins of the reaction times and trial times, as well as the histograms.

		NOTE: Need to add reaction time computation and plotting.
		'''

		trial_times = (self.state_time[self.ind_reward_states] - self.state_time[self.ind_reward_states-5])/60.		# gives trial lengths in seconds (calculated as diff between reward state and center state)
		transition_trials = np.ravel(np.nonzero(self.stress_trial[1:] - self.stress_trial[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials)-1)									# add end of final block
		num_blocks = len(transition_trials)

		block_length = np.zeros(num_blocks)  															# holder for length of blocks in minutes

		cmap = mpl.cm.hsv
		plt.figure()

		for i, trial_end in enumerate(transition_trials):
			if i==0:
				block_trial_times = trial_times[0:trial_end+1]
				num_trials_prev_block = 0
				trial_ind = np.arange(0,trial_end+1)
			else:
				block_trial_times = trial_times[transition_trials[i-1]+1:trial_end+1]
				num_trials_prev_block = transition_trials[i-1]
				trial_ind = np.arange(transition_trials[i-1]+1,trial_end+1)

			
			# compute average mins/trial
			sliding_avg_trial_length = trial_sliding_avg(block_trial_times, num_trials_slide)   	# gives the average num of sec/trial

			max_trial_length = np.max(block_trial_times)
			bins_trial_time = np.linspace(0,max_trial_length+1,10)

			# compute reaction times
			rt_per_trial, total_vel = self.compute_rt_per_trial_FreeChoiceTask(trial_ind)
			# compute sliding average of reaction times
			sliding_avg_rt = trial_sliding_avg(rt_per_trial, num_trials_slide)
			#max_rt = np.max(rt_per_trial)
			max_rt = 0.5
			bins_rt = np.linspace(0, max_rt+.1,10)
			
			
			plt.figure(1)
			plt.subplot(211)
			plt.plot(np.arange(len(sliding_avg_trial_length)) + num_trials_prev_block, sliding_avg_trial_length,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			
			plt.subplot(212)
			plt.hist(block_trial_times, bins_trial_time, alpha=0.5, label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))

			plt.figure(2)
			plt.subplot(211)
			plt.plot(np.arange(len(sliding_avg_rt)) + num_trials_prev_block, sliding_avg_rt,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			
			plt.subplot(212)
			plt.hist(rt_per_trial, bins_rt, alpha=0.5, label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))

		# Figure showing the average trial length in secs as a function of trials
		plt.figure(1)
		ax1 = plt.subplot(211)
		plt.xlabel('Trials')
		plt.ylabel('Avg Trial Length (s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		ax1 = plt.subplot(212)
		plt.xlabel('Trial Lengths (s)')
		plt.ylabel('Frequency')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		plt.figure(2)
		ax1 = plt.subplot(211)
		plt.xlabel('Trials')
		plt.ylabel('Avg Reaction Time (s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		ax1 = plt.subplot(212)
		plt.xlabel('Reaction Time (s)')
		plt.ylabel('Frequency')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		plt.show()

		return 

class StressBehaviorWithDrugs_CenterOut():
	'''
	For use with JoystickMulti
	'''

	def __init__(self, hdf_file):
		self.filename =  hdf_file
		self.table = tables.openFile(self.filename)

		self.state = self.table.root.task_msgs[:]['msg']
		self.state_time = self.table.root.task_msgs[:]['time']
		#self.stress_type = self.table.root.task[:]['stress_trial']
	  
		self.ind_wait_states = np.ravel(np.nonzero(self.state == 'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_reward_states = np.ravel(np.nonzero(self.state == 'reward'))
		#self.ind_hold_center_states = np.ravel(np.nonzero(self.state == 'hold_center'))
		self.ind_hold_center_states = self.ind_reward_states-5
		self.ind_target_states = np.ravel(np.nonzero(self.state == 'target'))
		
		#self.stress_trial = np.ravel(self.stress_type[self.state_time[self.ind_reward_states-4]])
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_reward_states.size
		self.trial_times = (self.state_time[self.ind_reward_states] - self.state_time[self.ind_reward_states-4])/60.		# gives trial lengths in seconds (calculated as diff between reward state and hold center state)
		

	
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


	def get_cursor_velocity(self, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
	    '''
	    hdf file -- task file generated from bmi3d
	    go_cue_ix -- list of go cue indices (units of hdf file row numbers)
	    before_cue_time -- time before go cue to inclue in trial (units of sec)
	    after_cue_time -- time after go cue to include in trial (units of sec)

	    returns a time x (x,y) x trials filtered velocity array
	    '''

	    ix = np.arange(-1*before_cue_time*fs, after_cue_time*fs).astype(int)


	    # Get trial trajectory: 
	    cursor = []
	    for g in go_cue_ix:
	        try:
	            #Get cursor
	            cursor.append(self.table.root.task[ix+g]['cursor'][:, [0, 2]])

	        except:
	            print 'skipping index: ', g, ' -- too close to beginning or end of file'
	    cursor = np.dstack((cursor))    # time x (x,y) x trial
	    
	    dt = 1./fs
	    vel = np.diff(cursor,axis=0)/dt

	    #Filter velocity: 
	    if use_filt_vel:
	        filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
	    else:
	        filt_vel = vel
	    total_vel = np.zeros((int(filt_vel.shape[0]),int(filt_vel.shape[2])))
	    for n in range(int(filt_vel.shape[2])):
	        total_vel[:,n] = np.sqrt(filt_vel[:,0,n]**2 + filt_vel[:,1,n]**2)

	    vel_bins = np.linspace(-1*before_cue_time, after_cue_time, vel.shape[0])

	    return filt_vel, total_vel, vel_bins
	
	def compute_rt_per_trial(self, trials): 
		'''
		Compute the reaction time for all trials in array trials.
		'''

		#Extract go_cue_indices in units of hdf file row number
		go_cue_ix = np.array([self.table.root.task_msgs[j-3]['time'] for j, i in enumerate(self.table.root.task_msgs) if i['msg']=='reward'])
		go_cue_ix = go_cue_ix[trials]

		# Calculate filtered velocity and 'velocity mag. in target direction'
		filt_vel, total_vel, vel_bins = self.get_cursor_velocity(go_cue_ix, 0., 2., use_filt_vel=False)
		## Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
		kin_feat = get_rt_change_deriv(total_vel.T, vel_bins, d_vel_thres = 0.3, fs=60)
		return kin_feat[:,1], total_vel

	def PlotReactionAndTrialTimes(self, num_trials_slide):
		'''
		This method computes the sliding average over num_trials_slide bins of the reaction times and trial times, as well as the histograms.

		NOTE: Need to add reaction time computation and plotting.
		'''

		trial_times = (self.state_time[self.ind_reward_states] - self.state_time[self.ind_reward_states-5])/60.		# gives trial lengths in seconds (calculated as diff between reward state and center state)
		transition_trials = np.ravel(np.nonzero(self.stress_trial[1:] - self.stress_trial[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials)-1)									# add end of final block
		num_blocks = len(transition_trials)

		block_length = np.zeros(num_blocks)  															# holder for length of blocks in minutes

		cmap = mpl.cm.hsv
		plt.figure()

		for i, trial_end in enumerate(transition_trials):
			if i==0:
				block_trial_times = trial_times[0:trial_end+1]
				num_trials_prev_block = 0
				trial_ind = np.arange(0,trial_end+1)
			else:
				block_trial_times = trial_times[transition_trials[i-1]+1:trial_end+1]
				num_trials_prev_block = transition_trials[i-1]
				trial_ind = np.arange(transition_trials[i-1]+1,trial_end+1)

			
			# compute average mins/trial
			sliding_avg_trial_length = trial_sliding_avg(block_trial_times, num_trials_slide)   	# gives the average num of sec/trial

			max_trial_length = np.max(block_trial_times)
			bins_trial_time = np.linspace(0,max_trial_length+1,10)

			# compute reaction times
			rt_per_trial, total_vel = self.compute_rt_per_trial_FreeChoiceTask(trial_ind)
			# compute sliding average of reaction times
			sliding_avg_rt = trial_sliding_avg(rt_per_trial, num_trials_slide)
			#max_rt = np.max(rt_per_trial)
			max_rt = 0.5
			bins_rt = np.linspace(0, max_rt+.1,10)
			
			
			plt.figure(1)
			plt.subplot(211)
			plt.plot(np.arange(len(sliding_avg_trial_length)) + num_trials_prev_block, sliding_avg_trial_length,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			
			plt.subplot(212)
			plt.hist(block_trial_times, bins_trial_time, alpha=0.5, label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))

			plt.figure(2)
			plt.subplot(211)
			plt.plot(np.arange(len(sliding_avg_rt)) + num_trials_prev_block, sliding_avg_rt,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			
			plt.subplot(212)
			plt.hist(rt_per_trial, bins_rt, alpha=0.5, label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))

		# Figure showing the average trial length in secs as a function of trials
		plt.figure(1)
		ax1 = plt.subplot(211)
		plt.xlabel('Trials')
		plt.ylabel('Avg Trial Length (s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		ax1 = plt.subplot(212)
		plt.xlabel('Trial Lengths (s)')
		plt.ylabel('Frequency')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		plt.figure(2)
		ax1 = plt.subplot(211)
		plt.xlabel('Trials')
		plt.ylabel('Avg Reaction Time (s)')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		ax1 = plt.subplot(212)
		plt.xlabel('Reaction Time (s)')
		plt.ylabel('Frequency')
		ax1.get_yaxis().set_tick_params(direction='out')
		ax1.get_xaxis().set_tick_params(direction='out')
		ax1.get_xaxis().tick_bottom()
		ax1.get_yaxis().tick_left()
		plt.legend()

		plt.show()

		return 