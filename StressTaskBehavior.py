import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv



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
		self.ind_target_states = np.ravel(np.nonzero(self.state == 'target'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == 'check_reward'))
		#self.trial_type = np.ravel(trial_type[state_time[ind_center_states]])
		#self.stress_type = np.ravel(stress_type[state_time[ind_center_states]])
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_check_reward_states.size

	'''
	Sliding average of number of successful trials/min, sliding average of probability of selecting better option, raster plots showing each of these (do for all blocks of task)
	'''


	def PlotTrialsCompleted(self, num_trials_slide):
		'''
		This method computes the sliding average over num_trials_slide bins of the total number of trials successfully completed in each block.
		'''
		trial_times = self.state_time[self.ind_check_reward_states]/(60.**2)
		trials = np.ravel(self.stress_type[self.state_time[self.ind_check_reward_states]])
		transition_trials = np.ravel(np.nonzero(trials[1:] - trials[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials))									# add end of final block
		num_blocks = len(transition_trials)

		block_length = np.zeros(num_blocks)  															# holder for length of blocks in minutes

		cmap = mpl.cm.hsv
		plt.figure()

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
			
			plt.figure(1)
			plt.subplot(211)
			plt.plot(block_trial_times + end_prev_block, sliding_avg_trial_rate,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			plt.plot(block_trial_times + end_prev_block,np.ones(len(block_trial_times)),'|', color=cmap(i/float(num_blocks)))
    		

			plt.figure(1)
			plt.subplot(212)
			plt.plot(block_trial_times/block_trial_times[-1] + i, sliding_avg_trial_rate,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			plt.plot(block_trial_times/block_trial_times[-1] + i,np.ones(len(block_trial_times)),'|', color=cmap(i/float(num_blocks)))

		# Figure showing the average rate of trials/min as a function of minutes in the task
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
		plt.show()

		return 


	def PlotTrialChoices(self, num_trials_slide):
		'''
		This method computes the sliding average over num_trials_slide trials of the number of choices for the HV target.
		'''
		trial_times = self.state_time[self.ind_check_reward_states]/(60.**2)
		trials = np.ravel(self.stress_type[self.state_time[self.ind_check_reward_states]])
		transition_trials = np.ravel(np.nonzero(trials[1:] - trials[:-1])) 								# these trial numbers mark the ends of a block
		transition_trials = np.append(transition_trials, len(trials))									# add end of final block
		num_blocks = len(transition_trials)

		target_choices = self.state[self.ind_check_reward_states - 2]

		block_length = np.zeros(num_blocks)  															# holder for length of blocks in minutes

		cmap = mpl.cm.hsv
		plt.figure()

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
			hv_choices = np.zeros(len(block_trial_times))
			hv_choices[choices=='hold_targetH'] = 1

			sliding_avg_hv_choices = trial_sliding_avg(hv_choices, num_trials_slide)   	# gives the average num of mins/trial
			
			plt.figure(1)
			plt.subplot(211)
			plt.plot(block_trial_times + end_prev_block, sliding_avg_hv_choices,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			plt.plot(block_trial_times + end_prev_block,0.1*hv_choices - 0.2,'|', color=cmap(i/float(num_blocks)))
    		

			plt.figure(1)
			plt.subplot(212)
			plt.plot(block_trial_times/block_trial_times[-1] + i, sliding_avg_hv_choices,label='Block %i - Stress type: %i' % (i+1, trials[trial_end-1]), color=cmap(i/float(num_blocks)))
			plt.plot(block_trial_times/block_trial_times[-1] + i,0.1*hv_choices - 0.2,'|', color=cmap(i/float(num_blocks)))

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

		return

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