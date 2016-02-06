'''
Stress Trial Days:
*11/24/2015 - Blocks 1, 2 (DIO data didn't save to recording)
*12/4/2015 - Blocks 1 (luig20151204_05.hdf), 2 (luig20151204_07.hdf), 3 (luig20151204_08.hdf)
12/6/2015 - Block 1 (luig20151206_04.hdf)
12/7/2015 - Block 1 (luig20151207_03.hdf)
12/17/2015 - Block 1 (luig20151217_05.hdf; reversal, not stress)
12/22/2015 - Block 1 (luig20151222_05.hdf)
12/23/2015 - Blocks 1 (luig20151223_03.hdf), 2 (luig20151223_05.hdf)
12/28/2015 - Blocks 1 (luig20151228_09.hdf), 2 (luig20151228_11.hdf)
12/29/2015 - Blocks 1 (luig20151229_02.hdf)
1/5/2016 - Block 1 (luig20160105_13.hdf)
1/6/2016 - Block 1 (luig20160106_03.hdf)
1/11/2016 - Block 1 (luig20160111_06.hdf)

12/23 - Block 1: best day
1/27 - Block 3: best day center out

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
- Average power in beta over time of trial (aligned to target hold) (trial x time x power plot)
- Power over all frequencies for entire trial (trial x freq x power plot)
'''

import numpy as np 
import scipy as sp
import tables
from neo import io
from PulseMonitorData import findIBIs
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / float(N) 

# Set up code for particular day and block
hdf_filename = 'luig20160106_03.hdf'
filename = 'Luigi20160106_HDEEG'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
#hdf_location = hdf_filename
block_num = 1

num_avg = 50 	# number of trials to compute running average of trial statistics over

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
  
ind_wait_states = np.ravel(np.nonzero(state == 'wait'))   # total number of unique trials
ind_center_states = np.ravel(np.nonzero(state == 'center'))   # total number of totals (includes repeats if trial was incomplete)
ind_target_states = np.ravel(np.nonzero(state == 'target'))
ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]	# free choice trial = 2, instructed = 1
all_instructed_or_freechoice = trial_type[state_time[ind_center_states]]
successful_stress_or_not = np.ravel(stress_type[state_time[ind_check_reward_states]])
all_stress_or_not = np.ravel(stress_type[state_time[ind_center_states]])
rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_check_reward_states]]
rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_check_reward_states]]

num_trials = ind_center_states.size
num_successful_trials = ind_check_reward_states.size
total_states = state.size

trial_success = np.zeros(num_trials)
target = np.zeros(num_trials)
reward = np.zeros(num_trials)
ind_successful_center_states = []
counter = 0 	# counter increments for all successful trials

for i in range(0,num_trials):
	if (state[np.minimum(ind_center_states[i]+5,total_states-1)] == 'check_reward'):	 
		trial_success[i] = 1
		target_state = state[ind_center_states[i] + 3]
		if (target_state == 'hold_targetL'):
			target[i] = 1
			reward[i] = rewarded_reward_scheduleL[counter]
		else:
			target[i] = 2
			reward[i] = rewarded_reward_scheduleH[counter]
		counter += 1
	else:
		trial_success[i] = 0
		target[i] = 0 	# no target selected
		reward[i] = 0 	# no reward givens

# Number of successful stress trials
tot_successful_stress = np.logical_and(trial_success,all_stress_or_not)
successful_stress_trials = float(np.sum(tot_successful_stress))/np.sum(all_stress_or_not)

# Number of successful non-stress trials
tot_successful_reg = np.logical_and(trial_success,np.logical_not(all_stress_or_not))
successful_reg_trials = float(np.sum(tot_successful_reg))/(num_trials - np.sum(all_stress_or_not))

# Response times for successful stress trials
ind_successful_stress = np.ravel(np.nonzero(tot_successful_stress))   	# gives trial index, not row index
row_ind_successful_stress = ind_center_states[ind_successful_stress]		# gives row index
ind_successful_stress_reward = np.ravel(np.nonzero(successful_stress_or_not))
row_ind_successful_stress_reward = ind_check_reward_states[ind_successful_stress_reward]
response_time_successful_stress = (state_time[row_ind_successful_stress_reward] - state_time[row_ind_successful_stress])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response times for successful regular trials
ind_successful_reg = np.ravel(np.nonzero(tot_successful_reg))
row_ind_successful_reg = ind_center_states[ind_successful_reg]
ind_successful_reg_reward = np.ravel(np.nonzero(np.logical_not(successful_stress_or_not)))
row_ind_successful_reg_reward = ind_check_reward_states[ind_successful_reg_reward]
response_time_successful_reg = (state_time[row_ind_successful_reg_reward] - state_time[row_ind_successful_reg])/float(60)

# Target choice for successful stress trials - look at free-choice trials only
tot_successful_fc_stress = np.logical_and(tot_successful_stress,np.ravel(np.equal(all_instructed_or_freechoice,2)))
ind_successful_fc_stress = np.ravel(np.nonzero(tot_successful_fc_stress))
target_choice_successful_stress = target[ind_successful_fc_stress]
reward_successful_stress = reward[ind_successful_fc_stress]
prob_choose_low_successful_stress = np.zeros(len(target_choice_successful_stress))
prob_choose_high_successful_stress = np.zeros(len(target_choice_successful_stress))
prob_reward_high_successful_stress = np.zeros(len(target_choice_successful_stress))
prob_reward_low_successful_stress = np.zeros(len(target_choice_successful_stress))
for i in range(0,len(target_choice_successful_stress)):
	chosen_high_freechoice = target_choice_successful_stress[range(np.maximum(0,i - num_avg),i+1)] == 2
	chosen_low_freechoice = target_choice_successful_stress[range(np.maximum(0,i - num_avg),i+1)] == 1
	reward_high_freechoice = np.logical_and(chosen_high_freechoice,reward_successful_stress[range(np.maximum(0,i - num_avg),i+1)])
	reward_low_freechoice = np.logical_and(chosen_low_freechoice,reward_successful_stress[range(np.maximum(0,i - num_avg),i+1)])
	
	prob_choose_low_successful_stress[i] = float(np.sum(chosen_low_freechoice))/chosen_low_freechoice.size
	prob_choose_high_successful_stress[i] = float(np.sum(chosen_high_freechoice))/chosen_high_freechoice.size
	prob_reward_high_successful_stress[i] = float(sum(reward_high_freechoice))/(sum(chosen_high_freechoice) + (sum(chosen_high_freechoice)==0))  # add logic statment to denominator so we never divide by 0
	prob_reward_low_successful_stress[i] = float(sum(reward_low_freechoice))/(sum(chosen_low_freechoice) + (sum(chosen_low_freechoice)==0))


# Target choice for successful regular trials - look at free-choice trials only
tot_successful_fc_reg = np.logical_and(tot_successful_reg,np.ravel(np.equal(all_instructed_or_freechoice,2)))
ind_successful_fc_reg = np.ravel(np.nonzero(tot_successful_fc_reg))
target_choice_successful_reg = target[ind_successful_fc_reg]
reward_successful_reg = reward[ind_successful_fc_reg]
prob_choose_low_successful_reg = np.zeros(len(target_choice_successful_reg))
prob_choose_high_successful_reg = np.zeros(len(target_choice_successful_reg))
prob_reward_high_successful_reg = np.zeros(len(target_choice_successful_reg))
prob_reward_low_successful_reg = np.zeros(len(target_choice_successful_reg))
for i in range(0,len(target_choice_successful_reg)):
	chosen_high_freechoice = target_choice_successful_reg[range(np.maximum(0,i - num_avg),i+1)] == 2
	chosen_low_freechoice = target_choice_successful_reg[range(np.maximum(0,i - num_avg),i+1)] == 1
	reward_high_freechoice = np.logical_and(chosen_high_freechoice,reward_successful_reg[range(np.maximum(0,i - num_avg),i+1)])
	reward_low_freechoice = np.logical_and(chosen_low_freechoice,reward_successful_reg[range(np.maximum(0,i - num_avg),i+1)])
	
	prob_choose_low_successful_reg[i] = float(np.sum(chosen_low_freechoice))/chosen_low_freechoice.size
	prob_choose_high_successful_reg[i] = float(np.sum(chosen_high_freechoice))/chosen_high_freechoice.size
	prob_reward_high_successful_reg[i] = float(sum(reward_high_freechoice))/(sum(chosen_high_freechoice) + (sum(chosen_high_freechoice)==0))  # add logic statment to denominator so we never divide by 0
	prob_reward_low_successful_reg[i] = float(sum(reward_low_freechoice))/(sum(chosen_low_freechoice) + (sum(chosen_low_freechoice)==0))



plt.figure()
plt.subplot(2,1,1)
plt.plot(range(0,len(prob_choose_low_successful_stress)),prob_choose_low_successful_stress,'r',label='Target - Low')
plt.plot(range(0,len(prob_choose_low_successful_stress)),prob_reward_low_successful_stress,'r--',label='Reward - Low')
plt.plot(range(0,len(prob_choose_low_successful_stress)),prob_choose_high_successful_stress,'b',label='Target - High')
plt.plot(range(0,len(prob_choose_low_successful_stress)),prob_reward_high_successful_stress,'b--',label='Reward - High')
plt.title('Stress Trial Performance: Trial Completion Rate %f' % (successful_stress_trials))
plt.xlabel('Trial Number')
plt.ylabel('Probability')
plt.ylim((0,1.1))
plt.xlim((0,len(prob_choose_low_successful_stress)))
#plt.legend()
plt.subplot(2,1,2)
plt.plot(range(0,len(prob_choose_low_successful_reg)),prob_choose_low_successful_reg,'r',label='Target - Low')
plt.plot(range(0,len(prob_choose_low_successful_reg)),prob_reward_low_successful_reg,'r--',label='Reward - Low')
plt.plot(range(0,len(prob_choose_low_successful_reg)),prob_choose_high_successful_reg,'b',label='Target - High')
plt.plot(range(0,len(prob_choose_low_successful_reg)),prob_reward_high_successful_reg,'b--',label='Reward - High')
plt.title('Regular Trial Performance: Trial Completion Rate %f' % successful_reg_trials)
plt.xlabel('Trial Number')
plt.ylabel('Probability')
plt.ylim((0,1.1))
plt.xlim((0,len(prob_choose_low_successful_reg)))
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_TrialPerformance.svg')


# Load syncing data for hdf file and TDT recording
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

r = io.TdtIO(TDT_tank)
bl = r.read_block(lazy=False,cascade=True)
hdeeg = dict()
# Get Pulse and Pupil Data
for sig in bl.segments[block_num-1].analogsignals:
	if (sig.name == 'PupD 1'):
		pupil_data = np.ravel(sig)
		pupil_samprate = sig.sampling_rate.item()
	if (sig.name == 'HrtR 1'):
		pulse_data = np.ravel(sig)
		pulse_samprate = sig.sampling_rate.item()
	if (sig.name[0:4] == 'EEGx'):
		channel = sig.channel_index
		if channel not in [4,6,8]:
			hdeeg_samprate = sig.sampling_rate.item()
			hdeeg[channel] = np.ravel(sig)

cutoff_f = 50
cutoff_f = float(cutoff_f)/(pupil_samprate/2)
num_taps = 100
lpf = signal.firwin(num_taps,cutoff_f,window='hamming')
#pupil_data = signal.lfilter(lpf,1,pupil_data[eyes_open])


# divide up analysis for regular trials before stress trials, stress trials, and regular trials after stress trials are introduced
hdf_rows = np.ravel(hdf_times['row_number'])
hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

# Convert DIO TDT sample numbers to for pupil and pulse data:
# if dio sample num is x, then data sample number is R*(x-1) + 1 where
# R = data_sample_rate/dio_sample_rate
pulse_dio_sample_num = (float(pulse_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
pupil_dio_sample_num = (float(pupil_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
hdeeg_dio_sample_num = (float(hdeeg_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1

state_row_ind_successful_stress = state_time[row_ind_successful_stress]
state_row_ind_successful_reg = state_time[row_ind_successful_reg]
pulse_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
pupil_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
hdeeg_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
pulse_ind_successful_reg_before = []
pulse_ind_successful_reg_after = []
pupil_ind_successful_reg_before = []
pupil_ind_successful_reg_after = []
hdeeg_ind_successful_reg_before = []
hdeeg_ind_successful_reg_after = []

for i in range(0,len(row_ind_successful_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
	pulse_ind_successful_stress[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_successful_stress[i] = pupil_dio_sample_num[hdf_index]
	hdeeg_ind_successful_stress[i] = hdeeg_dio_sample_num[hdf_index]

ind_start_stress = row_ind_successful_stress[0]
for i in range(0,len(state_row_ind_successful_reg)):
	if (row_ind_successful_reg[i] < ind_start_stress):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		pulse_ind_successful_reg_before.append(pulse_dio_sample_num[hdf_index])
		pupil_ind_successful_reg_before.append(pupil_dio_sample_num[hdf_index])
		hdeeg_ind_successful_reg_before.append(hdeeg_dio_sample_num[hdf_index])
	else:
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		pulse_ind_successful_reg_after.append(pulse_dio_sample_num[hdf_index])
		pupil_ind_successful_reg_after.append(pupil_dio_sample_num[hdf_index])
		hdeeg_ind_successful_reg_after.append(hdeeg_dio_sample_num[hdf_index])

# Find IBIs and pupil data for all stress trials. 
samples_pulse_successful_stress = np.floor(response_time_successful_stress*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil_successful_stress = np.floor(response_time_successful_stress*pupil_samprate)
samples_hdeeg_successful_stress = np.floor(response_time_successful_stress*hdeeg_samprate)
#ibi_stress = dict()
#pupil_stress = dict()
ibi_stress_mean = []
ibi_stress_std = []
pupil_stress_mean = []
pupil_stress_std = []
all_ibi_stress = []
all_pupil_stress = []
for i in range(0,len(row_ind_successful_stress)):
	pulse_snippet = pulse_data[pulse_ind_successful_stress[i]:pulse_ind_successful_stress[i]+samples_pulse_successful_stress[i]]
	ibi_snippet = findIBIs(pulse_snippet,pulse_samprate)
	all_ibi_stress += ibi_snippet.tolist()
	ibi_stress_mean.append(np.mean(ibi_snippet))
	ibi_stress_std.append(np.mean(ibi_snippet))
	#ibi_stress['i'] = ibi_snippet
	pupil_snippet = pupil_data[pupil_ind_successful_stress[i]:pupil_ind_successful_stress[i]+samples_pupil_successful_stress[i]]
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
			rm_range = range(np.maximum(eyes_closed_ind[i-1]-20,0),np.minimum(eyes_closed_ind[i] + 20,len(pupil_snippet)-1))
			rm_indices = [pupil_snippet_range.index(rm_range[ind]) for ind in range(0,len(rm_range)) if (rm_range[ind] in pupil_snippet_range)]
			pupil_snippet_range = np.delete(pupil_snippet_range,rm_indices)
			pupil_snippet_range = pupil_snippet_range.tolist()
	#pupil_snippet = signal.lfilter(lpf,1,pupil_snippet[eyes_open])
	pupil_snippet = pupil_snippet[pupil_snippet_range]
	pupil_snippet_mean = np.mean(pupil_snippet)
	pupil_snippet_std = np.std(pupil_snippet)
	window = np.floor(pupil_samprate/10) # sample window equal to ~100 ms
	pupil_snippet = (pupil_snippet[0:window]- pupil_snippet_mean)/float(pupil_snippet_std)
	all_pupil_stress += pupil_snippet.tolist()
	#pupil_stress['i'] = pupil_snippet
	pupil_stress_mean.append(np.mean(pupil_snippet))
	pupil_stress_std.append(np.mean(pupil_snippet))

Fs = hdeeg_samprate
density_length = 30
for chann in hdeeg.keys():
	trial_power = np.zeros([density_length,len(row_ind_successful_stress)])
	beta_power = np.zeros([len(row_ind_successful_stress),10])
	low_power = np.zeros([len(row_ind_successful_stress),10])
	for i in range(0,len(row_ind_successful_stress)):	
		hdeeg_snippet = hdeeg[channel][hdeeg_ind_successful_stress[i]:hdeeg_ind_successful_stress[i]+samples_hdeeg_successful_stress[i]]
		#num_timedom_samples = hdeeg_snippet.size
		#time = [float(t)/Fs for t in range(0,num_timedom_samples)]
 		#freq, Pxx_den = signal.welch(hdeeg_snippet, Fs, nperseg=1024)
 		#trial_power[:,i] = Pxx_den[0:density_length]
 		hdeeg_snippet_aligned_to_end = hdeeg_snippet[-hdeeg_samprate:]
 		hdeeg_snippet_aligned_to_beginning = hdeeg_snippet[0:hdeeg_samprate]
 		num_timedom_samples = hdeeg_snippet_aligned_to_end.size
 		time = [float(t)/Fs for t in range(0,num_timedom_samples)]
 		#Pxx, freqs, bins, im = plt.specgram(hdeeg_snippet_aligned_to_beginning, NFFT=512, Fs=Fs, noverlap=256)
 		Pxx, freqs, bins = mlab.specgram(hdeeg_snippet_aligned_to_end, NFFT=512, Fs=Fs, noverlap=256)
 		#plt.pcolor(Pxx)
 		freq_low = np.less_equal(freqs,5)
 		freq_beta = np.logical_and(np.greater(freqs,50),np.less(freqs,100))
 		freq_low_ind = np.ravel(np.nonzero(freq_low))
 		freq_beta_ind = np.ravel(np.nonzero(freq_beta))
 		Pxx_low = np.sum(Pxx[freq_low_ind],axis=0)/np.sum(Pxx,axis=0)
 		Pxx_beta = np.sum(Pxx[freq_beta_ind],axis=0)/np.sum(Pxx,axis=0)
 		beta_power[i,:] = Pxx_beta
 		low_power[i,:] = Pxx_low
 	# plot figures here
 	z_min, z_max = -np.abs(trial_power).max(), np.abs(trial_power).max()
	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(low_power, cmap='RdBu')
	#plt.xticks(np.arange(0.5,density_length+0.5),freq[0:density_length])
	#plt.yticks(range(0,len(row_ind_successful_stress)))
	plt.title('Channel %i - Spectrogram: 0 - 10 Hz power' % (chann))
	plt.axis('auto')
	plt.colorbar()
	#z_min, z_max = -np.abs(beta_power).max(), np.abs(beta_power).max()
	plt.subplot(1, 2, 2)
	plt.imshow(beta_power, cmap='RdBu')
	#plt.xticks(np.arange(0.5,len(bins)+0.5),bins)
	#plt.yticks(range(0,len(row_ind_successful_stress)))
	plt.title('Spectrogram: 10 - 30 Hz power')
	plt.axis('auto')
	# set the limits of the plot to the limits of the data
	#plt.axis([x.min(), x.max(), y.min(), y.max()])
	plt.colorbar()
	#plt.show()
	#plt.close()

mean_ibi_stress = np.mean(all_ibi_stress)
std_ibi_stress = np.std(all_ibi_stress)
nbins_ibi_stress = np.arange(mean_ibi_stress-10*std_ibi_stress,mean_ibi_stress+10*std_ibi_stress,float(std_ibi_stress)/2)
ibi_stress_hist,nbins_ibi_stress = np.histogram(all_ibi_stress,bins=nbins_ibi_stress)
nbins_ibi_stress = nbins_ibi_stress[1:]
ibi_stress_hist = ibi_stress_hist/float(len(all_ibi_stress))

mean_pupil_stress = np.mean(all_pupil_stress)
std_pupil_stress = np.std(all_pupil_stress)
nbins_pupil_stress = np.arange(mean_pupil_stress-10*std_pupil_stress,mean_pupil_stress+10*std_pupil_stress,float(std_pupil_stress)/2)
pupil_stress_hist,nbins_pupil_stress = np.histogram(all_pupil_stress,bins=nbins_pupil_stress)
nbins_pupil_stress = nbins_pupil_stress[1:]
pupil_stress_hist = pupil_stress_hist/float(len(all_pupil_stress))

# Find IBIs and pupil data for all regular trials. 
samples_pulse_successful_reg = np.floor(response_time_successful_reg*pulse_samprate)
samples_pupil_successful_reg = np.floor(response_time_successful_reg*pupil_samprate)
#ibi_reg_before = dict()
#pupil_reg_before = dict()
#ibi_reg_after = dict()
#pupil_reg_after = dict()

ibi_reg_before_mean = []
ibi_reg_before_std = []
ibi_reg_after_mean = []
ibi_reg_after_std = []
pupil_reg_before_mean = []
pupil_reg_before_std = []
pupil_reg_after_mean = []
pupil_reg_after_std = []
all_ibi_reg_before = []
all_ibi_reg_after = []
all_pupil_reg_before = []
all_pupil_reg_after = []
count_before = 0
for i in range(0,len(row_ind_successful_reg)):
	if (row_ind_successful_reg[i] < ind_start_stress):
		pulse_snippet = pulse_data[pulse_ind_successful_reg_before[i]:pulse_ind_successful_reg_before[i]+samples_pulse_successful_reg[i]]
		ibi_snippet = findIBIs(pulse_snippet,pulse_samprate)
		all_ibi_reg_before += ibi_snippet.tolist()
		ibi_reg_before_mean.append(np.mean(ibi_snippet))
		ibi_reg_before_std.append(np.std(ibi_snippet))
		#ibi_reg_before[num2str(i)] = ibi_snippet
		pupil_snippet = pupil_data[pupil_ind_successful_reg_before[i]:pupil_ind_successful_reg_before[i]+samples_pupil_successful_reg[i]]
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
				rm_range = range(np.maximum(eyes_closed_ind[i-1]-20,0),np.minimum(eyes_closed_ind[i] + 20,len(pupil_snippet)-1))
				rm_indices = [pupil_snippet_range.index(rm_range[ind]) for ind in range(0,len(rm_range)) if (rm_range[ind] in pupil_snippet_range)]
				pupil_snippet_range = np.delete(pupil_snippet_range,rm_indices)
				pupil_snippet_range = pupil_snippet_range.tolist()
	
		pupil_snippet = pupil_snippet[pupil_snippet_range]
		#eyes_open = np.nonzero(np.greater(pupil_snippet,0.5))
		#eyes_open = np.ravel(eyes_open)
		#pupil_snippet = signal.lfilter(lpf,1,pupil_snippet[eyes_open])
		pupil_snippet_mean = np.mean(pupil_snippet)
		pupil_snippet_std = np.std(pupil_snippet)
		window = np.floor(pupil_samprate/10) # sample window equal to ~100 ms
		pupil_snippet = (pupil_snippet[0:window]- pupil_snippet_mean)/float(pupil_snippet_std)
		all_pupil_reg_before += pupil_snippet.tolist()
		pupil_reg_before_mean.append(np.mean(pupil_snippet))
		pupil_reg_before_std.append(np.std(pupil_snippet))
		#pupil_reg_before[num2str(i)] = pupil_snippet
		count_before += 1
	else:
		pulse_snippet = pulse_data[pulse_ind_successful_reg_after[i-count_before]:pulse_ind_successful_reg_after[i-count_before]+samples_pulse_successful_reg[i]]
		ibi_snippet = findIBIs(pulse_snippet,pulse_samprate)
		all_ibi_reg_after += ibi_snippet.tolist()
		ibi_reg_after_mean.append(np.mean(ibi_snippet))
		ibi_reg_after_std.append(np.mean(ibi_snippet))
		#ibi_reg_after['i-count_before'] = ibi_snippet
		pupil_snippet = pupil_data[pupil_ind_successful_reg_after[i-count_before]:pupil_ind_successful_reg_after[i-count_before]+samples_pupil_successful_reg[i]]
		#eyes_open = np.nonzero(np.greater(pupil_snippet,0.5))
		#eyes_open = np.ravel(eyes_open)
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
				rm_range = range(np.maximum(eyes_closed_ind[i-1]-20,0),np.minimum(eyes_closed_ind[i] + 20,len(pupil_snippet)-1))
				rm_indices = [pupil_snippet_range.index(rm_range[ind]) for ind in range(0,len(rm_range)) if (rm_range[ind] in pupil_snippet_range)]
				pupil_snippet_range = np.delete(pupil_snippet_range,rm_indices)
				pupil_snippet_range = pupil_snippet_range.tolist()
	
		pupil_snippet = pupil_snippet[pupil_snippet_range]
		pupil_snippet_mean = np.mean(pupil_snippet)
		pupil_snippet_std = np.std(pupil_snippet)
		window = np.floor(pupil_samprate/10) # sample window equal to ~100 ms
		pupil_snippet = (pupil_snippet[0:window]- pupil_snippet_mean)/float(pupil_snippet_std)
		#pupil_snippet = signal.lfilter(lpf,1,pupil_snippet[eyes_open])
		all_pupil_reg_after += pupil_snippet.tolist()
		#pupil_reg_after['i-count_before'] = pupil_snippet
		pupil_reg_after_mean.append(np.mean(pupil_snippet))
		pupil_reg_after_std.append(np.std(pupil_snippet))

mean_ibi_reg_before = np.mean(all_ibi_reg_before)
std_ibi_reg_before = np.std(all_ibi_reg_before)
nbins_ibi_reg_before = np.arange(mean_ibi_reg_before-10*std_ibi_reg_before,mean_ibi_reg_before+10*std_ibi_reg_before,float(std_ibi_reg_before)/2)
ibi_reg_before_hist,nbins_ibi_reg_before = np.histogram(all_ibi_reg_before,bins=nbins_ibi_reg_before)
nbins_ibi_reg_before = nbins_ibi_reg_before[1:]
ibi_reg_before_hist = ibi_reg_before_hist/float(len(all_ibi_reg_before))

mean_pupil_reg_before = np.mean(all_pupil_reg_before)
std_pupil_reg_before = np.std(all_pupil_reg_before)
nbins_pupil_reg_before = np.arange(mean_pupil_reg_before-10*std_pupil_reg_before,mean_pupil_reg_before+10*std_pupil_reg_before,float(std_pupil_reg_before)/2)
pupil_reg_before_hist,nbins_pupil_reg_before = np.histogram(all_pupil_reg_before,bins=nbins_pupil_reg_before)
nbins_pupil_reg_before = nbins_pupil_reg_before[1:]
pupil_reg_before_hist = pupil_reg_before_hist/float(len(all_pupil_reg_before))

mean_ibi_reg_after = np.mean(all_ibi_reg_after)
std_ibi_reg_after = np.std(all_ibi_reg_after)
nbins_ibi_reg_after = np.arange(mean_ibi_reg_after-10*std_ibi_reg_after,mean_ibi_reg_after+10*std_ibi_reg_after,float(std_ibi_reg_after)/2)
ibi_reg_after_hist,nbins_ibi_reg_after = np.histogram(all_ibi_reg_after,bins=nbins_ibi_reg_after)
nbins_ibi_reg_after = nbins_ibi_reg_after[1:]
ibi_reg_after_hist = ibi_reg_after_hist/float(len(all_ibi_reg_after))

mean_pupil_reg_after = np.mean(all_pupil_reg_after)
std_pupil_reg_after = np.std(all_pupil_reg_after)
nbins_pupil_reg_after = np.arange(mean_pupil_reg_after-10*std_pupil_reg_after,mean_pupil_reg_after+10*std_pupil_reg_after,float(std_pupil_reg_after)/2)
pupil_reg_after_hist,nbins_pupil_reg_after = np.histogram(all_pupil_reg_after,bins=nbins_pupil_reg_after)
nbins_pupil_reg_after = nbins_pupil_reg_after[1:]
pupil_reg_after_hist = pupil_reg_after_hist/float(len(all_pupil_reg_after))

# Pearson correlations
r_before, r_p_before = stats.pearsonr(ibi_reg_before_mean, pupil_reg_before_mean)
r_stress, r_p_stress = stats.pearsonr(ibi_stress_mean,pupil_stress_mean)
r_after, r_p_after = stats.pearsonr(ibi_reg_after_mean,pupil_reg_after_mean)

# Linear fits to go along with plots
m,b = np.polyfit(range(1,len(ibi_stress_mean)+1), ibi_stress_mean, 1)
ibi_stress_mean_fit = m*np.arange(1,len(ibi_stress_mean)+1) + b

m,b = np.polyfit(range(1,len(ibi_reg_before_mean)+1), ibi_reg_before_mean, 1)
ibi_reg_before_mean_fit = m*np.arange(1,len(ibi_reg_before_mean)+1) + b

m,b = np.polyfit(range(1,len(ibi_reg_after_mean)+1), ibi_reg_after_mean, 1)
ibi_reg_after_mean_fit = m*np.arange(1,len(ibi_reg_after_mean)+1) + b

m,b = np.polyfit(range(1,len(pupil_stress_mean)+1), pupil_stress_mean, 1)
pupil_stress_mean_fit = m*np.arange(1,len(pupil_stress_mean)+1) + b

m,b = np.polyfit(range(1,len(pupil_reg_before_mean)+1), pupil_reg_before_mean, 1)
pupil_reg_before_mean_fit = m*np.arange(1,len(pupil_reg_before_mean)+1) + b

m,b = np.polyfit(range(1,len(pupil_reg_after_mean)+1), pupil_reg_after_mean, 1)
pupil_reg_after_mean_fit = m*np.arange(1,len(pupil_reg_after_mean)+1) + b

plt.figure()
plt.subplot(3,1,2)
plt.plot(range(1,len(ibi_stress_mean)+1),ibi_stress_mean,'r')
plt.plot(range(1,len(ibi_stress_mean)+1),ibi_stress_mean_fit,'r--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Stress Trials')

plt.subplot(3,1,1)
plt.plot(range(1,len(ibi_reg_before_mean)+1),ibi_reg_before_mean,'b')
plt.plot(range(1,len(ibi_reg_before_mean)+1),ibi_reg_before_mean_fit,'b--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Regular Trials before Stress')

plt.subplot(3,1,3)
plt.plot(range(1,len(ibi_reg_after_mean)+1),ibi_reg_after_mean,'k')
plt.plot(range(1,len(ibi_reg_after_mean)+1),ibi_reg_after_mean_fit,'k--')
plt.xlabel('Trial')
plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Regular Trials after Stress')
plt.tight_layout()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_TrialIBI.svg')

plt.figure()
plt.subplot(3,1,2)
plt.plot(range(1,len(pupil_stress_mean)+1),pupil_stress_mean,'r')
plt.plot(range(1,len(pupil_stress_mean)+1),pupil_stress_mean_fit,'r--')
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Stress Trials')
plt.text(1,np.max(pupil_stress_mean),'Pulse v. Pupil:r=%f \n p=%f' % (r_stress,r_p_stress))

plt.subplot(3,1,1)
plt.plot(range(1,len(pupil_reg_before_mean)+1),pupil_reg_before_mean,'b')
plt.plot(range(1,len(pupil_reg_before_mean)+1),pupil_reg_before_mean_fit,'b--')
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Regular Trials before Stress')
plt.text(1,np.max(pupil_reg_before_mean),'Pulse v. Pupil:r=%f \n p=%f' % (r_before,r_p_before))

plt.subplot(3,1,3)
plt.plot(range(1,len(pupil_reg_after_mean)+1),pupil_reg_after_mean,'k')
plt.plot(range(1,len(pupil_reg_after_mean)+1),pupil_reg_after_mean_fit,'k--')
plt.xlabel('Trial')
plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Regular Trials after Stress')
plt.text(1,np.max(pupil_reg_after_mean),'Pulse v. Pupil:r=%f \n p=%f' % (r_after,r_p_after))
plt.tight_layout()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_TrialPupil-DeleteBlinks.svg')

# IBI plots
# Compute significance
F_ibi,p_ibi = stats.f_oneway(all_ibi_reg_before,all_ibi_reg_after,all_ibi_stress)
F_pupil,p_pupil = stats.f_oneway(all_pupil_stress,all_ibi_reg_after,all_ibi_reg_before)

plt.figure()
plt.plot(nbins_ibi_reg_before,ibi_reg_before_hist,'b',label='Before Stress')
plt.fill_between(nbins_ibi_reg_before[17:22],ibi_reg_before_hist[17:22],np.zeros(5),facecolor='blue',linewidth=0.1,alpha=0.5)
plt.plot([nbins_ibi_reg_before[19],nbins_ibi_reg_before[19]],[0,ibi_reg_before_hist[19]],'b--')
plt.plot(nbins_ibi_stress,ibi_stress_hist,'r',label='Stress')
plt.fill_between(nbins_ibi_stress[17:22],ibi_stress_hist[17:22],np.zeros(5),facecolor='red',linewidth=0.1,alpha=0.5)
plt.plot([nbins_ibi_stress[19],nbins_ibi_stress[19]],[0,ibi_stress_hist[19]],'r--')
plt.xlabel('IBI (s)')
plt.ylabel('Frequency')
plt.xlim((0.1,0.6))
plt.title('IBI Distribtions')

plt.plot(nbins_ibi_reg_after,ibi_reg_after_hist,'k',label='After Stress')
plt.fill_between(nbins_ibi_reg_after[17:22],ibi_reg_after_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1,alpha=0.5)
plt.plot([nbins_ibi_reg_after[19],nbins_ibi_reg_after[19]],[0,ibi_reg_after_hist[19]],'k--')
if (p_ibi < 0.05):
	t_before_stress,p_before_stress = stats.ttest_ind(all_ibi_reg_before,all_ibi_stress, axis=0, equal_var=True)
	t_after_stress,p_after_stress = stats.ttest_ind(all_ibi_reg_after,all_ibi_stress,axis=0,equal_var=True)
	t_before_after,p_before_after = stats.ttest_ind(all_ibi_reg_before,all_ibi_reg_after,axis=0,equal_var=True)
	plt.text(0.12,np.max(ibi_reg_after_hist),'Before v. Stress:p=%f \n Before v. After: p=%f \n Stress v. After: p=%f' % (p_before_stress,p_before_after,p_after_stress))
plt.legend()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_IBIDistribution.svg')

plt.figure()
plt.plot(nbins_pupil_reg_before,pupil_reg_before_hist,'b',label='Regular Before')
plt.fill_between(nbins_pupil_reg_before[17:22],pupil_reg_before_hist[17:22],np.zeros(5),facecolor='blue',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_reg_before[19],nbins_pupil_reg_before[19]],[0,pupil_reg_before_hist[19]],'b--')
plt.plot(nbins_pupil_stress,pupil_stress_hist,'r',label='Stress')
plt.fill_between(nbins_pupil_stress[17:22],pupil_stress_hist[17:22],np.zeros(5),facecolor='red',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_stress[19],nbins_pupil_stress[19]],[0,pupil_stress_hist[19]],'r--')
plt.xlabel('Diameter (AU)')
plt.ylabel('Frequency')
plt.title('Distribution of Pupil Diameters')

plt.plot(nbins_pupil_reg_after,pupil_reg_after_hist,'k',label='Regular After')
plt.fill_between(nbins_pupil_reg_after[17:22],pupil_reg_after_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_reg_after[19],nbins_pupil_reg_after[19]],[0,pupil_reg_after_hist[19]],'k--')
if (p_pupil < 0.05):
	t_before_stress,p_before_stress = stats.ttest_ind(all_pupil_reg_before,all_pupil_stress, axis=0, equal_var=True)
	t_after_stress,p_after_stress = stats.ttest_ind(all_pupil_reg_after,all_pupil_stress,axis=0,equal_var=True)
	t_before_after,p_before_after = stats.ttest_ind(all_pupil_reg_before,all_pupil_reg_after,axis=0,equal_var=True)
	plt.text(nbins_pupil_reg_after[1],np.max(pupil_reg_after_hist)-0.05,'Before v. Stress:p=%f \n Before v. After: p=%f \n Stress v. After: p=%f' % (p_before_stress,p_before_after,p_after_stress))
plt.legend()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_PupilDistribution-DeleteBlinks.svg')
plt.close("all")
hdf.close()




