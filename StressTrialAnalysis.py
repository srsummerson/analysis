import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
from neo import io
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD


# Set up code for particular day and block
hdf_filename = 'mari20160409_07_te1961.hdf'
filename = 'Mario20160409'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
#hdf_location = hdffilename
block_num = 3
stim_freq = 100

lfp1_channels = [34, 39, 44, 45, 71, 76, 80, 82, 84, 90, 93, 94, 95, 96]

num_avg = 50 	# number of trials to compute running average of trial statistics over

'''
Load behavior data
'''
## self.target_index = 1 for instructed, 2 for free choice
## self.stress_trial =1 for stress trial, 0 for regular trial
state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward = FreeChoiceBehavior_withStressTrials(hdf_location)

# Total number of trials
num_trials = ind_center_states.size
total_states = state_time.size

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

# Response time for all stress trials
ind_stress = np.ravel(np.nonzero(all_stress_or_not))
row_ind_stress = ind_center_states[ind_stress]  # gives row index
row_ind_end_stress = np.zeros(len(row_ind_stress))
row_ind_end_stress = row_ind_stress + 2  # targ_transition state occurs two states later for unsuccessful trials
row_ind_end_stress[-1] = np.min([row_ind_end_stress[-1],len(state_time)-1])  # correct final incomplete trial

for i in range(0,len(row_ind_successful_stress)):
	ind = np.where(row_ind_stress == row_ind_successful_stress[i])[0]
	row_ind_end_stress[ind] = row_ind_successful_stress_reward[i]  # for successful trials, update with real end of trial
response_time_stress = (state_time[row_ind_end_stress] - state_time[row_ind_stress])/float(60)

# Response times for successful regular trials
ind_successful_reg = np.ravel(np.nonzero(tot_successful_reg))
row_ind_successful_reg = ind_center_states[ind_successful_reg]
ind_successful_reg_reward = np.ravel(np.nonzero(np.logical_not(successful_stress_or_not)))
row_ind_successful_reg_reward = ind_check_reward_states[ind_successful_reg_reward]
response_time_successful_reg = (state_time[row_ind_successful_reg_reward] - state_time[row_ind_successful_reg])/float(60)

# Response time for all regular trials
ind_reg = np.ravel(np.nonzero(np.logical_not(all_stress_or_not)))
row_ind_reg = ind_center_states[ind_reg]
row_ind_end_reg = np.zeros(len(row_ind_reg))
row_ind_end_reg = np.minimum(row_ind_reg + 5,total_states-1)  # target_transition state occues two states later for successful trials
for i in range(0,len(row_ind_successful_reg)):
	ind = np.where(row_ind_reg == row_ind_successful_reg[i])[0]
	row_ind_end_reg[ind] = row_ind_successful_reg_reward[i]
response_time_reg = (state_time[row_ind_end_reg] - state_time[row_ind_reg])/float(60)

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

'''
Load syncing data for behavior and TDT recording
'''
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

'''
Load pupil dilation and heart rate data
'''
if filename == 'Mario20160320':
	PupD_filename = '/home/srsummerson/storage/tdt/Mario20160320_plex/Mario20160320_Block-1_PupD.csv'
	HrtR_filename = '/home/srsummerson/storage/tdt/Mario20160320_plex/Mario20160320_Block-1_HrtR.csv'
	pupil_data = get_csv_data_singlechannel(PupD_filename)
	pupil_samprate = 3051.8
	pulse_data = get_csv_data_singlechannel(HrtR_filename)
	pulse_samprate = 3051.8
	hdeeg = dict()
	hdeeg_samprate = 3051.8
else:
	r = io.TdtIO(TDT_tank)
	bl = r.read_block(lazy=False,cascade=True)
	print "File read."
	hdeeg = dict()
	# Get Pulse and Pupil Data
	for sig in bl.segments[block_num-1].analogsignals:
		if (sig.name == 'PupD 1'):
			pupil_data = np.ravel(sig)
			pupil_samprate = sig.sampling_rate.item()
		if (sig.name == 'HrtR 1'):
			pulse_data = np.ravel(sig)
			pulse_samprate = sig.sampling_rate.item()
		if (sig.name[0:4] == 'LFP1'):
			channel = sig.channel_index
			if channel in lfp1_channels:
				hdeeg_samprate = sig.sampling_rate.item()
				hdeeg[channel] = np.ravel(sig)


'''
Convert DIO TDT samples for pupil and pulse data for regular and stress trials
'''
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
state_row_ind_stress = state_time[row_ind_stress]
state_row_ind_reg = state_time[row_ind_reg]
pulse_ind_stress = np.zeros(row_ind_stress.size)
pupil_ind_stress = np.zeros(row_ind_stress.size)
hdeeg_ind_stress = np.zeros(row_ind_stress.size)
pulse_ind_reg_before = []
pulse_ind_reg_after = []
pupil_ind_reg_before = []
pupil_ind_reg_after = []
hdeeg_ind_reg_before = []
hdeeg_ind_reg_after = []

for i in range(0,len(row_ind_successful_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
	pulse_ind_successful_stress[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_successful_stress[i] = pupil_dio_sample_num[hdf_index]
	hdeeg_ind_successful_stress[i] = hdeeg_dio_sample_num[hdf_index]
for i in range(0,len(row_ind_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_stress[i]))
	pulse_ind_stress[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_stress[i] = pupil_dio_sample_num[hdf_index]
	hdeeg_ind_stress[i] = hdeeg_dio_sample_num[hdf_index]

if len(row_ind_successful_stress) > 0: 
	ind_start_stress = row_ind_successful_stress[0]
else:
	ind_start_stress = np.inf

ind_start_all_stress = row_ind_stress[0]
for i in range(0,len(state_row_ind_successful_reg)):
	if (row_ind_successful_reg[i] < ind_start_all_stress):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		pulse_ind_successful_reg_before.append(pulse_dio_sample_num[hdf_index])
		pupil_ind_successful_reg_before.append(pupil_dio_sample_num[hdf_index])
		hdeeg_ind_successful_reg_before.append(hdeeg_dio_sample_num[hdf_index])
	else:
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		pulse_ind_successful_reg_after.append(pulse_dio_sample_num[hdf_index])
		pupil_ind_successful_reg_after.append(pupil_dio_sample_num[hdf_index])
		hdeeg_ind_successful_reg_after.append(hdeeg_dio_sample_num[hdf_index])
for i in range(0,len(state_row_ind_reg)):
	if (row_ind_reg[i] < ind_start_all_stress):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
		pulse_ind_reg_before.append(pulse_dio_sample_num[hdf_index])
		pupil_ind_reg_before.append(pupil_dio_sample_num[hdf_index])
		hdeeg_ind_reg_before.append(hdeeg_dio_sample_num[hdf_index])
	else:
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
		pulse_ind_reg_after.append(pulse_dio_sample_num[hdf_index])
		pupil_ind_reg_after.append(pupil_dio_sample_num[hdf_index])
		hdeeg_ind_reg_after.append(hdeeg_dio_sample_num[hdf_index])

'''
Process pupil and pulse data
'''

# Find IBIs and pupil data for all successful stress trials. 
samples_pulse_successful_stress = np.floor(response_time_successful_stress*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil_successful_stress = np.floor(response_time_successful_stress*pupil_samprate)
samples_hdeeg_successful_stress = np.floor(response_time_successful_stress*hdeeg_samprate)
samples_hdeeg_successful_reg = np.floor(response_time_successful_reg*hdeeg_samprate)

ibi_stress_mean, ibi_stress_std, pupil_stress_mean, pupil_stress_std, nbins_ibi_stress, ibi_stress_hist, nbins_pupil_stress, pupil_stress_hist = getIBIandPuilDilation(pulse_data, pulse_ind_successful_stress,samples_pulse_successful_stress, pulse_samprate,pupil_data, pupil_ind_successful_stress,samples_pupil_successful_stress,pupil_samprate)


# Find IBIs and pupil data for all stress trials
samples_pulse_stress = np.floor(response_time_stress*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil_stress = np.floor(response_time_stress*pupil_samprate)
samples_hdeeg_stress = np.floor(response_time_stress*hdeeg_samprate)

ibi_all_stress_mean, ibi_all_stress_std, pupil_all_stress_mean, pupil_all_stress_std, nbins_ibi_all_stress, ibi_all_stress_hist, nbins_pupil_all_stress, pupil_all_stress_hist = getIBIandPuilDilation(pulse_data, pulse_ind_stress,samples_pulse_stress, pulse_samprate,pupil_data, pupil_ind_stress,samples_pupil_stress,pupil_samprate)

# Find IBIs and pupil data for successful and all regular trials. 
samples_pulse_successful_reg = np.floor(response_time_successful_reg*pulse_samprate)
samples_pupil_successful_reg = np.floor(response_time_successful_reg*pupil_samprate)
samples_pulse_reg = np.floor(response_time_reg*pulse_samprate)
samples_pupil_reg = np.floor(response_time_reg*pupil_samprate)

samples_pupil_successful_reg_before = [samples_pupil_successful_reg[i] for i in range(len(row_ind_successful_reg)) if (row_ind_successful_reg[i] < ind_start_stress)]
samples_pupil_successful_reg_after = [samples_pupil_successful_reg[i] for i in range(len(row_ind_successful_reg)) if (row_ind_successful_reg[i] > ind_start_stress)]
samples_pulse_successful_reg_before = [samples_pulse_successful_reg[i] for i in range(len(row_ind_successful_reg)) if (row_ind_successful_reg[i] < ind_start_stress)]
samples_pulse_successful_reg_after = [samples_pulse_successful_reg[i] for i in range(len(row_ind_successful_reg)) if (row_ind_successful_reg[i] > ind_start_stress)]

samples_pupil_reg_before = [samples_pupil_reg[i] for i in range(len(row_ind_reg)) if (row_ind_reg[i] < ind_start_stress)]
samples_pupil_reg_after = [samples_pupil_reg[i] for i in range(len(row_ind_reg)) if (row_ind_reg[i] > ind_start_stress)]
samples_pulse_reg_before = [samples_pulse_reg[i] for i in range(len(row_ind_reg)) if (row_ind_reg[i] < ind_start_stress)]
samples_pulse_reg_after = [samples_pulse_reg[i] for i in range(len(row_ind_reg)) if (row_ind_reg[i] > ind_start_stress)]

ibi_reg_before_mean, ibi_reg_before_std, pupil_reg_before_mean, pupil_reg_before_std, nbins_ibi_reg_before, ibi_reg_before_hist, nbins_pupil_reg_before, pupil_reg_before_hist = getIBIandPuilDilation(pulse_data, pulse_ind_successful_reg_before,samples_pulse_successful_reg_before, pulse_samprate,pupil_data, pupil_ind_successful_reg_before,samples_pupil_successful_reg_before,pupil_samprate)
ibi_reg_after_mean, ibi_reg_after_std, pupil_reg_after_mean, pupil_reg_after_std, nbins_ibi_reg_after, ibi_reg_after_hist, nbins_pupil_reg_after, pupil_reg_after_hist = getIBIandPuilDilation(pulse_data, pulse_ind_successful_reg_after,samples_pulse_successful_reg_after, pulse_samprate,pupil_data, pupil_ind_successful_reg_after,samples_pupil_successful_reg_after,pupil_samprate)
ibi_all_reg_before_mean, ibi_all_reg_before_std, pupil_all_reg_before_mean, pupil_all_reg_before_std, nbins_ibi_all_reg_before, ibi_all_reg_before_hist, nbins_pupil_all_reg_before, pupil_all_reg_before_hist = getIBIandPuilDilation(pulse_data, pulse_ind_reg_before,samples_pulse_reg_before, pulse_samprate,pupil_data, pupil_ind_reg_before,samples_pupil_reg_before,pupil_samprate)
ibi_all_reg_after_mean, ibi_all_reg_after_std, pupil_all_reg_after_mean, pupil_all_reg_after_std, nbins_ibi_all_reg_after, ibi_all_reg_after_hist, nbins_pupil_all_reg_after, pupil_all_reg_after_hist = getIBIandPuilDilation(pulse_data, pulse_ind_reg_after,samples_pulse_reg_after, pulse_samprate,pupil_data, pupil_ind_reg_after,samples_pupil_reg_after,pupil_samprate)



'''
Process LFP data and find PSDs.
'''
Fs = hdeeg_samprate
density_length = 30
plt.figure()
for chann in hdeeg.keys():
	freq, trial_power_stress = TrialAveragedPSD(hdeeg, chann, Fs, hdeeg_ind_successful_stress, samples_hdeeg_successful_stress, row_ind_successful_stress, stim_freq)
	freq, trial_power_reg = TrialAveragedPSD(hdeeg, chann, Fs, hdeeg_ind_successful_reg_before, samples_hdeeg_successful_reg, row_ind_successful_reg, stim_freq)

	trial_power_avg_stress = np.nanmean(trial_power_stress,axis=1)
	trial_power_std_stress = np.nanstd(trial_power_stress,axis=1)
	trial_power_avg_reg = np.nanmean(trial_power_reg,axis=1)
	trial_power_std_reg = np.nanstd(trial_power_reg,axis=1)

	plt.plot(freq[0:density_length],trial_power_avg_stress,'r',label='Stress Trials') # plotting the spectrum
	plt.fill_between(freq[0:density_length],trial_power_avg_stress - trial_power_std_stress,trial_power_avg_stress +trial_power_std_stress,facecolor='red',linewidth=0.1,alpha=0.5)
	plt.plot(freq[0:density_length],trial_power_avg_reg,'b',label='Regular Trials')
	plt.fill_between(freq[0:density_length],trial_power_avg_reg - trial_power_std_reg,trial_power_avg_reg +trial_power_std_reg,facecolor='blue',linewidth=0.1,alpha=0.5)
	plt.xlim((0, 50))
	plt.xlabel('Freq (Hz)')
	plt.ylabel('Normalized PSD')
	plt.title('Channel ' +str(chann) + '- Successful Trial-Averaged PSD')
	plt.legend()
	plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_Spectrogram_Ch'+str(chann)+'.svg')
#plt.show()
#plt.close()

'''
stopped here 
'''

# Pearson correlations
r_before, r_p_before = stats.pearsonr(ibi_reg_before_mean, pupil_reg_before_mean)
r_stress, r_p_stress = stats.pearsonr(ibi_stress_mean,pupil_stress_mean)
r_after, r_p_after = stats.pearsonr(ibi_reg_after_mean,pupil_reg_after_mean)

# Linear fits to go along with plots
if len(ibi_stress_mean) > 0:
	m,b = np.polyfit(range(1,len(ibi_stress_mean)+1), ibi_stress_mean, 1)
	ibi_stress_mean_fit = m*np.arange(1,len(ibi_stress_mean)+1) + b

	m,b = np.polyfit(range(1,len(pupil_stress_mean)+1), pupil_stress_mean, 1)
	pupil_stress_mean_fit = m*np.arange(1,len(pupil_stress_mean)+1) + b

m,b = np.polyfit(range(1,len(ibi_reg_before_mean)+1), ibi_reg_before_mean, 1)
ibi_reg_before_mean_fit = m*np.arange(1,len(ibi_reg_before_mean)+1) + b

m,b = np.polyfit(range(1,len(pupil_reg_before_mean)+1), pupil_reg_before_mean, 1)
pupil_reg_before_mean_fit = m*np.arange(1,len(pupil_reg_before_mean)+1) + b


# Linear fits to go along with plots of all trial results
get_all_not_nan = np.logical_not(np.isnan(ibi_all_stress_mean))
get_all_not_nan_ind = np.ravel(np.nonzero(get_all_not_nan))
m_test,b_test = np.polyfit(get_all_not_nan_ind+1, np.array(ibi_all_stress_mean)[get_all_not_nan_ind], 1)
ibi_all_stress_mean_fit = m_test*np.arange(1,len(get_all_not_nan_ind)+1) + b_test

get_all_not_nan = np.logical_not(np.isnan(ibi_all_reg_before_mean))
get_all_not_nan_ind = np.ravel(np.nonzero(get_all_not_nan))
m,b = np.polyfit(get_all_not_nan_ind+1, np.array(ibi_all_reg_before_mean)[get_all_not_nan_ind], 1)
ibi_all_reg_before_mean_fit = m*np.arange(1,len(get_all_not_nan_ind)+1) + b

m,b = np.polyfit(range(1,len(pupil_all_stress_mean)+1),pupil_all_stress_mean, 1)
pupil_all_stress_mean_fit = m*np.arange(1,len(pupil_all_stress_mean)+1) + b

m,b = np.polyfit(range(1,len(pupil_all_reg_before_mean)+1), pupil_all_reg_before_mean, 1)
pupil_all_reg_before_mean_fit = m*np.arange(1,len(pupil_all_reg_before_mean)+1) + b


plt.figure()
plt.subplot(2,2,3)
if len(ibi_stress_mean) > 0:
	plt.plot(range(1,len(ibi_stress_mean)+1),ibi_stress_mean,'r')
	plt.plot(range(1,len(ibi_stress_mean)+1),ibi_stress_mean_fit,'r--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Stress Trials')

plt.subplot(2,2,1)
plt.plot(range(1,len(ibi_reg_before_mean)+1),ibi_reg_before_mean,'b')
plt.plot(range(1,len(ibi_reg_before_mean)+1),ibi_reg_before_mean_fit,'b--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Regular Trials before Stress')

plt.subplot(2,2,4)
plt.plot(range(1,len(ibi_all_stress_mean)+1),ibi_all_stress_mean,'r')
plt.plot(range(1,len(ibi_all_stress_mean_fit)+1),ibi_all_stress_mean_fit,'r--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Stress Trials')

plt.subplot(2,2,2)
plt.plot(range(1,len(ibi_all_reg_before_mean)+1),ibi_all_reg_before_mean,'b')
plt.plot(range(1,len(ibi_all_reg_before_mean_fit)+1),ibi_all_reg_before_mean_fit,'b--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
plt.ylim((0.28,0.5))
#plt.ylim((0.32,0.5))
plt.title('Pulse in Regular Trials before Stress')

#plt.subplot(3,1,3)
#plt.plot(range(1,len(ibi_reg_after_mean)+1),ibi_reg_after_mean,'k')
#plt.plot(range(1,len(ibi_reg_after_mean)+1),ibi_reg_after_mean_fit,'k--')
#plt.xlabel('Trial')
#plt.ylabel('Average IBI')
#plt.ylim((0.28,0.5))
##plt.ylim((0.32,0.5))
#plt.title('Pulse in Regular Trials after Stress')
plt.tight_layout()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_TrialIBI.svg')

plt.figure()
plt.subplot(2,2,3)
if len(pupil_stress_mean) > 0: 
	plt.plot(range(1,len(pupil_stress_mean)+1),pupil_stress_mean,'r')
	plt.plot(range(1,len(pupil_stress_mean)+1),pupil_stress_mean_fit,'r--')
	plt.text(1,np.max(pupil_stress_mean),'Pulse v. Pupil:r=%f \n p=%f' % (r_stress,r_p_stress))
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Stress Trials')


plt.subplot(2,2,1)
plt.plot(range(1,len(pupil_reg_before_mean)+1),pupil_reg_before_mean,'b')
plt.plot(range(1,len(pupil_reg_before_mean)+1),pupil_reg_before_mean_fit,'b--')
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Regular Trials before Stress')
plt.text(1,np.max(pupil_reg_before_mean),'Pulse v. Pupil:r=%f \n p=%f' % (r_before,r_p_before))

plt.subplot(2,2,4)
plt.plot(range(1,len(pupil_all_stress_mean)+1),pupil_all_stress_mean,'r')
plt.plot(range(1,len(pupil_all_stress_mean)+1),pupil_all_stress_mean_fit,'r--')
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Stress Trials')

plt.subplot(2,2,2)
plt.plot(range(1,len(pupil_all_reg_before_mean)+1),pupil_all_reg_before_mean,'b')
plt.plot(range(1,len(pupil_all_reg_before_mean)+1),pupil_all_reg_before_mean_fit,'b--')
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
#plt.ylim((0.70,1.15))
plt.ylim((-3,3))
plt.title('Pupil Diameter in Regular Trials before Stress')
#plt.subplot(3,1,3)
#plt.plot(range(1,len(pupil_reg_after_mean)+1),pupil_reg_after_mean,'k')
#plt.plot(range(1,len(pupil_reg_after_mean)+1),pupil_reg_after_mean_fit,'k--')
#plt.xlabel('Trial')
#plt.ylabel('Average Pupil Diameter')
##plt.ylim((0.70,1.15))
#plt.ylim((-3,3))
#plt.title('Pupil Diameter in Regular Trials after Stress')
#plt.text(1,np.max(pupil_reg_after_mean),'Pulse v. Pupil:r=%f \n p=%f' % (r_after,r_p_after))
plt.tight_layout()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_TrialPupil-DeleteBlinks.svg')

# IBI plots
# Compute significance
#F_ibi,p_ibi = stats.f_oneway(all_ibi_reg_before,all_ibi_reg_after,all_ibi_stress)
#F_pupil,p_pupil = stats.f_oneway(all_pupil_stress,all_ibi_reg_after,all_ibi_reg_before)

#F_all_ibi,p_all_ibi = stats.f_oneway(all_ibi_all_reg_before,all_ibi_all_reg_after,all_ibi_all_stress)
#F_all_pupil,p_all_pupil = stats.f_oneway(all_pupil_all_stress,all_ibi_all_reg_after,all_ibi_all_reg_before)

plt.figure()
'''
plt.subplot(1,2,1)
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

plt.subplot(1,2,2)
'''
plt.plot(nbins_ibi_all_reg_before,ibi_all_reg_before_hist,'b',label='Before Stress')
plt.fill_between(nbins_ibi_all_reg_before[17:22],ibi_all_reg_before_hist[17:22],np.zeros(5),facecolor='blue',linewidth=0.1,alpha=0.5)
plt.plot([nbins_ibi_all_reg_before[19],nbins_ibi_all_reg_before[19]],[0,ibi_all_reg_before_hist[19]],'b--')
plt.plot(nbins_ibi_all_stress,ibi_all_stress_hist,'r',label='Stress')
plt.fill_between(nbins_ibi_all_stress[17:22],ibi_all_stress_hist[17:22],np.zeros(5),facecolor='red',linewidth=0.1,alpha=0.5)
plt.plot([nbins_ibi_all_stress[19],nbins_ibi_all_stress[19]],[0,ibi_all_stress_hist[19]],'r--')
plt.xlabel('IBI (s)')
plt.ylabel('Frequency')
plt.xlim((0.1,0.6))
plt.title('IBI Distribtions for All Trials')

#plt.plot(nbins_ibi_reg_after,ibi_reg_after_hist,'k',label='After Stress')
#plt.fill_between(nbins_ibi_reg_after[17:22],ibi_reg_after_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1,alpha=0.5)
#plt.plot([nbins_ibi_reg_after[19],nbins_ibi_reg_after[19]],[0,ibi_reg_after_hist[19]],'k--')
#if (p_ibi < 0.05):
#	t_before_stress,p_before_stress = stats.ttest_ind(all_ibi_reg_before,all_ibi_stress, axis=0, equal_var=True)
#	t_after_stress,p_after_stress = stats.ttest_ind(all_ibi_reg_after,all_ibi_stress,axis=0,equal_var=True)
#	t_before_after,p_before_after = stats.ttest_ind(all_ibi_reg_before,all_ibi_reg_after,axis=0,equal_var=True)
#	plt.text(0.12,np.max(ibi_reg_after_hist),'Before v. Stress:p=%f \n Before v. After: p=%f \n Stress v. After: p=%f' % (p_before_stress,p_before_after,p_after_stress))
plt.legend()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_IBIDistribution.svg')

plt.figure()
'''
plt.subplot(1,2,1)
plt.plot(nbins_pupil_reg_before,pupil_reg_before_hist,'b',label='Regular Before')
plt.fill_between(nbins_pupil_reg_before[17:22],pupil_reg_before_hist[17:22],np.zeros(5),facecolor='blue',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_reg_before[19],nbins_pupil_reg_before[19]],[0,pupil_reg_before_hist[19]],'b--')
plt.plot(nbins_pupil_stress,pupil_stress_hist,'r',label='Stress')
plt.fill_between(nbins_pupil_stress[17:22],pupil_stress_hist[17:22],np.zeros(5),facecolor='red',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_stress[19],nbins_pupil_stress[19]],[0,pupil_stress_hist[19]],'r--')
plt.xlabel('Diameter (AU)')
plt.ylabel('Frequency')
plt.title('Distribution of Pupil Diameters')

plt.subplot(1,2,2)
'''
plt.plot(nbins_pupil_all_reg_before,pupil_all_reg_before_hist,'b',label='Regular Before')
plt.fill_between(nbins_pupil_all_reg_before[17:22],pupil_all_reg_before_hist[17:22],np.zeros(5),facecolor='blue',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_all_reg_before[19],nbins_pupil_all_reg_before[19]],[0,pupil_all_reg_before_hist[19]],'b--')
plt.plot(nbins_pupil_all_stress,pupil_all_stress_hist,'r',label='Stress')
plt.fill_between(nbins_pupil_all_stress[17:22],pupil_all_stress_hist[17:22],np.zeros(5),facecolor='red',linewidth=0.1,alpha=0.5)
plt.plot([nbins_pupil_all_stress[19],nbins_pupil_all_stress[19]],[0,pupil_all_stress_hist[19]],'r--')
plt.xlabel('Diameter (AU)')
plt.ylabel('Frequency')
plt.title('Distribution of Pupil Diameters')

#plt.plot(nbins_pupil_reg_after,pupil_reg_after_hist,'k',label='Regular After')
#plt.fill_between(nbins_pupil_reg_after[17:22],pupil_reg_after_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1,alpha=0.5)
#plt.plot([nbins_pupil_reg_after[19],nbins_pupil_reg_after[19]],[0,pupil_reg_after_hist[19]],'k--')
#if (p_pupil < 0.05):
#	t_before_stress,p_before_stress = stats.ttest_ind(all_pupil_reg_before,all_pupil_stress, axis=0, equal_var=True)
#	t_after_stress,p_after_stress = stats.ttest_ind(all_pupil_reg_after,all_pupil_stress,axis=0,equal_var=True)
#	t_before_after,p_before_after = stats.ttest_ind(all_pupil_reg_before,all_pupil_reg_after,axis=0,equal_var=True)
#	plt.text(nbins_pupil_reg_after[1],np.max(pupil_reg_after_hist)-0.05,'Before v. Stress:p=%f \n Before v. After: p=%f \n Stress v. After: p=%f' % (p_before_stress,p_before_after,p_after_stress))
plt.legend()
#plt.show()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_PupilDistribution-DeleteBlinks.svg')


norm_ibi_all_stress_mean = ibi_all_stress_mean 
norm_pupil_all_stress_mean = pupil_all_stress_mean
norm_ibi_all_reg_before_mean = ibi_all_reg_before_mean
norm_pupil_all_reg_before_mean = pupil_all_reg_before_mean

#norm_ibi_all_stress_mean = (ibi_all_stress_mean - np.nanmin(ibi_all_stress_mean + ibi_all_reg_before_mean))/np.nanmax(ibi_all_stress_mean + ibi_all_reg_before_mean - np.nanmin(ibi_all_stress_mean + ibi_all_reg_before_mean))
#norm_pupil_all_stress_mean = (pupil_all_stress_mean - np.nanmin(pupil_all_stress_mean + pupil_all_reg_before_mean))/np.nanmax(pupil_all_stress_mean + pupil_all_reg_before_mean - np.nanmin(pupil_all_stress_mean + pupil_all_reg_before_mean))
#norm_ibi_all_reg_before_mean = (ibi_all_reg_before_mean - np.nanmin(ibi_all_stress_mean + ibi_all_reg_before_mean))/np.nanmax(ibi_all_stress_mean + ibi_all_reg_before_mean - np.nanmin(ibi_all_stress_mean + ibi_all_reg_before_mean))
#norm_pupil_all_reg_before_mean = (pupil_all_reg_before_mean - np.nanmin(pupil_all_stress_mean + pupil_all_reg_before_mean))/np.nanmax(pupil_all_stress_mean + pupil_all_reg_before_mean - np.nanmin(pupil_all_stress_mean + pupil_all_reg_before_mean))

points_all_stress = np.array([norm_ibi_all_stress_mean,norm_pupil_all_stress_mean])
points_all_reg_before = np.array([norm_ibi_all_reg_before_mean,norm_pupil_all_reg_before_mean])
cov_all_stress = np.cov(points_all_stress)
cov_all_reg_before = np.cov(points_all_reg_before)
mean_vec_all_stress = [np.nanmean(norm_ibi_all_stress_mean),np.nanmean(norm_pupil_all_stress_mean)]
mean_vec_all_reg_before = [np.nanmean(norm_ibi_all_reg_before_mean),np.nanmean(norm_pupil_all_reg_before_mean)]

cmap_stress = mpl.cm.autumn
cmap_reg_before = mpl.cm.winter

plt.figure()
#plt.plot(ibi_all_stress_mean,pupil_all_stress_mean,'ro',label='Stress')
for i in range(0,len(ibi_all_stress_mean)):
    plt.plot(norm_ibi_all_stress_mean[i],norm_pupil_all_stress_mean[i],color=cmap_stress(i/float(len(ibi_all_stress_mean))),marker='o')
plot_cov_ellipse(cov_all_stress,mean_vec_all_stress,fc='r',ec='None',a=0.2)
#plt.plot(ibi_all_reg_before_mean,pupil_all_reg_before_mean,'bo',label='Reg Before')
for i in range(0,len(ibi_all_reg_before_mean)):
	plt.plot(norm_ibi_all_reg_before_mean[i],norm_pupil_all_reg_before_mean[i],color=cmap_reg_before(i/float(len(ibi_all_reg_before_mean))),marker='o')
plot_cov_ellipse(cov_all_reg_before,mean_vec_all_reg_before,fc='b',ec='None',a=0.2)
#plt.legend()
plt.xlabel('Mean Trial IBI (s)')
plt.ylabel('Mean Trial PD (AU)')
plt.title('All Trials')
sm_reg_before = plt.cm.ScalarMappable(cmap=cmap_reg_before, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_reg_before._A = []
cbar = plt.colorbar(sm_reg_before,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
sm_stress = plt.cm.ScalarMappable(cmap=cmap_stress, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_stress._A = []
cbar = plt.colorbar(sm_stress,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
#plt.ylim((-0.05,1.05))
#plt.xlim((-0.05,1.05))
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_IBIPupilCovariance_alltrials.svg')

norm_ibi_stress_mean = ibi_stress_mean 
norm_pupil_stress_mean = pupil_stress_mean 
norm_ibi_reg_before_mean = ibi_reg_before_mean 
norm_pupil_reg_before_mean = pupil_reg_before_mean 

#norm_ibi_stress_mean = (ibi_stress_mean - np.nanmin(ibi_stress_mean + ibi_reg_before_mean))/np.nanmax(ibi_stress_mean + ibi_reg_before_mean - np.nanmin(ibi_stress_mean + ibi_reg_before_mean))
#norm_pupil_stress_mean = (pupil_stress_mean - np.nanmin(pupil_stress_mean + pupil_reg_before_mean))/np.nanmax(pupil_stress_mean + pupil_reg_before_mean - np.nanmin(pupil_stress_mean + pupil_reg_before_mean))
#norm_ibi_reg_before_mean = (ibi_reg_before_mean - np.nanmin(ibi_stress_mean + ibi_reg_before_mean))/np.nanmax(ibi_stress_mean + ibi_reg_before_mean - np.nanmin(ibi_stress_mean + ibi_reg_before_mean))
#norm_pupil_reg_before_mean = (pupil_reg_before_mean - np.nanmin(pupil_stress_mean + pupil_reg_before_mean))/np.nanmax(pupil_stress_mean + pupil_reg_before_mean - np.nanmin(pupil_stress_mean + pupil_reg_before_mean))

points_stress = np.array([norm_ibi_stress_mean,norm_pupil_stress_mean])
points_reg_before = np.array([norm_ibi_reg_before_mean,norm_pupil_reg_before_mean])
cov_stress = np.cov(points_stress)
cov_reg_before = np.cov(points_reg_before)
mean_vec_stress = [np.nanmean(norm_ibi_stress_mean),np.nanmean(norm_pupil_stress_mean)]
mean_vec_reg_before = [np.nanmean(norm_ibi_reg_before_mean),np.nanmean(norm_pupil_reg_before_mean)]

plt.figure()
for i in range(0,len(ibi_stress_mean)):
    #plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o',markeredgecolor=None,markeredgewidth=0.0)
    plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o')
plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
for i in range(0,len(ibi_reg_before_mean)):
	plt.plot(norm_ibi_reg_before_mean[i],norm_pupil_reg_before_mean[i],color=cmap_reg_before(i/float(len(ibi_reg_before_mean))),marker='o')
plot_cov_ellipse(cov_reg_before,mean_vec_reg_before,fc='b',ec='None',a=0.2)
#plt.legend()
plt.xlabel('Mean Trial IBI (s)')
plt.ylabel('Mean Trial PD (AU)')
plt.title('Successful Trials')
sm_reg_before = plt.cm.ScalarMappable(cmap=cmap_reg_before, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_reg_before._A = []
cbar = plt.colorbar(sm_reg_before,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
sm_stress = plt.cm.ScalarMappable(cmap=cmap_stress, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_stress._A = []
cbar = plt.colorbar(sm_stress,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
#plt.ylim((-0.05,1.05))
#plt.xlim((-0.05,1.05))
plt.ylim((-4,3))
plt.xlim((0.32,0.46))
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_IBIPupilCovariance.svg')

plt.figure()
for i in range(0,len(ibi_stress_mean)):
    #plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o',markeredgecolor=None,markeredgewidth=0.0)
    plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o')
plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
for i in range(len(ibi_reg_before_mean)/2,len(ibi_reg_before_mean)):
	plt.plot(norm_ibi_reg_before_mean[i],norm_pupil_reg_before_mean[i],color=cmap_reg_before(i/float(len(ibi_reg_before_mean))),marker='o')
plot_cov_ellipse(cov_reg_before,mean_vec_reg_before,fc='b',ec='None',a=0.2)
#plt.legend()
plt.xlabel('Mean Trial IBI (s)')
plt.ylabel('Mean Trial PD (AU)')
plt.title('Block A - late trials only')
sm_reg_before = plt.cm.ScalarMappable(cmap=cmap_reg_before, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_reg_before._A = []
cbar = plt.colorbar(sm_reg_before,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
sm_stress = plt.cm.ScalarMappable(cmap=cmap_stress, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_stress._A = []
cbar = plt.colorbar(sm_stress,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
#plt.ylim((-0.05,1.05))
#plt.xlim((-0.05,1.05))
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_IBIPupilCovariance_latetrials.svg')


plt.close("all")
