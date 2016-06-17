import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
import sys
import statsmodels.api as sm
from neo import io
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse, LDAforFeatureSelection
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score




hdf_filename = 'mari20160614_03_te2237.hdf'
hdf_filename_stim = 'mari20160614_09_te2243.hdf'
filename = 'Mario20160614'
block_num = 1
print filename
#TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
hdf_location_stim = '/storage/rawdata/hdf/'+hdf_filename_stim
#hdf_location = hdffilename

stim_freq = 100

lfp_channels = [29, 13, 27, 11, 25, 9, 10, 26, 12, 28, 14, 30, 20, 4, 18, 2, 63, 1, 17, 3]
lfp_channels = [11]
#bands = [[1,8],[8,12],[12,30],[30,55],[65,100]]
bands = [[30,55]]

'''
Load behavior data
'''
state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward = FreeChoiceBehavior_withStressTrials(hdf_location)
state_time_stim, ind_center_states_stim, ind_check_reward_states_stim, all_instructed_or_freechoice_stim, all_stress_or_not_stim, successful_stress_or_not_stim,trial_success_stim, target_stim, reward_stim = FreeChoiceBehavior_withStressTrials(hdf_location_stim)

print "Behavior data loaded."

# Total number of trials
num_trials = ind_center_states.size
total_states = state_time.size

num_trials_stim = ind_center_states_stim.size
total_states_stim = state_time_stim.size

# Number of successful stress trials
tot_successful_stress = np.logical_and(trial_success,all_stress_or_not)
successful_stress_trials = float(np.sum(tot_successful_stress))/np.sum(all_stress_or_not)

tot_successful_stress_stim = np.logical_and(trial_success_stim,all_stress_or_not_stim)
successful_stress_trials_stim = float(np.sum(tot_successful_stress_stim))/np.sum(all_stress_or_not_stim)

# Number of successful non-stress trials
tot_successful_reg = np.logical_and(trial_success,np.logical_not(all_stress_or_not))
successful_reg_trials = float(np.sum(tot_successful_reg))/(num_trials - np.sum(all_stress_or_not))

# Response times for successful stress trials
ind_successful_stress = np.ravel(np.nonzero(tot_successful_stress))   	# gives trial index, not row index
row_ind_successful_stress = ind_center_states[ind_successful_stress]		# gives row index
ind_successful_stress_reward = np.ravel(np.nonzero(successful_stress_or_not))
row_ind_successful_stress_reward = ind_check_reward_states[ind_successful_stress_reward]
response_time_successful_stress = (state_time[row_ind_successful_stress_reward] - state_time[row_ind_successful_stress])/float(60)		# hdf rows are written at a rate of 60 Hz

ind_successful_stress_stim = np.ravel(np.nonzero(tot_successful_stress_stim))   	# gives trial index, not row index
row_ind_successful_stress_stim = ind_center_states_stim[ind_successful_stress_stim]		# gives row index
ind_successful_stress_reward_stim = np.ravel(np.nonzero(successful_stress_or_not_stim))
row_ind_successful_stress_reward_stim = ind_check_reward_states_stim[ind_successful_stress_reward_stim]
response_time_successful_stress_stim = (state_time_stim[row_ind_successful_stress_reward_stim] - state_time_stim[row_ind_successful_stress_stim])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response time for all stress trials
ind_stress = np.ravel(np.nonzero(all_stress_or_not))
row_ind_stress = ind_center_states[ind_stress]  # gives row index
row_ind_end_stress = np.zeros(len(row_ind_stress))
row_ind_end_stress = row_ind_stress + 2  # targ_transition state occurs two states later for unsuccessful trials
row_ind_end_stress[-1] = np.min([row_ind_end_stress[-1],len(state_time)-1])  # correct final incomplete trial

ind_stress_stim = np.ravel(np.nonzero(all_stress_or_not_stim))
row_ind_stress_stim = ind_center_states_stim[ind_stress_stim]  # gives row index
row_ind_end_stress_stim = np.zeros(len(row_ind_stress_stim))
row_ind_end_stress_stim = row_ind_stress_stim + 2  # targ_transition state occurs two states later for unsuccessful trials
row_ind_end_stress_stim[-1] = np.min([row_ind_end_stress_stim[-1],len(state_time_stim)-1])  # correct final incomplete trial

for i in range(0,len(row_ind_successful_stress)):
	ind = np.where(row_ind_stress == row_ind_successful_stress[i])[0]
	row_ind_end_stress[ind] = row_ind_successful_stress_reward[i]  # for successful trials, update with real end of trial
response_time_stress = (state_time[row_ind_end_stress] - state_time[row_ind_stress])/float(60)

for i in range(0,len(row_ind_successful_stress_stim)):
	ind = np.where(row_ind_stress_stim == row_ind_successful_stress_stim[i])[0]
	row_ind_end_stress_stim[ind] = row_ind_successful_stress_reward_stim[i]  # for successful trials, update with real end of trial
response_time_stress_stim = (state_time_stim[row_ind_end_stress_stim] - state_time_stim[row_ind_stress_stim])/float(60)

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


'''
Load syncing data for behavior and TDT recording
'''
print "Loading syncing data."

hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

hdf_times_stim = dict()
mat_filename_stim = filename+'_b'+str(block_num + 1)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename_stim,hdf_times_stim)

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
	lfp = dict()
	lfp_samprate = 3051.8
else:
	r = io.TdtIO(TDT_tank)
	bl = r.read_block(lazy=False,cascade=True)
	print "File read."
	lfp = dict()
	lfp_stim = dict()
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
			if channel in lfp_channels:
				lfp_samprate = sig.sampling_rate.item()
				lfp[channel] = np.ravel(sig)
	for sig in bl.segments[block_num].analogsignals:
		if (sig.name == 'PupD 1'):
			pupil_data_stim = np.ravel(sig)
			pupil_samprate_stim = sig.sampling_rate.item()
		if (sig.name == 'HrtR 1'):
			pulse_data_stim = np.ravel(sig)
			pulse_samprate_stim = sig.sampling_rate.item()
		if (sig.name == 'Hold 1'):
			hold_cue = np.ravel(sig)
			hold_samprate = sig.sampling_rate.item()
		if (sig.name == 'Hold 2'):
			stim_state = np.ravel(sig)
			stim_state_samprate = sig.sampling_rate.item()
		if (sig.name[0:4] == 'LFP1'):
			channel = sig.channel_index
			if channel in lfp_channels:
				lfp_samprate_stim = sig.sampling_rate.item()
				lfp_stim[channel] = np.ravel(sig)


'''
Convert DIO TDT samples for pupil and pulse data for regular and stress trials
'''
# divide up analysis for regular trials before stress trials, stress trials, and regular trials after stress trials are introduced
hdf_rows = np.ravel(hdf_times['row_number'])
hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

hdf_rows_stim = np.ravel(hdf_times_stim['row_number'])
hdf_rows_stim = [val for val in hdf_rows_stim]	# turn into a list so that the index method can be used later
dio_tdt_sample_stim = np.ravel(hdf_times_stim['tdt_samplenumber'])
dio_freq_stim = np.ravel(hdf_times_stim['tdt_dio_samplerate'])

# Convert DIO TDT sample numbers to for pupil and pulse data:
# if dio sample num is x, then data sample number is R*(x-1) + 1 where
# R = data_sample_rate/dio_sample_rate
pulse_dio_sample_num = (float(pulse_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
pupil_dio_sample_num = (float(pupil_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
lfp_dio_sample_num = (float(lfp_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1

pulse_dio_sample_num_stim = (float(pulse_samprate_stim)/float(dio_freq_stim))*(dio_tdt_sample_stim - 1) + 1
pupil_dio_sample_num_stim = (float(pupil_samprate_stim)/float(dio_freq_stim))*(dio_tdt_sample_stim - 1) + 1
lfp_dio_sample_num_stim = (float(lfp_samprate_stim)/float(dio_freq_stim))*(dio_tdt_sample_stim - 1) + 1
hold_cue_dio_sample_num_stim = (float(hold_samprate)/float(dio_freq_stim))*(dio_tdt_sample_stim - 1) + 1
stim_state_dio_sample_num_stim = (float(stim_state_samprate)/float(dio_freq_stim))*(dio_tdt_sample_stim - 1) + 1

state_row_ind_successful_stress = state_time[row_ind_successful_stress]
state_row_ind_successful_reg = state_time[row_ind_successful_reg]
pulse_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
pupil_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
lfp_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
pulse_ind_successful_reg = []
pupil_ind_successful_reg = []
lfp_ind_successful_reg = []
state_row_ind_stress = state_time[row_ind_stress]
state_row_ind_reg = state_time[row_ind_reg]
pulse_ind_stress = np.zeros(row_ind_stress.size)
pupil_ind_stress = np.zeros(row_ind_stress.size)
lfp_ind_stress = np.zeros(row_ind_stress.size)
pulse_ind_reg = []
pupil_ind_reg = []
lfp_ind_reg = []

state_row_ind_successful_stress_stim = state_time_stim[row_ind_successful_stress_stim]
pulse_ind_successful_stress_stim = np.zeros(row_ind_successful_stress_stim.size)
pupil_ind_successful_stress_stim = np.zeros(row_ind_successful_stress_stim.size)
lfp_ind_successful_stress_stim = np.zeros(row_ind_successful_stress_stim.size)
hold_cue_ind_successful_stress_stim = np.zeros(row_ind_successful_stress_stim.size)
stim_state_ind_successful_stress_stim = np.zeros(row_ind_successful_stress_stim.size)
state_row_ind_stress_stim = state_time_stim[row_ind_stress_stim]
pulse_ind_stress_stim = np.zeros(row_ind_stress_stim.size)
pupil_ind_stress_stim = np.zeros(row_ind_stress_stim.size)
lfp_ind_stress_stim = np.zeros(row_ind_stress_stim.size)
hold_cue_ind_stress_stim = np.zeros(row_ind_stress_stim.size)
stim_state_ind_stress_stim = np.zeros(row_ind_stress_stim.size)


for i in range(0,len(row_ind_successful_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
	pulse_ind_successful_stress[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_successful_stress[i] = pupil_dio_sample_num[hdf_index]
	lfp_ind_successful_stress[i] = lfp_dio_sample_num[hdf_index]
for i in range(0,len(row_ind_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_stress[i]))
	pulse_ind_stress[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_stress[i] = pupil_dio_sample_num[hdf_index]
	lfp_ind_stress[i] = lfp_dio_sample_num[hdf_index]

if len(row_ind_successful_stress) > 0: 
	ind_start_stress = row_ind_successful_stress[0]
else:
	ind_start_stress = np.inf

ind_start_all_stress = row_ind_stress[0]
for i in range(0,len(state_row_ind_successful_reg)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
	pulse_ind_successful_reg.append(pulse_dio_sample_num[hdf_index])
	pupil_ind_successful_reg.append(pupil_dio_sample_num[hdf_index])
	lfp_ind_successful_reg.append(lfp_dio_sample_num[hdf_index])
	
for i in range(0,len(state_row_ind_reg)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
	pulse_ind_reg.append(pulse_dio_sample_num[hdf_index])
	pupil_ind_reg.append(pupil_dio_sample_num[hdf_index])
	lfp_ind_reg.append(lfp_dio_sample_num[hdf_index])

for i in range(0,len(row_ind_successful_stress_stim)):
	hdf_index = np.argmin(np.abs(hdf_rows_stim - state_row_ind_successful_stress_stim[i]))
	pulse_ind_successful_stress_stim[i] = pulse_dio_sample_num_stim[hdf_index]
	pupil_ind_successful_stress_stim[i] = pupil_dio_sample_num_stim[hdf_index]
	lfp_ind_successful_stress_stim[i] = lfp_dio_sample_num_stim[hdf_index]
	hold_cue_ind_successful_stress_stim[i] = hold_cue_dio_sample_num_stim[hdf_index]
	stim_state_ind_successful_stress_stim[i] = stim_state_dio_sample_num_stim[hdf_index]
for i in range(0,len(row_ind_stress_stim)):
	hdf_index = np.argmin(np.abs(hdf_rows_stim - state_row_ind_stress_stim[i]))
	pulse_ind_stress_stim[i] = pulse_dio_sample_num_stim[hdf_index]
	pupil_ind_stress_stim[i] = pupil_dio_sample_num_stim[hdf_index]
	lfp_ind_stress_stim[i] = lfp_dio_sample_num_stim[hdf_index]
	hold_cue_ind_stress_stim[i] = hold_cue_dio_sample_num_stim[hdf_index]
	stim_state_ind_stress_stim[i] = stim_state_dio_sample_num_stim[hdf_index]

'''
Process pupil and pulse data
'''

# Find IBIs and pupil data for all successful stress trials. 
samples_pulse_successful_stress = np.floor(response_time_successful_stress*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil_successful_stress = np.floor(response_time_successful_stress*pupil_samprate)
samples_lfp_successful_stress = np.floor(response_time_successful_stress*lfp_samprate)

ibi_stress_mean, ibi_stress_std, pupil_stress_mean, pupil_stress_std, nbins_ibi_stress, ibi_stress_hist, nbins_pupil_stress, pupil_stress_hist = getIBIandPuilDilation(pulse_data, pulse_ind_successful_stress,samples_pulse_successful_stress, pulse_samprate,pupil_data, pupil_ind_successful_stress,samples_pupil_successful_stress,pupil_samprate)


# Find IBIs and pupil data for all stress trials
samples_pulse_stress = np.floor(response_time_stress*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil_stress = np.floor(response_time_stress*pupil_samprate)
samples_lfp_stress = np.floor(response_time_stress*lfp_samprate)

ibi_all_stress_mean, ibi_all_stress_std, pupil_all_stress_mean, pupil_all_stress_std, nbins_ibi_all_stress, ibi_all_stress_hist, nbins_pupil_all_stress, pupil_all_stress_hist = getIBIandPuilDilation(pulse_data, pulse_ind_stress,samples_pulse_stress, pulse_samprate,pupil_data, pupil_ind_stress,samples_pupil_stress,pupil_samprate)

# Find IBIs and pupil data for successful and all regular trials. 
samples_pulse_successful_reg = np.floor(response_time_successful_reg*pulse_samprate)
samples_pupil_successful_reg = np.floor(response_time_successful_reg*pupil_samprate)
samples_lfp_successful_reg = np.floor(response_time_successful_reg*lfp_samprate)

ibi_reg_mean, ibi_reg_std, pupil_reg_mean, pupil_reg_std, nbins_ibi_reg, ibi_reg_hist, nbins_pupil_reg, pupil_reg_hist = getIBIandPuilDilation(pulse_data, pulse_ind_successful_reg,samples_pulse_successful_reg, pulse_samprate,pupil_data, pupil_ind_successful_reg,samples_pupil_successful_reg,pupil_samprate)

samples_pulse_reg = np.floor(response_time_reg*pulse_samprate)
samples_pupil_reg = np.floor(response_time_reg*pupil_samprate)
samples_lfp_reg = np.floor(response_time_reg*lfp_samprate)

ibi_all_reg_mean, ibi_all_reg_std, pupil_all_reg_mean, pupil_all_reg_std, nbins_ibi_all_reg, ibi_all_reg_hist, nbins_pupil_all_reg, pupil_all_reg_hist = getIBIandPuilDilation(pulse_data, pulse_ind_reg,samples_pulse_reg, pulse_samprate,pupil_data, pupil_ind_reg,samples_pupil_reg,pupil_samprate)

# Find IBIs and pupil data for all successful stress trials with stimulation. 
samples_pulse_successful_stress_stim = np.floor(response_time_successful_stress_stim*pulse_samprate_stim) 	#number of samples in trial interval for pulse signal
samples_pupil_successful_stress_stim = np.floor(response_time_successful_stress_stim*pupil_samprate_stim)
samples_lfp_successful_stress_stim = np.floor(response_time_successful_stress_stim*lfp_samprate_stim)

ibi_stress_mean_stim, ibi_stress_std_stim, pupil_stress_mean_stim, pupil_stress_std_stim, nbins_ibi_stress_stim, ibi_stress_hist_stim, nbins_pupil_stress_stim, pupil_stress_hist_stim = getIBIandPuilDilation(pulse_data_stim, pulse_ind_successful_stress_stim,samples_pulse_successful_stress_stim, pulse_samprate_stim,pupil_data_stim, pupil_ind_successful_stress_stim,samples_pupil_successful_stress_stim,pupil_samprate_stim)

# Find IBIs and pupil data for all stress trials with stimulation.
samples_pulse_stress_stim = np.floor(response_time_stress_stim*pulse_samprate_stim) 	#number of samples in trial interval for pulse signal
samples_pupil_stress_stim = np.floor(response_time_stress_stim*pupil_samprate_stim)
samples_lfp_stress_stim = np.floor(response_time_stress_stim*lfp_samprate_stim)

ibi_all_stress_mean_stim, ibi_all_stress_std_stim, pupil_all_stress_mean_stim, pupil_all_stress_std_stim, nbins_ibi_all_stress_stim, ibi_all_stress_hist_stim, nbins_pupil_all_stress_stim, pupil_all_stress_hist_stim = getIBIandPuilDilation(pulse_data_stim, pulse_ind_stress_stim,samples_pulse_stress_stim, pulse_samprate_stim,pupil_data_stim, pupil_ind_stress_stim,samples_pupil_stress_stim,pupil_samprate_stim)

'''
Get power in designated frequency bands per trial
'''
lfp_power_successful_stress = []
lfp_power_stress = []
lfp_power_successful_reg = []
lfp_power_reg = []
X_successful_stress = []
X_stress = []
X_successful_reg = []
X_reg = []
X_successful_stim = []
X_stim = []

for i, ind in enumerate(lfp_ind_successful_stress):
	trial_array = []
	trial_array.append(pupil_stress_mean[i])
	trial_array.append(ibi_stress_mean[i])
	
	
	for chann in lfp_channels:
		#freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_successful_stress[i]], lfp_samprate, nperseg=1024)
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+lfp_samprate/2], lfp_samprate, nperseg=1024)  # take 0.5 s of data 
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			#trial_array.append(np.sum(freq_band))
		lfp_power_successful_stress.append(np.sum(freq_band))
	
	X_successful_stress.append(trial_array)

for i, ind in enumerate(lfp_ind_stress):
	trial_array = []
	trial_array.append(pupil_all_stress_mean[i])
	trial_array.append(ibi_all_stress_mean[i])
	
	
	for chann in lfp_channels:
		#freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_stress[i]], lfp_samprate, nperseg=1024)
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+lfp_samprate/2], lfp_samprate, nperseg=1024)  # take 0.5 s of data 
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			#trial_array.append(np.sum(freq_band))
		lfp_power_stress.append(np.sum(freq_band))
	
	X_stress.append(trial_array)

for i, ind in enumerate(lfp_ind_successful_reg):
	trial_array = []
	trial_array.append(pupil_reg_mean[i])
	trial_array.append(ibi_reg_mean[i])
	
	
	for chann in lfp_channels:
		#freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_successful_reg[i]], lfp_samprate, nperseg=1024)
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+lfp_samprate/2], lfp_samprate, nperseg=1024)  # take 0.5 s of data 
		
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			#trial_array.append(np.sum(freq_band))
		lfp_power_successful_reg.append(np.sum(freq_band))
	
	X_successful_reg.append(trial_array)

for i, ind in enumerate(lfp_ind_reg):
	trial_array = []
	trial_array.append(pupil_all_reg_mean[i])
	trial_array.append(ibi_all_reg_mean[i])
	
	for chann in lfp_channels:
		#freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_reg[i]], lfp_samprate, nperseg=1024)
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+lfp_samprate/2], lfp_samprate, nperseg=1024)  # take 0.5 s of data 
		
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			#trial_array.append(np.sum(freq_band))
	lfp_power_reg.append(np.sum(freq_band))
	
	X_reg.append(trial_array)

trial_success_stim_state = []
trial_stim_state = []

for ind in stim_state_ind_successful_stress_stim:
	trial_success_stim_state.append(stim_state[ind])

for ind in stim_state_ind_stress_stim:
	trial_stim_state.append(stim_state[ind])

for i, ind in enumerate(lfp_ind_successful_stress_stim):
	trial_array = []
	trial_array.append(pupil_stress_mean_stim[i])
	trial_array.append(ibi_stress_mean_stim[i])
	
	X_successful_stim.append(trial_array)


for i, ind in enumerate(lfp_ind_stress_stim):
	trial_array = []
	trial_array.append(pupil_all_stress_mean_stim[i])
	trial_array.append(ibi_all_stress_mean_stim[i])
	
	X_stim.append(trial_array)

'''
Set up IBI/PD data for logisitic regression on trial type
'''

# Labels: 0 = regular, 1 = stress
X_successful_stress = np.array(X_successful_stress)
num_successful_stress = X_successful_stress.shape[0]
y_successful_stress = np.ones(num_successful_stress)

X_stress = np.array(X_stress)
num_stress = X_stress.shape[0]
y_stress = np.ones(num_stress)

X_successful_reg = np.array(X_successful_reg)
num_successful_reg = X_successful_reg.shape[0]
y_successful_reg = np.zeros(num_successful_reg)

X_reg = np.array(X_reg)
num_reg = X_reg.shape[0]
y_reg = np.zeros(num_reg)

X_successful_stim = np.array(X_successful_stim)
X_stim = np.array(X_stim)

X_successful = np.vstack([X_successful_reg, X_successful_stress])
y_successful = np.append(y_successful_reg,y_successful_stress)

X_all = np.vstack([X_reg, X_stress])
y_all = np.append(y_reg, y_stress)

'''
Logistic Regression of trial type using data from Blocks A and B
'''

x = np.vstack((np.append(ibi_all_reg_mean, ibi_all_stress_mean), np.append(pupil_all_reg_mean, pupil_all_stress_mean)))
x = np.transpose(x)
x = sm.add_constant(x,prepend='False')

x_successful = np.vstack((np.append(ibi_reg_mean, ibi_stress_mean), np.append(pupil_reg_mean, pupil_stress_mean)))
x_successful = np.transpose(x_successful)
x_successful = sm.add_constant(x_successful,prepend='False')

'''
print "Regression with all trials"
model_glm = sm.Logit(y_all,x)
fit_glm = model_glm.fit()
print fit_glm.summary()
'''

print "Regression with successful trials"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()

'''
Classify Block C trials
'''
regression_params = fit_glm.params 
p_trial_type = regression_params[1]*ibi_stress_mean_stim + regression_params[2]*pupil_stress_mean_stim + regression_params[0]
y_stress_stim = (p_trial_type > 0.5)
fraction_stress_stim = np.sum(y_stress_stim)/len(y_stress_stim)
print "Fraction of Block C trials classified as stress:", fraction_stress_stim

'''
IBI-PD Covariance plots: need to adjust for trials classified as regular/stress
'''
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

'''
Add analysis of what trials were classified online as stress and which were classified offline as stress
'''
