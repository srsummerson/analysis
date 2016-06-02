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


# Set up code for particular day and block
hdf_filename = 'mari20160517_07_te2097.hdf'
filename = 'Mario20160517'
TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
#hdf_location = hdffilename
block_num = 1
stim_freq = 100

lfp_channels = [29, 13, 27, 11, 25, 9, 10, 26, 12, 28, 14, 30, 20, 4, 18, 2, 63, 1, 17, 3]
lfp_channels = [26]
#bands = [[1,8],[8,12],[12,30],[30,55],[65,100]]
bands = [[8,12]]

'''
Load behavior data
'''
state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward = FreeChoiceBehavior_withStressTrials(hdf_location)

print "Behavior data loaded."

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


'''
Load syncing data for behavior and TDT recording
'''
print "Loading syncing data."

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
	lfp = dict()
	lfp_samprate = 3051.8
else:
	r = io.TdtIO(TDT_tank)
	bl = r.read_block(lazy=False,cascade=True)
	print "File read."
	lfp = dict()
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
lfp_dio_sample_num = (float(lfp_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1

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

for i, ind in enumerate(lfp_ind_successful_stress):
	trial_array = []
	trial_array.append(pupil_stress_mean[i])
	trial_array.append(ibi_stress_mean[i])
	
	
	for chann in lfp_channels:
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_successful_stress[i]], lfp_samprate, nperseg=1024)
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			trial_array.append(np.sum(freq_band))
		lfp_power_successful_stress.append(np.sum(freq_band))
	X_successful_stress.append(trial_array)

for i, ind in enumerate(lfp_ind_stress):
	trial_array = []
	trial_array.append(pupil_all_stress_mean[i])
	trial_array.append(ibi_all_stress_mean[i])
	
	for chann in lfp_channels:
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_stress[i]], lfp_samprate, nperseg=1024)
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			trial_array.append(np.sum(freq_band))
		lfp_power_stress.append(np.sum(freq_band))
	X_stress.append(trial_array)

for i, ind in enumerate(lfp_ind_successful_reg):
	trial_array = []
	trial_array.append(pupil_reg_mean[i])
	trial_array.append(ibi_reg_mean[i])
	
	for chann in lfp_channels:
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_successful_reg[i]], lfp_samprate, nperseg=1024)
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			trial_array.append(np.sum(freq_band))
		lfp_power_successful_reg.append(np.sum(freq_band))

	X_successful_reg.append(trial_array)

for i, ind in enumerate(lfp_ind_reg):
	trial_array = []
	trial_array.append(pupil_all_reg_mean[i])
	trial_array.append(ibi_all_reg_mean[i])
	
	for chann in lfp_channels:
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_reg[i]], lfp_samprate, nperseg=1024)
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			trial_array.append(np.sum(freq_band))
	lfp_power_reg.append(np.sum(freq_band))

	X_reg.append(trial_array)

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

X_successful = np.vstack([X_successful_reg, X_successful_stress])
y_successful = np.append(y_successful_reg,y_successful_stress)

X_all = np.vstack([X_reg, X_stress])
y_all = np.append(y_reg, y_stress)

clf_successful = LinearDiscriminantAnalysis()
clf_successful.fit(X_successful, y_successful)

LDAforFeatureSelection(X_successful,y_successful,filename,block_num)

'''
Do regression as well
'''

x = np.vstack((np.append(ibi_all_reg_mean, ibi_all_stress_mean), np.append(pupil_all_reg_mean, pupil_all_stress_mean), 
				np.append(lfp_power_reg, lfp_power_stress)))
x = np.transpose(x)
x = sm.add_constant(x,prepend='False')

model_glm = sm.Logit(y_all,x)
fit_glm = model_glm.fit()
print fil_glm.summary()