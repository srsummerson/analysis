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
from sklearn.cluster import KMeans



hdf_filename = 'mari20160610_13_te2210.hdf'
filename = 'Mario20160610'
block_num = 1
print filename
TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename

lfp_channels = [29, 13, 27, 11, 25, 9, 10, 26, 12, 28, 14, 30, 20, 4, 18, 2, 63, 1, 17, 3]
lfp_channels = [11]
#bands = [[1,8],[8,12],[12,30],[30,55],[65,100]]
bands = [[30,55]]

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
pulse_ind_successful_reg = np.zeros(row_ind_successful_reg.size)
pupil_ind_successful_reg = np.zeros(row_ind_successful_reg.size)
lfp_ind_successful_reg = np.zeros(row_ind_successful_reg.size)
state_row_ind_stress = state_time[row_ind_stress]
state_row_ind_reg = state_time[row_ind_reg]
pulse_ind_stress = np.zeros(row_ind_stress.size)
pupil_ind_stress = np.zeros(row_ind_stress.size)
lfp_ind_stress = np.zeros(row_ind_stress.size)
pulse_ind_reg = np.zeros(row_ind_reg.size)
pupil_ind_reg = np.zeros(row_ind_reg.size)
lfp_ind_reg = np.zeros(row_ind_reg.size)


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
	pulse_ind_successful_reg[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_successful_reg[i] = pupil_dio_sample_num[hdf_index]
	lfp_ind_successful_reg[i] = lfp_dio_sample_num[hdf_index]
	
for i in range(0,len(state_row_ind_reg)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
	pulse_ind_reg[i] = pulse_dio_sample_num[hdf_index]
	pupil_ind_reg[i] = pupil_dio_sample_num[hdf_index]
	lfp_ind_reg[i] = lfp_dio_sample_num[hdf_index]


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
Need to do norm of data first: no normalizing now 
'''
norm_ibi_all_stress_mean = np.array(ibi_all_stress_mean)
norm_pupil_all_stress_mean = np.array(pupil_all_stress_mean)
norm_ibi_all_reg_before_mean = np.array(ibi_all_reg_mean)
norm_pupil_all_reg_before_mean = np.array(pupil_all_reg_mean)

norm_ibi_stress_mean = np.array(ibi_stress_mean)
norm_pupil_stress_mean = np.array(pupil_stress_mean)
norm_ibi_reg_before_mean = np.array(ibi_reg_mean)
norm_pupil_reg_before_mean = np.array(pupil_reg_mean)


'''
Do K-means to get well-clustered trials
'''
print "Starting K-means using successful trials"
# K Means of IBI and PD data
total_len = len(norm_ibi_stress_mean) + len(norm_ibi_reg_before_mean)
X_kmeans = np.zeros([total_len,2])
X_kmeans[:,0] = np.append(norm_ibi_stress_mean, norm_ibi_reg_before_mean)
X_kmeans[:,1] = np.append(norm_pupil_stress_mean, norm_pupil_reg_before_mean)
y_kmeans_pred = KMeans(n_clusters=2).fit_predict(X_kmeans)

plt.figure()
plt.subplot(211)
plt.scatter(X_kmeans[:,0],X_kmeans[:,1],c = y_kmeans_pred)
plt.title('K-means Clusters')
plt.subplot(212)
plt.scatter(norm_ibi_stress_mean,norm_pupil_stress_mean,c='r')
plt.scatter(norm_ibi_reg_before_mean,norm_pupil_reg_before_mean,c='b')
plt.title('Trial labels')
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_KMeans.svg')

print "Plotted K-means using successful trials"

# Assign label to majority cluster for each trial type
cluster_vals = set(y_kmeans_pred)
cluster_vals = [val for val in cluster_vals]
stress_cluster_assignment = y_kmeans_pred[0:len(norm_ibi_stress_mean)]
stress_cluster_vals = [list(stress_cluster_assignment).count(x) for x in cluster_vals]
stress_majority_cluster = cluster_vals[np.argmax(stress_cluster_vals)]
reg_cluster_assignment = y_kmeans_pred[len(norm_ibi_stress_mean):]
reg_cluster_vals = [list(reg_cluster_assignment).count(x) for x in cluster_vals]
reg_majority_cluster = cluster_vals[np.argmax(reg_cluster_vals)]

# Put out trials of each type that are categorized in the labeled majority cluster
if stress_majority_cluster != reg_majority_cluster:
	stress_well_clustered_ind = np.ravel(np.nonzero(stress_cluster_assignment == stress_majority_cluster))
	well_clustered_ibi_stress = norm_ibi_stress_mean[stress_well_clustered_ind]
	well_clustered_pupil_stress = norm_pupil_stress_mean[stress_well_clustered_ind]

	reg_well_clustered_ind = np.ravel(np.nonzero(reg_cluster_assignment == reg_majority_cluster))
	well_clustered_ibi_reg = norm_ibi_reg_before_mean[reg_well_clustered_ind]
	well_clustered_pupil_reg = norm_pupil_reg_before_mean[reg_well_clustered_ind]

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

for i, ind in enumerate(lfp_ind_successful_stress[stress_well_clustered_ind]):
	trial_array = []
	trial_array.append(well_clustered_pupil_stress[i])
	trial_array.append(well_clustered_ibi_stress[i])
	
	
	for chann in lfp_channels:
		#freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_successful_stress[i]], lfp_samprate, nperseg=1024)
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+lfp_samprate/2], lfp_samprate, nperseg=1024)  # take 0.5 s of data 
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			#trial_array.append(np.sum(freq_band))
		lfp_power_successful_stress.append(np.sum(freq_band))
	
	X_successful_stress.append(trial_array)

for i, ind in enumerate(lfp_ind_successful_reg[reg_well_clustered_ind]):
	trial_array = []
	trial_array.append(well_clustered_pupil_reg[i])
	trial_array.append(well_clustered_ibi_reg[i])
	
	
	for chann in lfp_channels:
		#freq, Pxx_den = signal.welch(lfp[chann][ind:ind+samples_lfp_successful_reg[i]], lfp_samprate, nperseg=1024)
		freq, Pxx_den = signal.welch(lfp[chann][ind:ind+lfp_samprate/2], lfp_samprate, nperseg=1024)  # take 0.5 s of data 
		
		for k, item in enumerate(bands):
			freq_band = [Pxx_den[j] for j in range(len(freq)) if (item[0] <= freq[j] <= item[1])]
			#trial_array.append(np.sum(freq_band))
		lfp_power_successful_reg.append(np.sum(freq_band))
	
	X_successful_reg.append(trial_array)


# Labels: 0 = regular, 1 = stress
X_successful_stress = np.array(X_successful_stress)
#X_successful_stress = (X_successful_stress - np.nanmean(X_successful_stress,axis=0))/np.nanstd(X_successful_stress,axis=0)
num_successful_stress = X_successful_stress.shape[0]
y_successful_stress = np.ones(num_successful_stress)
X_successful_reg = np.array(X_successful_reg)
#X_successful_reg = (X_successful_reg - np.nanmean(X_successful_reg,axis=0))/np.nanstd(X_successful_reg,axis=0)
num_successful_reg = X_successful_reg.shape[0]
y_successful_reg = np.zeros(num_successful_reg)

X_successful = np.vstack([X_successful_reg, X_successful_stress])
y_successful = np.append(y_successful_reg,y_successful_stress)

clf_all = LinearDiscriminantAnalysis()
clf_all.fit(X_successful, y_successful)
scores = cross_val_score(LinearDiscriminantAnalysis(),X_successful,y_successful,scoring='accuracy',cv=10)
print "CV (10-fold) scores:", scores
print "Avg CV score:", scores.mean()

predict_stress = clf_all.predict(X_successful_stress)
print "Fraction of stress trials classified as stress:", np.sum(predict_stress)/len(predict_stress)

x_successful = sm.add_constant(X_successful,prepend='False')

print "Regression with successful, well-clustered trials"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()

"""
Decision boundary given by:
np.dot(clf.coef_, x) - clf.intercept_ = 0 according to 
http://stackoverflow.com/questions/36745480/how-to-get-the-equation-of-the-boundary-line-in-linear-discriminant-analysis-wit
"""

#LDAforFeatureSelection(X_successful,y_successful,filename,block_num)

'''
Do regression as well: power is total power in beta band per trial
'''
'''
lfp_power_successful_reg = (lfp_power_successful_reg - np.nanmean(lfp_power_successful_reg))/np.nanstd(lfp_power_successful_reg)
lfp_power_successful_stress = (lfp_power_successful_stress - np.nanmean(lfp_power_successful_stress))/np.nanstd(lfp_power_successful_stress)

x_successful = np.vstack((np.append(ibi_reg_mean, ibi_stress_mean), np.append(pupil_reg_mean, pupil_stress_mean), 
				np.append(lfp_power_successful_reg, lfp_power_successful_stress)))
x_successful = np.transpose(x_successful)
x_successful = sm.add_constant(x_successful,prepend='False')

print "Regression with successful trials"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()

'''






