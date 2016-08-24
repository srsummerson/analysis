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
from spectralAnalysis import TrialAveragedPSD, computePowerFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
import os.path

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet


hdf_filename = 'mari20160713_10_te2348.hdf'
filename = 'Mario20160713'
block_num = 1
print filename
TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
#hdf_location = hdffilename
pf_location = '/home/srsummerson/storage/PowerFeatures/'
pf_filename = pf_location + filename+'_b'+str(block_num)+'_PowerFeatures.mat'


lfp_channels = range(0,160)
lfp_channels.pop(129)  # delete channel 129
lfp_channels.pop(130)  # delete channel 131
lfp_channels.pop(143)  # delete channel 145
#bands = [[1,8],[8,12],[12,30],[30,55],[65,100]]
bands = [[0,20],[20,40],[40,60]]

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
row_ind_successful_stress_check_reward = ind_check_reward_states[ind_successful_stress_reward]
response_time_successful_stress = (state_time[row_ind_successful_stress_reward] - state_time[row_ind_successful_stress])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response time for all stress trials
ind_stress = np.ravel(np.nonzero(all_stress_or_not))
row_ind_stress = ind_center_states[ind_stress]  # gives row index
#row_ind_stress_check_reward = ind_check_reward_states[ind_stress]
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
row_ind_successful_reg_check_reward = ind_check_reward_states[ind_successful_reg_reward]
response_time_successful_reg = (state_time[row_ind_successful_reg_reward] - state_time[row_ind_successful_reg])/float(60)

# Response time for all regular trials
ind_reg = np.ravel(np.nonzero(np.logical_not(all_stress_or_not)))
row_ind_reg = ind_center_states[ind_reg]
#row_ind_reg_check_reward = ind_check_reward_states[ind_reg]
row_ind_end_reg = np.zeros(len(row_ind_reg))
row_ind_end_reg = np.minimum(row_ind_reg + 5,total_states-1)  # target_transition state occues two states later for successful trials
for i in range(0,len(row_ind_successful_reg)):
	ind = np.where(row_ind_reg == row_ind_successful_reg[i])[0]
	row_ind_end_reg[ind] = row_ind_successful_reg_reward[i]
response_time_reg = (state_time[row_ind_end_reg] - state_time[row_ind_reg])/float(60)

if os.path.exists(pf_filename):
	print "Power features previously computed. Loading now."
	lfp_features = dict()
	sp.io.loadmat(pf_filename,lfp_features)
else:

	'''
	Load syncing data for behavior and TDT recording
	'''
	print "Loading syncing data."

	hdf_times = dict()
	mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
	sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

	print "Loading TDT data."
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
				if (channel in lfp_channels)&(channel < 97):
					lfp_samprate = sig.sampling_rate.item()
					lfp[channel] = np.ravel(sig)
			if (sig.name[0:4] == 'LFP2'):
				channel = sig.channel_index
				if (channel % 96) in lfp_channels:
					channel_name = channel + 96
					lfp[channel_name] = np.ravel(sig)

	print "Finished loading TDT data."
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
	state_row_ind_successful_stress_check_reward = state_time[row_ind_successful_stress_check_reward]
	state_row_ind_successful_reg = state_time[row_ind_successful_reg]
	state_row_ind_successful_reg_check_reward = state_time[row_ind_successful_reg_check_reward]
	pulse_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
	pupil_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
	lfp_ind_successful_stress_center = np.zeros(row_ind_successful_stress.size)
	lfp_ind_successful_stress_check_reward = np.zeros(row_ind_successful_stress.size)
	pulse_ind_successful_reg = []
	pupil_ind_successful_reg = []
	lfp_ind_successful_reg_center = np.zeros(row_ind_successful_reg.size)
	lfp_ind_successful_reg_check_reward = np.zeros(row_ind_successful_reg.size)
	state_row_ind_stress = state_time[row_ind_stress]
	state_row_ind_reg = state_time[row_ind_reg]
	pulse_ind_stress = np.zeros(row_ind_stress.size)
	pupil_ind_stress = np.zeros(row_ind_stress.size)
	lfp_ind_stress = np.zeros(row_ind_stress.size)
	pulse_ind_reg = []
	pupil_ind_reg = []
	lfp_ind_reg = np.zeros(row_ind_reg.size)


	for i in range(0,len(row_ind_successful_stress)):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
		#pulse_ind_successful_stress[i] = pulse_dio_sample_num[hdf_index]
		#pupil_ind_successful_stress[i] = pupil_dio_sample_num[hdf_index]
		lfp_ind_successful_stress_center[i] = lfp_dio_sample_num[hdf_index]
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress_check_reward[i]))
		lfp_ind_successful_stress_check_reward[i] = lfp_dio_sample_num[hdf_index]
	for i in range(0,len(row_ind_stress)):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_stress[i]))
		#pulse_ind_stress[i] = pulse_dio_sample_num[hdf_index]
		#pupil_ind_stress[i] = pupil_dio_sample_num[hdf_index]
		lfp_ind_stress[i] = lfp_dio_sample_num[hdf_index]

	if len(row_ind_successful_stress) > 0: 
		ind_start_stress = row_ind_successful_stress[0]
	else:
		ind_start_stress = np.inf

	ind_start_all_stress = row_ind_stress[0]
	for i in range(0,len(state_row_ind_successful_reg)):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		#pulse_ind_successful_reg.append(pulse_dio_sample_num[hdf_index])
		#pupil_ind_successful_reg.append(pupil_dio_sample_num[hdf_index])
		lfp_ind_successful_reg_center[i] = lfp_dio_sample_num[hdf_index]
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg_check_reward[i]))
		lfp_ind_successful_reg_check_reward[i] = lfp_dio_sample_num[hdf_index]
		
	for i in range(0,len(state_row_ind_reg)):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
		#pulse_ind_reg.append(pulse_dio_sample_num[hdf_index])
		#pupil_ind_reg.append(pupil_dio_sample_num[hdf_index])
		lfp_ind_reg[i] = lfp_dio_sample_num[hdf_index]


	'''
	Process pupil and pulse data
	'''
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

	#### event_indices: N x M array of event indices, where N is the number of trials and M is the number of different events 
	lfp_center_states = np.append(lfp_ind_successful_stress_center, lfp_ind_successful_reg_center)
	lfp_before_reward_states = np.append(lfp_ind_successful_stress_check_reward - int(0.5*lfp_samprate), lfp_ind_successful_reg_check_reward - int(0.5*lfp_samprate))
	lfp_after_reward_states = np.append(lfp_ind_successful_stress_check_reward, lfp_ind_successful_reg_check_reward)

	event_indices = np.vstack([lfp_center_states,lfp_before_reward_states,lfp_after_reward_states]).T
	t_window = [0.5,0.5,0.5]
	print "Computing LFP features."
	lfp_features = computePowerFeatures(lfp, lfp_samprate, bands, event_indices, t_window)
	#pf_filename = filename+'_b'+str(block_num)+'_PowerFeatures.mat'
	#sp.io.savemat('/home/srsummerson/storage/PowerFeatures/'+pf_filename,lfp_features)
	sp.io.savemat(pf_filename,lfp_features)

X_successful = []
lfp_features_keys = lfp_features.keys()
skip_keys = ['__globals__','__header__','__version__']
for key in lfp_features_keys:
	if key not in skip_keys:
		trial_features = lfp_features[key]
		trial_features = trial_features[:,0:2*len(bands)].flatten()  # take only powers from first event
		X_successful.append(trial_features)

X_successful_mean = np.abs(np.mean(X_successful))
X_successful_std = np.abs(np.std(X_successful))

X_successful = (X_successful - X_successful_mean)/X_successful_std

y_successful_reg = np.zeros(len(ind_successful_reg))
y_successful_stress = np.ones(len(ind_successful_stress))
y_successful = np.append(y_successful_reg,y_successful_stress)

print "LDA using Power Features:"
clf_all = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto')
clf_all.fit(X_successful, y_successful)
scores = cross_val_score(LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto'),X_successful,y_successful,scoring='accuracy',cv=10)
print "CV (10-fold) scores:", scores
print "Avg CV score:", scores.mean()

print "Artificial Neural Network with Power Features:"
# Create a dummy dataset with pybrain

num_trials, num_features = X_successful.shape
alldata = SupervisedDataSet(num_features,1) 

# add the features and target locations into the dataset
for xnum in xrange(num_trials): 
    alldata.addSample(X_successful[xnum,:],y_successful[xnum])

# split the data into testing and training data
tstdata_temp, trndata_temp = alldata.splitWithProportion(0.2)

# small bug with _convertToOneOfMany function.  This fixes that
tstdata = ClassificationDataSet(num_features,1,nb_classes=2)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample(tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])

trndata = ClassificationDataSet(num_features,1,nb_classes=2)
for n in xrange(0,trndata_temp.getLength()):
    trndata.addSample(trndata_temp.getSample(n)[0],trndata_temp.getSample(n)[1])

# organizes dataset for pybrain
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

# sample printouts before running classifier
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

# build the ANN
# 50 hidden layers (52 layers total)
fnn = buildNetwork(trndata.indim, 2, trndata.outdim, outclass=SoftmaxLayer)
 # create the trainer
trainer = BackpropTrainer(fnn, dataset=trndata)
    
for i in xrange(1000): # given how many features there are, lots of iterations are required
    # classify the data
    trainer.train() # can choose how many epochs to train on using trainEpochs()
    
trnresult = percentError(trainer.testOnClassData(), trndata['class'])
tstresult = percentError(trainer.testOnClassData(dataset = tstdata), tstdata['class'])
print "epoch: %4d" % trainer.totalepochs, \
    " train error: %5.2f%%" % trnresult, \
    " test error: %5.2f%%" % tstresult


x_successful = sm.add_constant(X_successful,prepend='False')
'''
print "Regression with all trials"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()
'''