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
from spectralAnalysis import TrialAveragedPSD, computePowerFeatures, computePowerFeatures_Chirplets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
import os.path
import time

from StressTaskBehavior import StressBehavior

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, TanhLayer
from pybrain.datasets import SupervisedDataSet


hdf_filename = 'mari20160517_07_te2097.hdf'
hdf_filename_stim = 'mari20160517_09_te2099.hdf'
filename = 'Mario20160517'
filename2 = 'Mario20160517'
block_num = 1
block_num_stim = 2
print filename
TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
hdf_location_stim = '/storage/rawdata/hdf/'+hdf_filename_stim
#hdf_location = hdffilename
pf_location = '/home/srsummerson/storage/PowerFeatures/'
pf_filename = pf_location + filename+'_b'+str(block_num)+'_PowerFeatures.mat'
pf_filename_stim = pf_location + filename+'_b'+str(block_num_stim)+'_PowerFeatures.mat'
phys_filename = pf_location + filename+'_b'+str(block_num)+'_PhysFeatures.mat'
phys_filename_stim = pf_location + filename+'_b'+str(block_num_stim)+'_PhysFeatures.mat'

lfp_channels = np.arange(1,161,dtype = int)					# channels 1 - 160
lfp_channels = np.delete(lfp_channels, [129, 131, 145])		# channels 129, 131, and 145 are open
lfp_channels = lfp_channels - 1								# correct for indexing to start at 0 instead of 1

bands = [[8,13],[13,30],[40,70],[70,200]]
# bands = [[8,18], [14-21],[101,298]]
'''
Load behavior data:
Note that FreeChoiceBehavior_withStressTrials has been updated so that the array ind_center_states contains the indices corresponding to when the center hold state begins.

state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward = FreeChoiceBehavior_withStressTrials(hdf_location)
state_time_stim, ind_center_states_stim, ind_check_reward_states_stim, all_instructed_or_freechoice_stim, all_stress_or_not_stim, successful_stress_or_not_stim,trial_success_stim, target_stim, reward_stim = FreeChoiceBehavior_withStressTrials(hdf_location_stim)
'''
BlockAB_behavior = StressBehavior(hdf_location)
if hdf_filename_stim != '':
	BlockCB_behavior = StressBehavior(hdf_location_stim)

print "Behavior data loaded."

if os.path.exists(pf_filename)&os.path.exists(pf_filename_stim)&os.path.exists(phys_filename)&os.path.exists(phys_filename_stim):
	print "Power features previously computed. Loading now."
	lfp_features = dict()
	sp.io.loadmat(pf_filename,lfp_features)
	lfp_features_stim = dict()
	sp.io.loadmat(pf_filename_stim,lfp_features_stim)

	phys_features = dict()
	sp.io.loadmat(phys_filename,phys_features)
	ibi_reg_mean = np.ravel(phys_features['ibi_reg_mean'] )
	ibi_stress_mean = np.ravel(phys_features['ibi_stress_mean'])
	pupil_reg_mean = np.ravel(phys_features['pupil_reg_mean'])
	pupil_stress_mean = np.ravel(phys_features['pupil_stress_mean'])

	phys_features_stim = dict()
	sp.io.loadmat(phys_filename_stim,phys_features_stim)
	ibi_stress_mean_stim = np.ravel(phys_features_stim['ibi_stress_mean_stim'])
	pupil_stress_mean_stim = np.ravel(phys_features_stim['pupil_stress_mean_stim'])
	
else:

	'''
	Load syncing data for behavior and TDT recording
	'''
	print "Determine stress and regular trials"
	trial_types = np.ravel(BlockAB_behavior.stress_type[BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states]])
	BlockAB_stress_trial_inds = np.ravel(np.nonzero(trial_types==1))
	BlockAB_reg_trial_inds = np.ravel(np.nonzero(trial_types==0))

	trial_types = np.ravel(BlockCB_behavior.stress_type[BlockCB_behavior.state_time[BlockCB_behavior.ind_check_reward_states]])
	BlockCB_stress_trial_inds = np.ravel(np.nonzero(trial_types==1))

	print "Loading syncing data."

	hdf_times = dict()
	mat_filename = '/home/srsummerson/storage/syncHDF/' + filename+'_b'+str(block_num)+'_syncHDF.mat'
	#sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)
	lfp_ind_hold_center_states_reg_trials = BlockAB_behavior.get_state_TDT_LFPvalues(BlockAB_behavior.ind_check_reward_states[BlockAB_reg_trial_inds] - 4,mat_filename)
	lfp_ind_hold_center_states_stress_trials = BlockAB_behavior.get_state_TDT_LFPvalues(BlockAB_behavior.ind_check_reward_states[BlockAB_stress_trial_inds] - 4,mat_filename)

	# NOTE: peripheral hold
	lfp_ind_hold_target_states_reg_trials = BlockAB_behavior.get_state_TDT_LFPvalues(BlockAB_behavior.ind_check_reward_states[BlockAB_reg_trial_inds] - 2,mat_filename)
	lfp_ind_hold_target_states_stress_trials = BlockAB_behavior.get_state_TDT_LFPvalues(BlockAB_behavior.ind_check_reward_states[BlockAB_stress_trial_inds] - 2,mat_filename)

	# NOTE: REWARD HOLD
	lfp_ind_check_reward_states_reg_trials = BlockAB_behavior.get_state_TDT_LFPvalues(BlockAB_behavior.ind_check_reward_states[BlockAB_reg_trial_inds],mat_filename)
	lfp_ind_check_reward_states_stress_trials = BlockAB_behavior.get_state_TDT_LFPvalues(BlockAB_behavior.ind_check_reward_states[BlockAB_stress_trial_inds],mat_filename)


	hdf_times_stim = dict()
	mat_filename_stim = filename+'_b'+str(block_num_stim)+'_syncHDF.mat'
	#sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename_stim,hdf_times_stim)
	lfp_ind_hold_center_states_stim_trials = BlockCB_behavior.get_state_TDT_LFPvalues(BlockCB_behavior.ind_check_reward_states[BlockCB_stress_trial_inds] - 4,mat_filename)
	lfp_ind_check_reward_states_stim_trials = BlockCB_behavior.get_state_TDT_LFPvalues(BlockCB_behavior.ind_check_reward_states[BlockCB_stress_trial_inds],mat_filename)

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
				if (channel + 96) in lfp_channels:
					channel_name = channel + 96
					lfp[channel_name] = np.ravel(sig)
		for sig in bl.segments[block_num_stim-1].analogsignals:
			if (sig.name == 'PupD 1'):
				pupil_data_stim = np.ravel(sig)
				pupil_samprate_stim = sig.sampling_rate.item()
			if (sig.name == 'HrtR 1'):
				pulse_data_stim = np.ravel(sig)
				pulse_samprate_stim = sig.sampling_rate.item()
			if (sig.name[0:4] == 'LFP1'):
				channel = sig.channel_index
				if (channel in lfp_channels)&(channel < 97):
					lfp_samprate_stim = sig.sampling_rate.item()
					lfp_stim[channel] = np.ravel(sig)
			if (sig.name[0:4] == 'LFP2'):
				channel = sig.channel_index
				if (channel + 96) in lfp_channels:
					channel_name = channel + 96
					lfp_stim[channel_name] = np.ravel(sig)


	print "Finished loading TDT data."

	'''
	Process pupil and pulse data
	'''
	
	# Find IBIs and pupil data for all successful stress trials. 
	samples_pulse_successful_stress = ((BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states[BlockAB_stress_trial_inds]] - BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states[BlockAB_stress_trial_inds]-4])/60.)*pulse_samprate 	#number of samples in trial interval for pulse signal
	samples_pulse_successful_stress = np.array([int(val) for val in samples_pulse_successful_stress])
	samples_pupil_successful_stress = 0.1*pupil_samprate*np.ones(len(BlockAB_stress_trial_inds))  # look at first 100 ms
	samples_pupil_successful_stress = np.array([int(val) for val in samples_pulse_successful_stress])

	ibi_stress_mean, ibi_stress_std, pupil_stress_mean, pupil_stress_std, nbins_ibi_stress, ibi_stress_hist, nbins_pupil_stress, pupil_stress_hist = getIBIandPuilDilation(pulse_data, lfp_ind_hold_center_states_stress_trials,samples_pulse_successful_stress, pulse_samprate,pupil_data, lfp_ind_hold_center_states_stress_trials,samples_pupil_successful_stress,pupil_samprate)

	# Find IBIs and pupil data for all successful regular trials.
	samples_pulse_successful_reg = ((BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states[BlockAB_reg_trial_inds]] - BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states[BlockAB_reg_trial_inds]-4])/60.)*pulse_samprate 	#number of samples in trial interval for pulse signal
	samples_pulse_successful_reg = np.array([int(val) for val in samples_pulse_successful_reg])
	samples_pupil_successful_reg = 0.1*pupil_samprate*np.ones(len(BlockAB_reg_trial_inds))  # look at first 100 ms
	samples_pupil_successful_reg = np.array([int(val) for val in samples_pupil_successful_reg])


	ibi_reg_mean, ibi_reg_std, pupil_reg_mean, pupil_reg_std, nbins_ibi_reg, ibi_reg_hist, nbins_pupil_reg, pupil_reg_hist = getIBIandPuilDilation(pulse_data, lfp_ind_hold_center_states_reg_trials,samples_pulse_successful_reg, pulse_samprate,pupil_data, lfp_ind_hold_center_states_reg_trials,samples_pupil_successful_reg,pupil_samprate)

	# Find IBIs and pupil data for all successful stress trials with stimulation. 
	samples_pulse_successful_stress_stim = ((BlockCB_behavior.state_time[BlockCB_behavior.ind_check_reward_states[BlockCB_stress_trial_inds]] - BlockCB_behavior.state_time[BlockCB_behavior.ind_check_reward_states[BlockCB_stress_trial_inds]-4])/60.)*pulse_samprate 	#number of samples in trial interval for pulse signal
	samples_pulse_successful_stress_stim = np.array([int(val) for val in samples_pulse_successful_stress_stim])
	samples_pupil_successful_stress_stim = 0.1*pupil_samprate*np.ones(len(BlockCB_stress_trial_inds))  # look at first 100 ms
	samples_pupil_successful_stress_stim = np.array([int(val) for val in samples_pupil_successful_stress_stim])

	ibi_stress_mean_stim, ibi_stress_std_stim, pupil_stress_mean_stim, pupil_stress_std_stim, nbins_ibi_stress_stim, ibi_stress_hist_stim, nbins_pupil_stress_stim, pupil_stress_hist_stim = getIBIandPuilDilation(pulse_data, lfp_ind_hold_center_states_stim_trials,samples_pulse_successful_stress_stim, pulse_samprate,pupil_data, lfp_ind_hold_center_states_stim_trials,samples_pupil_successful_stress_stim,pupil_samprate)

	'''
	Get power in designated frequency bands per trial
	'''
	lfp_power_successful_stress = []
	lfp_power_successful_reg = []
	lfp_power_successful_stress_stim = []
	X_successful_stress = []
	X_successful_reg = []
	X_successful_stim = []
	
	#### event_indices: N x M array of event indices, where N is the number of trials and M is the number of different events 
	lfp_center_states = np.append(lfp_ind_hold_center_states_reg_trials - int(0.4*lfp_samprate), lfp_ind_hold_center_states_stress_trials - int(0.4*lfp_samprate))
	lfp_before_reward_states = np.append(lfp_ind_hold_target_states_reg_trials, lfp_ind_hold_target_states_stress_trials)
	lfp_after_reward_states = np.append(lfp_ind_check_reward_states_reg_trials, lfp_ind_check_reward_states_stress_trials)

	event_indices = np.vstack([lfp_center_states,lfp_before_reward_states,lfp_after_reward_states]).T
	t_window = [0.4,0.5,0.4]
	event_indices.shape
	
	print "Computing LFP features."
	lfp_features = computePowerFeatures(lfp, lfp_samprate, bands, event_indices, t_window)
	sp.io.savemat(pf_filename,lfp_features)

	phys_features = dict()
	phys_features['ibi_reg_mean'] = ibi_reg_mean
	phys_features['ibi_stress_mean'] = ibi_stress_mean
	phys_features['pupil_reg_mean'] = pupil_reg_mean
	phys_features['pupil_stress_mean'] = pupil_stress_mean
	sp.io.savemat(phys_filename,phys_features)

	#### event_indices: N x M array of event indices, where N is the number of trials and M is the number of different events 
	lfp_center_states_stim = lfp_ind_hold_center_states_stim_trials - int(0.4*lfp_samprate)
	lfp_before_reward_states_stim = lfp_ind_check_reward_states_stim_trials - int(0.5*lfp_samprate)
	lfp_after_reward_states_stim = lfp_ind_check_reward_states_stim_trials

	event_indices_stim = np.vstack([lfp_center_states_stim,lfp_before_reward_states_stim,lfp_after_reward_states_stim]).T
	t_window = [0.4,0.5,0.4]
	
	print "Computing stim LFP features."
	lfp_features_stim = computePowerFeatures(lfp_stim, lfp_samprate, bands, event_indices_stim, t_window)
	sp.io.savemat(pf_filename_stim,lfp_features_stim)

	phys_features_stim = dict()
	phys_features_stim['ibi_stress_mean_stim'] = ibi_stress_mean_stim
	phys_features_stim['pupil_stress_mean_stim'] = pupil_stress_mean_stim
	sp.io.savemat(phys_filename_stim,phys_features_stim)

ibi_mean = np.append(np.array(ibi_reg_mean), np.array(ibi_stress_mean))
pupil_mean = np.append(np.array(pupil_reg_mean), np.array(pupil_stress_mean))

X_successful = []
X_successful_stim = []
lfp_features_keys = lfp_features.keys()
lfp_features_stim_keys = lfp_features_stim.keys()
skip_keys = ['__globals__','__header__','__version__']
#### Check if keys are in order

lfp_features_keys = [int(key) for key in lfp_features_keys if (key not in skip_keys)]
lfp_features_keys.sort()
lfp_features_stim_keys = [int(key) for key in lfp_features_stim_keys if (key not in skip_keys)]
lfp_features_stim_keys.sort()

for key in lfp_features_keys:
	trial_features = lfp_features[str(key)].flatten()
	trial_features = np.append(trial_features, [ibi_mean[key], pupil_mean[key]])
	X_successful.append(trial_features)
for key in lfp_features_stim_keys:
	trial_features = lfp_features_stim[str(key)].flatten()
	trial_features = np.append(trial_features, [ibi_stress_mean_stim[key], pupil_stress_mean_stim[key]])
	X_successful_stim.append(trial_features)

X_successful_mean = np.abs(np.mean(X_successful))
X_successful_std = np.abs(np.std(X_successful))

#X_successful = (X_successful - X_successful_mean)/X_successful_std
X_successful = np.array(X_successful)

X_successful_stim_mean = np.abs(np.mean(X_successful_stim))
X_successful_stim_std = np.abs(np.std(X_successful_stim))

#X_successful_stim = (X_successful_stim - X_successful_stim_mean)/X_successful_stim_std
X_successful_stim = np.array(X_successful_stim)

y_successful_reg = np.zeros(len(ibi_reg_mean))
y_successful_stress = np.ones(len(ibi_stress_mean))
y_successful = np.append(y_successful_reg,y_successful_stress)

y_successful_stim = np.ones(X_successful_stim.shape[0])
'''
print "LDA using Power Features:"
clf_all = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto')
clf_all.fit(X_successful, y_successful)
scores = cross_val_score(LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto'),X_successful,y_successful,scoring='accuracy',cv=10)
print "CV (10-fold) scores:", scores
print "Avg CV score:", scores.mean()
'''
'''
Using ANN to predict classes.
'''
print "Artificial Neural Network with Power Features:"
# Create a dummy dataset with pybrain

num_trials, num_features = X_successful.shape
alldata = SupervisedDataSet(num_features,1) 

stimalldata = SupervisedDataSet(num_features,1)

# add the features and class labels into the dataset
for xnum in xrange(num_trials): 
    alldata.addSample(X_successful[xnum,:],y_successful[xnum])

# add the features and dummy class labels into the stim dataset
for xnum in xrange(len(y_successful_stim)):
	stimalldata.addSample(X_successful_stim[xnum,:],y_successful_stim[xnum])

# split the data into testing and training data
tstdata_temp, trndata_temp = alldata.splitWithProportion(0.15)

# small bug with _convertToOneOfMany function.  This fixes that
tstdata = ClassificationDataSet(num_features,1,nb_classes=2)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample(tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])

trndata = ClassificationDataSet(num_features,1,nb_classes=2)
for n in xrange(0,trndata_temp.getLength()):
    trndata.addSample(trndata_temp.getSample(n)[0],trndata_temp.getSample(n)[1])

valdata = ClassificationDataSet(num_features,1,nb_classes=2)
for n in xrange(0,stimalldata.getLength()):
    valdata.addSample(stimalldata.getSample(n)[0],stimalldata.getSample(n)[1])

# organizes dataset for pybrain
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

valdata._convertToOneOfMany()

# sample printouts before running classifier
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

# build the ANN
# 2 hidden layers (4 layers total)
fnn = buildNetwork(trndata.indim, trndata.indim/2, trndata.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
 # create the trainer
trainer = BackpropTrainer(fnn, dataset=trndata)

num_epochs = 4
epoch_error = np.zeros(num_epochs)
epoch_tst_error = np.zeros(num_epochs)
start_time = time.time()
for i in xrange(num_epochs): # given how many features there are, lots of iterations are required
    # classify the data
    #trainer.trainEpochs(50) # can choose how many epochs to train on using trainEpochs()
    trainer.train()
    trnresult = percentError(trainer.testOnClassData(), trndata['class'])
    tstresult = percentError(trainer.testOnClassData(dataset = tstdata), tstdata['class'])
    epoch_error[i] = trnresult
    epoch_tst_error[i] = tstresult
    #print "\n epoch: %4d" % trainer.totalepochs
    print "Epoch: %i, train error: %5.2f%%" % (i,trnresult)
    print "Length time: ", (time.time() - start_time)/60.
    #print "\n test error: %5.2f%%" % tstresult
end_time = time.time()
print "Training time:", (end_time - start_time)/60.
	
trnresult = percentError(trainer.testOnClassData(), trndata['class'])
tstresult = percentError(trainer.testOnClassData(dataset = tstdata), tstdata['class'])
valresult = percentError(trainer.testOnClassData(dataset = valdata), valdata['class'])

# trndata_data = trndata['input']
'''
print 'LDA Analysis:'
print "\n CV (10-fold) scores:", scores
print "\n Avg CV score:", scores.mean()
'''
print '\n\n ANN Analysis:'
print "\n epoch: %4d" % trainer.totalepochs
print "\n train error: %5.2f%%" % trnresult
print "\n test error: %5.2f%%" % tstresult
print "\n stim trial rate of reg classification: %5.2f%%" % valresult
'''
orig_std = sys.stdout
f = file(pf_location + filename + '.txt', 'w')
sys.stdout = f

print 'LDA Analysis:'
print "\n CV (10-fold) scores:", scores
print "\n Avg CV score:", scores.mean()

print '\n\n ANN Analysis:'
print "epoch: %4d" % trainer.totalepochs
print "\n train error: %5.2f%%" % trnresult
print "\n test error: %5.2f%%" % tstresult
print "\n stim trial rate of reg classification: %5.2f%%" % valresult

sys.stdout = orig_std
f.close()
'''
plt.figure()
plt.plot(xrange(num_epochs),epoch_error,'b',label='Training Error')
plt.plot(xrange(num_epochs),epoch_tst_error,'r',label='Test set Error')
plt.xlabel('Num Epochs')
plt.ylabel('Percent Error')
plt.legend()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_ANNPerformance.svg')

'''
x_successful = sm.add_constant(X_successful,prepend='False')

print "Regression with all trials"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()
'''