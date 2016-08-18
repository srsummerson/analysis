import numpy as np
import scipy as sp
from scipy import signal
import re
from neo import io
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
from OMNI_methods import convert_OMNI, get_stim_sync_sig, computePowersWithChirplets, powersWithFFT, powersWithSpecgram
from spectralAnalysis import LFPPowerPerTrial_SingleBand_PerChannel_Timestamps
import tables

blocks = [1,2,3]

#chans = [2, 9, 12, 33, 34, 62, 71, 78, 83, 86, 92, 94]
#chans = [9,34, 83, 86] 
channels = [89]  		# picking subset of channels to analyze
freq_window = [20,40]	# defining frequency window of interest
center_freq = 20.
num_trials = np.zeros(len(blocks))

filename_prefix = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_Device/Data/streams7_20/'
tdt_filename = 'Mario20160720-OMNI'
hdf_prefix = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_device/Data/' 

for ind in blocks:

	block_num = ind

	if block_num ==1:
		filename = filename_prefix + '20160720-163020.csv'
		omni_start_samp = 7237
		tdt_start_samp = 16437
		hdf_filename = hdf_prefix + 'mari20160720_02_te2381.hdf'
	elif block_num ==2:

		filename = filename_prefix + '20160720-171300.csv'
		omni_start_samp = 5246
		tdt_start_samp = 27849
		hdf_filename = hdf_prefix + 'mari20160720_17_te2396.hdf'
	elif block_num ==3:
		filename = filename_prefix + '20160720-174338.csv'
		omni_start_samp = 4142
		tdt_start_samp = 21011
		hdf_filename = hdf_prefix + 'mari20160720_18_te2397.hdf'


	'''
	Load syncing data for behavior and TDT recording
	'''
	print "Loading syncing data."

	hdf_times = dict()
	mat_filename = tdt_filename+'_b'+str(block_num)+'_syncHDF.mat'
	#sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)
	sp.io.loadmat('C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_device/Data/' + mat_filename,hdf_times)

	hdf_rows = np.ravel(hdf_times['row_number'])
	hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
	dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
	dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])
	hdf_row_timestamps = dio_tdt_sample/float(dio_freq)

	'''
	Get OMNI data
	'''
	data = pd.read_csv(filename,sep=',',header=None,skiprows=[0,1])
	print "Data read."

	corrected_counter, corrected_channel_data, channel_data, miss_samp_index = convert_OMNI(data)
	# saved corrected_channel data
	mat_filename = tdt_filename +'_b'+str(block_num)+'_CorrectedData.mat'
	#sp.io.savemat('C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_Device/Data/'+mat_filename,corrected_channel_data)


	Avg_Fs = 944.  # based on offline inspection of stim pulse timings in OMNI recordings
	num_samps, num_chans = corrected_channel_data.shape
	'''
	Get sync data
	'''
	#stim_signal, stim_on_trig, stim_delivered, stwv_samprate = get_stim_sync_sig(TDT_location)
	stwv_samprate = 24414.0625
	# stim wave samples that correspond to dio samples
	stim_dio_sample_num = (float(stwv_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1
	# DIO TDT sample number that corresponds to first stim pulse
	dio_start_samp = int(dio_freq[0]*tdt_start_samp/stwv_samprate)
	# translate DIO vals to OMNI sample numbers
	omni_tdt_sample_num = (Avg_Fs/dio_freq[0])*(dio_tdt_sample - dio_start_samp) + omni_start_samp
	omni_tdt_sample_num = np.array([int(samp) for samp in omni_tdt_sample_num])
	

	'''
	Translate behavioral timestamps to timestamps recorded with OMNI device
	'''
	hdf = tables.openFile(hdf_filename)

	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	ind_reward_states = np.ravel(np.nonzero(state == 'reward'))   # total number of unique trials
	ind_gocue_states = ind_reward_states - 4			  # index of target state
	state_time_ind_reward_states = state_time[ind_reward_states]
	state_time_ind_gocue_states = state_time[ind_gocue_states]
	num_trials[ind-1] = len(state_time_ind_gocue_states)

	response_time = state_time_ind_reward_states[:-1] - state_time_ind_gocue_states[:-1]
	sort_response_time = np.argsort(response_time)

	print "Beginning to find sample numbers for rows of relevant behavioral events"
	if ind == 1:
		ntrials = len(state_time_ind_reward_states)-68
	else:
		ntrials = len(state_time_ind_reward_states)-1

	omni_ind_reward = np.zeros(ntrials)
	omni_ind_gocue = np.zeros(ntrials)
	save_hdf_index = np.zeros(ntrials)

	for i in range(0,ntrials):  # exclude final trial in case it includes stim pulses in after period
		hdf_index = np.argmin(np.abs(hdf_rows - state_time_ind_reward_states[i]))
		save_hdf_index[i] = hdf_index
		#omni_ind_reward[i] = int(omni_tdt_sample_num[stim_dio_sample_num[hdf_index]])
		omni_ind_reward[i] = int(omni_tdt_sample_num[hdf_index])
		hdf_index = np.argmin(np.abs(hdf_rows - state_time_ind_gocue_states[i]))
		#omni_ind_gocue[i] = int(omni_tdt_sample_num[stim_dio_sample_num[hdf_index]])
		omni_ind_gocue[i] = int(omni_tdt_sample_num[hdf_index])

	hdf.close()
	print "Compute powers."
	#trialPowers, timestamps, complex_power, examp_data, window = computePowersWithChirplets(channel_data,Avg_Fs,channels,omni_ind_reward,2,2,center_freq)
	print "Compute FFT"
	#trialPowers_FFT, freq = powersWithFFT(channel_data,Avg_Fs,channels,omni_ind_reward, 2, 2)
	"""
	omni_data = dict()
	omni_data['corrected_data'] = corrected_channel_data
	omni_data['omni_data_rate'] = Avg_Fs
	omni_data['reward_index'] = omni_ind_reward
	omni_data['gocue_index'] = omni_ind_gocue
	mat_filename = tdt_filename +'_b'+str(block_num)+'_IndReward.mat'
	sp.io.savemat('C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_Device/Data/'+mat_filename,omni_data)
	"""
	if block_num==1:
		trialPowers_Spec,t, f = powersWithSpecgram(corrected_channel_data,Avg_Fs,channels,omni_ind_reward,1, 3)
		trialPowers_Spec_gocue, t, f = powersWithSpecgram(corrected_channel_data,Avg_Fs,channels,omni_ind_gocue,1, 3)
		time_to_reward = (omni_ind_reward- omni_ind_gocue)/Avg_Fs
	else:
		trialPowers_Spec_holder,t, f = powersWithSpecgram(corrected_channel_data,Avg_Fs,channels,omni_ind_reward,1, 3)
		trialPowers_Spec_holder_gocue,t, f = powersWithSpecgram(corrected_channel_data,Avg_Fs,channels,omni_ind_gocue,1, 3)
		trialPowers_Spec = np.vstack([trialPowers_Spec,trialPowers_Spec_holder])
		trialPowers_Spec_gocue = np.vstack([trialPowers_Spec_gocue, trialPowers_Spec_holder_gocue])
		time_to_reward = np.append(time_to_reward, (omni_ind_reward- omni_ind_gocue)/Avg_Fs)
	

max_power = np.argmax(trialPowers_Spec[:,7:8],axis = 1)
sort_max_power = np.argsort(max_power)
norm = np.abs(np.sum(trialPowers_Spec))

col,row = trialPowers_Spec.shape
sorted_ind = np.argsort(time_to_reward)

ticks_time_for_reward = time_to_reward[sorted_ind]

norm_per_trial = np.abs(np.sum(trialPowers_Spec,axis=1))
norm_per_trial = np.tile(norm_per_trial,(row,1)).T
#norm = 1
fig = plt.figure()
ax = plt.imshow(trialPowers_Spec[sort_max_power,:]/norm_per_trial, aspect='auto',extent = [-1,3,col,0])
plt.plot(np.zeros(col),range(0,col),'k')
fig.colorbar(ax)
plt.xlim((-1,3))
plt.ylim((0,col))
plt.title('power sorted - channel %i' % channels[0])
plt.show()

fig = plt.figure()
ax = plt.imshow(trialPowers_Spec[:,:]/norm_per_trial, aspect='auto',extent = [-1,3,col,0])
plt.plot(np.zeros(col),range(0,col),'k')
fig.colorbar(ax)
plt.xlim((-1,3))
plt.ylim((0,col))
plt.title('no sort - reward -  channel %i' % channels[0])
plt.show()

fig = plt.figure()
ax = plt.imshow(trialPowers_Spec_gocue[:,:]/norm, aspect='auto',extent = [-1,3,col,0])
plt.plot(np.zeros(col),range(0,col),'k')
fig.colorbar(ax)
plt.xlim((-1,3))
plt.ylim((0,col))
plt.title('no sort - gocue -  channel %i' % channels[0])
plt.show()

"""
fig = plt.figure()

norm_per_trial = np.abs(np.max(trialPowers_Spec_gocue,axis=1))
norm_per_trial = np.tile(norm_per_trial,(row,1)).T

plt.subplot(121)
ax = plt.imshow(trialPowers_Spec_gocue[sorted_ind,:]/norm, aspect='auto',extent = [-1,3,col,0])
plt.plot(np.zeros(col),range(0,col),'k')
plt.plot(ticks_time_for_reward,range(0,col),'b')
plt.plot(ticks_time_for_reward + 1,range(0,col),'b')
fig.colorbar(ax)
plt.title('sorted from time to reward - norm max - gocue -  channel %i' % channels[0])
plt.xlim((-1,3))
plt.ylim((0,col))
"""
norm_per_trial = np.abs(np.sum(trialPowers_Spec_gocue,axis=1))
norm_per_trial = np.tile(norm_per_trial,(row,1)).T
fig = plt.figure()
#plt.subplot(122)
ax = plt.imshow(trialPowers_Spec_gocue[sorted_ind,:]/norm_per_trial, aspect='auto',extent = [-1,3,col,0])
plt.plot(np.zeros(col),range(0,col),'k')
plt.plot(ticks_time_for_reward,range(0,col),'b')
plt.plot(ticks_time_for_reward + 1,range(0,col),'b')
fig.colorbar(ax)
plt.title('sorted from time to reward - norm per trial total - gocue -  channel %i' % channels[0])
plt.xlim((-1,3))
plt.ylim((0,col))
plt.show()
'''
look at omni_ind_reward - anything different about last trial?
look at window
'''