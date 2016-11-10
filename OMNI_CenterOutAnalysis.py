import numpy as np
import scipy as sp
from scipy import signal
from scipy.ndimage import filters
import re
#from neo import io
from scipy import io
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
from OMNI_methods import convert_OMNI, get_stim_sync_sig, computePowersWithChirplets, powersWithFFT, powersWithSpecgram
from spectralAnalysis import LFPPowerPerTrial_SingleBand_PerChannel_Timestamps
import tables
import os.path
from scipy.interpolate import spline
from spectrogram.spectrogram_methods import make_spectrogram
from neo import io
from basicAnalysis import notchFilterData


"""
Also do high-gamma: should see high-gamma increase during movement

"""

blocks = [1,2,3]
channel = 86  		# picking subset of channels to analyze
fmax = 200.

filename_prefix = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_Device/Data/'

print 'Loading TDT data'
tdt_filename = 'Mario20161103'
r = io.TdtIO(filename_prefix + tdt_filename)
bl = r.read_block(lazy=False,cascade=True)
print "File read."
tdt_lfp = dict()
for sig in bl.segments[0].analogsignals:
	if (sig.name[0:4] == 'LFP1'):
		chann = sig.channel_index
		lfp_samprate = sig.sampling_rate.item()
		tdt_lfp[chann] = np.ravel(sig)
	if (sig.name[0:4] == 'LFP2'):
		chann = sig.channel_index + 96
		tdt_lfp[chann] = np.ravel(sig)


tdt_filename = 'Mario20160720-OMNI'

mat_filename1 = filename_prefix + tdt_filename + '_b1.mat'
mat_filename2 = filename_prefix + tdt_filename + '_b2.mat'
mat_filename3 = filename_prefix + tdt_filename + '_b3.mat'

mat_files = [mat_filename1, mat_filename2, mat_filename3]
#mat_files = [mat_filename1]
print "Loading .mat files with data"

# Get all power data
for i in range(len(mat_files)):
	omnib1 = dict()
	mat_file = mat_files[i]
	print "Loading data for Block %i." % (i+1)
	sp.io.loadmat(mat_file,omnib1)

	data = omnib1['corrected_data']
	fs = float(omnib1['omni_data_rate'])
	omni_ind_reward = np.ravel(omnib1['reward_index'])
	omni_ind_gocue = np.ravel(omnib1['gocue_index'])

	num_time_samps = 8*fs   # adding padding
	num_begin_pad = 2*fs
	num_end_pad = 2*fs
	lfp = np.zeros([len(omni_ind_gocue), num_time_samps])

	for j, ind in enumerate(omni_ind_gocue):
		lfp[j,:] = data[ind - 3*fs:ind + 5*fs, channel-1]
	"""
	if i == 0:
		print "Computing powers for Block 1."
		powers, Power, cf_list = make_spectrogram(lfp, fs, fmax, 1, 0, num_begin_pad, num_end_pad)
		time_to_reward = omni_ind_reward- omni_ind_gocue  # in samples with rate 944 Hz
		all_powers = powers
		all_times_to_reward = time_to_reward
	else:
		print "Computing powers for Block %i." % (i+1)
		powers, Power, cf_list = make_spectrogram(lfp, fs, fmax, 1, 0, num_begin_pad, num_end_pad)
		time_to_reward = omni_ind_reward- omni_ind_gocue  # in samples with rate 944 Hz
		all_powers = np.vstack([all_powers,powers])
		all_times_to_reward = np.append(all_times_to_reward, time_to_reward)
	"""
'''
# Compute average low (19.51 - 25.75 Hz) and high (33.47 - 43.59 Hz) beta. When fmax = 100, these are the appropriate cf_list indices
avg_low_beta_per_trial = np.nanmean(all_powers[:,20:26,2*fs:-2*fs], axis = 1)
avg_high_beta_per_trial = np.nanmean(all_powers[:,30:36,2*fs:-2*fs], axis = 1)
avg_all_beta_per_trial = np.nanmean(all_powers[:,20:36,2*fs:-2*fs], axis = 1)
'''
"""
# Compute average low (19.51 - 25.75 Hz) and high (33.47 - 43.59 Hz) beta 
avg_low_beta_per_trial = np.nanmean(all_powers[:,18:23,2*fs:-2*fs], axis = 1)
avg_high_beta_per_trial = np.nanmean(all_powers[:,26:31,2*fs:-2*fs], axis = 1)
avg_all_beta_per_trial = np.nanmean(all_powers[:,18:31,2*fs:-2*fs], axis = 1)
avg_high_gamma_per_trial = np.nanmean(all_powers[:,37:, 2*fs:-2*fs], axis = 1)

num_trials, num_samples = avg_low_beta_per_trial.shape

# Compute trial-averaged powers in high and low beta bands
trial_avg_low_beta = np.nanmean(avg_low_beta_per_trial,axis = 0)
trial_avg_high_beta = np.nanmean(avg_high_beta_per_trial,axis = 0)
trial_avg_all_beta = np.nanmean(avg_all_beta_per_trial,axis = 0)
trial_avg_high_gamma = np.nanmean(avg_high_gamma_per_trial, axis = 0)

x = np.linspace(-1,3,len(trial_avg_low_beta))
#xnew = np.linspace(-1,3,100)
#low_beta_smooth = spline(x,trial_avg_low_beta,xnew)
#high_beta_smooth = spline(x,trial_avg_high_beta,xnew)
#all_beta_smooth = spline(x,trial_avg_all_beta,xnew)
#high_gamma_smooth = spline(x,trial_avg_high_gamma,xnew)

# Gaussian smoothing
b = signal.gaussian(39,3)
low_beta_smooth = filters.convolve1d(trial_avg_low_beta, b/b.sum())
high_beta_smooth = filters.convolve1d(trial_avg_high_beta, b/b.sum())
high_gamma_smooth = filters.convolve1d(trial_avg_high_gamma, b/b.sum())
all_beta_smooth = filters.convolve1d(trial_avg_all_beta, b/b.sum())
xnew = np.linspace(-1, 3, len(low_beta_smooth))

adjusted_times_to_reward = np.array([int(fs + all_times_to_reward[i]) for i in range(num_trials)])  # gives sample number in x array

# convert times from samples to secs
adjusted_times_to_reward = (1./fs)*adjusted_times_to_reward - 1
adjusted_times_to_target_hold = adjusted_times_to_reward - 0.5 			# target hold begins 0.5 s before reward
adjusted_time_following_reward = adjusted_times_to_reward + 1 			# reward duration is 1 s following reward commencement

# Get rid of outlier trials
min_high_gamma = np.min(avg_high_gamma_per_trial,axis = 1)
bad_trials = np.ravel(np.nonzero(np.less(min_high_gamma, -39)))
sort_ind = np.argsort(all_times_to_reward)
sort_ind = np.array([ind for ind in sort_ind if ind not in bad_trials])
num_trials = len(sort_ind)

# Make plot for low-beta

fig = plt.figure()
ax1 = plt.subplot(211)
#plt.plot(x,trial_avg_low_beta,'b')
plt.plot(xnew,low_beta_smooth,'k')
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.title('Trial-averaged Low Beta: n = %f' % (num_trials))
ax1 = plt.subplot(212)
ax = plt.imshow(avg_low_beta_per_trial[sort_ind,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [-1,int((num_samples)/fs)-1,0, num_trials])
plt.plot(np.zeros(num_trials),range(num_trials),'k')								# Go cue line (x = 0)
plt.plot(adjusted_times_to_target_hold[sort_ind],range(num_trials),'k')			# Peripheral hold begin
plt.plot(adjusted_times_to_reward[sort_ind],range(num_trials),'k--')				# Reward begin following peripheral hold
plt.plot(adjusted_time_following_reward[sort_ind],range(num_trials),'k--')		# Reward end 
#yticks = np.arange(0, len(cf_list), 5)
#yticks = np.append(yticks,len(cf_list)-1)
#yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#plt.yticks(yticks, yticklabels)
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.xlim((-1,int((num_samples)/fs)-1))
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Average Low Beta Power: 19 - 26 Hz')
fig.colorbar(ax)
plt.show()

# Make plot for high-beta

fig = plt.figure()
ax1 = plt.subplot(211)
#plt.plot(trial_avg_high_beta)
plt.plot(xnew,high_beta_smooth,'k')
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.title('Trial-averaged High Beta: n = %f' % (num_trials))
ax1 = plt.subplot(212)
ax = plt.imshow(avg_high_beta_per_trial[sort_ind,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0-1,int((num_samples)/fs)-1,0, num_trials])
plt.plot(np.zeros(num_trials),range(num_trials),'k')								# Go cue line (x = 0)
plt.plot(adjusted_times_to_target_hold[sort_ind],range(num_trials),'k')			# Peripheral hold begin
plt.plot(adjusted_times_to_reward[sort_ind],range(num_trials),'k--')				# Reward begin following peripheral hold
plt.plot(adjusted_time_following_reward[sort_ind],range(num_trials),'k--')		# Reward end 
#yticks = np.arange(0, len(cf_list), 5)
#yticks = np.append(yticks,len(cf_list)-1)
#yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#plt.yticks(yticks, yticklabels)
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.xlim((-1,int((num_samples)/fs)-1))
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Average High Beta Power: 32 - 45 Hz')
fig.colorbar(ax)
plt.show()

fig = plt.figure()
ax1 = plt.subplot(211)
#plt.plot(trial_avg_high_beta)
plt.plot(xnew,all_beta_smooth,'k')
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.title('Trial-averaged All Beta: n = %f' % (num_trials))
ax1 = plt.subplot(212)
ax = plt.imshow(avg_all_beta_per_trial[sort_ind,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0-1,int((num_samples)/fs)-1,0, num_trials])
plt.plot(np.zeros(num_trials),range(num_trials),'k')								# Go cue line (x = 0)
plt.plot(adjusted_times_to_target_hold[sort_ind],range(num_trials),'k')			# Peripheral hold begin
plt.plot(adjusted_times_to_reward[sort_ind],range(num_trials),'k--')				# Reward begin following peripheral hold
plt.plot(adjusted_time_following_reward[sort_ind],range(num_trials),'k--')		# Reward end 
#yticks = np.arange(0, len(cf_list), 5)
#yticks = np.append(yticks,len(cf_list)-1)
#yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#plt.yticks(yticks, yticklabels)
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.xlim((-1,int((num_samples)/fs)-1))
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Average Beta Power: 19 - 45 Hz')
fig.colorbar(ax)
plt.show()

fig = plt.figure()
ax1 = plt.subplot(211)	
#plt.plot(trial_avg_high_beta)
plt.plot(xnew,high_gamma_smooth,'k')
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.title('Trial-averaged High Gamma: n = %f' % (num_trials))
ax1 = plt.subplot(212)
ax = plt.imshow(avg_high_gamma_per_trial[sort_ind,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0-1,int((num_samples)/fs)-1,0, num_trials])
plt.plot(np.zeros(num_trials),range(num_trials),'k')								# Go cue line (x = 0)
plt.plot(adjusted_times_to_target_hold[sort_ind],range(num_trials),'k')			# Peripheral hold begin
plt.plot(adjusted_times_to_reward[sort_ind],range(num_trials),'k--')				# Reward begin following peripheral hold
plt.plot(adjusted_time_following_reward[sort_ind],range(num_trials),'k--')		# Reward end 
#yticks = np.arange(0, len(cf_list), 5)
#yticks = np.append(yticks,len(cf_list)-1)
#yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#plt.yticks(yticks, yticklabels)
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.xlim((-1,int((num_samples)/fs)-1))
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Average High Gamma Power: 70 - 200 Hz')
fig.colorbar(ax)
plt.show()
"""

'''
Compute PSDs 

'''
# NOTE: Need to add in TDT recording data. Need to add in conversion from OMNI mapping to TDT mapping
# data variable already has time samples by channel array

tdt_channels = [95, 79, 93, 77, 91, 75, 89, 73, 96, 80, 94, 78, 92, #13
				76, 90, 74, 88, 72, 86, 70, 84, 68, 82, 66, 87, 71, #13
				85, 69, 83, 67, 81, 65, 127, 111, 125, 109, 123, 107, #12
				121, 105, 128, 112, 126, 110, 124, 108, 122, 106, 120,  #11
				104, 118, 102, 116, 100, 114, 98, 119, 103, 117, 101, # 11
				115, 99, 113, 97, 159, 143, 157, 141, 155, 139, 153,  # 11
				137, 160, 144, 158, 142, 156, 140, 154, 138, 152, 136, # 11
				150, 134, 148, 132, 146, 130, 151, 135, 149, 133, 147] # 11

print "TDT data loaded. Beginning computation of PSDs"
#channels = range(93)
#channels = range(72, 93)
channels = np.array([16, 22, 23, 6, 38, 39, 26, 62, 65, 70, 50, 58, 87, 78, 80, 81])
for i, chan in enumerate(channels):
	lfp_snippet = data[5*60*fs:10*60*fs, chan]
	tdt_lfp_snippet = tdt_lfp[tdt_channels[chan]][5*60*lfp_samprate:10*60*lfp_samprate]
	lfp_snippet_notch = notchFilterData(lfp_snippet, fs, 60)
	tdt_lfp_snippet_notch = notchFilterData(tdt_lfp_snippet, lfp_samprate, 60)
	freq, Pxx_den = signal.welch(lfp_snippet, fs, nperseg=512, noverlap=256)
	#freq, Pxx_den_notch = signal.welch(lfp_snippet_notch, fs, nperseg=512, noverlap = 256)
	freq_tdt, Pxx_den_tdt = signal.welch(tdt_lfp_snippet, lfp_samprate, nperseg = 512*2, noverlap = 256*2)
	#freq_tdt, Pxx_den_tdt_notch = signal.welch(tdt_lfp_snippet_notch, lfp_samprate, nperseg=512*2, noverlap = 256*2)
	
	# smooth tdt PSD
	x = freq_tdt[:35]
	xnew = np.linspace(0,100,100)
	Pxx_den_tdt_smooth = spline(x,Pxx_den_tdt[:35]/np.sum(Pxx_den_tdt),xnew)

	# smooth omni PSD
	x = freq[:57]
	Pxx_den_smooth = spline(x,Pxx_den[:57]/np.sum(Pxx_den),xnew)

	fig_num = i /16 + 1
	plt.figure(fig_num)
	ax = plt.subplot(4,4,(i % 16) + 1)
	ax.semilogy(freq,Pxx_den/np.sum(Pxx_den),color = 'b', label = 'OMNI')
	#ax.semilogy(freq,Pxx_den_notch/np.sum(Pxx_den_notch),color='c', label= 'OMNI notch')
	ax.semilogy(xnew,Pxx_den_smooth, color = 'c', label = 'OMNI smooth')
	ax.semilogy(freq_tdt,Pxx_den_tdt/np.sum(Pxx_den_tdt), color = 'r', label = 'TDT')
	#ax.semilogy(freq_tdt,Pxx_den_tdt_notch/np.sum(Pxx_den_tdt_notch), color = 'm', label = 'TDT notch')
	ax.semilogy(xnew, Pxx_den_tdt_smooth, color = 'm', label = 'TDT smooth')
	plt.title('Channel %i' % (chan))
	plt.legend()
	plt.ylim((10**-4, 10**0))
	plt.xlim((0,100))

plt.show()


# normalize TDT 0 - 100 Hz power (:35): small handful of channels match 
# normalize TDT 0 - 200 Hz power (:68): ch 16, 22, 23, 6, 38, 39, 26, 62, 65, 70, 50, 58
# normalize TDT all power: 16, 22, 23, 6
# should notch filter for 60 Hz noise
