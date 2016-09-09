#from neo import io
from numpy import sin, linspace, pi
import matplotlib
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import cohere
from scipy import fft, arange, signal
from pylab import specgram
from scipy import signal

def LFPSpectrumSingleChannel(tankname,channel):
	"""
	Adopted from: http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
	"""
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	tank = tankname[-13:]  # extracts last 13 values, which should be LuigiYYYYMMDD
	block_num = 0
	for block in bl.segments:
		block_num += 1
		for analogsig in block.analogsignals:
			if analogsig.name[:4]=='LFP2':
				analogsig.channel_index +=96
			if (analogsig.name[:3]=='LFP')&(analogsig.channel_index==channel):
				Fs = analogsig.sampling_rate.item()
				data = analogsig
				num_timedom_samples = data.size
				time = [float(t)/Fs for t in range(0,num_timedom_samples)]
 				freq, Pxx_den = signal.welch(data, Fs, nperseg=1024)

 				plt.figure()
 				plt.subplot(2,1,1)
 				plt.plot(freq,Pxx_den/np.sum(Pxx_den),'r') # plotting the spectrum
 				plt.xlim((0, 100))
 				plt.xlabel('Freq (Hz)')
 				plt.ylabel('PSD')
 				plt.title('Channel ' +str(channel))
 				
 				plt.subplot(2,1,2)
				plt.plot(time[0:np.int(Fs)*10],data[0:np.int(Fs)*10],'r') # plotting LFP snippet
				plt.xlabel('Time (s)')
				plt.ylabel('LFP (uv)')
				plt.title('LFP Snippet')
 				plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/PowerSpec_'+tank+'_'+str(block_num)+'_Ch'+str(channel)+'.png')
 				plt.close()
 	return 


def LFPSpectrumAllChannel(tankname,num_channels):
	"""
	Adopted from: http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
	"""
	tank = tankname[-13:]  # extracts last 13 values, which should be LuigiYYYYMMDD
	block_num = 0
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	
	matplotlib.rcParams.update({'font.size': 6})
	for block in bl.segments:
		block_num += 1
		for analogsig in block.analogsignals:
			if analogsig.name[:4]=='LFP2':
				analogsig.channel_index +=96
			if (analogsig.name[:3]=='LFP'):
				Fs = analogsig.sampling_rate
				data = analogsig
				
 				freq, Pxx_den = signal.welch(data, Fs, nperseg=1024)
 				plt.figure(2*block_num-1)
 				if num_channels==96:
 					ax1 = plt.subplot(8,12,analogsig.channel_index)
 				else:
 					ax1 = plt.subplot(10,16,analogsig.channel_index)
 					
 				plt.plot(freq,Pxx_den/np.sum(Pxx_den),'r')
 				ax1.set_xlim([0, 40])
 				ax1.set_xticklabels([])
				ax1.set_ylim([0, 0.8])
				ax1.set_yticklabels([])
				plt.title(str(analogsig.channel_index))
 				
 				plt.figure(2*block_num)
 				if num_channels==96:
 					ax2 = plt.subplot(8,12,analogsig.channel_index)
 				else:
 					ax2 = plt.subplot(10,16,analogsig.channel_index)
				plt.semilogy(freq,Pxx_den,'r')
 				ax2.set_xlim([0, 40])
 				ax2.set_xticklabels([])
				#ax2.set_ylim([0, 1.0e-8])
				ax2.set_yticklabels([])
				plt.title(str(analogsig.channel_index))
 		plt.figure(1)
 		plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/NormalizedPowerSpec_'+tank+'_'+str(block_num)+'.png')
 		plt.close()
 		plt.figure(2)
 		plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/PowerSpec_'+tank+'_'+str(block_num)+'.png')
 		plt.close()
 	return 

def gen_spcgrm(tankname,channel,cutoffs=(0,250),binsize=50):
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	for analogsig in bl.segments[0].analogsignals:
		if analogsig.name[:4]=='LFP2':
			analogsig.channel_index +=96
		if (analogsig.name[:3]=='LFP')&(analogsig.channel_index==channel):
			data = analogsig
			srate = analogsig.sampling_rate
			spec,freqs,bins,im=specgram(data,Fs=srate,NFFT=binsize,noverlap=0)
	return 

def TrialAveragedPSD(lfp_data, chann, Fs, lfp_ind, samples_lfp, row_ind, stim_freq):
	'''
	Computes PSD per channel, with data averaged over trials. 
	'''
	density_length = 30
	
	trial_power = np.zeros([density_length,len(row_ind)])
	freq = np.zeros(257)
	
	for i in range(0,len(row_ind)):	
		lfp_snippet = lfp_data[chann][lfp_ind[i]:lfp_ind[i]+samples_lfp[i]]
		num_timedom_samples = lfp_snippet.size
		time = [float(t)/Fs for t in range(0,num_timedom_samples)]
 		freq, Pxx_den = signal.welch(lfp_snippet, Fs, nperseg=512, noverlap=256)
 		norm_freq = np.append(np.ravel(np.nonzero(np.less(freq,stim_freq-3))),np.ravel(np.nonzero(np.less(freq,stim_freq+3))))
 		total_power_Pxx_den = np.sum(Pxx_den[norm_freq])
 		Pxx_den = Pxx_den/total_power_Pxx_den
 		trial_power[:,i] = Pxx_den[0:density_length]

 	'''
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
	plt.title('Spectrogram: 50 - 100 Hz power')
	plt.axis('auto')
	# set the limits of the plot to the limits of the data
	#plt.axis([x.min(), x.max(), y.min(), y.max()])
	plt.colorbar()
	'''
	
	return freq, trial_power

def TrialAveragedPeakPower(lfp, Fs, lfp_ind, samples_lfp, freq_window, stim_freq):
	'''
	Computes PSD per channel, finds the peak power in the frequency window indicated, and then data is averaged over trials. 

	Input:
		- lfp: dictionary with one entry per channel of array of lfp samples
		- Fs: sample frequency in Hz
		- lfp_ind: sample index for the beginning of a trial
		- samples_lfp: the number of lfp samples per trial
		- stim_freq: frequency to notch out when normalizing spectral power
		- freq_window: frequency band over which to look for peak power, should be of form [f_low,f_high]
	Output:
		- trial_averaged_peak_power: an array of length equal to the number of channels, containing the trial-averaged peak power 
									of each channel in the designated frequency band

	'''
	channels = [int(item) for item in lfp.keys()]
	channels.sort()
	f_low = freq_window[0]
	f_high = freq_window[1]
	counter = 0
	peak_power = np.zeros([len(channels),len(lfp_ind)])
	density_length = 30

	for i in range(0,len(lfp_ind)):	
		for chann in channels:
			lfp_snippet = lfp[chann][lfp_ind[i]:lfp_ind[i]+samples_lfp[i]]
			num_timedom_samples = lfp_snippet.size
			freq, Pxx_den = signal.welch(lfp_snippet, Fs, nperseg=512, noverlap=256)
	 		norm_freq = np.append(np.ravel(np.nonzero(np.less(freq,stim_freq-3))),np.ravel(np.nonzero(np.less(freq,stim_freq+3))))
	 		total_power_Pxx_den = np.sum(Pxx_den[norm_freq])
	 		Pxx_den = Pxx_den/total_power_Pxx_den
	 		
	 		freq_band = np.less(freq,f_high)&np.greater(freq,f_low)
 			freq_band_ind = np.ravel(np.nonzero(freq_band))
 			peak_power[chann-1,i] = np.max(Pxx_den[freq_band_ind])

 	trial_averaged_peak_power = np.nanmean(peak_power,axis=1)

 	return trial_averaged_peak_power

def TrialAveragedPeakPowerDuringHold(lfp, Fs, lfp_ind, samples_lfp, hold_duration, freq_window, stim_freq):
	'''
	Computes PSD per channel, finds the peak power in the frequency window indicated, and then data is averaged over trials. 

	Input:
		- lfp: dictionary with one entry per channel of array of lfp samples
		- Fs: sample frequency in Hz
		- lfp_ind: sample index for the beginning of a trial
		- samples_lfp: the number of lfp samples per trial
		- hold_duration: the length of the hold period in seconds
		- stim_freq: frequency to notch out when normalizing spectral power
		- freq_window: frequency band over which to look for peak power, should be of form [f_low,f_high]
	Output:
		- trial_averaged_peak_power: an array of length equal to the number of channels, containing the trial-averaged peak power 
									of each channel in the designated frequency band

	'''
	channels = [int(item) for item in lfp.keys()]
	channels.sort()
	f_low = freq_window[0]
	f_high = freq_window[1]
	counter = 0
	peak_power = np.zeros([len(channels),len(lfp_ind)])
	density_length = 30
	hold = Fs*hold_duration 	# length of hold period in samples

	for i in range(0,len(lfp_ind)):	
		for chann in channels:
			lfp_snippet = lfp[chann][lfp_ind[i]:lfp_ind[i]+np.minimum(hold, samples_lfp[i])]
			num_timedom_samples = lfp_snippet.size
			freq, Pxx_den = signal.welch(lfp_snippet, Fs, nperseg=512, noverlap=256)
	 		norm_freq = np.append(np.ravel(np.nonzero(np.less(freq,stim_freq-3))),np.ravel(np.nonzero(np.less(freq,stim_freq+3))))
	 		total_power_Pxx_den = np.sum(Pxx_den[norm_freq])
	 		Pxx_den = Pxx_den/total_power_Pxx_den
	 		
	 		freq_band = np.less(freq,f_high)&np.greater(freq,f_low)
 			freq_band_ind = np.ravel(np.nonzero(freq_band))
 			peak_power[chann-1,i] = np.max(Pxx_den[freq_band_ind])

 	trial_averaged_peak_power = np.nanmean(peak_power,axis=1)

 	return trial_averaged_peak_power

def computePeakPowerPerChannel(lfp,Fs,stim_freq,t_start,t_end,freq_window):
	'''
	Input:
		- lfp: dictionary with one entry per channel of array of lfp samples
		- Fs: sample frequency in Hz
		- stim_freq: frequency to notch out when normalizing spectral power
		- t_start: time window start in units of sample number
		- t_end: time window end in units of sample number
		- freq_window: frequency band over which to look for peak power, should be of form [f_low,f_high]
	Output:
		- peak_power: an array of length equal to the number of channels, containing the peak power of each channel in 
					  the designated frequency band
	'''
	channels = lfp.keys()
	f_low = freq_window[0]
	f_high = freq_window[1]
	counter = 0
	peak_power = np.zeros(len(channels))
	
	for chann in channels:
		lfp_snippet = lfp[chann][t_start:t_end]
		num_timedom_samples = lfp_snippet.size
		freq, Pxx_den = signal.welch(lfp_snippet, Fs, nperseg=512, noverlap=256)
 		norm_freq = np.append(np.ravel(np.nonzero(np.less(freq,stim_freq-3))),np.ravel(np.nonzero(np.less(freq,stim_freq+3))))
 		total_power_Pxx_den = np.sum(Pxx_den[norm_freq])
 		Pxx_den = Pxx_den/total_power_Pxx_den

 		freq_band = np.less(freq,f_high)&np.greater(freq,f_low)
 		freq_band_ind = np.ravel(np.nonzero(freq_band))
 		peak_power[counter] = np.max(Pxx_den[freq_band_ind])
 		counter += 1

	return peak_power

def LFPPowerPerTrial_SingleBand_PerChannel(lfp,Fs,channels,window_start_times,t_before,t_after,freq_window):
	'''
	This method computes the power in a single band defined by power_band over sliding windows in the range 
	[window_start_times-time_before,window_start_times + time_after]. This is done per trial and then a plot of 
	power in the specified band is produce with time in the x-axis and trial in the y-axis. This is done per channel.

	Inputs:
		- lfp: lfp data arrange in an array of data x channel
		- channels: list of channels for which to apply this method
		- Fs: sampling rate of data in lfp_data array in Hz
		- window_start_times: window start times in units of sample numbers 
		- t_before: length of time in s to look at before the alignment times in window_start_times
		- t_after: length of time in s to look at after the alignment times 
		- freq_window: list defining the window of frequencies to look at, should be of the form power_band = [f_min,f_max]
	'''
	# Initialize parameters: convert to units of samples
	t_before_samp = np.floor(t_before*Fs)  	# convert to units of samples
	t_after_samp = np.floor(t_after*Fs)

	# Set up plotting variables
	# Set up matrix for plotting peak powers

	for chann in channels:
		trial_power = []
		for i, time in enumerate(window_start_times):
			Sxx,f,t = specgram(lfp[time - t_before_samp:time + t_after_samp,chann],Fs=Fs)
			f_band_ind = [ind for ind in range(0,len(f)) if (f[ind] <= freq_window[1])&(freq_window[0] <= f[ind])]
			f_band = np.sum(Sxx[f_band_ind,:],axis=0)
			trial_power.append(f_band)
		
		power_mat = np.zeros([len(window_start_times),len(trial_power[0])])
		for ind in range(0,len(window_start_times)):
			power_mat[i,:] = trial_power[i]

		dx, dy = float(t_before+t_after)/len(t), 1
		y, x = np.mgrid[slice(0,len(window_start_times),dy),
			slice(-t_before,t_after,dx)]
		cmap = plt.get_cmap('RdBu')
		plt.figure()
		plt.title('Channel %i - Power in band [%f,%f]' % (chann,freq_window[0],freq_window[1]))
		plt.pcolormesh(x,y,power_mat,cmap=cmap)
		plt.ylabel('Trial num')
		plt.xlabel('Time (s)')
		plt.axis([x.min(),x.max(),y.min(),y.max()])
		plt.show()

def LFPPowerPerTrial_SingleBand_PerChannel_Timestamps(lfp,timestamps,Avg_Fs,channels,t_start,t_before,t_after,freq_window):
	'''
	This method computes the power in a single band defined by power_band over sliding windows in the range 
	[window_start_times-time_before,window_start_times + time_after]. This is done per trial and then a plot of 
	power in the specified band is produce with time in the x-axis and trial in the y-axis. This is done per channel.

	Main difference with LFPPowerPerTrial_SingleBand_PerChannel is that t_start is in seconds and we don't assume
	a fixed time between samples.

	Inputs:
		- lfp: lfp data arrange in an array of data x channel
		- timestamps: time stamps for lfp samples, which may occur at irregular intervals
		- channels: list of channels for which to apply this method
		- Fs: sampling rate of data in lfp_data array in Hz
		- t_start: window start times in units of s 
		- t_before: length of time in s to look at before the alignment times in window_start_times
		- t_after: length of time in s to look at after the alignment times 
		- freq_window: list defining the window of frequencies to look at, should be of the form power_band = [f_min,f_max]
	'''

	# Set up plotting variables
	# Set up matrix for plotting peak powers

	for chann in channels:
		trial_power = []
		trial_times = []
		for i, time in enumerate(t_start):
			lfp_snippet = []
			lfp_snippet = [lfp[ind,chann] for ind in range(0,len(timestamps)) if (timestamps[ind] <= time + t_after)&(time - t_before <= timestamps[ind])]
			lfp_snippet = np.array(lfp_snippet)
			
			Sxx,f,t, fig = specgram(lfp_snippet,NFFT = 128,Fs=Avg_Fs,noverlap=56,scale_by_freq=False)
			f_band_ind = [ind for ind in range(0,len(f)) if (f[ind] <= freq_window[1])&(freq_window[0] <= f[ind])]
			f_band = np.sum(Sxx[f_band_ind,:],axis=0)
			f_band_norm = f_band/np.sum(f_band)
			trial_power.append(f_band_norm)
			trial_times.append(t)
		
		power_mat = np.zeros([len(t_start),len(trial_power[0])])
		
		for ind in range(0,len(t_start)):
			if len(trial_power[ind]) == len(trial_power[0]):
				power_mat[ind,:] = trial_power[ind]
			else:
				power_mat[ind,:] = np.append(trial_power[ind],np.zeros(len(trial_power[0])-len(trial_power[ind])))
		
		dx, dy = float(t_before+t_after)/len(t), 1
		y, x = np.mgrid[slice(0,len(t_start),dy),
			slice(-t_before,t_after,dx)]
		cmap = plt.get_cmap('RdBu')
		plt.figure()
		plt.title('Channel %i - Power in band [%f,%f]' % (chann,freq_window[0],freq_window[1]))
		plt.pcolormesh(x,y,power_mat,cmap=cmap)
		plt.ylabel('Trial num')
		plt.xlabel('Time (s)')
		plt.axis([x.min(),x.max(),y.min(),y.max()])
		plt.show()
		
	return Sxx, f, t

def powersWithSpecgram(channel_data,Avg_Fs,channel,event_indices,t_before, t_after):

	win_before = int(t_before*Avg_Fs)
	win_after = int(t_after*Avg_Fs)
	channel = np.array(channel) - 1 	# adjust so that counting starts at 0

	times = np.arange(-t_before,t_after,float(t_after + t_before)/(win_after + win_before))
	spec = dict()

	for j,ind in enumerate(event_indices):
		data = channel_data[ind - win_before:ind + win_after,channel]
		data = np.ravel(data)
		Sxx, f, t, fig = specgram(data,Fs=Avg_Fs)
		Sxx = Sxx/np.sum(Sxx)
		Sxx = 10*np.log10(Sxx)
		spec[str(j)] = Sxx

	return spec, t, f

def computePowerFeatures(lfp_data, Fs, power_bands, event_indices, t_window):
	'''
	Inputs
		- data: dictionary of data, with one entry for each channel 
		- Fs: sampling frequency 
		- power_bands: list of power bands 
		- event_indices: N x M array of event indices, where N is the number of trials and M is the number of 
		                 different events 
		- t_window: length M array of time window (in seconds) to compute features over, one element for each feature 
	Outputs
		- features: dictionary with N entries (one per trial), with a C x K matric which C is the number of channels 
					and K is the number of features (number of power bands times M)
	'''
	NFFT = int(Fs*0.25)
	noverlap = int(Fs*0.125)
	t_window = [int(Fs*time) for time in t_window]  # changing seconds into samples

	N, M = event_indices.shape
	times = np.ones([N,M])
	for t,time in enumerate(t_window):
		times[:,t] = time*np.ones(N)

	features = dict()

	channels = lfp_data.keys()

	for trial in range(0,N):
		events = event_indices[trial,:]  # should be array of length M
		trial_powers = np.zeros([len(channels),M*len(power_bands)])
		for j, chann in enumerate(channels):
			chann_data = lfp_data[chann]
			feat_counter = 0
			for i,ind in enumerate(events):
				data = chann_data[ind:ind + times[trial,i]]
				data = np.ravel(data)
				Sxx, f, t, fig = specgram(data, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
				Sxx = Sxx/np.sum(Sxx)
				Sxx = 10*np.log10(Sxx)
				for k in range(0,len(power_bands)):
					low_band, high_band = power_bands[k]
					freqs = np.ravel(np.nonzero(np.greater(f,low_band)&np.less_equal(f,high_band)))
					tot_power_band = np.sum(Sxx[freqs,:],axis=0)
					trial_powers[j,feat_counter] = np.sum(tot_power_band)/float(len(tot_power_band))
					#trial_powers[j,i*len(power_bands) + k] = np.sum(tot_power_band)/float(len(tot_power_band))
					feat_counter += 1
		features[str(trial)] = trial_powers

	return features

def computeCoherenceFeatures(lfp_data, channel_pairs, Fs, power_bands, event_indices, t_window):
	'''
	Inputs
		- lfp_data: dictionary of data, with one entry for each channel
		- channel_pairs: list of channel pairs
		- Fs: sampling frequency 
		- power_bands: list of power bands 
		- event_indices: N x M array of event indices, where N is the number of trials and M is the number of 
		                 different events 
		- t_window: length M array of time window (in seconds) to compute features over, one element for each feature 
	Outputs
		- features: dictionary with N entries (one per trial), with a C x K matric which C is the number of channel pairs 
					and K is the number of features (number of power bands times M)
	'''
	nperseg = int(Fs*0.25)
	noverlap = int(Fs*0.125)
	t_window = [int(Fs*time) for time in t_window]  # changing seconds into samples

	N, M = event_indices.shape
	times = np.ones([N,M])
	for t,time in enumerate(t_window):
		times[:,t] = time*np.ones(N)

	features = dict()

	channels = lfp_data.keys()
	num_channel_pairs = len(channel_pairs)

	for trial in range(0,N):
		events = event_indices[trial,:]  # should be array of length M
		trial_powers = np.zeros([len(channels),M*len(power_bands)])
		for j, pair in enumerate(channel_pairs):
			chann1 = pair[0] - 1
			chann2 = pair[1] - 1
			chann_data1 = lfp_data[chann1]
			chann_data2 = lfp_data[chann2]
			feat_counter = 0
			for i,ind in enumerate(events):
				data1 = chann_data1[ind:ind + times[trial,i]]
				data1 = np.ravel(data1)
				data2 = chann_data2[ind:ind + times[trial,i]]
				data2 = np.ravel(data2)
				Cxy, f = cohere(data1, data2, NFFT = nperseg, Fs=Fs, noverlap=noverlap)
				Cxy = Cxy/np.sum(Cxy)
				Cxy = 10*np.log10(Cxy)
				for k in range(0,len(power_bands)):
					low_band, high_band = power_bands[k]
					freqs = np.ravel(np.nonzero(np.greater(f,low_band)&np.less_equal(f,high_band)))
					tot_power_band = np.sum(Cxy[freqs])
					trial_powers[j,feat_counter] = tot_power_band
					#trial_powers[j,i*len(power_bands) + k] = np.sum(tot_power_band)/float(len(tot_power_band))
					feat_counter += 1
		features[str(trial)] = trial_powers

	return features

def computeAllCoherenceFeatures(lfp_data, Fs, power_bands, event_indices, t_window):
	'''
	Generate the list of all possible channel combinations and uses the computeCoherenceFeatures method
	to extract all of the coherence features.

	Inputs
		- lfp_data: dictionary of data, with one entry for each channel
		- Fs: sampling frequency 
		- power_bands: list of power bands 
		- event_indices: N x M array of event indices, where N is the number of trials and M is the number of 
		                 different events 
		- t_window: length M array of time window (in seconds) to compute features over, one element for each feature 
	Outputs
		- features: dictionary with N entries (one per trial), with a C x K matric which C is the number of channel pairs 
					and K is the number of features (number of power bands times M)
	'''
	channels = [int(item) for item in lfp.keys()]
	channels.sort()
	channel_pairs = []
	for i, chan1 in enumerate(channels[:-1]):
		for chan2 in channels[i+1:]:
			channel_pairs.append([chan1, chan2])

	features = computeCoherenceFeatures(lfp_data, channel_pairs, Fs, power_bands, event_indices, t_window)
	return features