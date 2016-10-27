import numpy as np
import scipy as sp
from scipy import signal
import re
#from neo import io
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
import tables
from pylab import specgram




def plotRawLFPTraces(data, **kwargs):
	'''
	This method plots the raw LFP data for all channels or a subset of channels in a single plot 
	for easy viewing. Data is normalized to have zero DC component. Spacing between traces is determined by 
	the max standard deviation across all channels.

	Input:
		- data: numpy array, as generated by the convert_OMNI method
	Optional input:
		- channs: list of channels to be plotted
		- filter_data: True/False if low-pass filter is applied to data, default is False
	'''
	num_channs = data.shape[1]
	 		
	if kwargs:
		channs = kwargs['channs']
	else:
		channs = range(1,num_channs+1,1) 	# recall one column is for timestamps, not channel data

	channs = [(chann - 1) for chann in channs]  # channel numbers are offset by 1 from index values

	'''
	if filter_data:
		cutoff_f = 100
		cutoff_f = cutoff_f/(1000./2)  # sampling rate is 1000 Hz
		num_taps = 100
		lpf = signal.firwin(num_taps,cutoff_f,window='hamming')
		for chann in channs:
			data[:][chann] = signal.lfilter(lpf,1,data[:][chann])
	'''

	mean_vec = np.mean(data[:,channs], axis = 0)
	std_vec = np.std(data[:,channs], axis = 0)

	for i, chann in enumerate(channs):
		test_above_thres = np.ravel(np.nonzero(np.greater(data[:,chann],mean_vec[i] + 8*std_vec[i])))
		test_below_thres = np.ravel(np.nonzero(np.less(data[:,chann],mean_vec[i] - 4*std_vec[i])))
		ind_to_be_corrected = np.append(test_above_thres, test_below_thres)
		ind_to_be_corrected = [ind for ind in ind_to_be_corrected]
		for ind in ind_to_be_corrected:
			if ind + 1 < data.shape[0]:
				data[ind,chann] = (data[ind-1, chann] + data[ind+1,chann])/2.
			else:
				data[ind,chann] = data[ind-1, chann]

	mean_vec = np.mean(data[:,channs], axis = 0)
	std_vec = np.std(data[:,channs], axis = 0)
	trace_dist = 0.25*np.max([std_vec])  # don't include std of time stamps when 

	times = data[:,num_channs-1]

	plt.figure()
	cmap = mpl.cm.brg
	for i, chann in enumerate(channs):
		plt.plot(times[:5000],data[:5000,chann] - mean_vec[i] + i*trace_dist,color=cmap(i/float(len(channs))), label=str(chann))
	plt.xlabel('Time (s)')
	plt.title('LFP Traces')
	plt.legend()

	plt.show()

	return

def get_stim_sync_sig(tdt_tank):
	r = io.TdtIO(dirname = tdt_tank)
	bl = r.read_block(lazy=False,cascade=True)

	for sig in bl.segments[0].analogsignals:
		if (sig.name == 'StWv 1'):
			# Signal is output of stimulator monitor recording voltage on channel stimulation is delivered on
			stim_monitor = np.ravel(sig)
		if (sig.name == 'StWv 2'):
			# Stimulation signal used: this is the base stimulation signal that is turned on/off
			stim_signal = np.ravel(sig)
		if (sig.name == 'StWv 4'):
			# Trigger signal for stimulation. High is ON, Low is OFF.
			stim_on_trig = np.ravel(sig)
		if (sig.name == 'StWv 3'):
			# Control signal for stimulator. When stim_on_trig = 1, this signal equals stim_signal
			stim_delivered = np.ravel(sig)
			stwv_samprate = sig.sampling_rate.item()

	return stim_signal, stim_on_trig, stim_delivered, stwv_samprate


def test_convert_OMNI(data, **kwargs):
	
	#data = pd.read_csv(filename,sep=',',header=None,skiprows=[0,1])
	data = np.array(data)
	time_samps, num_col = data.shape
	crc_flag = np.array(data[:,0])
	ind_crc_pass = [ind for ind in range(0,len(crc_flag)) if crc_flag[ind]==170]

	channel_data = np.zeros([len(ind_crc_pass),num_col-3])  # 3 fewer columns since one is crv flag, one is ramp, and one is time samples
	
	for col in range(0,num_col-3):
		channel_data[:,col] = data[ind_crc_pass,col+1]
	timestamps = data[ind_crc_pass,num_col-1]
	counter_ramp = data[ind_crc_pass,num_col-2]

	corrected_channel_data = channel_data[0,:]
	corrected_counter = [counter_ramp[0]]
	num_cycle = 0
	samps_ind = []

	for i in range(1,len(counter_ramp)):
	#for i in range(1,17000):
		diff = counter_ramp[i] - counter_ramp[i-1]
		diff = int(diff)
		if (i % 1000 == 0):
			print i, diff
		elif (diff > 1):
			print i, diff
		if (diff==1):
			corrected_counter.append(counter_ramp[i] + num_cycle*(2**16))
			#corrected_channel_data = np.vstack([corrected_channel_data,channel_data[i,:]])
		elif (diff == -2**16 +1):
			num_cycle += 1
			corrected_counter.append(counter_ramp[i] + num_cycle*(2**16))
			#corrected_channel_data = np.vstack([corrected_channel_data,channel_data[i,:]])
		else:
			samps_ind.append(i)
			num_samples_insert = diff - 1.
			corrected_counter.extend((counter_ramp[i-1] + range(1,diff+1) + num_cycle*(2**16)).tolist())
			#inter_mat = np.zeros([num_samples_insert+1,num_col-3])
			#for j in range(0,num_col-3):
			#	y = np.interp(range(1,diff),[0, diff], [channel_data[i-1,j], channel_data[i,j]])
			#	y = np.append(y,channel_data[i,j])
			#	inter_mat[:,j] = y
			#corrected_channel_data = np.vstack([corrected_channel_data,inter_mat])
		

	return corrected_counter, corrected_channel_data, samps_ind

def convert_OMNI(data, **kwargs):
	'''
	This method converts csv files saved using the OMNI device to a pandas DataFrame for easy
	analysis in Python.

	Input:
		- filename: string containing the file path for a csv file saved with the OMNI device
	
		
	Output:
		- data: pandas DataFrame, M rows x N columns, M = number of data points, N = number of channels + 1, 
				first N -1 columns corresponds to data from the differnt channels while the Nth column 
				contains the timestamps 

	'''
	#data = pd.read_csv(filename,sep=',',header=None,skiprows=[0,1])
	data = np.array(data)
	time_samps, num_col = data.shape
	crc_flag = np.array(data[:,0])
	ind_crc_pass = [ind for ind in range(0,len(crc_flag)) if crc_flag[ind]==170]

	channel_data = np.zeros([len(ind_crc_pass),num_col-3])  # 3 fewer columns since one is crv flag, one is ramp, and one is time samples
	
	for col in range(0,num_col-3):
		channel_data[:,col] = data[ind_crc_pass,col+1]
	timestamps = data[ind_crc_pass,num_col-1]
	counter_ramp = data[ind_crc_pass,num_col-2]

	corrected_counter = [counter_ramp[0]]
	num_cycle = 0

	# Counter is 16 bit and resets at 2**16. This loop unwraps this cycling so that values are monotonically increasing.
	for i in range(1,len(counter_ramp)):
	#for i in range(1,17000):
		diff = counter_ramp[i] - counter_ramp[i-1]
		diff = int(diff)
		
		if (diff == -2**16 +1):
			num_cycle += 1
			corrected_counter.append(counter_ramp[i] + num_cycle*(2**16))
		else:
			corrected_counter.append(counter_ramp[i] + num_cycle*(2**16))
	
	corrected_counter = np.array(corrected_counter)
	diff_corrected_counter = corrected_counter[1:] - corrected_counter[:-1]
	miss_samp_index = np.ravel(np.nonzero(np.greater(diff_corrected_counter,1)))

	corrected_channel_data = channel_data[0:miss_samp_index[0]+1,:]
	diff = corrected_counter[miss_samp_index[0]+1] - corrected_counter[miss_samp_index[0]]
	diff = int(diff)
	inter_mat = np.zeros([diff,num_col-3])
	for j in range(0,num_col-3):
		y = np.interp(range(1,diff),[0, diff], [channel_data[miss_samp_index[0],j], channel_data[miss_samp_index[0]+1,j]])
		y = np.append(y,channel_data[miss_samp_index[0]+1,j])
		inter_mat[:,j] = y
	corrected_channel_data = np.vstack([corrected_channel_data, inter_mat])
	
	print "There are %i instances of missed samples." % len(miss_samp_index)
	print "Beginning looping through regeneration of data"
	for i in range(1,len(miss_samp_index)):
		print i
		# pad with good data first
		corrected_channel_data = np.vstack([corrected_channel_data, channel_data[miss_samp_index[i-1] + 2:miss_samp_index[i]+1,:]])
		# check number of samples that were skipped and need to be regenerated
		diff = corrected_counter[miss_samp_index[i]+1] - corrected_counter[miss_samp_index[i]]
		diff = int(diff)
		inter_mat = np.zeros([diff,num_col-3])
		# interpolate values to regenerate missing data
		for j in range(0,num_col-3):
			y = np.interp(range(1,diff),[0, diff], [channel_data[miss_samp_index[i],j], channel_data[miss_samp_index[i]+1,j]])
			y = np.append(y,channel_data[miss_samp_index[i]+1,j])
			inter_mat[:,j] = y
		corrected_channel_data = np.vstack([corrected_channel_data,inter_mat])

	if (miss_samp_index[-1]+1 != len(ind_crc_pass)-1):
		print "adding last zeros"
		corrected_channel_data = np.vstack([corrected_channel_data, channel_data[miss_samp_index[-1] + 2:,:]])
		

	return corrected_counter, corrected_channel_data, channel_data, miss_samp_index

def convert_OMNI_from_hdf(hdf_filename, **kwargs):
	'''
	This method converts csv files saved using the OMNI device to a pandas DataFrame for easy
	analysis in Python.

	Input:
		- filename: string containing the file path for a csv file saved with the OMNI device
	
		
	Output:
		- data: pandas DataFrame, M rows x N columns, M = number of data points, N = number of channels + 1, 
				first N -1 columns corresponds to data from the differnt channels while the Nth column 
				contains the timestamps 

	'''
	hdf = tables.openFile(hdf_filename)
	print "Loading data."
	data = hdf.root.dataGroup.dataTable[:]['out']
	#time_stamps = hdf.root.dataGroup.dataTable[:]['time']
	print "Loaded data."
	time_samps, num_col = data.shape
	crc_flag = np.array(data[:,0])
	ind_crc_pass = [ind for ind in range(0,len(crc_flag)) if crc_flag[ind]==0]
	print "Found which inds pass CRC"
	#channel_data = np.zeros([len(ind_crc_pass),num_col-2])  # 2 fewer columns since one is crv flag and one is ramp

	'''
	check what columns of 'out' entry are, stopped here
	'''
	#for col in range(0,num_col-2):
	#	channel_data[:,col] = data[ind_crc_pass,col+1]
	channel_data = data[ind_crc_pass,1:-1]

	#timestamps = time_stamps[ind_crc_pass]
	counter_ramp = data[ind_crc_pass,-1]

	corrected_counter = [counter_ramp[0]]
	num_cycle = 0
	print "Finding missed samples in ramp."
	# Counter is 16 bit and resets at 2**16. This loop unwraps this cycling so that values are monotonically increasing.
	for i in range(1,len(counter_ramp)):
	#for i in range(1,17000):
		print float(i)/len(counter_ramp)
		diff = counter_ramp[i] - counter_ramp[i-1]
		diff = int(diff)
		
		if (diff == -2**16 +1):
			num_cycle += 1
			corrected_counter.append(counter_ramp[i] + num_cycle*(2**16))
		else:
			corrected_counter.append(counter_ramp[i] + num_cycle*(2**16))
	
	corrected_counter = np.array(corrected_counter)
	diff_corrected_counter = corrected_counter[1:] - corrected_counter[:-1]
	miss_samp_index = np.ravel(np.nonzero(np.greater(diff_corrected_counter,1)))

	corrected_channel_data = channel_data[0:miss_samp_index[0]+1,:]
	diff = corrected_counter[miss_samp_index[0]+1] - corrected_counter[miss_samp_index[0]]
	diff = int(diff)
	inter_mat = np.zeros([diff,num_col-2])
	for j in range(0,num_col-2):
		y = np.interp(range(1,diff),[0, diff], [channel_data[miss_samp_index[0],j], channel_data[miss_samp_index[0]+1,j]])
		y = np.append(y,channel_data[miss_samp_index[0]+1,j])
		inter_mat[:,j] = y
	corrected_channel_data = np.vstack([corrected_channel_data, inter_mat])
	
	print "There are %i instances of missed samples." % len(miss_samp_index)
	print "Beginning looping through regeneration of data"
	for i in range(1,len(miss_samp_index)):
		print i
		# pad with good data first
		corrected_channel_data = np.vstack([corrected_channel_data, channel_data[miss_samp_index[i-1] + 2:miss_samp_index[i]+1,:]])
		# check number of samples that were skipped and need to be regenerated
		diff = corrected_counter[miss_samp_index[i]+1] - corrected_counter[miss_samp_index[i]]
		diff = int(diff)
		inter_mat = np.zeros([diff,num_col-2])
		# interpolate values to regenerate missing data
		for j in range(0,num_col-2):
			y = np.interp(range(1,diff),[0, diff], [channel_data[miss_samp_index[i],j], channel_data[miss_samp_index[i]+1,j]])
			y = np.append(y,channel_data[miss_samp_index[i]+1,j])
			inter_mat[:,j] = y
		corrected_channel_data = np.vstack([corrected_channel_data,inter_mat])

	if (miss_samp_index[-1]+1 != len(ind_crc_pass)-1):
		print "adding last zeros"
		corrected_channel_data = np.vstack([corrected_channel_data, channel_data[miss_samp_index[-1] + 2:,:]])
		
	hdf.close()
	return corrected_counter, corrected_channel_data


def computePowersWithChirplets(channel_data,Avg_Fs,channel,event_indices,t_before, t_after, center_freq):
	'''
	This method extracts spectral amplitudes around a defined center frequency by convolving raw LFP data with Gabor 
	time-frequency basis functions (Gaussian envelope).

	Inputs:
		- channel_data: raw LFP data formatted as a multi-dimensional array of size N_samps x N_channels
		- Avg_Fs: sampling frequency of LFP data
		- channel: channel to perform analysis on, all values should be in range [1,N_channels]
		- event_indices: sample indices that correspond to trial events data is aligned to, one-dimensional of length N_trials 
		- t_before: time before event index to include in analysis, measured in seconds
		- t_after: time after event index to include in analysis, measured in seconds
		- center_freq: center frequency parameter that is used in definition of Gabor atom
	Outputs:
		- trial_powers: power amplitudes formatterd a multi-dimensional array of size N_trials x N_timepoints, where N_timepoints is defined by the window
		                size dictated by the t_before and t_after parameter
		- times: vector of time points for easy plotting after analysis is complete
	'''
	# Define Gabor atom parameters
	v_0 = center_freq
	s_0 = -5.075
	t_0 = 0

	# Define other parameters
	win_before = int(t_before*Avg_Fs)
	win_after = int(t_after*Avg_Fs)
	channel = np.array(channel) - 1 	# adjust so that counting starts at 0
	print "Defining variables"
	times = np.arange(-t_before,t_after,float(t_after + t_before)/(win_after + win_before))
	trial_powers = np.zeros([len(event_indices),2*944])
	windows = np.zeros([len(event_indices),len(times)])

	for i,ind in enumerate(event_indices):
		print i,'/',len(event_indices)
		window = range(int(ind) - win_before,int(ind) + win_after)
		windows[i,:] = window
		#t_0 = -t_before + 1
		gabor_atom = (2**0.25)*np.exp(-0.25*s_0 - np.pi*((times - t_0)**2)*np.exp(-s_0) + 1j*np.pi*(times - t_0)*(2*v_0))
		complex_power = np.convolve(channel_data[window,channel],gabor_atom,mode='full')
		trial_powers[i,:] = np.absolute(complex_power[3*944:5*944]) 	# get power magnitudes
		#trial_powers[i,:] = np.absolute(complex_power) 	# get power magnitudes

	#mlab.specgram(channel_data[window,channel], NFFT=256, Fs=Avg_Fs)
	print "Done looping"
	return trial_powers, times, complex_power, channel_data[window,channel], windows

def powersWithFFT(channel_data,Avg_Fs,channel,event_indices,t_before, t_after):

	win_before = int(t_before*Avg_Fs)
	win_after = int(t_after*Avg_Fs)
	channel = np.array(channel) - 1 	# adjust so that counting starts at 0

	times = np.arange(-t_before,t_after,float(t_after + t_before)/(win_after + win_before))
	window_times = np.arange(-win_before,win_after)
	trial_powers = np.zeros([len(event_indices),len(times)])


	T = 1./Avg_Fs
	N = 256
	x = np.linspace(0.0, N*T, N)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	for j,ind in enumerate(event_indices[0:10]):
		for i in range(0,len(times)):
			print i,'/',len(times)
			t = np.arange(ind + times[i],ind+256 + times[i],1./Avg_Fs)
			data = channel_data[ind + window_times[i]:ind+256 + window_times[i],channel]/np.sum(channel_data[ind + window_times[i]:ind+256 + window_times[i],channel])
			sp = np.fft.fft(data)
			trial_powers[j,i] = np.absolute(sp[5])**2
	print len(xf)
	print len(sp)
	
	return trial_powers, xf

def powersWithSpecgram(channel_data,Avg_Fs,channel,event_indices,t_before, t_after):

	win_before = int(t_before*Avg_Fs)
	win_after = int(t_after*Avg_Fs)
	channel = np.array(channel) - 1 	# adjust so that counting starts at 0

	times = np.arange(-t_before,t_after,float(t_after + t_before)/(win_after + win_before))
	trial_powers = np.zeros([len(event_indices),28])

	for j,ind in enumerate(event_indices):
		data = channel_data[ind - win_before:ind + win_after,channel]
		data = np.ravel(data)
		Sxx, f, t, fig = specgram(data,Fs=Avg_Fs)
		#Sxx = Sxx/np.sum(Sxx)
		Sxx = 10*np.log10(Sxx)
		trial_powers[j,:] = np.sum(Sxx[3:5,:],axis=0)/2.
		#trial_powers[j,:] = Sxx[5,:]
	return trial_powers, t, f
	#return trial_powers, data

'''
filename_prefix = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/OMNI_Device/Data/streams7_20/'
filename = filename_prefix + '20160720-163020.csv'
#filename = filename_prefix + '20160720-171300.csv'
#filename = filename_prefix + '20160720-174338.csv'
data = pd.read_csv(filename,sep=',',header=None,skiprows=[0,1])
print "Data read."
#test_corrected_counter, test_corrected_channel_data, counter = test_convert_OMNI(data)
corrected_counter, corrected_channel_data, channel_data, miss_samp_index = convert_OMNI(data)
'''



