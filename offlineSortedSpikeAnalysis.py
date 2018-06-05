import numpy as np 
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import filters
from scipy.interpolate import spline

class OfflineSorted_CSVFile():
	'''
	Class for CSV file of offline-sorted units recorded with TDT system. Units are offline-sorted with OpenSorter and then exported to a CSV files
	using OpenBrowser. Each entry is separated by a comma and new rows are indicated with a return character.
	'''

	def __init__(self, csv_file):
		self.filename =  csv_file
		# Read offline sorted data into pandas dataframe. Note that first row in csv file contains the columns headers.
		self.df = pd.read_csv(self.filename, sep=',', header = 0)
		self.event = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'EVENT'])))[0]
		self.times = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'TIME'])))
		self.channel = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'CHAN'])))
		# Adjust the channel numbers if this is for channels 97 - 160 that are recorded on the second RZ2.
		if self.event == 'eNe2':
			self.channel += 96
		self.sort_code = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'SORT'])))
		self.samp_rate = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'Sampling_Freq'])))[0]
		self.num_waveform_pts = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'NumOfPoints'])))[0]
		self.waveforms = np.array(pd.DataFrame(self.df, columns = [self.df.columns[-self.num_waveform_pts-1:-1]]))
		self.sample_num = np.rint(self.times*self.samp_rate)

		# Find units with non-noisy recorded data. Recall that sort code 31 is for noise events. 
		self.good_units = np.ravel(np.nonzero(np.logical_and(np.greater(self.sort_code, 0),np.less(self.sort_code, 31))))
		self.good_channels = np.unique(self.channel[self.good_units])

	def find_chan_sc(self, chan):
		'''
		Method that returns the sort codes for the indicated channel.
		Input:
			- chan: integer indicating which recording channel is in question
		Output:
			- sc_chan: array containing all sort codes for units on the channel chan
		'''

		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
		sc_chan = np.unique(self.sort_code[unit_chan])
		sc_chan = np.array([code for code in sc_chan if (code != 31)&(code!=0)])

		return sc_chan

	def find_unit_sc(self,channs):
		'''
		Method that returns the unit sort codes for the channels in channs.

		Input: 
		- channs: array containing integer values corresponding to channel numbers

		Output:
		- sc: dictionary containing an entry for each channel in channs. Each entry contains an array corresponding
				to the sort codes found for that channel.
		'''
		sc = dict()
		total_units = 0
		for chan in channs:
			# First find number of units recorded on this channel
			sc_chan = self.find_chan_sc(chan)
			total_units += len(sc_chan)
			sc[chan] = sc_chan
		return sc, total_units

	def get_avg_firing_rates(self,channs):
		'''
		Method that returns the average firing rates of the channels listed in channs.
		'''
		avg_firing_rates = dict()
		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([sc for sc in sc_chan if (sc != 31)&(sc!=0)])
			if sc_chan.size == 0:
				avg_firing_rates[chan] = np.array([np.nan])
			else:
				unit_rates = np.zeros(len(sc_chan))
				for i, sc in enumerate(sc_chan):
					sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
					sc_times = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
					unit_rates[i] = len(sc_times)/float(self.times[-1] - self.times[0])

				avg_firing_rates[chan] = unit_rates

		return avg_firing_rates

	def get_avg_firing_rates_range(self,channs, t_start, t_stop):
		'''
		Method that returns the average firing rates of the channels listed in channs over the time range defined
		by t_start and t_stop.

		Input:
		- channs: list or array of channels (all integers)
		- t_start: float representing time (s) at which to begin counting spikes
		- t_stop: float representing time (s) at whic to stop counting spikes
		'''
		avg_firing_rates = []
		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([sc for sc in sc_chan if (sc != 31)&(sc != 0)])
			if sc_chan.size == 0:
				avg_firing_rates[chan] = np.array([np.nan])
			else:
				unit_rates = np.zeros(len(sc_chan))
				for i, sc in enumerate(sc_chan):
					sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
					data = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
					spikes = (data > t_start)&(data < t_stop)  	# find spikes in this window
					unit_rates[i] = np.sum(spikes)/float(t_stop - t_start)		# count spikes and divide by window length
				avg_firing_rates += [unit_rates]


		return avg_firing_rates

	def plot_avg_waveform(self,chann, sc):
		'''
		Method that plots the average waveform of the unit, as well as example traces.
		'''
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.waveforms[unit_chan[sc_unit],:]

		avg_waveform = 10**6*np.nanmean(channel_data, axis = 0)	 # give amplitude in uV
		std_waveform = 10**6*np.nanstd(channel_data, axis = 0) 		# give amplitude in uV

		max_amp = np.max(channel_data)*10**6
		min_amp = np.min(channel_data)*10**6

		t = range(len(avg_waveform))

		plt.figure()
		plt.subplot(121)
		plt.plot(t,avg_waveform,'k')
		plt.fill_between(t, avg_waveform - std_waveform, avg_waveform + std_waveform, facecolor='gray', alpha = 0.5)
		#plt.xlabel('Time (ms)')
		plt.ylabel('Amplitude (uV)')
		plt.ylim((min_amp, max_amp))
		plt.subplot(122)
		plt.plot(t,10**6*channel_data[:200][:].T)
		plt.ylabel('Amplitude (uV)')
		plt.ylim((min_amp, max_amp))
		plt.show()

		return

	def compute_psth(self,chann,sc,times_align,t_before,t_after,t_resolution):
		'''
		Method that returns an array of psths for spiking activity aligned to the sample numbers indicated in samples_align
		with firing rates quantized to bins of size samp_resolution.

		Input:
		- chann: integer representing the channel number
		- sc: integer representing the sort code for the channel
		- times_align: array of T time points (s) corresponding to the time points for which activity should be aligned
		- t_before: integer indicating the length of time (s) to be included prior to the alignment time point
		- t_after: integer indicating the length of time (s) to be included after the alignment time point
		- t_resolution: the size of the time bins in terms of seconds, i.e. 0.1 = 100 ms and 1  = 1 s

		Output: 
		- psth: T x N array containing the average firing rate over a window of total length N samples for T different
				time points
		'''
		psth_length = int(np.rint((t_before + t_after)/t_resolution))
		num_timepoints = len(times_align)
		psth = np.zeros((num_timepoints, psth_length-1))
		smooth_psth = np.zeros(psth.shape)

		boxcar_length = 5.
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
		b = signal.gaussian(39, 1)
		
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 
		for i, tp in enumerate(times_align):
			data = channel_data
			t_window = np.arange(tp - t_before, tp + t_after, t_resolution)
			hist, bins = np.histogram(data, bins = t_window)
			hist_fr = hist/t_resolution
			psth[i,:] = hist_fr[:psth_length-1]
			smooth_psth[i,:] = np.convolve(hist_fr[:psth_length-1], boxcar_window,mode='same')/boxcar_length
			smooth_psth[i,:] = filters.convolve1d(hist_fr[:psth_length-1], b/b.sum())
		return psth, smooth_psth

	def compute_sliding_psth(self,chann,sc,times_align,t_before,t_after,t_resolution, t_overlap):
		'''
		Method that returns an array of psths for spiking activity aligned to the sample numbers indicated in samples_align
		with firing rates quantized to bins of size t_resolution. Bins overlap in time with overlap defined by
		t_overlap.

		Input:
		- chann: integer representing the channel number
		- sc: integer representing the sort code for the channel
		- times_align: array of T time points (s) corresponding to the time points for which activity should be aligned
		- t_before: integer indicating the length of time (s) to be included prior to the alignment time point
		- t_after: integer indicating the length of time (s) to be included after the alignment time point
		- t_resolution: the size of the time bins in terms of seconds, i.e. 0.1 = 100 ms and 1  = 1 s
		- t_overlap: the size of the overlap of the time bins in terms of seconds, i.e. 0.05 = 50 ms, should be less than t_resolution

		Output: 
		- psth: T x N array containing the average firing rate over a window of total length N samples for T different
				time points
		'''
		psth_length = int(np.rint((t_before + t_after - t_overlap)/(t_resolution - t_overlap)))
		num_timepoints = len(times_align)
		psth = np.zeros((num_timepoints, psth_length))
		

		boxcar_length = 4.
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
		b = signal.gaussian(39, 1)
		
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 
		for i, tp in enumerate(times_align):
			data = channel_data
			t_start = tp - t_before
			t_end = tp + t_after
			for k in range(psth_length):
				data_window = np.ravel(np.greater(data, t_start + k*t_overlap)&np.less(data, t_start + k*t_overlap + t_resolution))
				psth[i,k] = np.sum(data_window)/t_resolution

			#smooth_psth[i,:] = filters.convolve1d(psth[i,:], b/b.sum())
			
		return psth

	def compute_raster(self,chann,sc,times_align,t_before,t_after):
		'''
		Method that returns times for spiking activity aligned to the sample numbers indicated in samples_align.

		Input:
		- chann: integer representing the channel number
		- sc: integer representing the sort code for the channel
		- times_align: array of T time points (s) corresponding to the time points for which activity should be aligned
		- t_before: integer indicating the length of time (s) to be included prior to the alignment time point
		- t_after: integer indicating the length of time (s) to be included after the alignment time point
		- t_resolution: the size of the time bins in terms of seconds, i.e. 0.1 = 100 ms and 1  = 1 s

		Output: 
		- psth: T x N array containing the average firing rate over a window of total length N samples for T different
				time points
		'''
		num_timepoints = len(times_align)
		raster = dict()
		
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 
		for i, tp in enumerate(times_align):
			data = channel_data
			data_window = np.ravel(np.nonzero(np.greater(data, tp - t_before)&np.less(data,tp + t_after)))
			raster_spikes = data[data_window] - tp
			raster[i] = raster_spikes
		return raster

	def compute_window_fr(self,chann,sc,times_align,t_before,t_after):
		'''
		Method that returns an array of avg firing rates for spiking activity aligned to the sample numbers indicated in samples_align
		with firing activity taking over the window [times_align - t_before, times_align + t_after]

		Input:
		- chann: integer representing the channel number
		- sc: integer representing the sort code for the channel
		- times_align: array of T time points (s) corresponding to the time points for which activity should be aligned
		- t_before: integer indicating the length of time (s) to be included prior to the alignment time point
		- t_after: integer indicating the length of time (s) to be included after the alignment time point
		
		Output: 
		- window_fr: length T array containing the average firing rate over the window size indicated aligned to T different
				time points
		'''
		num_timepoints = len(times_align)
		window_fr = np.zeros(num_timepoints)
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 
		for i, tp in enumerate(times_align):
			data = channel_data
			spikes = (data > (tp - t_before))&(data < (tp + t_after))  	# find spikes in this window
			window_fr[i] = np.sum(spikes)/float(t_after + t_before)		# count spikes and divide by window length

		return window_fr

	def compute_window_fr_smooth(self,chann,sc,times_align,t_before,t_after):
		'''
		Method that returns an array of avg firing rates for spiking activity aligned to the sample numbers indicated in samples_align
		with firing activity taking over the window [times_align - t_before, times_align + t_after]

		Input:
		- chann: integer representing the channel number
		- sc: integer representing the sort code for the channel
		- times_align: array of T time points (s) corresponding to the time points for which activity should be aligned
		- t_before: integer indicating the length of time (s) to be included prior to the alignment time point
		- t_after: integer indicating the length of time (s) to be included after the alignment time point
		
		Output: 
		- window_fr: length T array containing the average firing rate over the window size indicated aligned to T different
				time points
		'''
		num_timepoints = len(times_align)
		window_fr = np.zeros(num_timepoints)
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 

		t_resolution = 0.1
		boxcar_length = 4.
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
		b = signal.gaussian(39,4)

		for i, tp in enumerate(times_align):
			data = channel_data
			# look at time window in question with padded 2 s before and after
			t_window = np.arange(tp - t_before - 2, tp + t_after + 2, t_resolution)
			hist, bins = np.histogram(data, bins = t_window)
			hist_fr = hist/t_resolution
			smooth_psth = np.convolve(hist_fr, boxcar_window,mode='same')/boxcar_length
			#smooth_psth = filters.convolve1d(hist_fr, b/b.sum())
			num_samps_in_window = np.rint((t_after + t_before)/t_resolution)
			len_psth = len(smooth_psth)
			window_fr[i] = 0.5*np.sum(smooth_psth[len_psth/2 - num_samps_in_window/2:len_psth/2 + num_samps_in_window/2])/(num_samps_in_window/2.)		# count spikes and divide by window length

		return window_fr

	def compute_window_peak_fr(self,chann,sc,times_align, window_length):
		'''
		Method that returns an array of avg firing rates around the peak spiking activity aligned to the sample numbers indicated in samples_align
		with firing activity taking over the window [times_align, times_align + window_length]. The peak firing rate
		returned is from the activity in the time window of length window_length.

		Input:
		- chann: integer representing the channel number
		- sc: integer representing the sort code for the channel
		- times_align: array of T time points (s) corresponding to the time points for which activity should be aligned
		- window_length: integer indicated the size of the time window (s) that the spiking activity will be taken from

		Output: 
		- window_fr: length T array containing the peak firing rate in the window size indicated aligned to T different
				time points
		- smooth_window_fr: length T array containing the peak firing rate taken from smoothed psth in the window size indicated aligned to T different
				time points
		- psth: size T by 39 array, containing psth values aligned to timepoints indicated in times_align array
		'''
		num_timepoints = len(times_align)
		window_fr = np.zeros(num_timepoints)
		smooth_window_fr = np.zeros(num_timepoints)
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 

		t_before = 2
		t_after = 2
		t_resolution = 0.1
		pts_in_linspace = 100
		t_resolution_linspace = (t_before + t_after)/float(pts_in_linspace)
		num_pts_in_window = int(window_length/t_resolution)
		num_pts_in_window_linspace = int(window_length/t_resolution_linspace)
		psth_length = np.rint((t_before + t_after)/t_resolution)
		num_timepoints = len(times_align)
		psth = np.zeros((num_timepoints, psth_length-1))
		
		xnew = np.linspace(0,39,100)

		for i, tp in enumerate(times_align):
			data = channel_data
			t_window = np.arange(tp - t_before, tp + t_after, t_resolution)
			hist, bins = np.histogram(data, bins = t_window)
			hist_fr = hist/t_resolution
			psth[i,:] = hist_fr[:psth_length-1]
			smooth_psth = spline(range(39),psth[i,:],xnew)
			window_fr[i] = np.amax(psth[i,psth_length/2:psth_length/2 + num_pts_in_window])
			smooth_window_fr[i] = np.amax(smooth_psth[pts_in_linspace/2:pts_in_linspace/2 + num_pts_in_window_linspace])
			
		return window_fr, smooth_window_fr, psth

	def compute_multiple_channel_avg_psth(self, channs, times_align,t_before,t_after,t_resolution):
		'''
		Method that returns the average psth of spiking activity for all channels in channs array. The activity for all 
		channels is aligned to the same event times in times_align.
		'''
		unit_list = []
		avg_psth = []
		smooth_avg_psth = []
		counter = 0

		boxcar_length = 4.
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing

		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([code for code in sc_chan if code != 31])

			for sc in sc_chan:
				psth_sc, smooth_psth_sc = self.compute_psth(chan,sc,times_align,t_before,t_after,t_resolution)
				avg_psth_sc = np.nanmean(psth_sc, axis = 0)
				smooth_avg_psth_sc = np.nanmean(smooth_psth_sc, axis = 0)
				#avg_psth.append([avg_psth_sc])
				if counter == 0:
					avg_psth = avg_psth_sc
					smooth_avg_psth = smooth_avg_psth_sc
				else:
					avg_psth = np.vstack([avg_psth,avg_psth_sc])
					smooth_avg_psth = np.vstack([smooth_avg_psth, smooth_avg_psth_sc])
				unit_list.append([chan, sc])
				counter += 1

		return avg_psth, smooth_avg_psth, np.array(unit_list)

class OfflineSorted_Spikes():
	'''
	Class for converting spike data from CSV file of offline-sorted units recorded with TDT system
	to .mat file with all spike data binned to 1 ms bins. Units are offline-sorted with OpenSorter and then 
	exported to a CSV files using OpenBrowser. Each entry is separated by a comma and new rows are indicated 
	with a return character. Only good channels are exported into the .mat file.
	'''

	def __init__(self, csv_file):
		self.filename =  csv_file
		# Read offline sorted data into pandas dataframe. Note that first row in csv file contains the columns headers.
		self.df = pd.read_csv(self.filename, sep=',', header = 0)
		self.event = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'EVENT'])))[0]
		self.times = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'TIME'])))
		self.channel = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'CHAN'])))
		# Adjust the channel numbers if this is for channels 97 - 160 that are recorded on the second RZ2.
		if self.event == 'eNe2':
			self.channel += 96
		self.sort_code = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'SORT'])))
		self.samp_rate = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'Sampling_Freq'])))[0]
		self.num_waveform_pts = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'NumOfPoints'])))[0]
		self.waveforms = np.array(pd.DataFrame(self.df, columns = [self.df.columns[-self.num_waveform_pts-1:-1]]))
		self.sample_num = np.rint(self.times*self.samp_rate)

		# Find units with non-noisy recorded data. Recall that sort code 31 is for noise events. 
		self.good_units = np.ravel(np.nonzero(np.less(self.sort_code, 31)))
		self.good_channels = np.unique(self.channel[self.good_units])

	def find_chan_sc(self, chan):
		'''
		Method that returns the sort codes for the indicated channel.
		Input:
			- chan: integer indicating which recording channel is in question
		Output:
			- sc_chan: array containing all sort codes for units on the channel chan
		'''

		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
		sc_chan = np.unique(self.sort_code[unit_chan])
		sc_chan = np.array([code for code in sc_chan if code != 31])

		return sc_chan

	def find_unit_sc(self,channs):
		'''
		Method that returns the unit sort codes for the channels in channs.

		Input: 
		- channs: array containing integer values corresponding to channel numbers

		Output:
		- sc: dictionary containing an entry for each channel in channs. Each entry contains an array corresponding
				to the sort codes found for that channel.
		'''
		sc = dict()
		total_units = 0
		for chan in channs:
			# First find number of units recorded on this channel
			sc_chan = self.find_chan_sc(chan)
			total_units += len(sc_chan)
			sc[chan] = sc_chan
		return sc, total_units
	
	def bin_data(self):
		data = dict()
		good_channels = self.good_channels

		t_max = self.times[-1]
		t_min = self.times[0]
		t_bins = np.arange(t_min, t_max, 0.001)
		t_bin_centers = (t_bins[1:] + t_bins[:-1])/2.
		X = t_bin_centers

		unit_labels = ['Time']

		for chan in good_channels:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([sc for sc in sc_chan if ((sc != 31) and (sc != 0))])
			
			unit_rates = np.zeros(len(sc_chan))
			for i, sc in enumerate(sc_chan):
				#unit_name = 'Ch' + str(chan) + '_' + str(sc)
				sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
				sc_times = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
				hist_spikes, bins = np.histogram(sc_times, t_bins)
				unit_labels += ['Ch' + str(chan) + '_' + str(sc)]
				X = np.vstack([X, hist_spikes])

		return X, unit_labels
	

	



