import numpy as np 
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
import tables
import seaborn as sns
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
		if 'GOODCHAN' in self.df.columns:
			self.sorted_good_chann = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'GOODCHAN'])))
			self.sorted_good_chann = self.sorted_good_chann[~np.isnan(self.sorted_good_chann)]
			self.sorted_good_chans_sc, self.total_sorted_good_units = self.find_sorted_good_chan_sc()
			self.sorted_good_channels = np.unique(self.sorted_good_chann)
			
		# Adjust the channel numbers if this is for channels 97 - 160 that are recorded on the second RZ2.
		if self.event == 'eNe2':
			self.channel += 96
		self.sort_code = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'SORT'])))
		self.samp_rate = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'Sampling_Freq'])))[0]
		self.num_waveform_pts = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'NumOfPoints'])))[0]
		waveform_cols = self.df.columns[-self.num_waveform_pts-1:-1]
		waveform_cols = np.array([u'%s' % (elem) for elem in waveform_cols])
		self.waveforms = np.array(pd.DataFrame(self.df, columns = waveform_cols))
		self.sample_num = np.rint(self.times*self.samp_rate)

		# Find units with non-noisy recorded data. Recall that sort code 31 is for noise events. 
		self.good_units = np.ravel(np.nonzero(np.logical_and(np.greater(self.sort_code, 0),np.less(self.sort_code, 31))))
		self.good_channels = np.unique(self.channel[self.good_units])

	def find_sorted_good_chan_sc(self):
		'''
		Method that returns dictionary with keys as the sorted_good_channels and entries as the associated
		good sort codes.
		'''
		sorted_good_sc = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'GOODSC'])))
		sorted_good_sc = sorted_good_sc[~np.isnan(sorted_good_sc)]
		sc = dict()
		total_units = 0
		for chan in np.unique(self.sorted_good_chann):
			sc_chan = sorted_good_sc[self.sorted_good_chann==chan]
			total_units += len(sc_chan)
			sc[chan] = sc_chan
		return sc, total_units


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

	def get_avg_firing_rates_scgiven(self,sc_dict):
		'''
		Method that returns the average firing rates of the channels and associated sort codes in the input dictionary.

		Input:
		- sc_dict: dict, keys are the channel numbers and elements are arrays with the associated sort codes for that channel

		Output:
		- avg_firing_rates: dict, keys are the channel numbers and elements are the firing rates for the different units
							on that channel, where the length of the array corresponds to the number of sort codes indicated
							in the input dictionary
		'''
		avg_firing_rates = dict()

		channs = sc_dict.keys()

		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = sc_dict[chan]
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
		Output:
		- avg_firing_rates: list, average firing rates for all units on reported channel
		'''
		avg_firing_rates = []
		for k, chan in enumerate(channs):
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([sc for sc in sc_chan if (sc != 31)&(sc != 0)])
			if sc_chan.size > 0:
				unit_rates = np.zeros(len(sc_chan))
				for i, sc in enumerate(sc_chan):
					sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
					data = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
					spikes = (data > t_start)&(data < t_stop)  	# find spikes in this window
					unit_rates[i] = np.sum(spikes)/float(t_stop - t_start)		# count spikes and divide by window length
				if k==0:
					avg_firing_rates = unit_rates
				else:
					avg_firing_rates = np.hstack([avg_firing_rates, unit_rates])
				#avg_firing_rates += [unit_rates]


		return avg_firing_rates

	def get_avg_firing_rates_scgiven_range(self,sc_dict, t_start, t_stop):
		'''
		Method that returns the average firing rates over a specified time window of the channels and 
		associated sort codes in the input dictionary.

		Inputs:
		- sc_dict: dict, keys are the channel numbers and elements are arrays with the associated sort codes for that channel
		- t_start: float representing time (s) at which to begin counting spikes
		- t_stop: float representing time (s) at whic to stop counting spikes

		Output:
		- avg_firing_rates: dict, keys are the channel numbers and elements are the firing rates for the different units
							on that channel, where the length of the array corresponds to the number of sort codes indicated
							in the input dictionary
		'''
		avg_firing_rates = dict()

		channs = sc_dict.keys()

		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = sc_dict[chan]
			if sc_chan.size == 0:
				avg_firing_rates[chan] = np.array([np.nan])
			else:
				unit_rates = np.zeros(len(sc_chan))
				for i, sc in enumerate(sc_chan):
					sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
					sc_times = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
					spikes = (sc_times > t_start)&(sc_times < t_stop)  	# find spikes in this window
					unit_rates[i] = np.sum(spikes)/float(t_stop - t_start)		# count spikes and divide by window length

				avg_firing_rates[chan] = unit_rates

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

	def plot_all_avg_waveform(self, **kwargs):
		'''
		Method that plots the average waveform of all units.
		'''
		fig_dir = kwargs.get('fig_dir', "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Paper/Figures/")
		all_channels = self.good_channels
		sort_codes, total_units = self.find_unit_sc(all_channels)

		num_figs = 10*np.ceil(total_units/10)
		num_cols = 10
		num_rows = int(num_figs/10)

		fig_num = 1		# initialize figure number

		fig = plt.figure()

		for chann in all_channels:
			scs = sort_codes[chann]

			for sc in scs:
				unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
				sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
				channel_data = self.waveforms[unit_chan[sc_unit],:]

				avg_waveform = 10**6*np.nanmean(channel_data, axis = 0)	 # give amplitude in uV
				std_waveform = 10**6*np.nanstd(channel_data, axis = 0) 		# give amplitude in uV

				max_amp = np.max(channel_data)*10**6
				min_amp = np.min(channel_data)*10**6

				t = range(len(avg_waveform))

				#plt.subplot(np.ceil(fig_num/10),(fig_num+9) % 10 + 1,fig_num)
				plt.subplot(num_rows, num_cols, fig_num)
				plt.plot(t,avg_waveform,'k')
				plt.fill_between(t, avg_waveform - std_waveform, avg_waveform + std_waveform, facecolor='gray', alpha = 0.5)
				#plt.xlabel('Time (ms)')
				plt.ylim((-100,100))
				plt.title('Channel %i - %i' % (chann, sc))
				plt.ylabel('Amplitude (uV)')
				#plt.ylim((min_amp, max_amp))
				
				fig_num += 1
		fig.set_size_inches((30, 30), forward=False)
		plt.savefig(fig_dir + self.filename[81:-12] + '.svg', dpi = 500)
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
		psth_length = int(np.rint((t_before + t_after)/t_resolution))+1
		num_timepoints = len(times_align)
		psth = np.zeros((num_timepoints, psth_length-1))
		smooth_psth = np.zeros(psth.shape)

		boxcar_length = int(5)
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
		b = signal.gaussian(39, 1)
		
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 
		for i, tp in enumerate(times_align):
			data = channel_data
			t_window = np.arange(tp - t_before, tp + t_after + t_resolution, t_resolution)
			hist, bins = np.histogram(data, bins = t_window)
			hist_fr = hist/t_resolution
			psth[i,:] = hist_fr[:psth_length-1]
			#smooth_psth[i,:] = np.convolve(hist_fr[:psth_length-1], boxcar_window,mode='same')/boxcar_length
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
		smooth_psth = np.zeros((num_timepoints, psth_length))
		
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
			smooth_psth[i,:] = filters.convolve1d(psth[i,:], b/b.sum())
			
		return psth, smooth_psth

	def bin_data(self, t_resolution, smoothed):
		'''
		Method to bin spike data of all good offline sorted neurons into a 2D array: spike-counts x num_neurons

		Input:
		- t_resolution: integer, temporal resolution of bins (s)
		- smoothed: Boolean, indicate if data should be smoothed or not

		Output:
		- X: ndarray, 2D array of spike counts with resolution of t_resolution, size is neurons x spike_counts
		- unit_labsl: list, labels for units with channel number and sort codes
		'''


		data = np.array([])
		good_channels = self.sorted_good_chans_sc

		t_max = self.times[-1]
		t_min = self.times[0]
		t_bins = np.arange(t_min, t_max, t_resolution)
		t_bin_centers = (t_bins[1:] + t_bins[:-1])/2.
		#X = np.array([])
		X = t_bin_centers
		unit_labels = []

		boxcar_length = 4
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing


		for chan in good_channels:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = good_channels[chan]
			
			unit_rates = np.zeros(len(sc_chan))
			for i, sc in enumerate(sc_chan):
				#unit_name = 'Ch' + str(chan) + '_' + str(sc)
				sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
				sc_times = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
				hist_spikes, bins = np.histogram(sc_times, t_bins)
				unit_labels += ['Ch' + str(chan) + '_' + str(sc)]
				if smoothed:
					hist_spikes	 = np.convolve(hist_spikes, boxcar_window, mode='same')/boxcar_length
				X = np.vstack([X, hist_spikes])

		return X[1:,:], unit_labels

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
		boxcar_length = 4
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
		b = signal.gaussian(39,4)

		for i, tp in enumerate(times_align):
			data = channel_data
			# look at time window in question with padded 2 s before and after
			t_window = np.arange(tp - t_before - 2, tp + t_after + 2-t_resolution, t_resolution)
			hist, bins = np.histogram(data, bins = t_window)
			
			hist_fr = hist/t_resolution
			smooth_psth = np.convolve(hist_fr, boxcar_window,mode='same')/boxcar_length
			#smooth_psth = filters.convolve1d(hist_fr, b/b.sum())
			num_samps_in_window = np.rint((t_after + t_before)/t_resolution).astype(int)
			len_psth = len(smooth_psth)
			abegin = len_psth/2 - num_samps_in_window/2
			aend = len_psth/2 + num_samps_in_window/2
			window_fr[i] = 0.5*np.sum(smooth_psth[int(abegin):int(aend)])/(num_samps_in_window/2.)		# count spikes and divide by window length

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

	def compute_multiple_channel_scgiven_avg_psth(self, sc_dict, times_align,t_before,t_after,t_resolution):
		'''
		Method that returns the average psth of spiking activity for all channels in channs array. The activity for all 
		channels is aligned to the same event times in times_align.

		Inputs:
		- sc_dict: dict, keys are the channel numbers and elements are arrays with the associated sort codes for that channel
		- times_align: array, 
		- t_before: float, amount of time (seconds) of activity to be included before align times
		- t_after: float, amount of time (seconds) of activity to be included after align times
		- t_resolution: float, temporal resolution (in seconds) of bins used for psth

		Outputs:
		- avg_psth: array, (number of unique units) x (number of time points), containing the activity for each
					channel unit averaged across the aligned timepoints
		- smooth_avg_psth: array, (number of unique units) x (number of time points), containing the activity
					for each channel averaged across the aligned timepoints and smoothed
		- unit_list:  array, (number of unique units) x 2, the first column contains the channels used and the
						second column contains the corresponding sort codes; the order that each identifier
						appears in this array corresponds to the same order that the data comes from for
						avg_psth and smooth_avg_psth
		'''
		unit_list = []
		avg_psth = []
		smooth_avg_psth = []
		counter = 0

		boxcar_length = 4
		boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing

		channs = sc_dict.keys()

		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = sc_dict[chan]

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

	def activity_heatmaps(self, sc_dict, times_align, t_before, t_after, t_resolution, **kwargs):
		'''
		This methods produces heatmaps that are essentially PSTHS for either multiple trials or multiple neurons
		put together. It produces one heatmap per unit that gives the z-scored activity over time across multiple
		trials, as indicated using the times_align input, and then also produces an heatmap of averaged activity
		from all the units indicated.

		Inputs:
		- spike: OfflineSorted_CSVFile class object, contains all the spiking data
		- units: dict, keys are the channels and entries are the sort codes for the corresponding channel
		- times_align: array, times that the neural activity is aligned to
		- t_before: float, time (in seconds) of activity that should be included before the times_align points
		- t_after: float, time (in seconds) of activity that should be included after the times_align points
		- t_resolution: float, temporal resolution (in seconds) that should be used to bin the neural activity
		- plot_suffix: string, added to the end of the default naming convention for the plots produced
		'''
		fig_dir = kwargs.get('fig_dir', "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Paper/Figures/")
		time_zero_label = kwargs.get('time_zero_label','unspecified')

		channs = sc_dict.keys()
		counter = 0

		# create x-axis time labels for plots
		x_vals = np.arange(-t_before,t_after,t_resolution)
		x_vals = [format(x,'.1f') for x in x_vals]
		

		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = sc_dict[chan]

			for sc in sc_chan:
				psth_sc, smooth_psth_sc = self.compute_psth(chan,sc,times_align,t_before,t_after,t_resolution)
				x_ind = np.arange(psth_sc.shape[1])
				# plot heatmap of trials x time bins for each unit
				# NEED TO CHECK WHEN Z-SCORING SHOULD OCCUR
				plt.figure()
				cmap = sns.color_palette("Blues", 256)
				sns.heatmap(smooth_psth_sc, cmap=cmap)
				plt.xlabel('Time')
				plt.ylabel('Trials')
				plt.title('Spiking Activity over Time - Chan %i - Unit %i - Aligned %s' % (int(chan),int(sc),time_zero_label))
				fig_name = 'PSTH_heatmap_chan_' + str(int(chan)) + '_sc_' + str(int(sc)) + '.png'
				plt.xticks(x_ind[::10],x_vals[::10])
				plt.savefig(fig_dir + self.filename[81:-12] + "_" + fig_name, dpi = 500)
				plt.close()
				
				avg_psth_sc = np.nanmean(psth_sc, axis = 0)
				smooth_avg_psth_sc = np.nanmean(smooth_psth_sc, axis = 0)
				#avg_psth.append([avg_psth_sc])
				if counter == 0:
					avg_psth = avg_psth_sc
					smooth_avg_psth = smooth_avg_psth_sc
				else:
					avg_psth = np.vstack([avg_psth,avg_psth_sc])
					smooth_avg_psth = np.vstack([smooth_avg_psth, smooth_avg_psth_sc])
				counter += 1

		# plot heatmap of neurons x time bins, with averaged psth activity stacked for all neurons
		# normalize by average firing rate per channel
		avg_fr = np.nanmean(smooth_avg_psth,axis = 1)
		repeats_array = np.transpose([avg_fr] * smooth_avg_psth.shape[1])
		norm_smooth_avg_psth = smooth_avg_psth/repeats_array

		# indices of time indices corresponding to time window of interest
		t_begin = round(t_before/t_resolution)
		t_end = round((t_before + 1)/t_resolution)
		windowed_averages = np.nanmean(norm_smooth_avg_psth[:,t_begin:t_end], axis = 1)
		sorted_ind = np.argsort(windowed_averages)

		plt.figure()
		plt.subplot(121)
		# normalized acitivity for neurons in order of indices
		cmap = sns.color_palette("Blues", 256)
		sns.heatmap(norm_smooth_avg_psth, cmap=cmap)
		plt.xlabel('Time')
		plt.ylabel('Neurons')
		plt.title('Normalized Average Activity over Time - Aligned %s' % (time_zero_label))
		plt.xticks(x_ind[::10],x_vals[::10])
		plt.subplot(122)
		# normalized activity for neurons ordered by peak activity during time window of interest
		sns.heatmap(norm_smooth_avg_psth[sorted_ind,:], cmap=cmap)
		plt.xlabel('Time')
		plt.ylabel('Neurons')
		plt.title('Normalized Average Activity over Time - Aligned %s' % (time_zero_label))
		plt.xticks(x_ind[::10],x_vals[::10])
		fig_name = 'PSTH_heatmap_all_channels' + '.png'
		plt.savefig(fig_dir + self.filename[81:-12] + fig_name) 
		
		return avg_psth, smooth_avg_psth

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
	

	



