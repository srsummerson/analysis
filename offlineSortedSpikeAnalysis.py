import numpy as np 
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt

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
		self.good_units = np.ravel(np.nonzero(np.less(self.sort_code, 31)))
		self.good_channels = np.unique(self.channel[self.good_units])

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
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([code for code in sc_chan if code != 31])
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
			sc_chan = np.array([sc for sc in sc_chan if sc != 31])
			if sc_chan.size == 0:
				avg_firing_rates[chan] = np.nan
			else:
				unit_rates = np.zeros(len(sc_chan))
				for i, sc in enumerate(sc_chan):
					sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
					sc_times = self.times[unit_chan[sc_unit]]  	# times that this sort code on this channel was recorded
					unit_rates[i] = len(sc_times)/float(self.times[-1] - self.times[0])

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
		psth_length = np.rint((t_before + t_after)/t_resolution)
		num_timepoints = len(times_align)
		psth = np.zeros((num_timepoints, psth_length-1))
		
		unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chann)))
		sc_unit = np.ravel(np.nonzero(np.equal(self.sort_code[unit_chan], sc)))
		channel_data = self.times[unit_chan[sc_unit]] 
		for i, tp in enumerate(times_align):
			data = channel_data
			t_window = np.arange(tp - t_before, tp + t_after, t_resolution)
			hist, bins = np.histogram(data, bins = t_window)
			hist_fr = hist/t_resolution
			psth[i,:] = hist_fr

		return psth

	def compute_multiple_channel_avg_psth(self, channs, times_align,t_before,t_after,t_resolution):
		'''
		Method that returns the average psth of spiking activity for all channels in channs array. The activity for all 
		channels is aligned to the same event times in times_align.
		'''
		unit_list = []
		avg_psth = []
		counter = 0
		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channel, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([code for code in sc_chan if code != 31])

			for sc in sc_chan:
				psth_sc = self.compute_psth(chan,sc,times_align,t_before,t_after,t_resolution)
				avg_psth_sc = np.nanmean(psth_sc, axis = 0)
				#avg_psth.append([avg_psth_sc])
				if counter == 0:
					avg_psth = avg_psth_sc
				else:
					avg_psth = np.vstack([avg_psth,avg_psth_sc])
				unit_list.append([chan, sc])
				counter += 1

		return avg_psth, np.array(unit_list)