from plexon import plexfile
import numpy as np
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from scipy import signal
from scipy.ndimage import filters
from scipy.interpolate import spline

class OfflineSorted_PlxFile():
	'''
	Class for plexon file of offline-sorted units recorded with Plexon system. Units are offline-sorted with Plexon Offline Sorter
	V2.8 and then saved to .plx files. 
	'''

	def __init__(self, plx_file):
		self.filename =  plx_file
		# Read offline sorted data
		self.plx = plexfile.openFile('filename.plx')
		self.spikes = self.plx.spikes[:].data  # Extract spike times and channel info. Format is (time, chan, unit).
		self.waveforms = self.plex.spikes[:].waveforms
		self.times = np.array([entry[0] for entry in self.spikes]) 
		self.channels = np.array([entry[1] for entry in self.spikes])
		self.sort_code = np.array([entry[2] for entry in self.spikes])
		# Find channels with good sorted spikes 
		self.good_channels = np.unique([entry[1] for entry in self.spikes if entry[2] > 0])

	def find_chan_sc(self, chan):
		'''
		Method that returns the sort codes for the indicated channel.
		Input:
			- chan: integer indicating which recording channel is in question
		Output:
			- sc_chan: array containing all sort codes for units on the channel chan
		'''

		unit_chan = np.ravel(np.nonzero(np.equal(self.channels, chan)))
		sc_chan = np.unique(self.sort_code[unit_chan])
		sc_chan = np.array([code for code in sc_chan if code>0])

		return sc_chan

	def get_avg_firing_rates(self,channs):
		'''
		Method that returns the average firing rates of the channels listed in channs. Average is computed 
		over the duration of the recording.
		'''
		avg_firing_rates = dict()
		for chan in channs:
			# First find number of units recorded on this channel
			unit_chan = np.ravel(np.nonzero(np.equal(self.channels, chan)))
			sc_chan = np.unique(self.sort_code[unit_chan])
			sc_chan = np.array([sc for sc in sc_chan if sc>0])
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

	def get_waveform_data(self, chan, sc, plot_data):
		'''
		Method that returns length 32 array containing the average and standard deviation of the spike waveform on
		the indicated channel with the indicated sort code. Note that default sampling rate for spike data is 40K Hz.

		Input:
		- chan: integer indicating channel number
		- sc: integer indicating sort code
		- plot data : Boolean, indicate if data plot should be saved
		'''
		inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
		sc_waveform = self.waveforms[inds]
		mean_waveform = np.mean(sc_waveform, axis = 0) 		# array of size 32
		std_waveform = np.std(sc_waveform, axis = 0)

		# Plot waveform
		if plot_data==True:
			time = np.arange(0,32./40000., 1./40000)
			fig = plt.figure()
			plt.plot(time, mean_waveform, 'b')
			plt.fill_between(time, mean_waveform - std_waveform, mean_waveform + std_waveform, color = 'b', alpha = 0.5)
			plt.title('Channel %i - Unit %i' % (chan, sc))
			plt_filename = self.filename[:-4] + '_Chan_' + str(chan) + '_Unit_' + str(sc) + '.svg'
			plt.savefig(plt_filename)
			plt.close()

		return sc_waveform, mean_waveform, std_waveform

	def peak_to_peak_vals(self, chan, sc):
		'''
		Finds the peak-to-trough amplitude of each spike waveform and then computes the average.
		'''
		sc_waveform, mean_waveform, std_waveform = self.get_waveform_data(chan, sc)
		p2p = np.max(sc_waveform, axis = 1) - np.min(sc_waveform, axis = 1)
		avg_p2p = np.mean(p2p)

		return p2p, avg_p2p

	def peak_to_peak_hist(self, plot_data = True):
		'''
		Creates a histogram of the peak-to-trough amplitudes across all sorted units.
		Parameters
		----------
		plot_data : Boolean, indicates if data should be plotted and saved

		Return
		------
		hist_all : float array, hist of all peak-to-trough voltage values
		bins_all : float array, bins used for hist_all
		hist_avg : float array, hist of avg (per channel) peak-to-trough voltage values
		bins_avg : float array, bins used for hist_avg
		'''
		peaks = np.array([])
		avg_peaks = np.array([])
		for chan in self.good_channels:
			sc_chan = self.find_chan_sc(chan)
			for sc in sc_chan:
				p2p, avg_p2p = self.peak_to_peak_vals(chan, sc)
				peaks = np.append(peaks, p2p)
				avg_peaks = np.append(avg_peaks, avg_p2p)

		hist_all, bins_all = np.histogram(peaks, bins = 10)
		hist_all = hist_all/float(len(peaks))
		hist_avg, bins_avg = np.histogram(avg_peaks, bins = 10)
		hist_avg = hist_avg/float(len(avg_peaks))

		if plot_data:
			bins_all_center = (bins_all[1:] + bins_all[:-1])/2.
			bins_avg_center = (bins_avg[1:] + bins_avg[:-1])/2.
			fig = plt.figure()
			plt.subplot(121)
			plt.plot(bins_all_center, hist_all)
			plt.xlabel('Peak-to-Trough Values (uV)')
			plt.ylabel('Fraction of Units')
			plt.title('All Waveforms')
			plt.subplot(122)
			plt.plot(bins_avg_center, hist_avg)
			plt.xlabel('Peak-to-Trough Values (uV)')
			plt.ylabel('Fraction of Units')
			plt.title('Mean Waveforms')

			plt_filename = self.filename[:-4] + '_PeakAmpHistogram.svg'
			plt.savefig(plt_filename)
			plt.close()


		return hist_all, bins_all, hist_avg, bins_avg

	def raster_with_spike_hist(self, chan, sc, t_resolution):
	    """
	    Creates a raster plot with a corresponding smoothed histogram.
	    Parameters
	    ----------
	    chan : int, specifies channel number
	    sc : int, specifies sort code number
		t_resolution : int, the size of the time bins in terms of seconds, i.e. 0.1 = 100 ms and 1  = 1 s
		plot_data : Boolean, indicator if data should be plotted and saved

	    Returns
	    -------
	    ax : an axis containing the raster plot
	    """
	    ax = plt.gca()
	    boxcar_length = 4.
	    boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
	    b = signal.gaussian(39, 0.6)

	    inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
	    event_times_list = self.spikes[inds]

	    bins = np.arange(self.spikes['ts'][0], self.spikes['ts'][-1], t_resolution)

	    hist, bins = np.histogram(event_times_list, bins = bins)
	    hist_fr = hist/t_resolution
	    #smooth_hist = np.convolve(hist_fr, boxcar_window, mode='same')/boxcar_length
	    smooth_hist = filters.convolve1d(hist_fr, b/b.sum())

	    for ith, trial in enumerate(event_times_list):
	    	plt.vlines(trial, ith + .5, ith + 1.5, color=color)
	    plt.ylim(.5, len(event_times_list) + .5)
	    plt.plot(bins,smooth_hist)

	    plt_filename = self.filename[:-4] + '_Chan_' + str(chan) + '_Unit_' + str(sc) + '_Raster.svg'
	    plt.savefig(plt_filename)
	    plt.close()

	    return ax
	
	def spike_rate_correlation(self, t_resolution, plot_data = True):
		"""
		Computes correlation of spike rates across channels.
		Parameters
		----------
		t_resolution : float, value in seconds indicating resolution of time bins

		Returns
		-------
		corr_mat : array, two-dimensional array containing correlation values
		"""

		# 1. Bin all firing activity and build array of binned data
		count = 0
		bins = np.arange(self.spikes['ts'][0], self.spikes['ts'][-1], t_resolution) 
		for chan in self.good_channels:
			sc_chan = self.find_chan_sc(chan)
			for sc in sc_chan:
				inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
				event_times_list = self.spikes[inds]
				hist, bins = np.histogram(event_times_list, bins = bins)
				hist_fr = hist/t_resolution
				if count == 0:
					hist_all = hist_fr
					count += 1
				else:
					hist_all = np.vstack([hist_all, hist_fr])
		# 2. Correlate binned spike data across all channels.
		corr_mat = hist_all.corr()

		if plot_data:
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			cmap = cm.get_cmap('jet', 30)
			cax = ax1.imshow(corr_mat, interpolation="nearest", cmap=cmap)
			ax1.grid(True)
			plt.title('Firing Rate Correlation')
			labels=[str(chan) for chan in self.good_channels]
			ax1.set_xticklabels(labels,fontsize=6)
			ax1.set_yticklabels(labels,fontsize=6)
			# Add colorbar, make sure to specify tick locations to match desired ticklabels
			fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])

			plt_filename = self.filename[:-4] + '_FiringRateCorrelation.svg'
			plt.savefig(plt_filename)
			plt.close()

		return corr_mat

