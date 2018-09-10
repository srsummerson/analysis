from plexon import plexfile
import numpy as np
import scipy as sp
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import matplotlib as mpl
from scipy import signal
from scipy.ndimage import filters
from scipy.interpolate import spline

###
### To add: heat map
###

class OfflineSorted_PlxFile():
	'''
	Class for plexon file of offline-sorted units recorded with Plexon system. Units are offline-sorted with Plexon Offline Sorter
	V2.8 and then saved to .plx files. 
	'''

	def __init__(self, plx_file):
		self.filename =  plx_file
		# Read offline sorted data
		self.plx = plexfile.openFile(self.filename)
		self.spikes = self.plx.spikes[:].data  # Extract spike times and channel info. Format is (time, chan, unit).
		self.waveforms = self.plx.spikes[:].waveforms
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

	def get_waveform_data(self, chan, sc, waveform_inds):
		'''
		Method that returns length 32 arrays containing the average and standard deviation of the spike waveform on
		the indicated channel with the indicated sort code. Note that default sampling rate for spike data is 40K Hz.

		Input:
		- chan: integer indicating channel number
		- sc: integer indicating sort code
		- waveform_inds : integer array, indices of waveforms to include in plot
		'''
		inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
		sc_waveform = self.waveforms[inds]
		mean_waveform = np.mean(sc_waveform, axis = 0) 		# array of size 32
		std_waveform = np.std(sc_waveform, axis = 0)
		vrms = np.sqrt(np.mean(np.square(mean_waveform)))

		cmap = mpl.cm.hsv
		if (sc==2)&(chan!=32):
			cmap = mpl.cm.terrain
		elif (sc==1)&(chan!=32):
			cmap = mpl.cm.terrain
		
		num_waveforms = float(len(waveform_inds))

		time = np.arange(0,32./40000., 1./40000)
		plt.figure()
		plt.plot(time, mean_waveform, 'k')
		plt.fill_between(time, mean_waveform - std_waveform, mean_waveform + std_waveform, color = 'k', alpha = 0.5, linewidth=0.0)
		for i,ind in enumerate(waveform_inds):
			plt.plot(time, sc_waveform[ind,:], color = cmap(i/num_waveforms))
		plt.title('Channel %i - Unit %i' % (chan, sc))
		plt.xlabel('Time (s)')
		plt.ylabel('Voltage (' + r'$\mu$' + 'V)')
		plt.ylim((-70,70))
		plt.text(time[-8],mean_waveform[20],'$V_{rms}=$ %f' % (vrms))
		plt_filename = self.filename[:-4] + '_Chan_' + str(chan) + '_Unit_' + str(sc) + '_ExTraces.svg'
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

	def peak_amp_heatmap(self):

		peaks = np.array([])
		avg_peaks = np.array([])
		chan_array = np.array([])
		for chan in self.good_channels:
			sc_chan = self.find_chan_sc(chan)
			for sc in sc_chan:
				chan_array = np.append(chan_array, chan)
				p2p, avg_p2p = self.peak_to_peak_vals(chan, sc)
				peaks = np.append(peaks, p2p)
				avg_peaks = np.append(avg_peaks, avg_p2p)

		'''
		make mapping from channels to distance. deal with fact that some channels may have more than one entry because of 
		multiple units on the channel.

		power_mat = np.zeros([15,14])
		power_mat[:,:] = np.nan 	# all entries initially nan until they are update with peak powers
		channels = np.arange(1,161)

		row_zero = np.array([31, 15, 29, 13, 27, 11, 25, 9])
		row_one = np.array([32, 16, 30, 14, 28, 12, 26, 10])
		row_two = np.array([24, 8, 22, 6, 20, 4, 18, 2, 23])
		row_three = np.array([7, 21, 5, 19, 3, 17, 1, 63, 47])
		row_four = np.array([61, 45, 59, 43, 57, 41, 64, 48, 62, 46])
		row_five = np.array([60, 44, 58, 42, 56, 40, 54, 38, 52, 36])
		row_six = np.array([50, 34, 55, 39, 53, 37, 51, 35, 49, 33, 95])
		row_seven = np.array([79, 93, 77, 91, 75, 89, 73, 96, 80, 94, 78])
		row_eight = np.array([92, 76, 90, 74, 88, 72, 86, 70, 84, 68, 82, 66])
		row_nine = np.array([87, 71, 85, 69, 83, 67, 81, 65, 127, 111, 125, 109])
		row_ten = np.array([123, 107, 121, 105, 128, 112, 126, 110, 124, 108, 122, 106, 120])
		row_eleven = np.array([104, 118, 102, 116, 100, 114, 98, 119, 103, 117, 101, 115])
		row_twelve = np.array([99, 113, 97, 159, 143, 157, 141, 155, 139, 153, 137, 160])
		row_thirteen = np.array([144, 158, 142, 156, 140, 154, 138, 152, 136, 150, 134])
		row_fourteen = np.array([148, 132, 146, 130, 151, 135, 149, 133, 147])

		power_mat[0,0:8] = powers[row_zero-1]
		power_mat[1,0:8] = powers[row_one-1]
		power_mat[2,0:9] = powers[row_two-1]
		power_mat[3,0:9] = powers[row_three-1]
		power_mat[4,0:10] = powers[row_four-1]
		power_mat[5,0:10] = powers[row_five-1]
		power_mat[6,0:11] = powers[row_six-1]
		power_mat[7,0:11] = powers[row_seven-1]
		power_mat[8,0:12] = powers[row_eight-1]
		power_mat[9,0:12] = powers[row_nine-1]
		power_mat[10,0:13] = powers[row_ten-1]
		power_mat[11,1:13] = powers[row_eleven-1]
		power_mat[12,2:14] = powers[row_twelve-1]
		power_mat[13,3:14] = powers[row_thirteen-1]
		power_mat[14,4:13] = powers[row_fourteen-1]
		'''


		return

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

		bins_all = np.arange(50,130,10)

		hist_all, bins_all = np.histogram(peaks, bins = bins_all)
		hist_all = hist_all/float(len(peaks))
		hist_avg, bins_avg = np.histogram(avg_peaks, bins = bins_all)
		hist_avg = hist_avg/float(len(avg_peaks))

		if plot_data:
			bins_all_center = (bins_all[1:] + bins_all[:-1])/2.
			bins_avg_center = (bins_avg[1:] + bins_avg[:-1])/2.
			width_all = bins_all[1] - bins_all[0]
			width_avg = bins_avg[1] - bins_avg[0]
			print width_all
			print width_avg
			fig = plt.figure()
			plt.subplot(121)
			plt.bar(bins_all_center, hist_all, width_all)
			plt.xlabel('Peak-to-Trough Values ($\mu$V)')
			plt.ylabel('Fraction of Units')
			plt.title('All Waveforms')
			plt.ylim((0,0.3))
			plt.subplot(122)
			plt.bar(bins_avg_center, hist_avg, width_avg)
			plt.xlabel('Peak-to-Trough Values ($\mu$V)')
			plt.ylabel('Fraction of Units')
			plt.title('Mean Waveforms')
			plt.ylim((0,0.3))

			plt_filename = self.filename[:-4] + '_PeakAmpHistogram.svg'
			plt.savefig(plt_filename)
			plt.close()


		return hist_all, bins_all, hist_avg, bins_avg, avg_peaks

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
	    boxcar_length = 4
	    boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
	    b = signal.gaussian(39, 0.6)

	    inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
	    event_times_list = self.spikes['ts'][inds]

	    bins = np.arange(self.spikes['ts'][0], self.spikes['ts'][-1], t_resolution)

	    hist, bins = np.histogram(event_times_list, bins = bins)
	    hist_fr = hist/t_resolution
	    #smooth_hist = np.convolve(hist_fr, boxcar_window, mode='same')/boxcar_length
	    smooth_hist = filters.convolve1d(hist_fr, b/b.sum())
	    bin_centers = (bins[1:] + bins[:-1])/2.

	    plt.figure()
	    plt.subplot(121)
	    for ith, trial in enumerate(event_times_list):
	    	plt.vlines(trial, .5, 1.5, color='k')
	    plt.xlabel('Time (s)')
	    #plt.ylim(.5, len(event_times_list) + .5)
	    plt.subplot(122)
	    plt.plot(bin_centers,smooth_hist)
	    plt.xlabel('Time (s)')
	    plt.ylabel('Instantaneous Firing Rate (Hz)')

	    plt_filename = self.filename[:-4] + '_Chan_' + str(chan) + '_Unit_' + str(sc) + '_Raster.svg'
	    plt.savefig(plt_filename)
	    plt.close()

	    return ax

	def all_channels_raster_with_spike_hist(self, t_resolution, t_window):
	    """
	    Creates a raster plot where each row is a different channel with a corresponding smoothed histogram of total
	    spiking activity.
	    Parameters
	    ----------
		t_resolution : int, the size of the time bins in terms of seconds, i.e. 0.1 = 100 ms and 1  = 1 s
		t_window: integer array, size two array indicating values (in seconds) for window to plot

	    Returns
	    -------
	    ax : an axis containing the raster plot
	    """
	    ax = plt.gca()
	    boxcar_length = 4
	    boxcar_window = signal.boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
	    b = signal.gaussian(39, 0.6)

	    count = 0
	    all_events = np.array([])
	    bins = np.arange(self.spikes['ts'][0], self.spikes['ts'][-1], t_resolution)
	    cmap = mpl.cm.hsv

	    for chan in self.good_channels:
	    	sc_chan = self.find_chan_sc(chan)
	    	for sc in sc_chan:

	    		inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
	    		event_times_list = self.spikes['ts'][inds]
	    		hist, abins = np.histogram(event_times_list, bins = bins)
	    		hist_fr = hist/t_resolution
	    		if count==0:
	    			all_events = hist_fr
	    		else:
	    			all_events = np.vstack([all_events, hist_fr])

	    		plt.figure(1)
	    		plt.subplot(211)
	    		for ith, trial in enumerate(event_times_list):
	    			plt.vlines(trial, count + .5, count + 1.3, color=cmap(count/30.))
	    		count += 1

	    #print all_events.shape		
	    all_hist_fr = np.mean(all_events, axis = 0)
	    #print len(all_hist_fr)
	    #smooth_hist = np.convolve(hist_fr, boxcar_window, mode='same')/boxcar_length
	    smooth_hist = filters.convolve1d(all_hist_fr, b/b.sum())
	    bin_centers = (abins[1:] + abins[:-1])/2.

	    plt.subplot(211)
	    plt.xlabel('Time (s)')
	    plt.xlim((t_window[0],t_window[1]))
	    #plt.ylim(.5, len(event_times_list) + .5)
	    plt.subplot(212)
	    plt.plot(bin_centers,smooth_hist)
	    plt.xlabel('Time (s)')
	    plt.ylabel('Avg Firing Rate (Hz)')
	    plt.xlim((t_window[0],t_window[1]))

	    plt_filename = self.filename[:-4] + '_Raster.svg'
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
		labels = [] 
		for chan in self.good_channels:
			sc_chan = self.find_chan_sc(chan)
			for sc in sc_chan:
				labels += ['Chan_' + str(chan) + '_Unit_' + str(sc)]
				inds, = np.nonzero((self.spikes['chan'] == chan) * (self.spikes['unit'] == sc))
				event_times_list = self.spikes['ts'][inds]
				hist, bins = np.histogram(event_times_list, bins = bins)
				hist_fr = hist/t_resolution
				if count == 0:
					hist_all = hist_fr
					count += 1
				else:
					hist_all = np.vstack([hist_all, hist_fr])
		# 2. Correlate binned spike data across all channels.
		print hist_all.shape
		#corr_mat = hist_all.corr()
		corr_mat = np.corrcoef(hist_all)
		print corr_mat.shape

		if plot_data:
			fig = plt.figure(1)
			ax1 = fig.add_subplot(111)
			cmap = cm.get_cmap('jet', 30)
			cax = ax1.imshow(corr_mat, interpolation="nearest", cmap=cmap, vmin = 0.0, vmax = 1.0)
			ax1.grid(True)
			plt.title('Firing Rate Correlation')
			#labels=[str(chan) for chan in self.good_channels]
			ax1.set_xticklabels(labels,fontsize=6)
			ax1.set_xticks(range(len(labels)))
			ax1.set_yticklabels(labels,fontsize=6)
			ax1.set_yticks(range(len(labels)))
			# Add colorbar, make sure to specify tick locations to match desired ticklabels
			fig.colorbar(cax, ticks=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

			plt_filename = self.filename[:-4] + '_FiringRateCorrelation.svg'
			plt.savefig(plt_filename)

			fig = plt.figure(2)
			ax1 = fig.add_subplot(111)
			cmap = cm.get_cmap('jet', 30)
			cax = ax1.imshow(corr_mat, interpolation="nearest", cmap=cmap, vmin = 0.5, vmax = 1.0)
			ax1.grid(True)
			plt.title('Firing Rate Correlation')
			#labels=[str(chan) for chan in self.good_channels]
			ax1.set_xticklabels(labels,fontsize=6)
			ax1.set_xticks(range(len(labels)))
			ax1.set_yticklabels(labels,fontsize=6)
			ax1.set_yticks(range(len(labels)))
			# Add colorbar, make sure to specify tick locations to match desired ticklabels
			fig.colorbar(cax, ticks=[.5,.6,.7,.8,.9,1])

			plt_filename = self.filename[:-4] + '_FiringRateCorrelation2.svg'
			plt.savefig(plt_filename)
			plt.close()

		return corr_mat

