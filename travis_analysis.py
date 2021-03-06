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
		elif (sc==1)&(chan==32):
			cmap = mpl.cm.terrain

		num_waveforms = float(len(waveform_inds))

		time = np.arange(0,32./40000., 1./40000)
		plt.figure()
		plt.plot(time, mean_waveform, 'k')
		plt.fill_between(time, mean_waveform - std_waveform, mean_waveform + std_waveform, color = 'k', alpha = 0.5, linewidth=0.0)
		for i,ind in enumerate(waveform_inds):
			plt.plot(time, sc_waveform[ind,:], color = cmap(i/num_waveforms))
		plt.plot(time, mean_waveform, color='k', linewidth = 2)
		plt.title('Channel %i - Unit %i' % (chan, sc))
		plt.xlabel('Time (s)')
		plt.ylabel('Voltage (' + r'$\mu$' + 'V)')
		plt.ylim((-50,30))
		plt.text(time[-8],mean_waveform[20],'$V_{rms}=$ %f' % (vrms))
		plt_filename = self.filename[:-4] + '_Chan_' + str(chan) + '_Unit_' + str(sc) + '_ExTraces.svg'
		plt.savefig(plt_filename)
		
		plt.close()

		return sc_waveform, mean_waveform, std_waveform

	def peak_to_peak_vals(self, chan, sc):
		'''
		Finds the peak-to-trough amplitude of each spike waveform and then computes the average.
		'''
		sc_waveform, mean_waveform, std_waveform = self.get_waveform_data(chan, sc, range(50))
		p2p = np.max(sc_waveform, axis = 1) - np.min(sc_waveform, axis = 1)
		avg_p2p = np.mean(p2p)

		return p2p, avg_p2p

	def peak_amp_heatmap(self):

		peaks = np.array([])
		mpowers = np.zeros(32) 	# array for max amplitude of any unit on channel
		powers = np.zeros(32) 	# array for amplitude of first sorted unit on channel
		chan_array = np.array([])
		for chan in self.good_channels:
			sc_chan = self.find_chan_sc(chan)
			avg_peaks = np.array([])
			for sc in sc_chan:
				chan_array = np.append(chan_array, chan)
				p2p, avg_p2p = self.peak_to_peak_vals(chan, sc)
				#peaks = np.append(peaks, p2p)
				avg_peaks = np.append(avg_peaks, avg_p2p)
			mpowers[chan-1] = np.max(avg_peaks)
			powers[chan-1] = avg_peaks[0]
			if (chan == 26)&(self.filename[:-4]=='Travis20180324-2-03'):
				powers[chan-1] = avg_peaks[1]

		mpowers = np.append(mpowers, np.nan)
		powers = np.append(powers, np.nan)  	# add fake 33rd entry as dummy entry for when filling out power matrix
		mpowers[mpowers ==0] = np.nan
		powers[powers ==0] = np.nan
		print len(powers)
		
		power_mat = np.zeros([6,6])
		power_mat[:,:] = np.nan 	# all entries initially nan until they are update with peak powers
		mpower_mat = np.zeros([6,6])
		mpower_mat[:,:] = np.nan

		row_zero = np.array([14, 15, 16, 17, 18, 19])
		row_one = np.array([30, 31, 32, 2, 3, 4])
		row_two = np.array([29, 13, 12, 1, 20, 5])
		row_three = np.array([28, 11, 33, 21, 22, 6])
		row_four = np.array([27, 26, 25, 33, 8, 7])
		row_five = np.array([10, 9, 33, 33, 24, 23])

		chan_mat = np.vstack([row_zero, row_one, row_two, row_three, row_four, row_five])

		power_mat[0,:] = powers[row_zero-1]
		power_mat[1,:] = powers[row_one-1]
		power_mat[2,:] = powers[row_two-1]
		power_mat[3,:] = powers[row_three-1]
		power_mat[4,:] = powers[row_four-1]
		power_mat[5,:] = powers[row_five-1]

		mpower_mat[0,:] = mpowers[row_zero-1]
		mpower_mat[1,:] = mpowers[row_one-1]
		mpower_mat[2,:] = mpowers[row_two-1]
		mpower_mat[3,:] = mpowers[row_three-1]
		mpower_mat[4,:] = mpowers[row_four-1]
		mpower_mat[5,:] = mpowers[row_five-1]

		# distances from position (2,1) for 2-03
		# distances from position (2,2) for 7-02

		if self.filename[:-4] == 'Travis20180324-2-03':
			ref_point = np.array([1,2])  	# originally 2,1
		else:
			ref_point = np.array([5,5]) 	# originally 2,2

		pitch = 38 		# microns

		dist = np.zeros(36)
		amps = np.zeros(36)
		mamps = np.zeros(36)
		counter = 0

		for i in range(6):
			for j in range(6):
				dist[counter] = pitch*np.linalg.norm(np.array([i,j]) - ref_point) 	# dist in microns
				amps[counter] = power_mat[i,j]
				mamps[counter] = mpower_mat[i,j]
				counter += 1

		arg_dist_sort = np.argsort(dist)
		dist_sorted = dist[arg_dist_sort]
		amps_sorted = amps[arg_dist_sort]
		mamps_sorted = mamps[arg_dist_sort]

		dist_unique, dist_inds = np.unique(dist_sorted, True)
		avg_amps = np.zeros(len(dist_unique))
		avg_mamps = np.zeros(len(dist_unique))
		sem_amps = np.zeros(len(dist_unique))
		sem_mamps = np.zeros(len(dist_unique))

		for k, ind in enumerate(dist_inds[:-1]):
			avg_amps[k] = np.nanmean(amps_sorted[ind:dist_inds[i+1]])
			avg_mamps[k] = np.nanmean(mamps_sorted[ind:dist_inds[i+1]])
			sem_amps[k] = np.nanstd(amps_sorted[ind:dist_inds[i+1]])/np.sqrt(len(amps_sorted[ind:dist_inds[i+1]]))
			sem_mamps[k] = np.nanstd(mamps_sorted[ind:dist_inds[i+1]])/np.sqrt(len(mamps_sorted[ind:dist_inds[i+1]]))

		avg_amps[-1] = np.nanmean(amps_sorted[dist_inds[-1]:])
		avg_mamps[-1] = np.nanmean(mamps_sorted[dist_inds[-1]:])

		plt.figure()
		plt.subplot(1,2,1)
		plt.errorbar(dist_unique[1:], avg_amps[1:], yerr = sem_amps[1:], color = 'k', ecolor = 'k')
		plt.xlabel('Distance (um)')
		plt.ylabel('Average Spike Amp - First unit')
		plt.subplot(1,2,2)
		plt.errorbar(dist_unique[1:], avg_mamps[1:], yerr = sem_mamps[1:], color = 'k', ecolor = 'k')
		plt.xlabel('Distance (um)')
		plt.ylabel('Average Spike Amp - Max unit')
		plt_filename = self.filename[:-4] + '_SpikeAmplitudeOverDistance.svg'
		print plt_filename
		plt.savefig(plt_filename)
		plt.close()
		
		
		plt.figure()
		cmap = cm.get_cmap('jet', 30)
		plt.subplot(1,2,1)
		cax = plt.imshow(power_mat, interpolation="nearest", cmap=cmap)
		plt.grid(False)
		plt.title('Avg Spike Amplitude Per Channel - First Sorted Unit')
		#labels=[str(chan) for chan in self.good_channels]
		#ax1.set_xticklabels(labels,fontsize=6)
		#ax1.set_xticks(range(len(labels)))
		#ax1.set_yticklabels(labels,fontsize=6)
		#ax1.set_yticks(range(len(labels)))
		# Add colorbar, make sure to specify tick locations to match desired ticklabels
		#fig.colorbar(cax, ticks=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
		plt.colorbar(cax)

		
		plt.subplot(1,2,2)
		cax2 = plt.imshow(mpower_mat, interpolation="nearest", cmap=cmap)
		plt.grid(False)
		plt.title('Avg Spike Amplitude Per Channel - Max Amplitude Unit')
		#labels=[str(chan) for chan in self.good_channels]
		#ax1.set_xticklabels(labels,fontsize=6)
		#ax1.set_xticks(range(len(labels)))
		#ax1.set_yticklabels(labels,fontsize=6)
		#ax1.set_yticks(range(len(labels)))
		# Add colorbar, make sure to specify tick locations to match desired ticklabels
		#fig.colorbar(cax, ticks=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
		plt.colorbar(cax2)
		

		plt_filename = self.filename[:-4] + '_SpikeAmplitudeHeatMap.svg'
		print plt_filename
		plt.savefig(plt_filename)
		plt.close()
		

		return dist_sorted, amps_sorted, mamps_sorted

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

		#bins_all = np.arange(50,130,10)
		bins_all = np.arange(30,130,10)

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
			#plt.ylim((0,0.3))
			plt.subplot(122)
			plt.bar(bins_avg_center, hist_avg, width_avg)
			plt.xlabel('Peak-to-Trough Values ($\mu$V)')
			plt.ylabel('Fraction of Units')
			plt.title('Mean Waveforms')
			#plt.ylim((0,0.3))

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
	
	def spike_rate_correlation(self, t_resolution):
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

		fig = plt.figure(1)
		ax1 = fig.add_subplot(111)
		cmap = cm.get_cmap('jet', 30)
		cax = plt.imshow(corr_mat, interpolation="nearest", cmap=cmap, vmin = 0.0, vmax = 1.0)
		plt.grid(False)
		plt.title('Firing Rate Correlation')
		#labels=[str(chan) for chan in self.good_channels]
		ax1.set_xticklabels(labels,fontsize=6)
		ax1.set_xticks(range(len(labels)))
		ax1.set_yticklabels(labels,fontsize=6)
		ax1.set_yticks(range(len(labels)))
		# Add colorbar, make sure to specify tick locations to match desired ticklabels
		plt.colorbar(cax, ticks=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

		plt_filename = self.filename[:-4] + '_FiringRateCorrelation.svg'
		plt.savefig(plt_filename)
		
		"""
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		
		cmap = cm.get_cmap('jet', 30)
		cax1 = plt.imshow(corr_mat, interpolation="nearest", cmap=cmap, vmin = 0.5, vmax = 1.0)
		plt.grid(True)
		plt.title('Firing Rate Correlation')
		#labels=[str(chan) for chan in self.good_channels]
		ax2.set_xticklabels(labels,fontsize=6)
		ax2.set_xticks(range(len(labels)))
		ax2.set_yticklabels(labels,fontsize=6)
		ax2.set_yticks(range(len(labels)))
		# Add colorbar, make sure to specify tick locations to match desired ticklabels
		plt.colorbar(cax1, ticks=[.5,.6,.7,.8,.9,1])

		plt_filename = self.filename[:-4] + '_FiringRateCorrelation2.svg'
		plt.savefig(plt_filename)
		"""
		plt.close()
		

		return corr_mat

