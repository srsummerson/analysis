from plexon import plexfile
from neo import io
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import signal

def computeSTA(spike_file,tdt_signal,channel,t_start,t_stop):
	'''
	Compute the spike-triggered average (STA) for a specific channel overa  designated time window
	[t_start,t_stop].

	spike_file should be the results of plx = plexfile.openFile('filename.plx') and spike_file = plx.spikes[:].data
	tdt_signal should be the array of time-stamped values just for this channel
	'''
	channel_spikes = [entry for entry in spike_file if (t_start <= entry[0] <= t_stop)&(entry[1]==channel)]
	units = [spike[2] for spike in channel_spikes]
	unit_vals = set(units)  # number of units
	unit_vals.remove(0) 	# value 0 are units marked as noise events
	unit_sta = dict()

	tdt_times = np.ravel(tdt_signal.times)
	tdt_data = np.ravel(tdt_signal)

	for unit in unit_vals:
		
		spike_times = [spike[0] for spike in channel_spikes if (spike[2]==unit)]
		start_avg = [(time - 1) for time in spike_times] 	# look 1 s back in time until 1 s forward in time from spike
		stop_avg = [(time + 1) for time in spike_times]
		epoch = np.logical_and(np.greater(tdt_times,start_avg[0]),np.less(tdt_times,stop_avg[0]))
		epoch_inds = np.ravel(np.nonzero(epoch))
		len_epoch = len(epoch_inds)
		sta = np.zeros(len_epoch)
		num_spikes = len(spike_times)
		for i in range(0,num_spikes):
			epoch = np.logical_and(np.greater(tdt_times,start_avg[i]),np.less(tdt_times,stop_avg[i]))
			epoch_inds = np.ravel(np.nonzero(epoch))
			if (len(epoch_inds) == len_epoch):
				sta += tdt_data[epoch_inds]
		unit_sta[unit] = sta/float(num_spikes)

	return unit_sta

def computePSTH(spike_file1,spike_file2,times,window_before=1,window_after=2, binsize=1):
	'''
	Input:
		- spike_file1: sorted spikes for Channels 1 - 96
		- spike_file2: sorted spikes for Channels 97 - 160
		- times: time points to align peri-stimulus time histograms to
		- window_before: amount of time before alignment points to include in time window, units in seconds
		- window_after: amount of time after alignment points to include in time window, units in seconds
		- binsize: time length of bins for estimating spike rates, units in milleseconds
	Output:
		- psth: peri-stimulus time histogram over window [window_before, window_after] averaged over trials
	'''
	channels = np.arange(1,161)
	binsize = float(binsize)/1000
	psth_time_window = np.arange(0,window_before+window_after-float(binsize),float(binsize))
	boxcar_window = signal.boxcar(4)  # 2 ms before, 2 ms after for boxcar smoothing
	psth = dict()
	smooth_psth = dict()
	unit_labels = []

	for channel in channels:
		if channel < 97: 
			channel_spikes = [entry for entry in spike_file1 if (entry[1]==channel)]
		else:
			channel2 = channel % 96
			channel_spikes = [entry for entry in spike_file2 if (entry[1]==channel2)]
		units = [spike[2] for spike in channel_spikes]
		unit_vals = set(units)  # number of units
		if len(unit_vals) > 0:
			unit_vals.remove(0) 	# value 0 are units marked as noise events

		for unit in unit_vals:
			unit_name = 'Ch'+str(channel) +'_' + str(unit)
			spike_times = [spike[0] for spike in channel_spikes if (spike[2]==unit)]
			psth[unit_name] = np.zeros(len(psth_time_window))
			unit_labels.append(unit_name)
			
			for time in times:
				epoch_bins = np.arange(time-window_before,time+window_after,float(binsize)) 
				counts, bins = np.histogram(spike_times,epoch_bins)
				psth[unit_name] += counts[0:len(psth_time_window)]/binsize	# collect all rates into a N-dim array

			psth[unit_name] = psth[unit_name]/len(times)
			smooth_psth[unit_name] = np.convolve(psth[unit_name], boxcar_window,mode='same')

	return psth, smooth_psth, unit_labels

def computeSpikeRatesPerChannel(spike_file1,spike_file2,t_start,t_end):
	'''
	Input:
		- spike_file1: sorted spikes for Channels 1 - 96
		- spike_file2: sorted spikes for Channels 97 - 160
		- t_start: time window start in units of seconds
		- t_end: time window end in units of seconds
	Output:
		- spike_rates: an array of length equal to the number of units, containing the spike rate of each channel
		- spike_sem: an array of length equal to the number of units, containing the SEM for the spike rate of each channel
	'''
	channels = np.arange(1,161)
	
	spike_rates = []
	spike_sem = []
	unit_labels = []

	epoch_bins = np.arange(t_start,t_end,1)
	num_bins = len(epoch_bins)

	for channel in channels:
		if channel < 97: 
			channel_spikes = [entry for entry in spike_file1 if (t_start <= entry[0] <= t_end)&(entry[1]==channel)]
		else:
			channel2 = channel % 96
			channel_spikes = [entry for entry in spike_file2 if (t_start <= entry[0] <= t_end)&(entry[1]==channel2)]
		units = [spike[2] for spike in channel_spikes]
		unit_vals = set(units)  # number of units
		if len(unit_vals) > 0:
			unit_vals.remove(0) 	# value 0 are units marked as noise events

		for unit in unit_vals:
			unit_name = 'Ch'+str(channel) +'_' + str(unit)
			print unit_name
			unit_labels.append(unit_name)
			spike_times = [spike[0] for spike in channel_spikes if (spike[2]==unit)]
			counts, bins = np.histogram(spike_times,epoch_bins)
			spike_rates.append(np.nanmean(counts))
			spike_sem.append(np.nanstd(counts)/float(num_bins))

	return spike_rates, spike_sem, unit_labels

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

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
	"""
	Plots an `nstd` sigma ellipse based on the mean and covariance of a point
	"cloud" (points, an Nx2 array).
	Parameters
	----------
	points : An Nx2 array of the data points.
	nstd : The radius of the ellipse in numbers of standard deviations.
	Defaults to 2 standard deviations.
	ax : The axis that the ellipse will be plotted on. Defaults to the
	current axis.
	Additional keyword arguments are pass on to the ellipse patch.
	Returns
	-------
	A matplotlib ellipse artist
	"""
	pos = points.mean(axis=0)
	cov = np.cov(points, rowvar=False)
	return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
'''
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
	"""
	Plots an `nstd` sigma error ellipse based on the specified covariance
	matrix (`cov`). Additional keyword arguments are passed on to the
	ellipse patch artist.
	Parameters
	----------
	cov : The 2x2 covariance matrix to base the ellipse on
	pos : The location of the center of the ellipse. Expects a 2-element
	sequence of [x0, y0].
	nstd : The radius of the ellipse in numbers of standard deviations.
	Defaults to 2 standard deviations.
	ax : The axis that the ellipse will be plotted on. Defaults to the
	current axis.
	Additional keyword arguments are pass on to the ellipse patch.
	Returns
	-------
	A matplotlib ellipse artist
	"""
	def eigsorted(cov):
		vals, vecs = np.linalg.eigh(cov)
		order = vals.argsort()[::-1]
		return vals[order], vecs[:,order]

	if ax is None:
		ax = plt.gca()

	vals, vecs = eigsorted(cov)
	theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
	# Width and height are "full" widths, not radius
	width, height = 2 * nstd * np.sqrt(vals)
	ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
	ax.add_artist(ellip)
	return ellip
'''
def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)
    return ellip

def ElectrodeGridMat(powers):
	'''
	Input: 
		- powers: array of peak powers, length equal to the number of channels
	Output:
		- peak powers arranged in a matrix according to their position in the semi-chronic array
	'''
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

	power_mat[0,0:8] = powers[row_zero]
	power_mat[1,0:8] = powers[row_one]
	power_mat[2,0:9] = powers[row_two]
	power_mat[3,0:9] = powers[row_three]
	power_mat[4,0:10] = powers[row_four]
	power_mat[5,0:10] = powers[row_five]
	power_mat[6,0:11] = powers[row_six]
	power_mat[7,0:11] = powers[row_seven]
	power_mat[8,0:12] = powers[row_eight]
	power_mat[9,0:12] = powers[row_nine]
	power_mat[10,0:13] = powers[row_ten]
	power_mat[11,1:13] = powers[row_eleven]
	power_mat[12,2:14] = powers[row_twelve]
	power_mat[13,3:14] = powers[row_thirteen]
	power_mat[14,4:13] = powers[row_fourteen]
	
	return power_mat
