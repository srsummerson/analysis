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
			channel_spikes = [entry for entry in spike_file2 if (entry[1]==channel)]
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
			channel_spikes = [entry for entry in spike_file2 if (t_start <= entry[0] <= t_end)&(entry[1]==channel)]
		units = [spike[2] for spike in channel_spikes]
		unit_vals = set(units)  # number of units
		if len(unit_vals) > 0:
			unit_vals.remove(0) 	# value 0 are units marked as noise events

		for unit in unit_vals:
			unit_name = 'Ch'+str(channel) +'_' + str(unit)
			unit_labels.append(unit_name)
			spike_times = [spike[0] for spike in channel_spikes if (spike[2]==unit)]
			counts, bins = np.histogram(spike_times,epoch_bins)
			spike_rates.append(np.nanmean(counts))
			spike_sem.append(np.nanstd(counts)/float(num_bins))

	return spike_rates, spike_sem, unit_labels

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
