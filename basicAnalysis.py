#from neo import io
import numpy as np
import scipy as sp
import sys
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

def contSignalTrialAvg(data, times, window_before, window_after, binsize):
	'''
	Input:
	- data: 2D array, contains data that is Time x Channels
	- times: 1D array, time indices to align trial averages to
	- window_before: float, amount of time (s) to include in time window
	- window_after: float, amount of time (s) to include in time window
	- binsize: float, time length (s) of data bins

	Output: 
	- data_trialavg: 2D array, contains the trial averaged signals that is Time x Channels

	'''
	num_chans = data.shape[1]
	num_bins_before = int(window_before/binsize)
	num_bins_after = int(window_after/binsize)

	window_length = num_bins_before	+ num_bins_after
	data_trialavg = np.empty([int(window_length), int(num_chans)])
	epoch_data = np.empty([int(window_length), len(times)])

	for i in range(num_chans):
		for j,time in enumerate(times):
			epoch_bins = np.arange(time-num_bins_before,time+num_bins_after,1) 
			epoch_data[:,j] = data[epoch_bins,j]
		data_trialavg[:,i] = np.mean(epoch_data, axis = 1)

	return data_trialavg

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
	boxcar_length = 4.
	channels = np.arange(1,161)
	binsize = float(binsize)/1000
	psth_time_window = np.arange(0,window_before+window_after-float(binsize),float(binsize))
	boxcar_window = signal.boxcar(boxcar_length)  # 2 ms before, 2 ms after for boxcar smoothing
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

			psth[unit_name] = psth[unit_name]/float(len(times))
			smooth_psth[unit_name] = np.convolve(psth[unit_name], boxcar_window,mode='same')/boxcar_length

	return psth, smooth_psth, unit_labels

def computePSTH_SingleChannel(spike_file,channel,times,window_before=1,window_after=2, binsize=1):
	'''
	Input:
		- spike_file: sorted spikes for Channels N; spike_file should be the results of 
			plx = plexfile.openFile('filename.plx') and spike_file = plx.spikes[:].data
		- times: time points to align peri-stimulus time histograms to
		- window_before: amount of time before alignment points to include in time window, units in seconds
		- window_after: amount of time after alignment points to include in time window, units in seconds
		- binsize: time length of bins for estimating spike rates, units in milleseconds
	Output:
		- psth: peri-stimulus time histogram over window [window_before, window_after] averaged over trials
		- smooth_psth: psth smoothed using boxcar filter
		- unit_labels: names of units on channel
	'''
	boxcar_length = 4.
	channel = channel
	binsize = float(binsize)/1000
	psth_time_window = np.arange(0,window_before+window_after-float(binsize),float(binsize))
	boxcar_window = signal.boxcar(boxcar_length)  # 2 ms before, 2 ms after for boxcar smoothing
	psth = dict()
	smooth_psth = dict()
	unit_labels = []

	units = [spike[2] for spike in spike_file]
	unit_vals = set(units)  # number of units

	if len(unit_vals) > 0:
		unit_vals.remove(0) 	# value 0 are units marked as noise events

	for unit in unit_vals:
		unit_name = 'Ch'+str(channel) +'_' + str(unit)
		spike_times = [spike[0] for spike in spike_file if (spike[2]==unit)]
		psth[unit_name] = np.zeros(len(psth_time_window))
		unit_labels.append(unit_name)
		
		for time in times:
			epoch_bins = np.arange(time-window_before,time+window_after,float(binsize)) 
			counts, bins = np.histogram(spike_times,epoch_bins)
			psth[unit_name] += counts[0:len(psth_time_window)]/binsize	# collect all rates into a N-dim array

		psth[unit_name] = psth[unit_name]/float(len(times))
		smooth_psth[unit_name] = np.convolve(psth[unit_name], boxcar_window,mode='same')/boxcar_length

	return psth, smooth_psth, unit_labels

def remap_spike_channels(spike_data,channel_mapping):
	'''
	This method takes an array of spike data, the result of plx = plexfile.openFile('filename.plx') and 
	plx.spike[:].data and remaps the channel numbers according to the channel mapping file.

	Inputs:
		- spike_data: an array of spike data, the result of spike_data = plexfile.openFile('filename.plx').spike[:].data
		- channel_mapping: array of floats indicated the new channel numbers, should be extracting from 'filename.txt'
	Outputs:
		- mapped_spiked_data: array of same size as spike_data, but with new channel numbers
	'''
	mapped_spiked_data = spike_data

	for i in range(len(spike_data)):
		mapped_spiked_data[i][1] = channel_mapping[spike_data[i][1]-1]

	return mapped_spiked_data

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
	
	# setting up arrays of zeros as placeholders, picked 200 because it's likely greater than the number of units per recording
	spike_rates = np.zeros(200)
	spike_sem = np.zeros(200)
	unit_labels = []

	epoch_bins = np.arange(t_start,t_end,1)
	num_bins = len(epoch_bins)
	counter = 0
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
			print(unit_name)
			unit_labels.append(unit_name)
			spike_times = [spike[0] for spike in channel_spikes if (spike[2]==unit)]
			counts, bins = np.histogram(spike_times,epoch_bins)
			spike_rates[counter] = np.nanmean(counts)
			spike_sem[counter] = np.nanstd(counts)/float(num_bins)
			counter += 1

	spike_rates = spike_rates[:counter]
	spike_sem = spike_sem[:counter]
	#unit_labels = unit_labels[:counter]

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
	
	return power_mat

def ComputeRSquared(xd,xm):
	'''
	This method computes the coefficient of determination, R^2, for data xd that has been fit with a model resulting in 
	predicted values xm.

	Input:
		- xd: array of data, length n
		- xm: model predictions based on fitting to data, length n
	Output:
		-r_squared: coefficient of determination
	'''
	ss_tot = np.sum((xd - np.nanmean(xd))**2)  	# total sum of squares ~ variance in data
	ss_reg = np.sum((xm - np.nanmean(xd))**2)	# explained sum of squares
	ss_res = np.sum((xd - xm)**2)				# residual sum of squares

	r_squared = 1 - ss_res/float(ss_tot)

	return r_squared

def ComputeEfronRSquared(xd,xm_prob):
	'''
	This method computes Efron's Pseudo-R^2, for data xd that has been fit with a model resulting in 
	predicted values xm. The probability output from the model is xm_prob.

	Input:
		- xd: array of data, length n
		- xm: model predictions based on fitting to data, length n
	Output:
		-r_squared: coefficient of determination
	'''
	ss_tot = np.sum((xd - np.nanmean(xd))**2)  	# total sum of squares ~ variance in data
	ss_res = np.sum((xd - xm_prob)**2)				# residual sum of squares

	r_squared = 1 - ss_res/float(ss_tot)

	return r_squared

def plot_step_lda(X_lda,y,labels):
	num_labels = len(np.unique(y))
	ax = plt.subplot(111)
	for label, marker, color in zip(range(0,num_labels), ('^','s','o'),('k','r','b')):
		plt.scatter(x=X_lda[:,0].real[y==label],
			y=X_lda[:,1].real[y==label],
			marker=marker,
			color=color,
			alpha=0.5,
			label=labels[label]
			)
	plt.xlabel('LD1')
	plt.ylabel('LD2')

	leg = plt.legend(loc='upper right', fancybox = True)
	leg.get_frame().set_alpha(0.5)
	plt.title('LDA: Data projection onto the first 2 linear discriminants')

	# hide axis ticks
	plt.tick_params(axis='both', which='both', bottom = 'off', top = 'off', labelbottom='on', left='off', right = 'off', labelleft = 'on')

	# remove axis spines
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["bottom"].set_visible(False)

	plt.grid()
	plt.tight_layout()

	return

def lowpassFilterData(data, Fs, cutoff):
	'''
	This method lowpass filters data using a butterworth filter.

	Inputs:
		- data: array of time-stamped values to be filtered 
		- Fs: sampling frequency of data 
		- cutoff: cutoff frequency of the LPF in Hz
	Outputs:
		- filtered_data: array contained the lowpass-filtered data

	'''
	nyq = 0.5*Fs
	order = 5
	normal_cutoff = cutoff / nyq

	b, a = signal.butter(order, normal_cutoff, btype= 'low', analog = False)
	filtered_data = signal.filtfilt(b,a,data)

	return filtered_data

def highpassFilterData(data, Fs, cutoff):
	'''
	This method highpass filters data using a butterworth filter.

	Inputs:
		- data: array of time-stamped values to be filtered 
		- Fs: sampling frequency of data 
		- cutoff: cutoff frequency of the LPF in Hz
	Outputs:
		- filtered_data: array contained the lowpass-filtered data

	'''
	nyq = 0.5*Fs
	order = 5
	normal_cutoff = cutoff / nyq

	b, a = signal.butter(order, normal_cutoff, btype= 'highpass', analog = False)
	filtered_data = signal.filtfilt(b,a,data)

	return filtered_data

def bandpassFilterData(data, Fs, cutoff_low, cutoff_high):
	'''
	This method lowpass filters data using a butterworth filter.

	Inputs:
		- data: array of time-stamped values to be filtered 
		- Fs: sampling frequency of data 
		- cutoff: cutoff frequency of the LPF in Hz
	Outputs:
		- filtered_data: array contained the lowpass-filtered data

	'''
	nyq = 0.5*Fs
	order = 2
	cutoff_low = cutoff_low / nyq
	cutoff_high = cutoff_high / nyq

	b, a = signal.butter(order, [cutoff_low, cutoff_high], btype= 'bandpass', analog = False)
	filtered_data = signal.filtfilt(b,a,data)

	return filtered_data

def notchFilterData(data, Fs, notch_freq):
	'''
	This method lowpass filters data using a butterworth filter.

	Inputs:
		- data: array of time-stamped values to be filtered 
		- Fs: sampling frequency of data 
		- cutoff: cutoff frequency of the LPF in Hz
	Outputs:
		- filtered_data: array contained the lowpass-filtered data

	'''
	nyq = 0.5*Fs
	order = 2
	notch_start = (notch_freq - 1) / nyq
	notch_stop = (notch_freq + 1) / nyq

	b, a = signal.butter(order, [notch_start, notch_stop], btype= 'bandstop', analog = False)
	filtered_data = signal.filtfilt(b,a,data)

	return filtered_data

def LDAforFeatureSelection(X,y,filename,block_num):
	'''
	This method performs linear discriminant analysis on the data (X,y) and computes the variances explained by the 
	features represented in the X matrix. This is based on: http://sebastianraschka.com/Articles/2014_python_lda.html#principal-component-analysis-vs-linear-discriminant-analysis
	
	Input:
		- X: data matrix of shape (num_samples, num_features)
		- y: class labels for each sample row in X (note: this version assumes classes are numerical and start at 0)

	'''
	classes = np.unique(y)
	num_classes = len(classes)
	num_samples, num_features = X.shape
	
	# Compute the num-features-dimensional mean vectors of the different classes
	mean_vectors = []
	for cl in classes:
		mean_vectors.append(np.nanmean(X[y==cl], axis=0))
		#print('Mean vector class %s: %s\n' %(cl, mean_vectors[int(cl)]))

	# Compute the scatter matrices

	# Within-class scatter matrix
	S_W = np.zeros((num_features, num_features))
	for cl, mv in zip(range(0,num_features), mean_vectors):
		class_sc_mat = np.zeros((num_features,num_features))  					# scatter matrix for every class
		for row in X[y==cl]:
			row, mv = row.reshape(num_features,1), mv.reshape(num_features,1)  	# make column vectors
			class_sc_mat += (row-mv).dot((row-mv).T)
		S_W += class_sc_mat 													# sum class scatter matrices
	#print('Within-class Scatter Matrix:\n', S_W)

	# Between-class scatter matrix
	overall_mean = np.nanmean(X, axis = 0)

	S_B = np.zeros((num_features,num_features))
	for i, mean_vec in enumerate(mean_vectors):
		n = X[y==i,:].shape[0]
		mean_vec = mean_vec.reshape(num_features,1) 			# make column vector
		overall_mean = overall_mean.reshape(num_features,1)		# make column vector
		S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
	#print('Between-class Scatter Matrix:\n', S_B)

	# Solving the generalized eigenvalue problem 
	
	eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

	for i in range(len(eig_vals)):
		eigvec_sc = eig_vecs[:,i].reshape(num_features,1) 	# make column vector
		#print('\nEigenvector {}: \n{}'.format(i+1,eigvec_sc.real))
		#print('Eigenvalue {:} {:.2e}'.format(i+1,eig_vals[i].real))

	# Selecting linear discriminants

	# Make a list of (eigvalue, eigvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
	eig_pairs_ind = sorted(range(len(eig_vals)), key=lambda k: eig_vals[k], reverse=True)

	#print('Eigenvalues in decreasing order:\n')
	#for ind, i in enumerate(eig_pairs):
		#print(i[0], eig_pairs_ind[ind])

	#print('Variance explained:\n')
	eigv_sum = sum(eig_vals)
	#for i,j in enumerate(eig_pairs):
		#print('eigenvalue {0:} {1: .2%}'.format(i+1, (j[0]/eigv_sum).real))

	# Choosing 2 eigenvectors with largest eigenvalues
	W = np.hstack((eig_pairs[0][1].reshape(num_features,1), eig_pairs[1][1].reshape(num_features,1)))
	#print('Matrix W:\n', W.real)

	# Transform the samples onto the new subspace
	X_lda = X.dot(W)

	plot_step_lda(X_lda,y,['Reg','Stress'])
	#plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_LDA.svg')
	#plt.close()
	plt.show()
	return W

def computeCursorPathLength(start_times,stop_times,cursor):
	'''
	Computes the cursor path lengths for trajectories beginning at times in start_times array and ending at stop_times

	Input: 
		- start_times: array of hdf row numbers (at 60 Hz rate) for beginning of trajectory
		- end_times: array of hdf row numbers (at 60 Hz rate) for end of trajectory
		- cursor: 3XN array of cursor position, corresponding to cursor = hdf.root.task[:]['cursor']
	Output:
		- traj_length: array of trajectory lengths, same shape as start_times and end_times arrays
	'''
	traj_length = np.zeros(len(start_times))
	for j, time in enumerate(start_times):
		row_nums = np.arange(time,stop_times[j]) 		# row numbers that occur during the course of the path trajectory
		traj_length[j] = np.sum(np.sqrt(np.sum((cursor[row_nums[1:]] - cursor[row_nums[:-1]])**2, axis=1)))

	return traj_length

def plot_raster(event_times_list, color='k'):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable, a list of event time iterables
    color : string, color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax



def computeFisherScore(data, class_ass, nb_classes):
	'''
	The Fisher Score assigns a rank to each of the features, with the goal of finding the subset of features of the data
	such that in the data space spanned by the selected features, the distance between data points in different classes are
	as large as possible and the distance between data points in the same class are as small as possible.

	Input
		- data: matrix of inputs, size N x M, where N is the number of trials and M is the number of features
		- class_ass: array of class assignments, size 1 x N, where N is the number of trials
		- nb_classes: number of classes
	Output
		- Fscores: array of scores, size 1 x M, for each of the features
	'''
	num_trials, num_features = data.shape
	within_class_mean = np.zeros([nb_classes,num_features]) 	# mean for each feature within each class
	within_class_var = np.zeros([nb_classes,num_features]) 		# variance for each feature within each class
	num_points_within_class = np.zeros([1,nb_classes])  			# number of points within each class 
	
	for i in range(nb_classes):
		in_class = np.ravel(np.nonzero(class_ass == i))
		num_points_within_class[0,i] = len(in_class)
		class_data = data[in_class,:]  	# extract trails classified as belonging to this class
		within_class_mean[i,:] = np.nanmean(class_data, axis=0)  # length of mean vector should be equal to M, the number of features
		within_class_var[i,:] = np.nanvar(class_data,axis=0)

	between_class_mean = np.asmatrix(np.mean(within_class_mean,axis=0))
	between_class_mean = np.dot(np.ones([nb_classes,1]), between_class_mean)

	Fscores = np.dot(num_points_within_class,np.square(within_class_mean - between_class_mean))/np.dot(num_points_within_class,within_class_var)

	return Fscores

def getConnectionWeightsMLP(net, num_in, num_hidden, num_out):
	'''
	This method extracts the connection weights for a multilayer perceptron network (feed forward neural network) with
	one hidden layer. With N inputs, L hidden units, and M output units, it assembles a matrix with the connection 
	weights for the input to the hidden layer and a separate matrix with the connection weights for the hidden layer
	and out units.

	Inputs
		- net: network object created using PyBrain library
	Outputs
		- W: N x L matrix of connection weights between the nth input and the lth unit in the hidden layer
		- V: L x M matrix of connection weights between the lth unit in the hidden layer and the mth unit 
		     of the output layer
	'''
	W = np.array([])
	V = np.array([])
	for mod in net.modules:
		for conn in net.connections[mod]:
			conn_name = str(conn)
			print(conn_name[-19:])
			if (conn_name[-19:] == "'hidden0' -> 'out'>"):
				# fill in weights from hidden layer to output layer
				weights = conn.params 
				V = np.reshape(weights, (num_out, num_hidden))
				V = V.T

			if (conn_name[-18:] == "'in' -> 'hidden0'>"):
				# fill in weights from input layer to hidden layer
				weights = conn.params
				W = np.reshape(weights, (num_hidden, num_in))
				W = W.T

	return W, V

def variableImportanceMLP(input_to_hidden_weights, hidden_to_output_weights):
	'''
	This method useing Garson's algorithm to determine the relative importance of inputs to a multilayer perceptron network (feed forward neural 
	network) with one hidden layer. The relative importance Q values are percentages that should add up to 100%.

	Inputs
		- input_to_hidden_weights: N x L matrix of weights of the connections between the N input features and the M hidden units 
		- hidden_to_output_weights: L x M matrix of weights of the connections between the M hidden units and the L output units
	Output
		- Qrelimport: N x L matrix contain the relative importance of each of the N inputs features to the L output units
	'''

	N, L = input_to_hidden_weights.shape
	M = hidden_to_output_weights.shape[1]

	W = np.abs(input_to_hidden_weights)
	V = np.abs(hidden_to_output_weights)

	sum_over_inputs = np.sum(W,axis=0)  # vector of length L
	score = np.ndarray(shape = (N, L, M))
	for i in range(N):
		for j in range(L):
			for k in range(M):
				score[i,j,k] = W[i,j]*V[j,k]/sum_over_inputs[j]

	Q_partial = np.sum(score,axis=1) 	# matrix of size N x M
	Q_partial_sum_over_inputs = np.sum(Q_partial,axis=0) # matrix of size M
	Q_partial_sum_over_inputs = np.tile(Q_partial_sum_over_inputs, (N,1)) 	# matrix of size N x M
	Qrelimport = Q_partial/Q_partial_sum_over_inputs

	return Qrelimport


def get_HDFstate_TDT_LFPsamples(ind_state,state_time,syncHDF_file):
		'''
		This method finds the TDT sample numbers that correspond to indicated task state using the syncHDF.mat file.

		Inputs:
			- ind_state: array with state numbers corresponding to which state we're interested in finding TDT sample numbers for, e.g. self.ind_hold_center_states
			- state_time: array of state times taken from corresponding hdf file
			- syncHDF_file: syncHDF.mat file path, e.g. '/home/srsummerson/storage/syncHDF/Mario20161104_b1_syncHDF.mat'
		Output:
			- lfp_state_row_ind: array of tdt sample numbers that correspond the the task state events in ind_state array
		'''
		# Load syncing data
		hdf_times = dict()
		sp.io.loadmat(syncHDF_file, hdf_times)
		hdf_rows = np.ravel(hdf_times['row_number'])
		hdf_rows = [val for val in hdf_rows]
		dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
		dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

		lfp_dio_sample_num = dio_tdt_sample  # assumes DIOx and LFPx are saved using the same sampling rate

		state_row_ind = state_time[ind_state]		# gives the hdf row number sampled at 60 Hz
		lfp_state_row_ind = np.zeros(state_row_ind.size)

		for i in range(len(state_row_ind)):
			hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind[i]))
			if np.abs(hdf_rows[hdf_index] - state_row_ind[i])==0:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
			elif hdf_rows[hdf_index] > state_row_ind[i]:
				hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
				m = (lfp_dio_sample_num[hdf_index]-lfp_dio_sample_num[hdf_index - 1])/hdf_row_diff
				b = lfp_dio_sample_num[hdf_index-1] - m*hdf_rows[hdf_index-1]
				lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
			elif (hdf_rows[hdf_index] < state_row_ind[i])&(hdf_index + 1 < len(hdf_rows)):
				hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
				if (hdf_row_diff > 0):
					m = (lfp_dio_sample_num[hdf_index + 1] - lfp_dio_sample_num[hdf_index])/hdf_row_diff
					b = lfp_dio_sample_num[hdf_index] - m*hdf_rows[hdf_index]
					lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
				else:
					lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
			else:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]

		return lfp_state_row_ind