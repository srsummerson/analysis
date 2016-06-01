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
			print unit_name
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

def LDAforFeatureSelection(X,y):
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
		print('Mean vector class %s: %s\n' %(cl, mean_vectors[cl]))

	# Compute the scatter matrices

	# Within-class scatter matrix
	S_W = np.zeros((num_features, num_features))
	for cl, mv in zip(range(1,num_features), mean_vectors):
		class_sc_mat = np.zeros((num_features,num_features))  					# scatter matrix for every class
		for row in X[y==cl]:
			row, mv = row.reshape(num_features,1), mv.reshape(num_features,1)  	# make column vectors
			class_sc_mat += (row-mv).dot((row-mv).T)
		S_W += class_sc_mat 													# sum class scatter matrices
	print('Within-class Scatter Matrix:\n', S_W)

	# Between-class scatter matrix
	overall_mean = np.mean(X, axis = 0)

	S_B = np.zeros((num_features,num_features))
	for i, mean_vec in enumerate(mean_vectors):
		n = X[y==i,:].shape[0]
		mean_vec = mean_vec.reshape(num_features,1) 			# make column vector
		overall_mean = overall_mean.reshape(num_features,1)		# make column vector
		S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
	print('Between-class Scatter Matrix:\n', S_B)

	# Solving the generalized eigenvalue problem 

	eig_vals, eig_vecs = np.linalg.eig(np.lingalg.inv(S_W).dot(S_B))

	for i in range(len(eig_vals)):
		eigvec_sc = eig_vecs[:,i].reshape(num_features,1) 	# make column vector
		print('\nEigenvector {}: \n{}'.format(i+1,eigvec_sc.real))
		print('Eigenvalue {:} {:.2e}'.format(i+1,eig_vals[i].real))

	# Selecting linear discriminants

	# Make a list of (eigvalue, eigvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

	print('Eigenvalues in decreasing order:\n')
	for i in eig_pairs:
		print(i[0])

	print('Variance explained:\n')
	eigv_sum = sum(eig_vals)
	for i,j in enumerate(eig_pairs):
		print('eigenvalue {0:} {1: .2%}'.format(i+1, (j[0]/eigv_sum).real))


	return