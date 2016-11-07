import numpy as np
from numpy import linalg
from numpy import fft
import matplotlib.pyplot as plt

class LFPSignal():
	'''
	Object to contain raw signal data, and generate basic time and frequency domain information about this signal. This information is used in the 
	filter_LFP method.
	'''

	def __init__(self, raw_signal, sampling_rate, number_points_time_domain, output_type):
		self.raw_signal =  raw_signal
		self.sampling_rate = sampling_rate
		self.number_points_time_domain = number_points_time_domain
		maxpower2 = 2**23

		if number_points_time_domain < maxpower2:
			self.number_points_frequency_domain = 2.**np.ceil(np.log2(number_points_time_domain)) # fixed parameter for computational ease
		else:
			self.number_points_frequency_domain = number_points_time_domain

		self.time_step_size = 1./self.sampling_rate
		self.frequency_step_size = sampling_rate/float(self.number_points_frequency_domain)
		self.time_support = np.float32(self.time_step_size*np.arange(self.number_points_time_domain))

		frequency_support = self.frequency_step_size*np.arange(self.number_points_frequency_domain)
		inds = np.ravel(np.nonzero(np.greater(frequency_support, self.sampling_rate/2.)))
		frequency_support[inds] -= self.sampling_rate
		self.frequency_support = np.float32(frequency_support)
		frequency_domain = fft.fft(self.raw_signal, int(self.number_points_frequency_domain))

		if np.sum(np.abs(np.imag(self.raw_signal))) != 0: 	# signal already complex
		    self.time_domain = raw_signal 					# do not change
		elif output_type == 'real':
		    self.time_domain = raw_signal
		elif output_type == 'analytic':
		    time_domain = fft.ifft(frequency_domain, self.number_points_frequency_domain)
		    self.time_domain = time_domain[:self.number_points_time_domain]
		    # this step scales analytic signal such that real(analytic_signal)=raw_signal, but note that analytic signal energy is double that of raw signal energy
		    # sum(abs(raw_signal).^2)=0.5*sum(abs(s.time_domain).^2)
		    less_than_inds = np.ravel(np.nonzero(self.frequency_support < 0))
		    greater_than_inds = np.ravel(np.nonzero(self.frequency_support > 0))
		    frequency_domain[less_than_inds] = 0
		    frequency_domain[greater_than_inds] = 2*frequency_domain[greater_than_inds]

		self.frequency_domain = frequency_domain

	def filter_LFP(self, freq_band, normalize, sampling_rate):
		'''
		This method filters the LFP signal using a chirplet transfrom in a designated frequency band.
		The fractional bandwith used for each center frequency is 20 percent of the center frequency.

		Inputs:
			- freq_band: array of length 2 with values [minFreq, maxFreq]
			- normalize: indicator of whether the power is to be normalized by the average power over time 
			- sampling_rate: sampling rate 
		Ouputs:
			- power: array of N x T entries with the power at each frequency over time, N is the number of samples in
					 the frequency domain and T is the number of samples in the time domain
			- cf_list: array of length N that contains the center frequencies used in the analysis
			- phase: array of size N x T that contains the instantaneus phase for each frequency as a function of time
		'''

		minimum_frequency=freq_band[0] # Hz
		maximum_frequency=freq_band[1] # Hz
		number_of_frequencies=50
		minimum_frequency_step_size=0.75
		
		cf_list = make_center_frequencies(minimum_frequency, maximum_frequency, number_of_frequencies, minimum_frequency_step_size)

		tfmat = np.zeros([number_of_frequencies,self.number_points_time_domain]) + 0j  # cast as complex array
		
		for f in range(number_of_frequencies):
		    g = ChirpletFilter(cf_list[f], 0.0, 0.2, 0.0)
		    g.make_chirplet(self)
		    fs = g.filter_with_chirplet(self)
		    
		    tfmat[f,:] = fs['time_domain']
		    
		power = np.abs(tfmat)**2.
		if normalize:
			# divides each frequency by the mean for that frequency
			num_freq, num_samples = power.shape
			mean_power = np.mean(power,axis = 1)
			mean_power = np.tile(mean_power, (num_samples,1)).T 	# make matrix of same size with mean power repeated
			power = power/mean_power

		# Get phase in radians
		phase = np.angle(tfmat, deg = 0)  

		return power, cf_list, phase


class ChirpletFilter():
	'''
	Object to contain Chriplet transform information. This information is used in the filter_with_chirplet method.
	'''

	def __init__(self, center_frequency, center_time, fractional_bandwidth, chirp_rate):
		self.center_frequency = center_frequency
		self.center_time = center_time
		self.fractional_bandwidth = fractional_bandwidth
		self.chirp_rate = chirp_rate

		t0 = self.center_time
		v0 = self.center_frequency
		c0 = self.chirp_rate

		fbw = self.fractional_bandwidth
		s0 = np.log((2.*np.log(2.))/((fbw**2.)*np.pi*(v0**2.)))
		self.duration_parameter = s0

		# Fixed parameters:
		self.std_multiple_for_support = 6.0
		# Calculate other quanties as a function of these constants
		self.time_domain_standard_deviation = np.sqrt(np.exp(s0)/(4.*np.pi))
		self.frequency_domain_standard_deviation = np.sqrt((np.exp(-s0)+c0**2.*np.exp(s0))/(4.*np.pi))
		self.time_frequency_covariance = (c0*np.exp(s0))/(4.*np.pi)
		self.time_frequency_correlation_coefficient = self.time_frequency_covariance*(self.time_domain_standard_deviation*self.frequency_domain_standard_deviation)**(-1.)
		self.fractional_bandwidth = 2.*np.sqrt(2.*np.log(2.))*self.frequency_domain_standard_deviation/v0 # fwhm/center_frequency

	def make_chirplet(self, signal):
		'''
		Method creates the chirplet used to extract power from the input signal.

		Inputs:
			- signal: LFPSignal structure containing fields describing basic signal parameters (sampling rate in time domain, number
				  of points in the time domain)

		'''
		# Use short variable names for equations
		t0 = self.center_time
		v0 = self.center_frequency
		s0 = self.duration_parameter
		c0 = self.chirp_rate
		tstd = self.time_domain_standard_deviation
		vstd = self.frequency_domain_standard_deviation

		time_support = signal.time_support
		# time support: shift g['center_time'] to fall on sp.time_support
		if (self.center_time < time_support[0]) or (self.center_time > time_support[-1]): 
		    self.center_time = self.center_time % time_support[-1]

		temp1 = np.abs(time_support-self.center_time)
		center_index = np.argmin(temp1) # sp.time_support index closest to g.center_time

		inds = np.arange(0, tstd*self.std_multiple_for_support, signal.time_step_size)
		numinds= len(inds)

		support_inds = center_index + np.arange(-numinds,numinds)
		support_inds = support_inds % signal.number_points_time_domain
		nonzero_support_inds = np.ravel(np.nonzero(np.equal(support_inds, 0)))
		support_inds[nonzero_support_inds] = signal.number_points_time_domain
		self.signal_time_support_indices = support_inds

		signal_time_support_indices = self.signal_time_support_indices
		t = time_support[center_index] + signal.time_step_size*np.arange(-numinds,numinds)
		self.ptime = t

		# Chirplet in time domain:
		self.time_domain = 2.**(1./4)*np.exp(-s0/4.)*np.exp(-np.exp(-s0)*np.pi*(t-t0)**2.0)*np.exp(2.*np.pi*1j*v0*(t-t0))*np.exp(np.pi*1j*c0*(t-t0)**2.)    					
		self.time_domain = self.time_domain/linalg.norm(self.time_domain) 	# need to normalize due to discrete sampling

		v = signal.frequency_support # in Hz
		freq_nonzero_support_inds = np.ravel(np.nonzero(np.logical_and(np.less_equal(v0-self.std_multiple_for_support*vstd, v),
										np.greater_equal(v0+self.std_multiple_for_support*vstd, v))))
		self.signal_frequency_support_indices = freq_nonzero_support_inds
		
		# Shorten to include only chirplet support
		v = v[freq_nonzero_support_inds]
		self.frequency_support = v
		self.pfrequency = self.frequency_support

		# Chirplet in frequency domain: 
		Gk = (2.**(1./4))*np.sqrt(-1j*c0+np.exp(-s0))**(-1.)*np.exp(-s0/4.+ (np.exp(s0)*np.pi*(v-v0)**2.0)/(-1+1j*c0*np.exp(s0)))
		n1 = np.sqrt(signal.number_points_frequency_domain)/linalg.norm(Gk)
		Gk = n1*Gk 		# because of discrete sampling and different time/freq sample numbers
		self.filter = Gk 	# at center time of zero, use this for convolution filtering
		self.frequency_domain = Gk*np.exp(-2.*np.pi*1j*v*t0) # translation in time to tk

		return

	def filter_with_chirplet(self, signal):
		'''
		Method to filter input signal with chirplet. 

		Inputs:
			- signal: LFPSignal structure containing fields describing basic signal parameters and signal data 
		Output:
			- fs: dictionary with two entries for filtered signal in time domain and frequency domain 
		'''

		# Initialize arrays
		fs = dict()
		fs_time_domain = np.zeros(len(signal.time_domain))
		fs_frequency_domain = np.zeros(len(signal.frequency_domain)) + 0j

		# Compute frequency domain samples
		frequency_domain = signal.frequency_domain
		signal_frequency_support_indices = self.signal_frequency_support_indices
		number_points_frequency_domain = signal.number_points_frequency_domain
		number_points_time_domain = signal.number_points_time_domain
		fs_frequency_domain[signal_frequency_support_indices] = frequency_domain[signal_frequency_support_indices]*self.filter
		fs['frequency_domain'] = fs_frequency_domain

		# Compute time domain samples
		fs_time_domain=fft.ifft(fs_frequency_domain, number_points_frequency_domain)
		fs['time_domain'] =fs_time_domain[:number_points_time_domain]

		return fs


def make_center_frequencies(minimum_frequency, maximum_frequency,number_of_frequencies,minimum_frequency_step_size):
	'''
	Generates array of frequencies used with chirplet method of extracting powers.

	Inputs:
		- minimum_frequency: number indicating min frequency considered (in Hz)
		- maximum_frequency: number indicating max frequency considered (in Hz)
		- number_of_frequencies: number of center frequencies, determines the resolution in the frequency domain
		- minimum_frequency_step_size: number indicating the step size between center frequencies 
	Output:
		- center_frequencies: array of length C containing the center frequencies where C is equal to the input
							variable number_of_frequencies

	'''

	temp1 = np.arange(0, number_of_frequencies*minimum_frequency_step_size, minimum_frequency_step_size)
	temp2 = np.logspace(np.log10(minimum_frequency), np.log10(maximum_frequency), number_of_frequencies)
	temp2 = (temp2 - temp2[0])*((temp2[-1] - temp1[-1])/temp2[-1]) + temp2[0]
	center_frequencies=temp1+temp2

	return center_frequencies



def make_spectrogram(lfp, sampling_rate, fmax, trialave, makeplot, num_begin_pad, num_end_pad):
	'''
	This a method for making spectrograms using Ryan Canolty's technique for extracting powers using chirplets.
	Methods have been adapted by Erin Rich.

	Inputs:
		- lfp: array of N x T, where N is the number of trials and T is the number of time points. lfp data is broadband.
		- sampling_rate: sampling frequency
		- fmax: maximum frequency to be used in the spectrogram
		- trialave: binary input indicating whether to average over trials
		- makeplot: binary input indicating whether to create a plot at the end. If makeplot == True, it will be
					trial-averaged
		- num_begin_pad: number of time domain samples padded at the beginning of the lfp data in order to deal with edge effects, these samples will not be plotted in final figure
		- num_end_pad: number of time domain samples padded the end of the lfp data in order to deal with edge effects, these samples will not be plotted in final figure
	Outputs:
		- powers: array of N x M x T, where N is the number of trials, M is the number of frequency domain
		points, and T is the number of time domain points
		- Power: if trialave==0: array of N x M x T, where N is the number of trials, M is the number of frequency domain
		points, and T is the number of time domain points, if trialave ==1: array of M x T, 
		where M is the number of points in the frequency domain and T is the number of time points.
		- cf_list: list of the center frequencies for which power was computed

	'''
	
	# Define variables
	num_trials, num_samples = lfp.shape

	freq_band=np.zeros(2)
	freq_band[0]= 1.
	freq_band[1]= fmax

	powers = np.empty([num_trials,50,num_samples])
	powers[:] = np.NAN
	
	# Iterate over trials to find power
	for k in range(num_trials):	
		signal = LFPSignal(lfp[k,:], sampling_rate, len(lfp[k,:]), 'analytic')
		power, cf_list, phase = signal.filter_LFP(freq_band,1,sampling_rate)
		powers[k,:,:] = 10*np.log10(power)  # dB scale

	# Compute trial-average power (if trialave is True)
	if bool(trialave):
		Power = np.nanmean(powers, axis = 0)
	else:
		Power = powers
	
	# Plot results. Power is only shown for the time domain sample points past the begin pad points (num_begin_pad) and before the end pad points (num_end_pad).
	if bool(makeplot):
		fig = plt.figure()
		trialavePower = np.nanmean(powers, axis = 0)
		ax = plt.imshow(trialavePower[:,num_begin_pad:-num_end_pad],interpolation = 'bicubic', aspect='auto', origin='lower', 
			extent = [0,int((num_samples - num_begin_pad - num_end_pad)/sampling_rate),0, len(cf_list)])
		yticks = np.arange(0, len(cf_list), 5)
		yticks = np.append(yticks,len(cf_list)-1)
		yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
		plt.yticks(yticks, yticklabels)
		plt.ylabel('Frequency (Hz)')
		plt.xlabel('Time (s)')
		plt.title('Trial-averaged Spectrogram')
		fig.colorbar(ax)
		plt.show()

	return powers, Power, cf_list