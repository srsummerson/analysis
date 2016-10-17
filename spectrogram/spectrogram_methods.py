import numpy as np
from numpy import linalg
from numpy import fft
import matplotlib.pyplot as plt


def filter_with_chirplet(s, sp, g):
	'''
	Method to filter input signal with chirplet. 

	Inputs:
		- s: dictionary containing signal 
		- sp: dictionary containing signal parameters
		- g: dictionary containing chirplet parameters
	Output:
		- fs: dictionary with two entries for time domain and frequency domain
	'''

	# Initialize
	fs = dict()
	fs_time_domain = np.zeros(len(s['time_domain']))
	fs_frequency_domain = np.zeros(len(s['frequency_domain'])) + 0j

	frequency_domain = s['frequency_domain']
	signal_frequency_support_indices = g['signal_frequency_support_indices']
	number_points_frequency_domain = sp['number_points_frequency_domain']
	number_points_time_domain = sp['number_points_time_domain']

	fs_frequency_domain[signal_frequency_support_indices] = frequency_domain[signal_frequency_support_indices]*g['filter']
	fs_time_domain=fft.ifft(fs_frequency_domain, number_points_frequency_domain)
	fs['time_domain'] =fs_time_domain[:number_points_time_domain]

	fs['frequency_domain'] = fs_frequency_domain

	return fs

def complete_chirplet_parameters(g):
	'''
	Method to fill in chirplet parameters that will be used when calling make_chirplet method.

	Input:
		- g: dictionary containing fields for existing parameters. This should have keys: 'center_frequency',
			'fractional_bandwidth', and 'chirp_rate'
	Output:
		- g: dictionary with added fields for added parameters
	'''
	# Check for existing entries
	chirplet_keys = g.keys()
	if 'center_time' in chirplet_keys:
		t0=g['center_time']
	else: 
		t0=0
	if 'center_frequency' in chirplet_keys:
		v0=g['center_frequency']
	if 'chirp_rate' in chirplet_keys:
		c0=g['chirp_rate']

	# assign or compute duration parameter:
	if 'duration_parameter' in chirplet_keys:
	    s0 = g['duration_parameter']
	elif ('fractional_bandwidth' in chirplet_keys)&(c0==0):
	    fbw = g['fractional_bandwidth']
	    s0 = np.log((2.*np.log(2.))/((fbw**2.)*np.pi*(v0**2.)))
	elif ('frequency_domain_standard_deviation' in chirplet_keys)&(c0==0):
	    fstd = g['frequency_domain_standard_deviation']
	    s0 = -np.log(4.*np.pi*(fstd**2.))
	elif 'time_domain_standard_deviation' in chirplet_keys:
	    tstd = g['time_domain_standard_deviation']
	    s0=log(4.*np.pi*(tstd**2.))

	g['center_time'] = t0
	g['center_frequency'] = v0
	g['duration_parameter'] = s0
	g['chirp_rate'] = c0
	# Fixed parameter:
	g['std_multiple_for_support']=6.0
	# Calculate other quanties as a function of these constants
	g['time_domain_standard_deviation'] = np.sqrt(np.exp(s0)/(4.*np.pi))
	g['frequency_domain_standard_deviation'] = np.sqrt((np.exp(-s0)+c0**2.*np.exp(s0))/(4.*np.pi))
	g['time_frequency_covariance'] = (c0*np.exp(s0))/(4.*np.pi)
	g['time_frequency_correlation_coefficient'] = g['time_frequency_covariance']*(g['time_domain_standard_deviation']*g['frequency_domain_standard_deviation'])**(-1.)
	g['fractional_bandwidth'] = 2.*np.sqrt(2.*np.log(2.))*g['frequency_domain_standard_deviation']/v0 # fwhm/center_frequency

	return g

def make_chirplet(chirplet_structure, sp):
	'''
	Method creates the chirplet used to extract power.

	Inputs:
		- chirplet_structure: dictionary containing fields describing basic cirplet parameters 
		- sp: dictionary containing fields describing basic signal parameters (sampling rate in time domain, number
			  of points in the time domain)

	Outputs:
		- g: chirplet 

	'''

	# Fill in rest of values used for chirplet
	g = complete_chirplet_parameters(chirplet_structure)

	# Use short variable names for equations
	t0 = g['center_time']
	v0 = g['center_frequency']
	s0 = g['duration_parameter']
	c0 = g['chirp_rate']
	tstd = g['time_domain_standard_deviation']
	vstd = g['frequency_domain_standard_deviation']

	time_support = sp['time_support']
	# time support:
	# shift g['center_time'] to fall on sp.time_support
	if (g['center_time'] < time_support[0]) or (g['center_time'] > time_support[-1]): 
	    g['center_time'] = g['center_time'] % time_support[-1]

	temp1 = np.abs(time_support-g['center_time'])
	center_index = np.argmin(temp1) # sp.time_support index closest to g.center_time

	inds = np.arange(0, tstd*g['std_multiple_for_support'], sp['time_step_size'])
	numinds= len(inds)

	support_inds = center_index + np.arange(-numinds,numinds)
	support_inds = support_inds % sp['number_points_time_domain']
	nonzero_support_inds = np.ravel(np.nonzero(np.equal(support_inds, 0)))
	support_inds[nonzero_support_inds] = sp['number_points_time_domain']
	g['signal_time_support_indices'] = support_inds

	signal_time_support_indices = g['signal_time_support_indices']
	#print signal_time_support_indices[0]
	#print signal_time_support_indices[-1]
	#g['time_support'] = time_support[signal_time_support_indices]

	t = time_support[center_index] + sp['time_step_size']*np.arange(-numinds,numinds)
	g['ptime'] = t

	# chirplet in time domain:
	g['time_domain'] = 2.**(1./4)*np.exp(-s0/4.)*np.exp(-np.exp(-s0)*np.pi*(t-t0)**2.0)*np.exp(2.*np.pi*1j*v0*(t-t0))*np.exp(np.pi*1j*c0*(t-t0)**2.)    					
	g['time_domain'] = g['time_domain']/linalg.norm(g['time_domain']) 	# need to normalize due to discrete sampling

	v = sp['frequency_support'] # in Hz
	freq_nonzero_support_inds = np.ravel(np.nonzero(np.logical_and(np.less_equal(v0-g['std_multiple_for_support']*vstd, v),
									np.greater_equal(v0+g['std_multiple_for_support']*vstd, v))))
	g['signal_frequency_support_indices'] = freq_nonzero_support_inds
	
	# shorten to include only chirplet support
	v = v[freq_nonzero_support_inds]
	g['frequency_support'] = v
	g['pfrequency'] = g['frequency_support']

	# chirplet in frequency domain: 
	Gk = (2.**(1./4))*np.sqrt(-1j*c0+np.exp(-s0))**(-1.)*np.exp(-s0/4.+ (np.exp(s0)*np.pi*(v-v0)**2.0)/(-1+1j*c0*np.exp(s0)))
	n1 = np.sqrt(sp['number_points_frequency_domain'])/linalg.norm(Gk)
	Gk = n1*Gk 		# because of discrete sampling and different time/freq sample numbers
	g['filter'] = Gk 	# at center time of zero, use this for convolution filtering
	g['frequency_domain'] = Gk*np.exp(-2.*np.pi*1j*v*t0) # translation in time to tk

	return g

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

def make_signal_structure(raw_signal, output_type, sp):
	'''
	Method to generate basic time and frequency domain information about the input, raw_signal. This is used in the 
	filter_LFP method.

	Inputs:
		- raw_signal: array of length T containing T time-domain samples of the signal of interest
		- output_type: string that should be either 'real' or 'analytic' 
		- sp: signal parameters as generated by the get_signal_parameters method 
	Output:
		- s: dictionary containing fields time_domain and frequency_domain

	'''
	s = dict()
	frequency_domain = fft.fft(raw_signal, int(sp['number_points_frequency_domain']))

	if np.sum(np.abs(np.imag(raw_signal))) != 0: 	# signal already complex
	    s['time_domain'] = raw_signal 					# do not change
	elif output_type == 'real':
	    s['time_domain'] = raw_signal
	elif output_type == 'analytic':
	    time_domain = fft.ifft(frequency_domain, sp['number_points_frequency_domain'])
	    s['time_domain'] = time_domain[:sp['number_points_time_domain']]
	    # this step scales analytic signal such that
	    # real(analytic_signal)=raw_signal, but note that
	    # analytic signal energy is double that of raw signal energy
	    # sum(abs(raw_signal).^2)=0.5*sum(abs(s.time_domain).^2)
	    less_than_inds = np.ravel(np.nonzero(sp['frequency_support'] < 0))
	    greater_than_inds = np.ravel(np.nonzero(sp['frequency_support'] > 0))
	    frequency_domain[less_than_inds] = 0
	    frequency_domain[greater_than_inds] = 2*frequency_domain[greater_than_inds]
	
	s['frequency_domain'] = frequency_domain

	return s

def get_signal_parameters(sampling_rate, number_points_time_domain):
	'''
	Method to generate a dictionary containing various signal parameters that are necessary for processing the 
	signal in the time and frequency domain. 

	Inputs:
		- sampling_rate: sampling rate of the signal of interest in Hz
		- number_points_time_domain: number indicating how many samples there are in the time domain 
	Outputs:
		- sp: dictionary with fields corresponding to the various signal parameters (sampling_rate, number_points_time_domain,
			  number_points_frequency_domain, frequency_step_size, time_step_size, time_support, frequency_support)
	'''
	
	sp = dict()
	sp['sampling_rate'] = sampling_rate
	sp['number_points_time_domain'] = number_points_time_domain

	maxpower2=2**23 # to avoid out-of-memory issues
	# check this on your machine, machine-specific threshold
	if number_points_time_domain < maxpower2:
	    sp['number_points_frequency_domain'] = 2.**np.ceil(np.log2(number_points_time_domain)) # fixed parameter for computational ease
	else:
	    sp['number_points_frequency_domain'] = number_points_time_domain

	sp['time_step_size'] = 1./sampling_rate
	sp['frequency_step_size'] = sampling_rate/float(sp['number_points_frequency_domain'])

	# raw time_support, need to shift if
	# time_at_first_point or time_at_center_point provided by user
	time_support = sp['time_step_size']*np.arange(sp['number_points_time_domain'])

	frequency_support = sp['frequency_step_size']*np.arange(sp['number_points_frequency_domain'])
	inds = np.ravel(np.nonzero(np.greater(frequency_support, sampling_rate/2.)))
	frequency_support[inds] -= sampling_rate

	sp['time_support'] = np.float32(time_support)
	sp['frequency_support'] = np.float32(frequency_support)

	return sp

def filter_LFP(signal, freq_band, normalize, sampling_rate):
	'''
	This method filters the LFP signal using a chirplet transfrom in a designated frequency band.
	The fractional bandwith used for each center frequency is 20 percent of the center frequency.

	Inputs:
		- signal: lfp data
		- freq_band: array of length 2 with values [minFreq, maxFreq]
		- normalize: indicator of whether the power is to be normalized by the average power over time 
		- sampling_rate: sampling rate 
	Ouputs:
		- power: array of N x T entries with the power at each frequency over time, N is the number of samples in
				 the frequency domain and T is the number of samples in the time domain
		- cf_list: array of length N that contains the center frequencies used in the analysis
		- phase: array of size N x T that contains the instantaneus phase for each frequency as a function of time
	'''

	numpoints=len(signal)
	sp = get_signal_parameters(sampling_rate, numpoints)
	s = make_signal_structure(signal, 'analytic', sp)
	#s = make_signal_structure(signal, 'real', sp)
	
	minimum_frequency=freq_band[0,0] # Hz
	maximum_frequency=freq_band[1,0] # Hz
	number_of_frequencies=50
	minimum_frequency_step_size=0.75
	
	center_frequencies = make_center_frequencies(minimum_frequency, maximum_frequency, number_of_frequencies, minimum_frequency_step_size)

	tfmat = np.zeros([number_of_frequencies,sp['number_points_time_domain']]) + 0j  # cast as complex array
	cf_list = center_frequencies

	for f in range(number_of_frequencies):
	    chirplet_structure = dict()
	    chirplet_structure['center_frequency'] = cf_list[f] # in Hz
	    chirplet_structure['fractional_bandwidth'] = 0.2
	    chirplet_structure['chirp_rate'] = 0.0
	    
	    g = make_chirplet(chirplet_structure, sp)
	  
	    fs = filter_with_chirplet(s, sp, g)
	    
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


def make_spectrogram(lfp, sampling_rate, fmax, trialave, makeplot):
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
	Outputs:
		- powers: array of N x M x T, where N is the number of trials, M is the number of frequency domain
		points, and T is the number of time domain points
		- Power: if trialave==0: array of N x M x T, where N is the number of trials, M is the number of frequency domain
		points, and T is the number of time domain points, if trialave ==1: array of M x T, 
		where M is the number of points in the frequency domain and T is the number of time points.
		- cf_list: list of the center frequencies for which power was computed

	'''
	num_trials, num_samples = lfp.shape

	freq_band=np.zeros([2,num_samples])
	freq_band[0,:]= np.ones(num_samples)
	freq_band[1,:]= fmax*np.ones(num_samples)

	powers = np.empty([num_trials,50,num_samples])
	powers[:] = np.NAN
	
	for k in range(num_trials):	# iterate over trials
		signal = lfp[k,:]
		power, cf_list, phase = filter_LFP(signal,freq_band,1,sampling_rate)
		powers[k,:,:] = 10*np.log10(power)  # dB scale

	
	if bool(trialave):
		Power = np.nanmean(powers, axis = 0)
	else:
		Power = powers
	
	if bool(makeplot):
		fig = plt.figure()
		trialavePower = np.nanmean(powers, axis = 0)
		ax = plt.imshow(trialavePower[:,sampling_rate:(int(num_samples/sampling_rate) - 1)*sampling_rate],interpolation = 'bicubic', aspect='auto', origin='lower', 
			extent = [0,int(num_samples/sampling_rate) - 2,0, len(cf_list)])
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