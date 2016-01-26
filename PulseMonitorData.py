import numpy as np
from neo import io

def findIBIs(pulse,sampling_rate):
	# Input to method is pulse data channel extracted from TDT recording file.
	# Method determines the times of the heart pulses and returns an array of the pulse times.
	pulse_signal = pulse[np.nonzero(pulse)] # only look at pulse signal when it was saving
	pulse_peak_amp = np.amax(pulse_signal)
	pulse_trough_amp = np.amin(pulse_signal)
	pulse_mean = np.mean(pulse_signal)
	pulse_std = np.std(pulse_signal)
	#thres = pulse_trough_amp + 0.6*(pulse_peak_amp - pulse_trough_amp)
	thres = pulse_mean + 0.2*pulse_std
	thresholded_pulse = (pulse_signal > thres)
	pulse_detect = ((thresholded_pulse[1:] - thresholded_pulse[:-1]) > 0.5)  # is 1 when pulse crosses threshold

	max_rate = 220 # beats per min
	min_rate = 100 # beats per min
	max_rate_hz = float(max_rate)/60 # beats per sec
	min_rate_hz = float(min_rate)/60
	min_ibi = float(1)/max_rate_hz # minimum time between beats in seconds
	max_ibi = float(1)/min_rate_hz
	min_ibi_samples = min_ibi*sampling_rate # minimum time between beats in samples
	max_ibi_samples = max_ibi*sampling_rate

	pulse_indices = np.nonzero(pulse_detect)
	pulse_indices = np.ravel(pulse_indices)
	#realpulse = ((pulse_indices[1:] - pulse_indices[:-1]) > min_ibi_samples)
	real_indices = [pulse_indices[0]]
	for ind in range(1,pulse_indices.size):
		if ((pulse_indices[ind] - real_indices[-1]) > min_ibi_samples):
			real_indices.append(pulse_indices[ind])
	#pulse_indices = pulse_indices[realpulse]
	#IBI = pulse_indices[1:] - pulse_indices[:-1]
	real_indices = np.ravel(real_indices)
	IBI = real_indices[1:] - real_indices[:-1]
	not_too_long_IBI = (IBI < max_ibi_samples)
	realIBI = IBI[np.nonzero(not_too_long_IBI)]
	realIBI = realIBI/sampling_rate # IBI is in s

	return realIBI

def findPulseTimes(pulse):
	# Input to method is pulse data channel extracted from TDT recording file.
	# Method determines the times of the heart pulses and returns an array of the pulse times.
	pulse_signal = pulse[np.nonzero(pulse)] # only look at pulse signal when it was saving
	pulse_peak_amp = np.amax(pulse_signal)
	pulse_trough_amp = np.amin(pulse_signal)
	pulse_mean = np.mean(pulse_signal)
	pulse_std = np.std(pulse_signal)
	#thres = pulse_trough_amp + 0.6*(pulse_peak_amp - pulse_trough_amp)
	thres = pulse_mean + 0.2*pulse_std
	thresholded_pulse = (pulse_signal > thres)
	pulse_detect = ((thresholded_pulse[1:] - thresholded_pulse[:-1]) > 0.5)  # is 1 when pulse crosses threshold

	max_rate = 220 # beats per min
	min_rate = 100 # beats per min
	max_rate_hz = float(max_rate)/60 # beats per sec
	min_rate_hz = float(min_rate)/60
	min_ibi = float(1)/max_rate_hz # minimum time between beats in seconds
	max_ibi = float(1)/min_rate_hz
	min_ibi_samples = min_ibi*pulse_signal.sampling_rate.item() # minimum time between beats in samples
	max_ibi_samples = max_ibi*pulse_signal.sampling_rate.item()

	pulse_indices = np.nonzero(pulse_detect)
	pulse_indices = np.ravel(pulse_indices)
	#realpulse = ((pulse_indices[1:] - pulse_indices[:-1]) > min_ibi_samples)
	real_indices = [pulse_indices[0]]
	for ind in range(1,pulse_indices.size):
		if ((pulse_indices[ind] - real_indices[-1]) > min_ibi_samples):
			real_indices.append(pulse_indices[ind])
	#pulse_indices = pulse_indices[realpulse]
	#IBI = pulse_indices[1:] - pulse_indices[:-1]
	real_indices = np.ravel(real_indices)
	pulse_times = pulse.times[real_indices]

	return pulse_times

def TrialAverageIBI(hdf, hdf_times, pulse_signal):
	# Method to compute average inter-beat interval (IBI) for successful regular trials and stress trials.
	# Inputs are the hdf file (hdf; used to determine behavior state), the hdf row numbers 
	# as recorded by the TDT system (output from syncHDFwithDIOx), and the pulse times
	# (pulse_times; used to determine the per trial IBI). The output of the method is the average IBI per
	# trial, indexed by the trial number (regardless of whether trial was successful or not).

	pulse_sample_ind = []
	pulse_sample_rate = pulse_signal.sampling_rate

	pulse_peak_amp = np.amax(pulse_signal)
	thresholded_pulse = (pulse_signal > 0.7*pulse_peak_amp)
	pulse_detect = 0.5*(pulse_signal[1:] - pulse_signal[:-1]) + 0.5  # is 1 when pulse crosses threshold


	hdf = tables.openFile(hdf_file)
	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
	#ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))

	hdf_times['tdt_timestamp']

	# Get TDT sample times that match the row numbers corresponding to wait states, match this to pulse signal samples
	for ind in ind_wait_states:
		hdfrow = ind 	# hdf row number in behavior code
		DIOx_sample = (np.abs(hdf_times['row_number']-hdfrow)).argmin() 	# find index in DIOx_hdfrow which is closest
		DIOx_sampletime_waitstate = hdf_times['tdt_timestamp'][DIOx_sample]
		pulse_sample_ind.append((np.abs(pulse_signal.times - DIOx_sampletime_waitstate)).argmin())	# find pulse signal ind closest to this sample time of the wait states

	max_trial_length = np.amax(pulse_sample_ind[1:] - pulse_sample_ind[:-1])
	averageIBI = np.zeros(max_trial_length)
	count_pulse_samples = np.ones(max_trial_length)

	for ind in range(0,pulse_sample_ind.size-1): # for all but the last wait states
		pulse_segment = pulse_detect[pulse_sample_ind[ind]:pulse_sample_ind[ind+1]]
		pulse_indices = np.nonzero(pulse_segment)
		IBI = float(pulse_indices[1:] - pulse_indices[:-1])/pulse_sample_rate	# interbeat intervals in s
		trialIBI[ind] = float(sum(IBI))/IBI.size

	return trialIBI

def syncHDFwithDIOx(TDT_tank,block_num):

	r = io.TdtIO(dirname = TDT_tank)	# create a reader
	bl = r.read_block(lazy=False,cascade=True)
	analogsig = bl.segments[block_num-1].analogsignals

	counter = 0 	# counter for number of cycles in hdf row numbers
	row = 0
	prev_row = 0

	hdf_times = dict()
	hdf_times['row_number'] = []
	hdf_times['tdt_timestamp'] = []
	hdf_times['tdt_samplenumber'] = []
	hdf_times['tdt_dio_samplerate'] = []

	row_number = []
	tdt_timestamp = []
	tdt_samplenumber = []

	# find channel index for DIOx 3 and DIOx 4
	for sig in analogsig:
		if (sig.name == 'DIOx 3'): # third channel indicates message type
			DIOx3 = sig
		if (sig.name == 'DIOx 4'): # fourth channels has row numbers plus other messages
			DIOx4 = sig
	length = DIOx3.size
	hdf_times['tdt_dio_samplerate'] = DIOx3.sampling_rate
	for ind in range(0,length):
		if (DIOx3[ind]==21):
			row = DIOx4[ind].item()
			cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
			counter += cycle
			row_number.append(counter*256 + row)
			tdt_timestamp.append(DIOx4.times[ind])
			tdt_samplenumber.append(ind)
			prev_row = row
		print float(ind)/length

	hdf_times['row_number'] = row_number
	hdf_times['tdt_samplenumber'] = tdt_samplenumber
	hdf_times['tdt_timestamp'] = tdt_timestamp

	return hdf_times







