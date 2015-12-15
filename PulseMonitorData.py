import numpy as np
from neo import io

def PulseTimes(data):
	# Input to method is array of timestamped data already extracted from TDT recording file.
	# Method determines the times of the heart pulses and returns an array of the pulse times.
	

	return pulse_times

def TrialAverageIBI(hdf, DIOx, pulse_times):
	# Method to compute average inter-beat interval (IBI) for successful regular trials and stress trials.
	# Inputs are the hdf file (hdf; used to determine behavior state), the hdf row numbers 
	# as recorded by the TDT system (DIOx; used to sychronize behavior with recording), and the pulse times
	# (pulse_times; used to determine the per trial IBI). The output of the method is the average IBI per
	# trial, indexed by the trial number (regardless of whether trial was successful or not).

	counter = 0 	# counter for number of cycles in hdf row numbers

	hdf = tables.openFile(hdf_file)
	state = hdf.root.task_msgs[:]['msg']
	state_time = hdf.root.task_msgs[:]['time']
	ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
	#ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))

	for ind in DIOx:
		row = DIOx[ind]
		cycle = DIOx[ind] < DIOx[ind-1] # row counter has cycled when the current row number is less than the previous
		counter += cycle
		DIOx_hdfrow = counter*256 + row
		# need to add DIOx_hdftimes

	for ind in ind_wait_states:
		hdfrow = ind 	# hdf row number in behavior code
		hdfrow_next = ind_wait_states[ind+1]
		DIOx_sample = (np.abs(DIOx_hdfrow-hdfrow)).argmin() 	# find index in DIOx_hdfrow which is closest
		DIOx_sample_next = (np.abs(DIOx_hdfrow-hdfrow_next)).argmin()  # find index in DIOx_hdfrow which is closest
		DIOx_sampletime = DIOx_hdftimes[DIOx_sample]
		DIOx_sampletime_next = DIOx_hdftimes[DIOx_sample_next]




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
			DIOx3_ind = sig.channel_index
		if (sig.name == 'DIOx 4'): # fourth channels has row numbers plus other messages
			DIOx4_ind = sig.channel_index
	DIOx3 = bl.segments[block_num-1].analogsignals[DIOx3_ind]
	DIOx4 = bl.segments[block_num-1].analogsignals[DIOx4_ind]
	length = DIOx3.size
	hdf_times['tdt_dio_samplerate'] = DIOx3.sampling_rate
	for ind in range(0,length):
		if (DIOx3[ind]==21):
			row = DIOx4[ind]
			cycle = row < prev_row # row counter has cycled when the current row number is less than the previous
			counter += cycle
			row_number.append(counter*256 + row)
			tdt_timestamp.append(DIOx4.times[ind])
			tdt_samplenumber.append(ind)
			prev_row = row

	hdf_times['row_number'] = row_number
	hdf_times['tdt_samplenumber'] = tdt_samplenumber
	hdf_times['tdt_timestamp'] = tdt_timestamp

	return hdf_times







