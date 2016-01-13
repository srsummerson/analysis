import numpy as np
from neo import io

def syncHDFwithDIOx(TDT_tank,block_num):

	# Can we get timestamps directly?

	r = io.TdtIO(dirname = TDT_tank)	# create a reader
	bl = r.read_block(lazy=False,cascade=True)
	print "File read."
	analogsig = bl.segments[block_num-1].analogsignals

	counter = 0 	# counter for number of cycles in hdf row numbers
	row = 0
	prev_row = 0

	hdf_times = dict()
	hdf_times['row_number'] = []
	#hdf_times['tdt_timestamp'] = []
	hdf_times['tdt_samplenumber'] = []
	hdf_times['tdt_dio_samplerate'] = []

	row_number = []
	tdt_timestamp = []
	tdt_samplenumber = []

	# find channel index for DIOx 3 and DIOx 4
	for sig in analogsig:
		if (sig.name == 'DIOx 3'): # third channel indicates message type
			DIOx3 = [sig[ind].item() for ind in range(0,sig.size)]
			hdf_times['tdt_dio_samplerate'] = sig.sampling_rate
		if (sig.name == 'DIOx 4'): # fourth channels has row numbers plus other messages
			DIOx4 = [sig[ind].item() for ind in range(0,sig.size)]
			DIOx4_times = sig.times
	length = len(DIOx3)
	find_rows = np.ravel(np.equal(DIOx3, 21))
	data_rows = np.ravel(np.nonzero(find_rows))
	
	#rows = [DIOx4[ind].item() for ind in find_rows] 
	rows = [DIOx4[row_num] for row_num in data_rows]
	times = [DIOx4_times[row_num] for row_num in data_rows]
	prev_row = rows[0]
	for ind in range(1,len(rows)):
		row = rows[ind]
		cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
		counter += cycle
		rows[ind] = counter*256 + row
		prev_row = row
		print float(ind)/length

	hdf_times['row_number'] = rows
	hdf_times['tdt_samplenumber'] = data_rows
	hdf_times['tdt_timestamp'] = times

	# if dio sample num is x, then data sample number is R(x-1) + 1 where
	# R = data_sample_rate/dio_sample_rate

	return hdf_times
