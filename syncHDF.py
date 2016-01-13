import numpy as np
from neo import io

def syncHDFwithDIOx(TDT_tank,block_num):

	# Can we get timestamps directly?

	r = io.TdtIO(dirname = TDT_tank)	# create a reader
	bl = r.read_block(lazy=False,cascade=True)
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
			DIOx3 = [sig[ind].item() for ind in range(0,sig.size)
		if (sig.name == 'DIOx 4'): # fourth channels has row numbers plus other messages
			DIOx4 = [sig[ind].item() for ind in range(0,sig.size)
	length = DIOx3.size
	find_rows = np.equal(DIOx3, 21)
	find_rows = np.ravel(find_rows)
	hdf_times['tdt_dio_samplerate'] = DIOx3.sampling_rate
	#rows = [DIOx4[ind].item() for ind in find_rows] 
	rows = DIOx4[find_rows]
	prev_row = rows[0]
	for ind in range(1,rows.size):
		row = rows[ind]
		cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
		counter += cycle
		rows[ind] = counter*256 + row
		prev_row = row
		print float(ind)/length

	hdf_times['row_number'] = rows
	hdf_times['tdt_samplenumber'] = find_rows
	#hdf_times['tdt_timestamp'] = tdt_timestamp

	return hdf_times
