import numpy as np
import scipy as sp
from neo import io

#def syncHDFwithDIOx(TDT_tank,block_num):

filename = 'Mario20160412'
#TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
block_num = 3

# Can we get timestamps directly?

r = io.TdtIO(dirname = TDT_tank)	# create a reader
bl = r.read_block(lazy=False,cascade=True)
print "File read."
analogsig = bl.segments[block_num-1].analogsignals


hdf_times = dict()
hdf_times['row_number'] = []
#hdf_times['tdt_timestamp'] = []
hdf_times['tdt_samplenumber'] = []
hdf_times['tdt_dio_samplerate'] = []
hdf_times['tdt_recording_start'] = []

row_number = []
tdt_timestamp = []
tdt_samplenumber = []

# find channel index for DIOx 3 and DIOx 4
for sig in analogsig:
	if (sig.name == 'DIOx 1'):
		DIOx1 = np.ravel(sig)
	if (sig.name == 'DIOx 2'):
		DIOx2 = np.ravel(sig)
	if (sig.name == 'DIOx 3'): # third channel indicates message type
		#DIOx3 = [sig[ind].item() for ind in range(0,sig.size)]
		DIOx3 = np.ravel(sig)
		hdf_times['tdt_dio_samplerate'] = sig.sampling_rate
	if (sig.name == 'DIOx 4'): # fourth channels has row numbers plus other messages
		DIOx4 = np.ravel(sig)
		#DIOx4 = [sig[ind].item() for ind in range(0,sig.size)]
		##DIOx4_times = sig.times

find_recording_start = np.ravel(np.nonzero(DIOx1))[0]

find_data_rows = np.logical_and(np.ravel(np.equal(DIOx3,13)),np.ravel(np.greater(DIOx2,0))) # when data corresponds to a row and strobe is on
find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))	

rows = DIOx4[find_data_rows_ind]
#rows = [DIOx4[row_num] for row_num in data_rows]
#times = [DIOx4_times[row_num] for row_num in data_rows]
prev_row = rows[0]
counter = 0 	# counter for number of cycles in hdf row numbers

for ind in range(1,len(rows)):
	row = rows[ind]
	cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
	counter += cycle
	rows[ind] = counter*256 + row
	prev_row = row
	#print float(ind)/length

hdf_times['row_number'] = rows
hdf_times['tdt_samplenumber'] = find_data_rows_ind
hdf_times['tdt_recording_start'] = find_recording_start
#hdf_times['tdt_timestamp'] = times

mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.savemat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

# if dio sample num is x, then data sample number is R*(x-1) + 1 where
# R = data_sample_rate/dio_sample_rate


