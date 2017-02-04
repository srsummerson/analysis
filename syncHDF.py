import numpy as np
import scipy as sp
from neo import io



filename = 'Mario20170124'
#TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
#block_num = 1

# Load TDT files.
r = io.TdtIO(dirname = TDT_tank)				# create a reader
bl = r.read_block(lazy=False,cascade=True)		# read data
print "File read."

for block_num in range(1,len(bl.segments)+1):
	analogsig = bl.segments[block_num-1].analogsignals

	# Create dictionary to store synchronization data
	hdf_times = dict()
	hdf_times['row_number'] = [] 			# PyTable row number 
	#hdf_times['tdt_timestamp'] = []
	hdf_times['tdt_samplenumber'] = []		# Corresponding TDT sample number
	hdf_times['tdt_dio_samplerate'] = []	# Sampling frequency of DIO signal recorded by TDT system
	hdf_times['tdt_recording_start'] = []	# TDT sample number when behavior recording begins

	row_number = []
	tdt_timestamp = []
	tdt_samplenumber = []

	# Pull out recorded DIO channels.
	for sig in analogsig:
		if (sig.name == 'DIOx 1'):					# first channel indicates when behavior recording is ON (=1)
			DIOx1 = np.ravel(sig)
		if (sig.name == 'DIOx 2'):					# second channel is strobe signal that indicates when DIO message is sent from behavior recording
			DIOx2 = np.ravel(sig)
		if (sig.name == 'DIOx 3'): 					# third channel indicates DIO message type
			DIOx3 = np.ravel(sig)
			hdf_times['tdt_dio_samplerate'] = sig.sampling_rate
		if (sig.name == 'DIOx 4'): 					# fourth channels has row numbers (mod 256) plus other messages
			DIOx4 = np.ravel(sig)
			#DIOx4 = [sig[ind].item() for ind in range(0,sig.size)]
			##DIOx4_times = sig.times

	find_recording_start = np.ravel(np.nonzero(DIOx1))[0]
	find_data_rows = np.logical_and(np.ravel(np.equal(DIOx3,13)),np.ravel(np.greater(DIOx2,0))) 	# samples when data corresponds to a row and strobe is on
	find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))	

	rows = DIOx4[find_data_rows_ind]		# row numbers (mod 256)
	
	prev_row = rows[0] 	# placeholder variable for previous row number
	counter = 0 		# counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers

	for ind in range(1,len(rows)):
		row = rows[ind]
		cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
		counter += cycle
		rows[ind] = counter*256 + row
		prev_row = row
		#print float(ind)/length

	# Load data into dictionary
	hdf_times['row_number'] = rows
	hdf_times['tdt_samplenumber'] = find_data_rows_ind
	hdf_times['tdt_recording_start'] = find_recording_start
	#hdf_times['tdt_timestamp'] = times

	# Save syncing data as .mat file
	mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
	sp.io.savemat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

# Note:
# If DIO sample num is x, then data sample number is R*(x-1) + 1 where
# R = data_sample_rate/dio_sample_rate


