
import numpy as np
import scipy as sp
import csv

def syncHDF_withCSV(filename, TDT_tank, DIO_csv_files):
	'''
	Input: 
	- filename: string, name of tank (e.g. 'Mario20181012')
	- TDT_tank: string, location of tank with name filename (e.g. '/home/srsummerson/storage/tdt/'+filename)
	- DIO_csv_files: list, list of file locations for four CSV files corresponding to the four DIO channels exported to CSV file format

	Output:
	- hdf_times: dict, contains all information in normal syncHDF mat file for synchronizing behavior and TDT data

	'''

	# Read in CSV data and save as arrays
	f = open(DIO_csv_files[0], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k if k!= '' else np.nan for i in data for k in i]
	DIOx1 = np.array([float(val) for val in datal])

	f = open(DIO_csv_files[1], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k if k!= '' else np.nan for i in data for k in i]
	DIOx2 = np.array([float(val) for val in datal])

	f = open(DIO_csv_files[2], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k if k!= '' else np.nan for i in data for k in i]
	DIOx3 = np.array([float(val) for val in datal])

	f = open(DIO_csv_files[3], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k if k!= '' else np.nan for i in data for k in i]
	DIOx4 = np.array([float(val) for val in datal])
	
	DIOx_samprate = 3051.757813
	
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

	# Get synchronizing data

	hdf_times['tdt_dio_samplerate'] = DIOx_samprate		# Sampling frequency of DIO signal recorded by TDT system 
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
	sp.io.savemat('/storage/syncHDF/'+mat_filename,hdf_times)
	#sp.io.savemat('./'+mat_filename,hdf_times)

	return hdf_times