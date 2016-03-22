import numpy as np
import scipy as sp
import re
from neo import io

def get_csv_data_singlechannel(filename):
	'''
	File for reading the TDT data converted to a csv file into an array for processing. The output will be an Nx1 array with data. 

	Current version of file assumes that the data is organized with just the data (no metadata) and return deliminaters between data samples.
	'''
	f = open(filename,'r')
	data = []

	data = [float(line.strip()) for line in f if line.strip() != '']
	'''
	for line in f:
		line = line.strip()	# strip \n character from line
		if line != '':
			data.append(float(line))
	'''
	data = np.array(data)

	f.close()
	return data

def syncHDF_from_csv(tank,dio2_filename, dio3_filename,dio4_filename, dio_samprate,block_num):

	'''
	This if for making the syncHDF file when the TDT tank is too large to open itself. 'Tank' is the string of 
	TDT tank that is used in creating the correct associated filename. The following two inputs are the string
	names of the .csv files corresponding to the DIOx Channels 3 and 4, respectively. The next is the DIO 
	sampling rate. Default is 3051.8 Hz.
	'''

	TDT_tank = '/home/srsummerson/storage/tdt/'
	dio2_location = TDT_tank + dio2_filename
	dio3_location = TDT_tank + dio3_filename
	dio4_location = TDT_tank + dio4_filename

	hdf_times = dict()
	hdf_times['row_number'] = []
	#hdf_times['tdt_timestamp'] = []
	hdf_times['tdt_samplenumber'] = []
	hdf_times['tdt_dio_samplerate'] = []

	row_number = []
	tdt_timestamp = []
	tdt_samplenumber = []

	DIOx2 = get_csv_data_singlechannel(dio2_location)
	DIOx3 = get_csv_data_singlechannel(dio3_location)
	DIOx4 = get_csv_data_singlechannel(dio4_location)

	hdf_times['tdt_dio_samplerate'] = dio_samprate
	
	find_data_rows = np.logical_and(np.ravel(np.equal(DIOx3,13)),np.ravel(np.greater(DIOx2,0))) # when data corresponds to a row and strobe is on
	find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))	

	rows = DIOx4[find_data_rows_ind]
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
	#hdf_times['tdt_timestamp'] = times

	mat_filename = tank +'_b'+str(block_num)+'_syncHDF.mat'
	sp.io.savemat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

	return



