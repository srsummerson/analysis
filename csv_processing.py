import numpy as np
import re

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
