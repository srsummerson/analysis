import numpy as np
import re

def get_iscan_data(filename):
	'''
	File for reading the ISCAN data to an array for processing. The output will be an array with sample numbers in the
	first column and the data in the subsequent columns. Note: the default acquisition rate for these files is 240 Hz.

	Based on the current triggering of the ISCAN data acquisition, the synchronized data is saved as the second run (Run 2), 
	so this method first searches for the beginning of this run and then writes the data from there oncwards into the 
	array.

	Current version of file assumes that the data is organized as: Sample number, Pupil H1, Pupil V1, Pupil D1, Pupil VD1, 
	Pupil A1, Eye AZ1, Eye EL1.
	'''
	f = open(filename,'r')
	eye_data = dict()
	eye_data['sample'] = []
	eye_data['pupil_h1'] = []
	eye_data['pupil_v1'] = []
	eye_data['pupil_d1'] = []
	eye_data['pupil_vd1'] = []
	eye_data['pupil_a1'] = []
	eye_data['eye_az1'] = []
	eye_data['eye_el1'] = []

	counter = 0	# counter used to count read lines in file
	counterRun2 = float('inf')

	for line in f:
		line = line.strip()	# strip \n character from line
		matchRun = re.match(r'Run   2',line,re.S)	# find header line for Run 2
		if matchRun:
			counterRun2 = counter
		if counter > counterRun2 + 1:	# start writing data two lines following header line
			line = line.split()			# splits line into array of strings
			eye_data['sample'].append(float(line[0]))
			eye_data['pupil_h1'].append(float(line[1]))
			eye_data['pupil_v1'].append(float(line[2]))
			eye_data['pupil_d1'].append(float(line[3]))
			eye_data['pupil_vd1'].append(float(line[4]))
			eye_data['pupil_a1'].append(float(line[5]))
			eye_data['eye_az1'].append(float(line[6]))
			eye_data['eye_el1'].append(float(line[7]))
		counter += 1

	f.close()
	return eye_data
