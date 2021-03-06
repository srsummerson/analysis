##parse_logs.py
##functions to operate on raw .txt log data
import h5py
import numpy as np
import os
import glob
import time


"""
Takes in as an argument a log.txt files
Returns a dictionary of arrays containing the 
timestamps of individual events
"""
def parse_log(f_in):
	##open the file
	f = open(f_in, 'r')
	##set up the dictionary
	results = {
	"top_rewarded":[],
	"bottom_rewarded":[],
	"trial_start":[],
	"session_length":0,
	"reward_primed":[],
	"reward_idle":[],
	"top_lever":[],
	"bottom_lever":[],
	"rewarded_poke":[],
	"unrewarded_poke":[]
	}
	##run through each line in the log
	label, timestamp = read_line(f.readline())
	while  label is not None:
		#print timestamp
		##now just put the timestamp in it's place!
		if label == "rewarded=top_lever":
			results['top_rewarded'].append(float(timestamp))
		elif label == "rewarded=bottom_lever":
			results['bottom_rewarded'].append(float(timestamp))
		elif label == "trial_begin":
			results['trial_start'].append(float(timestamp))
		elif label == "session_end":
			results['session_length'] = [float(timestamp)]
		elif label == "reward_primed":
			results['reward_primed'].append(float(timestamp))
		elif label == "reward_idle":
			results['reward_idle'].append(float(timestamp))
		elif label == "top_lever":
			results['top_lever'].append(float(timestamp))
		elif label == "bottom_lever":
			results['bottom_lever'].append(float(timestamp))
		elif label == "rewarded_poke":
			results['rewarded_poke'].append(float(timestamp))
		elif label == "unrewarded_poke":
			results['unrewarded_poke'].append(float(timestamp))
		else:
			print("unknown label: " + label)
		##go to the next line
		label, timestamp = read_line(f.readline())
	f.close()
	return results

##a sub-function to parse a single line in a log, 
##and return the timestamp and label components seperately
def read_line(string):
	label = None
	timestamp = None
	if not string == '':
		##figure out where the comma is that separates
		##the timestamp and the event label
		comma_idx = string.index(',')
		##the timestamp is everything in front of the comma
		timestamp = string[:comma_idx]
		##the label is everything after but not the return character
		label = string[comma_idx+1:-1]
	return label, timestamp

##takes in a dictionary created by the log_parse function and 
##saves it as an hdf5 file
def dict_to_h5(d, path):
	f_out = h5py.File(path, 'w-')
	##make each list into an array, then a dataset
	for key in d.keys():
		##create a dataset with the same name that contains the data
		f_out.create_dataset(key, data = np.asarray(d[key]))
	##and... that's it.
	f_out.close()

##converts all the txt logs in a given directory to hdf5 files
def batch_log_to_h5(directory):
	log_files = get_log_file_names(directory)
	for log in log_files:
		##generate the dictionary
		result = pt.parse_log(log)
		##save it as an hdf5 file with the same name
		new_path = os.path.splitext(log)[0]+'.hdf5'
		dict_to_h5(result, new_path)
	print('Save complete!')

##offsets all timestamps in a log by a given value
##in a h5 file like the one produced by the above function
def offset_log_ts(h5_file, offset):
	f = h5py.File(h5_file, 'r+')
	for key in list(f):
		new_data = np.asarray(f[key])+offset
		f[key][:] = new_data
	f.close()

##a function to extract the creation date (expressed as the 
##julian date) in integer format of a given filepath
def get_cdate(path):
	return int(time.strftime("%j", time.localtime(os.path.getmtime(path))))


##takes in a dictionary returned by parse_log and returns the 
##percent correct. Chance is the rewarded chance rate for the active lever.
##function assumes the best possible performance is the chance rate.
def get_p_correct(result_dict, chance = 0.9):
	total_trials = len(result_dict['top_lever'])+len(result_dict['bottom_lever'])
	correct_trials = len(result_dict['rewarded_poke'])
	return (float(correct_trials)/total_trials)/chance

##takes in a dictionary returned by parse_log and returns the 
##success rate (mean for the whole session)
def get_success_rate(result_dict):
	correct_trials = len(result_dict['rewarded_poke'])
	session_len = result_dict['session_length'][0]/60.0
	return float(correct_trials)/session_len

##returns a list of file paths for all log files in a directory
def get_log_file_names(directory):
	##get the current dir so you can return to it
	cd = os.getcwd()
	filepaths = []
	os.chdir(directory)
	for f in glob.glob("*.txt"):
		filepaths.append(os.path.join(directory,f))
	os.chdir(cd)
	return filepaths