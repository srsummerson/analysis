import numpy as np
import scipy as sp
from scipy import signal
import re
from neo import io
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
from OMNI_methods import convert_OMNI, get_stim_sync_sig
from spectralAnalysis import LFPPowerPerTrial_SingleBand_PerChannel_Timestamps

omni_filename = 'stream_behavioral.csv'
hdf_filename = 'mari20160718_17_te2379.hdf'
filename = 'Mario20160718-OMNI'
block_num = 1

omni_location = '/storage/omni_data/' + omni_filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
TDT_location = '/home/srsummerson/storage/tdt/'+filename

channels = [10]  		# picking subset of channels to analyze
freq_window = [10,30]	# defining frequency window of interest

'''
Load syncing data for behavior and TDT recording
'''
print "Loading syncing data."

hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

hdf_rows = np.ravel(hdf_times['row_number'])
hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])
hdf_row_timestamps = dio_tdt_sample/float(dio_freq)

'''
Get OMNI data
'''
channel_data, timestamps, crc_flag = convert_OMNI(omni_location)
Avg_Fs = len(timestamps)/float(timestamps[-1] - timestamps[0])  	# estimate sampling freq in Hz

'''
Get sync data
'''
stim_signal, stim_on_trig, stim_delivered, stwv_samprate = get_stim_sync_sig(TDT_location)

first_pulse = timestamps[4063]  # time in s (found by inspection)
tdt_first_pulse = 23477./stwv_samprate  # time in s (found by inspection)
omni_time_offset = first_pulse - tdt_first_pulse

'''
Get behavior data
'''
hdf = tables.openFile(hdf_location)

state = hdf.root.task_msgs[:]['msg']
state_time = hdf.root.task_msgs[:]['time']
ind_wait_states = np.ravel(np.nonzero(state == 'wait'))   # total number of unique trials
ind_center_hold_states = ind_wait_states + 2 			  # index of center hold states
state_time_ind_wait_states = state_time[ind_wait_states]  # row numbers for wait states 

tdt_time_ind = np.zeros(len(state_time_ind_wait_states))

for i in range(0,len(state_time_ind_wait_states)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_time_ind_wait_states[i]))
	tdt_time_ind[i] = hdf_row_timestamps[hdf_index]

'''
Translate behavioral timestamps to timestamps recorded with OMNI device
'''
# times in s on the OMNI device for when the task state occurs
omni_time_ind = tdt_time_ind + omni_time_offset

LFPPowerPerTrial_SingleBand_PerChannel_Timestamps(channel_data,timestamps,Avg_Fs,channels,omni_time_ind,1,2,freq_window)
