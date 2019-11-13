from csv_processing import get_csv_data_singlechannel
from basicAnalysis import highpassFilterData, lowpassFilterData, bandpassFilterData
from PulseMonitorData import findIBIs, running_mean
import numpy as np
import scipy as sp 
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

pilot_dir = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Pilot drug testing/'
pilot_files_HrtR = [['Mario20180908/Mario20180908_Block-1_HrtR.csv', 'Mario20180908/Mario20180908_Block-2_HrtR.csv'], \
					['Mario20180911/Mario20180911_Block-1_HrtR.csv', 'Mario20180911/Mario20180911_Block-2_HrtR.csv', 'Mario20180911-1/Mario20180911-1_Block-1_HrtR.csv'], \
					['Mario20180912/Mario20180912_Block-1_HrtR.csv', 'Mario20180912/Mario20180912_Block-2_HrtR.csv', 'Mario20180912-1/Mario20180912-1_Block-1_HrtR.csv'], \
					]
pilot_files_PupD = [['Mario20180908/Mario20180908_Block-1_PupD.csv', 'Mario20180908/Mario20180908_Block-2_PupD.csv'], \
					['Mario20180911/Mario20180911_Block-1_PupD.csv', 'Mario20180911/Mario20180911_Block-2_PupD.csv', 'Mario20180911-1/Mario20180911-1_Block-1_PupD.csv'], \
					['Mario20180912/Mario20180912_Block-1_PupD.csv', 'Mario20180912/Mario20180912_Block-2_PupD.csv', 'Mario20180912-1/Mario20180912-1_Block-1_PupD.csv'], \
					]

Fsamp = 3043.

def process_HrtR(csv_filenames, Fs):
	'''
	Method to extract the data from the indicated csv file(s), filter it, and find the 
	IBIs over time.

	Inputs:
	- csv_filenames: list; list containing locations of all csv_files that contained heartrate data, 
			this data will be appended at the end of this method
	- Fs: float; sampling rate in Hz

	Outputs:
	- data_all: array; filtered pulse signal concatenated from all files in input list
	- times_all: array; time stamp array to go along with data_all signal
	- ibis: array; inter-beat intervals for pulse data in data_all
	'''

	# Define variables
	data_all = np.array([])
	times_all = np.array([])

	# Process each file in the list
	for file in csv_filenames:
		# Extract data from csv file
		data = get_csv_data_singlechannel(file)
		data = data[np.nonzero(data)]
		times = np.arange(len(data))/Fs
		if len(times_all) > 0:
			times = times + times_all[-1] + 1./Fs
		# High-pass filter to get rid of baseline offset
		data = highpassFilterData(data, Fs, 2)
		# Concatenate across the files
		data_all = np.append(data_all,data)
		times_all = np.append(times_all,times)

	# Find the inter-beat intervals for pulse signal data_all
	ibis = findIBIs(data_all,Fs)

	return data_all, times_all, ibis

def process_PupD(csv_filenames, Fs):
	'''
	Method to extract the data from the indicated csv file(s), filter it, and find the 
	pupil dilation over time.

	Inputs:
	- csv_filenames: list; list containing locations of all csv_files that contained heartrate data, 
			this data will be appended at the end of this method
	- Fs: float; sampling rate in Hz

	Outputs:
	- data_all: array; filtered pupil diameter signal concatenated from all files in input list
	- times_all: array; time stamp array to go along with data_all signal
	'''

	# Define variables
	data_all = np.array([])
	times_all = np.array([])
	data_all_unfiltered = np.array([])

	# Process each file in the list
	for file in csv_filenames:
		# Extract data from csv file
		Fsamp = Fs
		data = get_csv_data_singlechannel(file)
		data = data[np.nonzero(data)]
		data = sp.signal.decimate(data, 50)
		Fsamp = Fsamp/50.
		
		# Get rid of blinks
		pupil_snippet_range = range(0,len(data))
		eyes_closed = np.nonzero(np.less(data,-3.0))
		eyes_closed = np.ravel(eyes_closed)
		t = time.time()
		if len(eyes_closed) > 1:
			find_blinks = eyes_closed[1:] - eyes_closed[:-1]
			blink_inds = np.ravel(np.nonzero(np.not_equal(find_blinks,1)))
			eyes_closed_ind = [eyes_closed[0]]
			eyes_closed_ind += eyes_closed[blink_inds].tolist()
			eyes_closed_ind += eyes_closed[blink_inds+1].tolist()
			eyes_closed_ind += [eyes_closed[-1]]
			eyes_closed_ind.sort()
			
			for i in np.arange(1,len(eyes_closed_ind),2):
				rm_range = range(np.nanmax(eyes_closed_ind[i-1]-100,0),np.minimum(eyes_closed_ind[i] + 100,len(data)-1))
				rm_indices = [pupil_snippet_range.index(rm_range[ind]) for ind in range(0,len(rm_range)) if (rm_range[ind] in pupil_snippet_range)]
				pupil_snippet_range = np.delete(pupil_snippet_range,rm_indices)
				pupil_snippet_range = pupil_snippet_range.tolist()

		print("took %f secs" % (time.time() - t))
		data_unfiltered = data[pupil_snippet_range]
		
		# High-pass filter to get rid of baseline offset
		# data_unfiltered = data
		
		data = lowpassFilterData(data_unfiltered, Fsamp,1)
		times = np.arange(len(data))/Fsamp
		if len(times_all) > 0:
			times = times + times_all[-1] + 1./Fs
		# Concatenate across the files
		data_all = np.append(data_all,data)
		data_all_unfiltered = np.append(data_all_unfiltered,data_unfiltered)
		times_all = np.append(times_all,times)

	return data_all, data_all_unfiltered, times_all

# Define data locations
csv_locations_HrtR = []
for i in range(len(pilot_files_HrtR)):
	csv_locations_HrtR += [[(pilot_dir+file) for file in pilot_files_HrtR[i]]]

csv_locations_PupD = []
for i in range(len(pilot_files_PupD)):
	csv_locations_PupD += [[(pilot_dir+file) for file in pilot_files_PupD[i]]]

# Set up global variables
labels = ['0.1 mg/kg', '0.3 mg/kg', '0.5 mgkg']
cmap = mpl.cm.brg

# Batch process pupil dilataion
for j in range(len(csv_locations_PupD)):
	data, data_unfiltered, times = process_PupD(csv_locations_PupD[j], 3043)
	ts_10min = np.argmin(np.abs(times - 600))
	data = (data - np.nanmean(data[:ts_10min]))/np.nanstd(data[:ts_10min])	# zscore based on baseline
	data_unfiltered = (data_unfiltered - np.nanmean(data_unfiltered[:ts_10min]))/np.nanstd(data_unfiltered[:ts_10min])
	
	t_start = times[0]
	t_stop = times[-1]
	
	data = running_mean(data, 1000)
	ts = np.linspace(t_start, t_stop, len(data))
	
	ts_10min = np.argmin(np.abs(ts - 600))
	ts_stop = np.argmin(np.abs(ts - 2820))
	p = np.polyfit(ts[ts_10min:ts_stop],data[ts_10min:ts_stop],deg = 1)
	data_fit = p[0]*ts[ts_10min:] + p[1]
	ts = np.linspace(t_start, t_stop, len(data))/60.
	
	plt.figure(1)
	#plt.plot(ts,data_unfiltered[:len(data)], 'r', label = 'unfiltered')
	plt.plot(ts,data,label = labels[j], color=cmap(j/5))
	plt.plot(ts[ts_10min:],data_fit[:len(ts[ts_10min:])], color = cmap(j/5))
	plt.xlabel('time (s)')
	plt.ylabel('Pupil Diameter (AU)')
#plt.ylim((-6,3))
plt.xlim((0,2820/60.))
plt.plot([600/60.,600/60.],[-6,3], 'k--')
plt.legend()
plt.show() 


# Batch process IBI
for j in range(len(csv_locations_HrtR)):
	data, times, ibi = process_HrtR(csv_locations_HrtR[j], Fs)
	ibi = (ibi - np.nanmean(ibi[:500]))/np.nanstd(ibi[:500])  # zscore based on baseline
	ibi = running_mean(ibi, 100)
	ibi_filtered = lowpassFilterData(ibi, Fs,2)
	# create linear fit for after injection
	t_start = times[0]
	t_stop = times[-1]
	ts = np.linspace(t_start, t_stop, len(ibi))
	ts_10min = np.argmin(np.abs(ts - 600))
	ts_stop = np.argmin(np.abs(ts - 2820))
	p = np.polyfit(ts[ts_10min:ts_stop],ibi[ts_10min:ts_stop],deg = 1)
	ibi_fit = p[0]*ts[ts_10min:] + p[1]

	ts = np.linspace(t_start, t_stop, len(ibi))/60.
	
	plt.figure(1)
	#plt.plot(ts,ibi, 'r', label = 'unfiltered')
	plt.plot(ts,ibi_filtered,label = labels[j], color=cmap(j/5))
	plt.plot(ts[ts_10min:],ibi_fit, color = cmap(j/5))
	plt.xlabel('time (s)')
	plt.ylabel('z-scored IBI (relative to baseline)')
plt.ylim((-6,3))
plt.xlim((0,2820/60.))
plt.plot([600/60.,600/60.],[-6,3], 'k--')
plt.legend()
plt.show() 
