import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
from neo import io
from PulseMonitorData import findIBIs
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse
from iscan_processing import get_iscan_data


# Set up code for particular day and block
hdf_filename = 'luig20160204_15_te1382.hdf'
filename = 'Luigi20160204_HDEEG'
#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
#hdf_location = hdf_filename
block_num = 2
eyedata_location = '/home/srsummerson/storage/eye-tracker/'+filename+'-Block'+str(block_num-1)+'.tda'

num_avg = 50 	# number of trials to compute running average of trial statistics over

# Load behavior data

hdf = tables.openFile(hdf_location)
counter_block1 = 0
counter_block3 = 0
running_avg_length = 20

state = hdf.root.task_msgs[:]['msg']
state_time = hdf.root.task_msgs[:]['time']
trial_type = hdf.root.task[:]['target_index']
# reward schedules
reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
# Target information: high-value target= targetH, low-value target= targetL
targetH = hdf.root.task[:]['targetH']
targetL = hdf.root.task[:]['targetL']
  
ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
ind_target_states = np.ravel(np.nonzero(state == 'target'))
ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
num_successful_trials = ind_check_reward_states.size
instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]
rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]

target1 = np.zeros(100)
reward1 = np.zeros(target1.size)
target2 = np.zeros(100)
reward2 = np.zeros(100)
target3 = np.zeros(ind_check_reward_states.size-200)
#target3 = np.zeros(np.min([num_successful_trials-200,100]))
reward3 = np.zeros(target3.size)
target_freechoice_block1 = np.zeros(70)
reward_freechoice_block1 = np.zeros(70)
target_freechoice_block3 = []
reward_freechoice_block3 = []
#reward_freechoice_block3 = np.zeros(np.min([num_successful_trials-200,100]))
trial1 = np.zeros(target1.size)
target_side1 = np.zeros(target1.size)
target_side2 = np.zeros(100)
trial3 = np.zeros(target3.size)
target_side3 = np.zeros(target3.size)
stim_trials = np.zeros(target3.size)

targetH_info = targetH[state_time[ind_target_states]]
targetL_info = targetL[state_time[ind_target_states]]

targetH_side = targetH_info[:,2]


'''
Target choices for all (free-choice only) and associated reward assignments
'''
for i in range(0,100):
    target_state1 = state[ind_check_reward_states[i] - 2]
    trial1[i] = instructed_or_freechoice[i]
    target_side1[i] = targetH_side[i]
    if target_state1 == 'hold_targetL':
        target1[i] = 1
        reward1[i] = rewarded_reward_scheduleL[i]
    else:
        target1[i] = 2
        reward1[i] = rewarded_reward_scheduleH[i]
    if trial1[i] == 2:
        target_freechoice_block1[counter_block1] = target1[i]
        reward_freechoice_block1[counter_block1] = reward1[i]
        counter_block1 += 1
for i in range(100,200):
    target_state2 = state[ind_check_reward_states[i] - 2]
    target_side2[i-100] = targetH_side[i]
    if target_state2 == 'hold_targetL':
        target2[i-100] = 1
        reward2[i-100] = rewarded_reward_scheduleL[i]
    else:
        target2[i-100] = 2
        reward2[i-100] = rewarded_reward_scheduleH[i]
   
for i in range(200,num_successful_trials):
#for i in range(200,np.min([num_successful_trials,300])):
    target_state3 = state[ind_check_reward_states[i] - 2]
    trial3[i-200] = instructed_or_freechoice[i]
    target_side3[i-200] = targetH_side[i]
    if target_state3 == 'hold_targetL':
        target3[i-200] = 1
        reward3[i-200] = rewarded_reward_scheduleL[i]
    else:
        target3[i-200] = 2
        reward3[i-200] = rewarded_reward_scheduleH[i]
        stim_trials[i-200] = 0
    if trial3[i-200]==1:   # instructed trial paired with stim
        stim_trials[i-200] = 1
    else:
        stim_trials[i-200] = 0
    if trial3[i-200] == 2:
        target_freechoice_block3.append(target3[i-200])
        reward_freechoice_block3.append(reward3[i-200])
        counter_block3 += 1
target_freechoice_block3 = np.array(target_freechoice_block3)
reward_freechoice_block3 = np.array(reward_freechoice_block3)

instructed_or_freechoice_block1 = instructed_or_freechoice[0:100]
instructed_or_freechoice_block3 = instructed_or_freechoice[200:num_successful_trials]

hdf.close()


# Number of stim trials in Block B
tot_stim_block2 = np.sum(np.equal(target2,1))
# Number of stim trials in Block A'
tot_stim_block3 = np.sum(stim_trials)

# Response times for stim trials in Block A'
ind_stim_block3 = np.ravel(np.nonzero(stim_trials)) + 200   	# gives trial index, not row index
row_ind_stim_block3 = ind_wait_states[ind_stim_block3]		# gives row index
row_ind_stim_block3_reward = ind_check_reward_states[ind_stim_block3]
response_time_stim_block3 = (state_time[row_ind_stim_block3_reward] - state_time[row_ind_stim_block3])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response time for non-stim trials in Block A'
ind_reg_block3 = np.ravel(np.equal(stim_trials,0)) + 200   	# gives trial index, not row index
row_ind_reg_block3 = ind_wait_states[ind_reg_block3]		# gives row index
row_ind_reg_block3_reward = ind_check_reward_states[ind_reg_block3]
response_time_reg_block3 = (state_time[row_ind_reg_block3_reward] - state_time[row_ind_reg_block3])/float(60)		# hdf rows are written at a rate of 60 Hz


# Load syncing data for hdf file and TDT recording
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
#mat_filename = 'Luigi20151229_HDEEG'+ '_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)


r = io.TdtIO(TDT_tank)
bl = r.read_block(lazy=False,cascade=True)
print "File read."
hdeeg = dict()
# Get Pulse and Pupil Data
for sig in bl.segments[block_num-1].analogsignals:
	if (sig.name == 'PupD 1'):
		pupil_data = np.ravel(sig)
		pupil_samprate = sig.sampling_rate.item()
	if (sig.name == 'HrtR 1'):
		pulse_data = np.ravel(sig)
		pulse_samprate = sig.sampling_rate.item()


# Load eye-tracker data
eye_data = get_iscan_data(eyedata_location)
eye_azimuth = eye_data['eye_az1']
eye_elevation = eye_data['eye_el1']
eye_diameter = eye_data['pupil_d1']  # include this for sanity check that eye data is synced properly
eye_tracker_samprate = 240

# divide up analysis for regular trials before stress trials, stress trials, and regular trials after stress trials are introduced
hdf_rows = np.ravel(hdf_times['row_number'])
hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])
dio_start = np.ravel(hdf_times['tdt_recording_start'])

# Convert DIO TDT sample numbers to for pupil and pulse data:
# if dio sample num is x, then data sample number is R*x where
# R = data_sample_rate/dio_sample_rate
pulse_dio_sample_num = (float(pulse_samprate)/float(dio_freq))*(dio_tdt_sample)
pupil_dio_sample_num = (float(pupil_samprate)/float(dio_freq))*(dio_tdt_sample)
eye_tracker_sample_num = (float(eye_tracker_samprate)/float(dio_freq))*(dio_tdt_sample - dio_start)
eye_tracker_sample_num = [int(eye_tracker_sample_num[i]) for i in range(len(eye_tracker_sample_num))]


state_row_ind_stim_block3 = state_time[row_ind_stim_block3]
state_row_ind_reg_block3 = state_time[row_ind_reg_block3]
pupil_ind_stim_block3 = np.zeros(row_ind_stim_block3.size)
pupil_ind_reg_block3 = np.zeros(row_ind_reg_block3.size)
eye_azimuth_ind_stim_block3 = []
eye_elevation_ind_stim_block3 = []
eye_azimuth_ind_reg_block3 = []
eye_elevation_ind_reg_block3 = []

for i in range(0,len(row_ind_stim_block3)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_stim_block3[i]))
	pupil_ind_stim_block3[i] = pupil_dio_sample_num[hdf_index]
	eye_azimuth_ind_stim_block3.append(eye_tracker_sample_num[hdf_index])
	eye_elevation_ind_stim_block3.append(eye_tracker_sample_num[hdf_index])
for i in range(0,len(row_ind_reg_block3)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg_block3[i]))
	pupil_ind_reg_block3[i] = pupil_dio_sample_num[hdf_index]
	eye_azimuth_ind_reg_block3.append(eye_tracker_sample_num[hdf_index])
	eye_elevation_ind_reg_block3.append(eye_tracker_sample_num[hdf_index])

'''
Notes: 
- filter pupil signal for dropping to zero (eyes closed)
- only consider data points for azimuth and elevation when pupil is non-zero
'''


cmap_stim_block3 = mpl.cm.winter
cmap_stim_block3_pupil = mpl.cm.autumn
plt.figure()
#for i in range(0,len(row_ind_stim_block3)):
for i in range(1,2):
	# Get data for 1000 ms hold period
	pupil_snippet = eye_diameter[eye_elevation_ind_stim_block3[i]:eye_elevation_ind_stim_block3[i]+eye_tracker_samprate]
	pupil_range = np.nanmax(pupil_snippet) - np.nanmin(pupil_snippet)
	pupil_snippet = pupil_snippet/pupil_range
	pupil_snippet = pupil_snippet - np.nanmean(pupil_snippet)
	azimuth = eye_azimuth[eye_azimuth_ind_stim_block3[i]:eye_azimuth_ind_stim_block3[i]+eye_tracker_samprate]
	elevation = eye_elevation[eye_elevation_ind_stim_block3[i]:eye_elevation_ind_stim_block3[i]+eye_tracker_samprate]
	
	'''
	plt.subplot(2,1,1)
	plt.plot(azimuth,elevation,color=cmap_stim_block3(i/float(len(row_ind_stim_block3))),linestyle='-.')
	plt.subplot(2,1,2)
	plt.plot(np.arange(len(diameter))/float(eye_tracker_samprate),diameter,color=cmap_stim_block3(i/float(len(row_ind_stim_block3))),linestyle='-')
	plt.plot(np.arange(len(pupil_snippet))/float(pupil_samprate),pupil_snippet,color=cmap_stim_block3_pupil(i/float(len(row_ind_stim_block3))),linestyle='-')
    
	'''
	pupil_snippet_range = range(0,len(pupil_snippet))
	eyes_closed = np.nonzero(np.less(pupil_snippet,1))
	eyes_closed = np.ravel(eyes_closed)
	if len(eyes_closed) > 1:
		find_blinks = eyes_closed[1:] - eyes_closed[:-1]
		blink_inds = np.ravel(np.nonzero(np.not_equal(find_blinks,1)))
		eyes_closed_ind = [eyes_closed[0]]
		eyes_closed_ind += eyes_closed[blink_inds].tolist()
		eyes_closed_ind += eyes_closed[blink_inds+1].tolist()
		eyes_closed_ind += [eyes_closed[-1]]
		eyes_closed_ind.sort()
		for i in np.arange(1,len(eyes_closed_ind),2):
			rm_range = range(np.maximum(eyes_closed_ind[i-1]-20,0),np.minimum(eyes_closed_ind[i] + 20,len(pupil_snippet)-1))
			rm_indices = [pupil_snippet_range.index(rm_range[ind]) for ind in range(0,len(rm_range)) if (rm_range[ind] in pupil_snippet_range)]
			pupil_snippet_range = np.delete(pupil_snippet_range,rm_indices)
			pupil_snippet_range = pupil_snippet_range.tolist()
	#pupil_snippet = signal.lfilter(lpf,1,pupil_snippet[eyes_open])
	pupil_snippet = pupil_snippet[pupil_snippet_range]
	azimuth = azimuth[pupil_snippet_range]
	elevation = elevation[pupil_snippet_range]
	
	# target_side3 is the side of the high-value target, so we should actually plot the opposite of that
	plt.plot(azimuth,elevation)
	plt.plot(target_side3[ind_stim_block3[i]-200],0,color='g',marker='o',markersize = 2.0)
	
#plt.show()	

#plt.close("all")
hdf.close()