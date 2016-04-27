import numpy as np 
import scipy as sp
import tables
from neo import io
from plexon import plexfile
from PulseMonitorData import findIBIs
from basicAnalysis import computeSTA, computePSTH, computeSpikeRatesPerChannel
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials



# Set up code for particular day and block
hdf_filename = 'mari20160418_04_te2002.hdf'
filename = 'Mario20160418'
plx_filename1 = 'Offline_eNe1.plx'
plx_filename2 = 'Offline_eNe2.plx'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename


#hdf_location = hdf_filename
block_num = 1

plx_location1 = '/home/srsummerson/storage/tdt/'+filename+'/'+'Block-'+ str(block_num) + '/'+plx_filename1
plx_location2 = '/home/srsummerson/storage/tdt/'+filename+'/'+'Block-'+ str(block_num) + '/'+plx_filename2

# Get spike data
plx1 = plexfile.openFile(plx_location1)
spike_file1 = plx1.spikes[:].data
plx2 = plexfile.openFile(plx_location2)
spike_file2 = plx2.spikes[:].data

# Load behavior data
## self.stress_trial =1 for stress trial, 0 for regular trial
state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward = FreeChoiceBehavior_withStressTrials(hdf_location)

# Total number of trials
num_trials = ind_center_states.size
total_states = state_time.size

# Number of successful stress trials
tot_successful_stress = np.logical_and(trial_success,all_stress_or_not)
successful_stress_trials = float(np.sum(tot_successful_stress))/np.sum(all_stress_or_not)

# Number of successful non-stress trials
tot_successful_reg = np.logical_and(trial_success,np.logical_not(all_stress_or_not))
successful_reg_trials = float(np.sum(tot_successful_reg))/(num_trials - np.sum(all_stress_or_not))

# Response times for successful stress trials
ind_successful_stress = np.ravel(np.nonzero(tot_successful_stress))   	# gives trial index, not row index
row_ind_successful_stress = ind_center_states[ind_successful_stress]		# gives row index
ind_successful_stress_reward = np.ravel(np.nonzero(successful_stress_or_not))
row_ind_successful_stress_reward = ind_check_reward_states[ind_successful_stress_reward]
response_time_successful_stress = (state_time[row_ind_successful_stress_reward] - state_time[row_ind_successful_stress])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response time for all stress trials
ind_stress = np.ravel(np.nonzero(all_stress_or_not))
row_ind_stress = ind_center_states[ind_stress]  # gives row index
row_ind_end_stress = np.zeros(len(row_ind_stress))
row_ind_end_stress = row_ind_stress + 2  # targ_transition state occurs two states later for unsuccessful trials
row_ind_end_stress[-1] = np.min([row_ind_end_stress[-1],len(state_time)-1])  # correct final incomplete trial

for i in range(0,len(row_ind_successful_stress)):
	ind = np.where(row_ind_stress == row_ind_successful_stress[i])[0]
	row_ind_end_stress[ind] = row_ind_successful_stress_reward[i]  # for successful trials, update with real end of trial
response_time_stress = (state_time[row_ind_end_stress] - state_time[row_ind_stress])/float(60)

# Response times for successful regular trials
ind_successful_reg = np.ravel(np.nonzero(tot_successful_reg))
row_ind_successful_reg = ind_center_states[ind_successful_reg]
ind_successful_reg_reward = np.ravel(np.nonzero(np.logical_not(successful_stress_or_not)))
row_ind_successful_reg_reward = ind_check_reward_states[ind_successful_reg_reward]
response_time_successful_reg = (state_time[row_ind_successful_reg_reward] - state_time[row_ind_successful_reg])/float(60)

# Response time for all regular trials
ind_reg = np.ravel(np.nonzero(np.logical_not(all_stress_or_not)))
row_ind_reg = ind_center_states[ind_reg]
row_ind_end_reg = np.zeros(len(row_ind_reg))
row_ind_end_reg = np.minimum(row_ind_reg + 5,total_states-1)  # target_transition state occues two states later for successful trials
for i in range(0,len(row_ind_successful_reg)):
	ind = np.where(row_ind_reg == row_ind_successful_reg[i])[0]
	row_ind_end_reg[ind] = row_ind_successful_reg_reward[i]
response_time_reg = (state_time[row_ind_end_reg] - state_time[row_ind_reg])/float(60)


# Load syncing data for hdf file and TDT recording
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

hdf_rows = np.ravel(hdf_times['row_number'])
hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])
dio_recording_start = hdf_times['tdt_recording_start']  # starting sample value
dio_tstart = dio_recording_start/dio_freq # starting time in seconds

state_row_ind_successful_stress = state_time[row_ind_successful_stress]
state_row_ind_successful_reg = state_time[row_ind_successful_reg]
state_row_ind_stress = state_time[row_ind_stress]
state_row_ind_reg = state_time[row_ind_reg]

time_successful_stress = np.zeros(len(row_ind_successful_stress))
time_successful_reg = np.zeros(len(row_ind_successful_reg))
	
for i in range(0,len(row_ind_successful_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
	time_successful_stress[i] = dio_tdt_sample[hdf_index]/dio_freq

ind_start_all_stress = row_ind_stress[0]
hdf_index_start_stress = np.argmin(np.abs(hdf_rows - state_row_ind_stress[0]))
time_start_stress = dio_tdt_sample[hdf_index_start_stress]/dio_freq
hdf_index_end_stress = np.argmin(np.abs(hdf_rows - state_row_ind_stress[-1]))
time_end_stress = dio_tdt_sample[hdf_index_end_stress]/dio_freq

hdf_index_start_reg = np.argmin(np.abs(hdf_rows - state_row_ind_reg[0]))
time_start_reg = dio_tdt_sample[hdf_index_start_reg]/dio_freq
hdf_index_end_reg = np.argmin(np.abs(hdf_rows - state_row_ind_reg[-1]))
time_end_reg = dio_tdt_sample[hdf_index_end_reg]/dio_freq

for i in range(0,len(row_ind_successful_reg)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
	time_successful_reg[i] = dio_tdt_sample[hdf_index]/dio_freq

window_before = 1
window_after = 2
binsize = 100

psth_stress, smooth_psth_stress = computePSTH(spike_file1,spike_file2,time_successful_stress,window_before,window_after, binsize)
psth_reg, smooth_psth_reg = computePSTH(spike_file1,spike_file2,time_successful_reg,window_before,window_after, binsize)
psth_time_window = np.arange(-window_before,window_after-float(binsize)/1000,float(binsize)/1000)

spikerates_stress, spikerates_sem_stress, labels_stress = computeSpikeRatesPerChannel(spike_file1,spike_file2,time_start_stress,time_end_stress)
spikerates_reg, spikerates_sem_reg, labels_reg = computeSpikeRatesPerChannel(spike_file1,spike_file2,time_start_reg,time_end_reg)

cmap_stress = mpl.cm.autumn
plt.figure()
for i in range(len(psth_stress)):
	unit_name = psth_stress.keys()[i]
	plt.subplot(1,2,1)
	plt.plot(psth_time_window,psth_stress[unit_name],color=cmap_stress(i/float(len(psth_stress))))
for i in range(len(psth_reg)):
	plt.subplot(1,2,2)
	plt.plot(psth_time_window,psth_reg[unit_name],color=cmap_stress(i/float(len(psth_stress))))
plt.subplot(1,2,1)
plt.title('Stress')
plt.ylabel('Firing Rate (Hz)')
plt.xlabel('Time (s)')
plt.subplot(1,2,2)
plt.title('Regular')
plt.xlabel('Time (s)')
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_PSTH-Stress.svg')

ind = np.range(len(spikerates_stress))
plt.figure()
plt.bar(ind, spikerates_stress, color = 'y', yerr = spikerates_sem_stress, label='Stress')
plt.xticks(ind, labels_stress)
plt.xlabel('Units')
plt.ylabel('Avg Firing Rate (Hz)')
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_AvgFiringRate-Stress.svg')

