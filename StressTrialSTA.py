'''
Center Out:
1/27/2016 - Block 1 (luig20160127_06_te1315.hdf), Block 3 (luig20160127_11_te1320.hdf)
1/28/2016 - Block 1 (luig20160128_02_te1325.hdf)
1/29/2016 - Block 1 (luig20160129_02_te1329.hdf)

Trial types:
1. Regular (before stress) and rewarded
2. Regular (before stress) and unrewarded
3. Regular (after stress) and rewarded
4. Regular (after stress) and unrewarded
5. Stress and rewarded
6. Stress and unrewarded
7. Stress and unsuccessful

Behavior:
- Fraction of selecting low vs high value target (reg vs stress trials)
- Rewards received for each selection
- Number of successful trials for each trial type (reg vs stress)
- Response time for each trial type, i.e time to successfully complete trial
Physiological:
- Distribution of IBIs for stress trials, regular trials before stress trials are introduced, and regular trials after stress 
  trials are introduced -- divide up between successful and unsuccessful trials
- Average pupil diameter during stress trials, regular trials before stress trials are introduced, and regular trials after stress 
  is introduced -- divide up between successful and unsuccessful trials
Neurological:
- Average power in beta over time of trial (aligned to target hold) (trial x time x power plot)
- Power over all frequencies for entire trial (trial x freq x power plot)
'''

import numpy as np 
import scipy as sp
import tables
from neo import io
from plexon import plexfile
from PulseMonitorData import findIBIs
from basicAnalysis import computeSTA
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / float(N) 

# Set up code for particular day and block
hdf_filename = 'mari20160128_04_te1327.hdf'
filename = 'Mario20160128'
plx_filename = 'Offline_eNe1.plx'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
plx_location = '/home/srsummerson/storage/tdt/'+filename+'/'+plx_filename
#hdf_location = hdf_filename
block_num = 1

plx = plexfile.openFile(plx_location)
spike_file = plx.spikes[:].data

# Load behavior data
## self.stress_trial =1 for stress trial, 0 for regular trial
hdf = tables.openFile(hdf_location)

state = hdf.root.task_msgs[:]['msg']
state_time = hdf.root.task_msgs[:]['time']
#trial_type = hdf.root.task[:]['target_index']
stress_type = hdf.root.task[:]['stress_trial']

  
ind_wait_states = np.ravel(np.nonzero(state == 'wait'))   # total number of unique trials
#ind_center_states = np.ravel(np.nonzero(state == 'center'))   
ind_target_states = np.ravel(np.nonzero(state == 'target')) # total number of trials (includes repeats if trial was incomplete)
ind_reward_states = np.ravel(np.nonzero(state == 'reward'))  # reward instead of check_reward
ind_target_transition_states = np.ravel(np.nonzero(state == 'targ_transition'))

successful_stress_or_not = np.ravel(stress_type[state_time[ind_reward_states]])
all_stress_or_not = np.ravel(stress_type[state_time[ind_target_states]])

num_trials = ind_target_states.size
num_successful_trials = ind_reward_states.size
total_states = state.size

trial_success = np.zeros(num_trials)
reward = np.zeros(num_trials)

for i in range(0,num_trials):
	if (state[np.minimum(ind_target_states[i]+3,total_states-1)] == 'reward'):	 
		trial_success[i] = 1
		reward[i] = 1
	else:
		trial_success[i] = 0
		reward[i] = 0 	# no reward givens

# Number of successful stress trials
tot_successful_stress = np.logical_and(trial_success,all_stress_or_not)
successful_stress_trials = float(np.sum(tot_successful_stress))/np.sum(all_stress_or_not)

# Number of successful non-stress trials
tot_successful_reg = np.logical_and(trial_success,np.logical_not(all_stress_or_not))
successful_reg_trials = float(np.sum(tot_successful_reg))/(num_trials - np.sum(all_stress_or_not))

# Response times for successful stress trials
ind_successful_stress = np.ravel(np.nonzero(tot_successful_stress))   	# gives trial index, not row index

row_ind_successful_stress = ind_target_states[ind_successful_stress]		# gives row index
ind_successful_stress_reward = np.ravel(np.nonzero(successful_stress_or_not))
row_ind_successful_stress_reward = ind_reward_states[ind_successful_stress_reward]
response_time_successful_stress = (state_time[row_ind_successful_stress_reward] - state_time[row_ind_successful_stress])/float(60)		# hdf rows are written at a rate of 60 Hz

# Response time for all stress trials
ind_stress = np.ravel(np.nonzero(all_stress_or_not))
row_ind_stress = ind_target_states[ind_stress]  # gives row index
row_ind_end_stress = np.zeros(len(row_ind_stress))
row_ind_end_stress = row_ind_stress + 2  # targ_transition state occurs two states later for unsuccessful trials
for i in range(0,len(row_ind_successful_stress)):
	ind = np.where(row_ind_stress == row_ind_successful_stress[i])[0]
	row_ind_end_stress[ind] = row_ind_successful_stress_reward[i]  # for successful trials, update with real end of trial
response_time_stress = (state_time[row_ind_end_stress] - state_time[row_ind_stress])/float(60)

# Response times for successful regular trials
ind_successful_reg = np.ravel(np.nonzero(tot_successful_reg))
row_ind_successful_reg = ind_target_states[ind_successful_reg]
ind_successful_reg_reward = np.ravel(np.nonzero(np.logical_not(successful_stress_or_not)))
row_ind_successful_reg_reward = ind_reward_states[ind_successful_reg_reward]
response_time_successful_reg = (state_time[row_ind_successful_reg_reward] - state_time[row_ind_successful_reg])/float(60)

# Response time for all regular trials
ind_reg = np.ravel(np.nonzero(np.logical_not(all_stress_or_not)))
row_ind_reg = ind_target_states[ind_reg]
row_ind_end_reg = np.zeros(len(row_ind_reg))
row_ind_end_reg = np.minimum(row_ind_reg + 2,total_states-1)  # target_transition state occues two states later for successful trials
for i in range(0,len(row_ind_successful_reg)):
	ind = np.where(row_ind_reg == row_ind_successful_reg[i])[0]
	row_ind_end_reg[ind] = row_ind_successful_reg_reward[i]
response_time_reg = (state_time[row_ind_end_reg] - state_time[row_ind_reg])/float(60)


# Load syncing data for hdf file and TDT recording
hdf_times = dict()
mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
sp.io.loadmat('/home/srsummerson/storage/syncHDF/'+mat_filename,hdf_times)

r = io.TdtIO(TDT_tank)
bl = r.read_block(lazy=False,cascade=True)
lfp = dict()
channels = [9,10,11,12,25,26,27,28]
# Get Pulse and Pupil Data
for sig in bl.segments[block_num-1].analogsignals:
	if (sig.name[0:4] == 'LFP1')&(sig.channel_index in channels):
		lfp[sig.channel_index] = sig
		lfp_samprate = sig.sampling_rate.item()

# divide up analysis for regular trials before stress trials, stress trials, and regular trials after stress trials are introduced
hdf_rows = np.ravel(hdf_times['row_number'])
hdf_rows = [val for val in hdf_rows]	# turn into a list so that the index method can be used later
dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

# Convert DIO TDT sample numbers to for pupil and pulse data:
# if dio sample num is x, then data sample number is R*(x-1) + 1 where
# R = data_sample_rate/dio_sample_rate
lfp_dio_sample_num = (float(lfp_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1

state_row_ind_successful_stress = state_time[row_ind_successful_stress]
state_row_ind_successful_reg = state_time[row_ind_successful_reg]
lfp_ind_successful_stress = np.zeros(row_ind_successful_stress.size)
lfp_ind_successful_reg_before = []
lfp_ind_successful_reg_after = []

state_row_ind_stress = state_time[row_ind_stress]
state_row_ind_reg = state_time[row_ind_reg]
lfp_ind_stress = np.zeros(row_ind_stress.size)
lfp_ind_reg_before = []
lfp_ind_reg_after = []


for i in range(0,len(row_ind_successful_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
	lfp_ind_successful_stress[i] = lfp_dio_sample_num[hdf_index]


for i in range(0,len(row_ind_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_stress[i]))
	lfp_ind_stress[i] = lfp_dio_sample_num[hdf_index]

ind_start_stress = row_ind_successful_stress[0]
ind_start_all_stress = row_ind_stress[0]
for i in range(0,len(state_row_ind_successful_reg)):
	if (row_ind_successful_reg[i] < ind_start_stress):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		lfp_ind_successful_reg_before.append(lfp_dio_sample_num[hdf_index])
	else:
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
		lfp_ind_successful_reg_after.append(lfp_dio_sample_num[hdf_index])

for i in range(0,len(state_row_ind_reg)):
	if (row_ind_reg[i] < ind_start_all_stress):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
		lfp_ind_reg_before.append(lfp_dio_sample_num[hdf_index])
	else:
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_reg[i]))
		lfp_ind_reg_after.append(lfp_dio_sample_num[hdf_index])
		
t_start_reg_before = float(lfp_ind_reg_before[0])/lfp_samprate
t_stop_reg_before = float(lfp_ind_reg_before[-1])/lfp_samprate
t_start_stress = float(lfp_ind_stress[0])/lfp_samprate
t_stop_stress = float(lfp_ind_stress[-1])/lfp_samprate
t_start_reg_after = float(lfp_ind_reg_after[0])/lfp_samprate
t_stop_reg_after = float(lfp_ind_reg_after[-1])/lfp_samprate

for channel in channels:
	sta_reg_before = computeSTA(spike_file,lfp[channel],channel,t_start_reg_before,t_stop_reg_before)
	sta_stress = computeSTA(spike_file,lfp[channel],channel,t_start_stress,t_stop_stress)
	sta_reg_after = computeSTA(spike_file,lfp[channel],channel,t_start_reg_after,t_stop_reg_after)
	plt.figure()
	plt.plot(sta_reg_before,'r',label='Reg Before')
	plt.plot(sta_stress,'b',label='Stress')
	plt.plot(sta_reg_after,'m',label='Reg After')
	plt.ylabel('STA (V)')
	plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+str(channel)+'_STA.svg')

'''
plots: hists of response times
'''
