import numpy as np 
import scipy as sp
import tables
from neo import io
from plexon import plexfile
from PulseMonitorData import findIBIs
from basicAnalysis import computePSTH, computeSpikeRatesPerChannel, computePeakPowerPerChannel, ElectrodeGridMat
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials



# Set up code for particular day and block
hdf_filename = 'mari20160424_03_te2035.hdf'
filename = 'Mario20160424'
#TDT_tank = '/home/srsummerson/storage/tdt/'+filename
TDT_tank = '/backup/subnetsrig/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
stim_freq = 100

#hdf_location = hdf_filename
block_num = 1

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

# Loading LFP data
r = io.TdtIO(TDT_tank)
bl = r.read_block(lazy=False,cascade=True)
print "File read."
lfp = dict()
# Get Pulse and Pupil Data
for sig in bl.segments[block_num-1].analogsignals:
	if (sig.name[0:4] == 'LFP1'):
		lfp_samprate = sig.sampling_rate.item()
		channel = sig.channel_index
		lfp[channel] = np.ravel(sig)
	if (sig.name[0:4] == 'LFP2'):
		channel = sig.channel_index + 96
		lfp[channel] = np.ravel(sig)

'''
stopped here 
'''

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
lfp_dio_sample_num = (float(lfp_samprate)/float(dio_freq))*(dio_tdt_sample - 1) + 1

state_row_ind_successful_stress = state_time[row_ind_successful_stress]
state_row_ind_successful_reg = state_time[row_ind_successful_reg]
state_row_ind_stress = state_time[row_ind_stress]
state_row_ind_reg = state_time[row_ind_reg]

time_successful_stress = np.zeros(len(row_ind_successful_stress))
time_successful_reg = np.zeros(len(row_ind_successful_reg))
lfp_ind_successful_stress = np.zeros(len(row_ind_successful_stress))
lfp_ind_successful_reg = np.zeros(len(row_ind_successful_reg))
	
for i in range(0,len(row_ind_successful_stress)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_stress[i]))
	lfp_ind_successful_stress[i] = lfp_dio_sample_num[hdf_index]
	time_successful_stress[i] = dio_tdt_sample[hdf_index]/dio_freq
for i in range(0,len(row_ind_successful_reg)):
	hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind_successful_reg[i]))
	lfp_ind_successful_reg[i] = lfp_dio_sample_num[hdf_index]
	time_successful_reg[i] = dio_tdt_sample[hdf_index]/dio_freq

ind_start_all_stress = row_ind_stress[0]
hdf_index_start_stress = np.argmin(np.abs(hdf_rows - state_row_ind_stress[0]))
sample_start_stress = dio_tdt_sample[hdf_index_start_stress]
time_start_stress = sample_start_stress/dio_freq
hdf_index_end_stress = np.argmin(np.abs(hdf_rows - state_row_ind_stress[-1]))
sample_end_stress = dio_tdt_sample[hdf_index_end_stress]
time_end_stress = sample_end_stress/dio_freq

hdf_index_start_reg = np.argmin(np.abs(hdf_rows - state_row_ind_reg[0]))
sample_start_reg = dio_tdt_sample[hdf_index_start_reg]
time_start_reg = sample_start_reg/dio_freq
hdf_index_end_reg = np.argmin(np.abs(hdf_rows - state_row_ind_reg[-1]))
sample_end_reg = dio_tdt_sample[hdf_index_end_reg]
time_end_reg = sample_end_reg/dio_freq

peak_power_stress = computePeakPowerPerChannel(lfp, lfp_samprate,stim_freq,sample_start_stress,sample_end_stress,[1,20])
peak_power_reg = computePeakPowerPerChannel(lfp, lfp_samprate,stim_freq,sample_start_reg,sample_end_reg,[1,20])

# Set colormap for pcolormesh plots so that nan values are plotted with black
cmap = mpl.cm.autumn
cmap.set_bad(color='k', alpha = 1.)

# Set up matrix for plotting peak powers
dx, dy = 1, 1
y, x = np.mgrid[slice(0,15,dy),
		slice(0,14,dx)]

power_mat_stress = ElectrodeGridMat(peak_power_stress)
power_mat_reg = ElectrodeGridMat(peak_power_reg)
power_mat_stress = np.ma.masked_invalid(power_mat_stress)
power_mat_reg = np.ma.masked_invalid(power_mat_reg)
power_mat_diff = ElectrodeGridMat(peak_power_stress - peak_power_reg)
power_mat_diff = np.ma.masked_invalid(power_mat_diff)

cmap = plt.get_cmap('RdBu')
cmap.set_bad(color='k', alpha = 1.)

z_max = np.max(np.append(peak_power_stress,peak_power_reg))
z_min = np.min(np.append(peak_power_stress,peak_power_reg))

#z_max_diff = np.max(peak_power_stress - peak_power_reg)
#z_min_diff = np.min(peak_power_stress - peak_power_reg)
z_max_diff = 0.03
z_min_diff = -0.03

plt.figure()
plt.subplot(1,2,1)
plt.pcolormesh(x,y,power_mat_reg,cmap=cmap,vmin=z_min,vmax = z_max)
plt.title('Regular Block')
plt.ylabel('Peak Power: 1-20 Hz')
plt.axis([x.min(),x.max(),y.min(),y.max()])
plt.subplot(1,2,2)
plt.pcolormesh(x,y,power_mat_stress,cmap=cmap,vmin=z_min,vmax = z_max)
plt.title('Stress Block')
plt.ylabel('Peak Power: 1-20 Hz')
plt.axis([x.min(),x.max(),y.min(),y.max()])
plt.colorbar()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_PeakPowers.svg')

plt.figure()
plt.subplot(1,1,1)
plt.pcolormesh(x,y,power_mat_diff,cmap=cmap,vmin= z_min_diff,vmax = z_max_diff)
plt.title('Difference in Peak Power: Stress - Regular')
plt.ylabel('Peak Power: 1-20 Hz')
plt.axis([x.min(),x.max(),y.min(),y.max()])
plt.colorbar()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_PeakPowerDifference.svg')


plt.close()



