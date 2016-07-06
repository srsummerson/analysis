import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
from neo import io
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD
from rt_calc import compute_rt_per_trial_StressTask

hdf_filename = 'mari20160614_03_te2237.hdf'
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
filename = 'Mario20160614'
block_num = 1


## self.target_index = 1 for instructed, 2 for free choice
## self.stress_trial =1 for stress trial, 0 for regular trial
state_time, ind_center_states, ind_check_reward_states, all_instructed_or_freechoice, all_stress_or_not, successful_stress_or_not,trial_success, target, reward = FreeChoiceBehavior_withStressTrials(hdf_location)

# Get reaction times for successful trials
reaction_time, total_vel, stress_indicator = compute_rt_per_trial_StressTask(hdf_location)

# Reaction time hists for successful stress versus regular trials
rt_stress_ind = np.ravel(np.nonzero(stress_indicator))
rt_reg_ind = np.ravel(np.nonzero(np.logical_not(stress_indicator)))
hist_successful_reg, bins_reg = np.histogram(reaction_time[rt_reg_ind],10)
hist_successful_reg = hist_successful_reg/float(len(reaction_time[rt_reg_ind]))

hist_successful_stress, bins_stress = np.histogram(reaction_time[rt_stress_ind],10)
hist_successful_stress = hist_successful_stress/float(len(reaction_time[rt_stress_ind]))

plt.figure()
plt.plot(bins_reg,hist_successful_reg,'r',label='Regular')
plt.plot(bins_stress,hist_successful_stress,'b',label='Stress')
plt.xlabel('Reaction time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_StressTaskReactionTimes.svg')
