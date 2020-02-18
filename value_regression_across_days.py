from DecisionMakingBehavior import ThreeTargetTask_RegressedFiringRatesWithValue_PictureOnset, ThreeTargetTask_RegressedFiringRatesWithRPE_RewardOnset
from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile
import numpy as np
from os import listdir
from scipy import io 
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Define units of interest
cd_units = [1, 3, 4, 17, 18, 20, 40, 41, 54, 56, 57, 63, 64, 72, 75, 81, 83, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
acc_units = [5, 6, 19, 22, 30, 39, 42, 43, 55, 58, 59, 69, 74, 77, 85, 90, 91, 102, 105, 121, 128]
all_units = np.append(cd_units, acc_units)
# List data

dir = "C:/Users/ss45436/Box/UC Berkeley/Cd Stim/Neural Correlates/Mario/spike_data/"


hdf_list_sham = [[dir + 'mari20161220_05_te2795.hdf'], \
			[dir + 'mari20170106_03_te2818.hdf'], \
			[dir + 'mari20170119_03_te2878.hdf', dir + 'mari20170119_05_te2880.hdf'], \
			[dir + 'mari20170126_03_te2931.hdf', dir + 'mari20170126_05_te2933.hdf'], \
			[dir + 'mari20170126_07_te2935.hdf', dir + 'mari20170126_11_te2939.hdf'], \
			[dir + 'mari20170204_03_te2996.hdf'], \
			[dir + 'mari20170207_05_te3018.hdf', dir + 'mari20170207_07_te3020.hdf', dir + 'mari20170207_09_te3022.hdf'], \
			[dir + 'mari20170207_13_te3026.hdf', dir + 'mari20170207_15_te3028.hdf', dir + 'mari20170207_17_te3030.hdf', dir + 'mari20170207_19_te3032.hdf'], \
			[dir + 'mari20170214_03_te3085.hdf', dir + 'mari20170214_07_te3089.hdf', dir + 'mari20170214_09_te3091.hdf', dir + 'mari20170214_11_te3093.hdf', dir + 'mari20170214_13_te3095.hdf', dir + 'mari20170214_16_te3098.hdf'], \
			[dir + 'mari20170215_03_te3101.hdf', dir + 'mari20170215_05_te3103.hdf', dir + 'mari20170215_07_te3105.hdf'], \
			[dir + 'mari20170220_07.hdf', dir + 'mari20170220_09.hdf', dir + 'mari20170220_11.hdf', dir + 'mari20170220_14.hdf'], \
			]
hdf_list_stim = [[dir + 'mari20161221_03_te2800.hdf'], \
			[dir + 'mari20161222_03_te2803.hdf'], \
			[dir + 'mari20170108_03_te2821.hdf'], \
			[dir + 'mari20170125_10_te2924.hdf', dir + 'mari20170125_12_te2926.hdf', dir + 'mari20170125_14_te2928.hdf'], \
			[dir + 'mari20170130_12_te2960.hdf', dir + 'mari20170130_13_te2961.hdf'], \
			[dir + 'mari20170131_05_te2972.hdf', dir + 'mari20170131_07_te2974.hdf'], \
			[dir + 'mari20170201_03_te2977.hdf'], \
			[dir + 'mari20170202_06_te2985.hdf', dir + 'mari20170202_08_te2987.hdf'], \
			[dir + 'mari20170209_03_te3047.hdf', dir + 'mari20170209_05_te3049.hdf', dir + 'mari20170209_08_te3052.hdf'], \
			[dir + 'mari20170216_03_te3108.hdf', dir + 'mari20170216_05_te3110.hdf', dir + 'mari20170216_08_te3113.hdf', dir + 'mari20170216_10_te3115.hdf'], \
			[dir + 'mari20170219_14.hdf', dir + 'mari20170219_16.hdf', dir + 'mari20170219_18.hdf']
			]

hdf_list = hdf_list_sham + hdf_list_stim

syncHDF_list_sham = [[dir + 'Mario20161220_b1_syncHDF.mat'], \
				[dir + 'Mario20170106_b1_syncHDF.mat'], \
				[dir + 'Mario20170119_b1_syncHDF.mat', dir + 'Mario20170119-2_b1_syncHDF.mat'], \
				[dir + 'Mario20170126_b1_syncHDF.mat', ''], \
				[dir + 'Mario20170126-2_b1_syncHDF.mat', dir + 'Mario20170126-3_b1_syncHDF.mat'], \
				[dir + 'Mario20170204_b1_syncHDF.mat'], \
				[dir + 'Mario20170207_b1_syncHDF.mat', dir + 'Mario20170207_b2_syncHDF.mat', dir + 'Mario20170207_b3_syncHDF.mat'], \
				[dir + 'Mario20170207-2_b1_syncHDF.mat', dir + 'Mario20170207-2_b2_syncHDF.mat', dir + 'Mario20170207-2_b3_syncHDF.mat', dir + 'Mario20170207-2_b4_syncHDF.mat'], \
				[dir + 'Mario20170214_b1_syncHDF.mat', dir + 'Mario20170214_b2_syncHDF.mat', dir + 'Mario20170214_b3_syncHDF.mat', dir + 'Mario20170214_b4_syncHDF.mat', dir + 'Mario20170214-2_b1_syncHDF.mat', dir + 'Mario20170214-2_b2_syncHDF.mat'], \
				[dir + 'Mario20170215_b1_syncHDF.mat', dir + 'Mario20170215_b2_syncHDF.mat', dir + 'Mario20170215_b3_syncHDF.mat'], \
				[dir + 'Mario20170220_b1_syncHDF.mat', dir + 'Mario20170220_b2_syncHDF.mat', dir + 'Mario20170220_b3_syncHDF.mat', dir + 'Mario20170220_b4_syncHDF.mat'], \
				]
syncHDF_list_stim = [[dir + 'Mario20161221_b1_syncHDF.mat'], \
				[dir + 'Mario20161222_b1_syncHDF.mat'], \
				[dir + 'Mario20170108_b1_syncHDF.mat'], \
				[dir + 'Mario20170125_b1_syncHDF.mat', dir + 'Mario20170125-2_b1_syncHDF.mat', dir + 'Mario20170125-3_b1_syncHDF.mat'], \
				['', dir + 'Mario20170130-2_b1_syncHDF.mat'], \
				[dir + 'Mario20170131_b1_syncHDF.mat', dir + 'Mario20170131-2_b1_syncHDF.mat'], \
				[dir + 'Mario20170201_b1_syncHDF.mat'], \
				[dir + 'Mario20170202_b1_syncHDF.mat', dir + 'Mario20170202-2_b1_syncHDF.mat'], \
				[dir + 'Mario20170209_b1_syncHDF.mat', dir + 'Mario20170209_b2_syncHDF.mat', dir + 'Mario20170209_b3_syncHDF.mat'], \
				[dir + 'Mario20170216_b1_syncHDF.mat', dir + 'Mario20170216_b2_syncHDF.mat', dir + 'Mario20170216_b3_syncHDF.mat', dir + 'Mario20170216_b4_syncHDF.mat'], \
				[dir + 'Mario20170219_b1_syncHDF.mat', dir + 'Mario20170219_b2_syncHDF.mat', dir + 'Mario20170219_b3_syncHDF.mat']
				]

syncHDF_list = syncHDF_list_sham + syncHDF_list_stim

spike_list_sham = [[[dir + 'Mario20161220_Block-1_eNe1_Offline.csv', dir + 'Mario20161220_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170106_Block-1_eNe1_Offline.csv', dir + 'Mario20170106_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170119_Block-1_eNe1_Offline.csv', dir + 'Mario20170119_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170119-2_Block-1_eNe1_Offline.csv', dir + 'Mario20170119-2_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170126_Block-1_eNe1_Offline.csv', dir + 'Mario20170126_Block-1_eNe2_Offline.csv'], ['']], \
			  [[dir + 'Mario20170126-2_Block-1_eNe1_Offline.csv', dir + 'Mario20170126-2_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170126-3_Block-1_eNe1_Offline.csv', dir + 'Mario20170126-3_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170204_Block-1_eNe1_Offline.csv', dir + 'Mario20170204_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170207_Block-1_eNe1_Offline.csv', dir + 'Mario20170207_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170207_Block-2_eNe1_Offline.csv', dir + 'Mario20170207_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170207_Block-3_eNe1_Offline.csv', dir + 'Mario20170207_Block-3_eNe2_Offline.csv']],\
			  [[dir + 'Mario20170207-2_Block-1_eNe1_Offline.csv', dir +'Mario20170207-2_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170207-2_Block-2_eNe1_Offline.csv', dir + 'Mario20170207-2_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170207-2_Block-3_eNe1_Offline.csv', dir + 'Mario20170207-2_Block-3_eNe2_Offline.csv'], [dir + 'Mario20170207-2_Block-4_eNe1_Offline.csv','']], \
			  [[dir + 'Mario20170214_Block-1_eNe1_Offline.csv', dir + 'Mario20170214_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170214_Block-2_eNe1_Offline.csv', dir + 'Mario20170214_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170214_Block-3_eNe1_Offline.csv', dir + 'Mario20170214_Block-3_eNe2_Offline.csv'], [dir + 'Mario20170214_Block-4_eNe1_Offline.csv', dir + 'Mario20170214_Block-4_eNe2_Offline.csv'], [dir + 'Mario20170214-2_Block-1_eNe1_Offline.csv', dir + 'Mario20170214-2_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170214-2_Block-2_eNe1_Offline.csv', dir + 'Mario20170214-2_Block-2_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170215_Block-1_eNe1_Offline.csv', dir + 'Mario20170215_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170215_Block-2_eNe1_Offline.csv', dir + 'Mario20170215_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170215_Block-3_eNe1_Offline.csv', dir + 'Mario20170215_Block-3_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170220_Block-1_eNe1_Offline.csv', dir + 'Mario20170220_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170220_Block-2_eNe1_Offline.csv', dir + 'Mario20170220_Block-2_eNe2_Offline.csv'], ['',''], [dir + 'Mario20170220_Block-4_eNe1_Offline.csv', dir + 'Mario20170220_Block-4_eNe2_Offline.csv']], \
			  ]
spike_list_stim = [[[dir + 'Mario20161221_Block-1_eNe1_Offline.csv', dir + 'Mario20161221_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20161222_Block-1_eNe1_Offline.csv', dir + 'Mario20161222_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170108_Block-1_eNe1_Offline.csv', dir + 'Mario20170108_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170125_Block-1_eNe1_Offline.csv', dir + 'Mario20170125_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170125-2_Block-1_eNe1_Offline.csv', dir + 'Mario20170125-2_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170125-3_Block-1_eNe1_Offline.csv', dir + 'Mario20170125-3_Block-1_eNe2_Offline.csv']], \
			  [[''], [dir + 'Mario20170130-2_Block-1_eNe1_Offline.csv', dir +'Mario20170130-2_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170131_Block-1_eNe1_Offline.csv', dir + 'Mario20170131_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170131-2_Block-1_eNe1_Offline.csv', dir + 'Mario20170131-2_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170201_Block-1_eNe1_Offline.csv', dir + 'Mario20170201_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170202_Block-1_eNe1_Offline.csv', dir + 'Mario20170202_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170202-2_Block-1_eNe1_Offline.csv', dir + 'Mario20170202-2_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170209_Block-1_eNe1_Offline.csv', dir + 'Mario20170209_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170209_Block-2_eNe1_Offline.csv', dir + 'Mario20170209_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170209_Block-3_eNe1_Offline.csv', dir + 'Mario20170209_Block-3_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170216_Block-1_eNe1_Offline.csv', dir + 'Mario20170216_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170216_Block-2_eNe1_Offline.csv', dir + 'Mario20170216_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170216_Block-3_eNe1_Offline.csv', dir + 'Mario20170216_Block-3_eNe2_Offline.csv'], [dir + 'Mario20170216_Block-4_eNe1_Offline.csv', dir + 'Mario20170216_Block-4_eNe2_Offline.csv']], \
			  [[dir + 'Mario20170219_Block-1_eNe1_Offline.csv', dir + 'Mario20170219_Block-1_eNe2_Offline.csv'], [dir + 'Mario20170219_Block-2_eNe1_Offline.csv', dir + 'Mario20170219_Block-2_eNe2_Offline.csv'], [dir + 'Mario20170219_Block-3_eNe1_Offline.csv', dir + 'Mario20170219_Block-3_eNe2_Offline.csv']], \
			  ]
spike_list = spike_list_sham + spike_list_stim


def BinChangeInFiringRatesByValue(Q_early, Q_late, Q_bins, FR_early, FR_late):
	'''
	This method sorts FRs by their associated values and bins firing rates accordingly, in order to make a 
	value versus FR curve where value is binned. 

	Input:
	- Q_early: list of arrays of Q-values, where each array corresponds to a particular unit and entry in the 
				array corresponds to a trial. data is taken only from first 100 - 150 "early" trials.
	- Q_late: list of arrays of Q-values, where each array corresponds to a particular unit and entry in the 
				array corresponds to a trial. data is taken only starting from trial 200 or 250 ("late" trials).
	- Q_bins: array defining bin edges for binning the Q-values
	- FR_early: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_early corresponds to an entry in Q_early.
	- FR_late: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_late corresponds to an entry in Q_late.
	'''
	num_units = len(Q_early)
	num_bins = len(Q_bins) - 1
	delta_FR = np.zeros((num_units,num_bins))
	norm_delta_FR = np.zeros((num_units,num_bins))
	total_nonnans_delta_FR = np.zeros(num_bins)
	total_nonnans_norm_delta_FR = np.zeros(num_bins)
	for i in range(num_units):
		FR_e_means, bin_edges, binnumber = stats.binned_statistic(Q_early[i], FR_early[i], statistic= 'mean', bins = Q_bins)
		FR_l_means, bin_edges, binnumber = stats.binned_statistic(Q_late[i], FR_late[i], statistic= 'mean', bins = Q_bins)
		delta_FR[i,:] = FR_l_means - FR_e_means
		norm_delta_FR[i,:] = delta_FR[i,:]/FR_e_means
		total_nonnans_delta_FR += np.greater(np.abs(delta_FR[i,:]),0)
		total_nonnans_norm_delta_FR += np.greater(np.abs(norm_delta_FR[i,:]), 0)
	
	avg_delta_FR = np.nanmean(delta_FR, axis = 0)
	sem_delta_FR = np.nanstd(delta_FR, axis = 0)/np.sqrt(total_nonnans_delta_FR)
	avg_norm_delta_FR = np.nanmean(norm_delta_FR, axis = 0)
	sem_norm_delta_FR = np.nanstd(norm_delta_FR, axis = 0)/np.sqrt(total_nonnans_norm_delta_FR)

	return Q_bins, delta_FR, norm_delta_FR, avg_delta_FR, sem_delta_FR, avg_norm_delta_FR, sem_norm_delta_FR

def zscore_FR_together(FR_early, FR_late):
	'''
	Z-score corresponding arrays together.

	Input:
	- FR_early: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_early corresponds to an entry in Q_early.
	- FR_late: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_late corresponds to an entry in Q_late.
	Output:
	- FR_early_zscore: same format as FR_early, but with entries z-scored using corresponding means and std from 
				early and late arrays
	- FR_late_zscore: same format as FR_late, but with entries z-scored using corresponding means and std from 
				early and late arrays
	'''
	FR_early_zscore = []
	FR_late_zscore = []
	num_units = len(FR_early)
	for i in range(num_units):
		all_rates = np.append(FR_early[i], FR_late[i])
		rate_mean = np.nanmean(all_rates)
		rate_std = np.nanstd(all_rates)
		FR_early_zscore += [(FR_early[i] - rate_mean)/float(rate_std)]
		FR_late_zscore += [(FR_late[i] - rate_mean)/float(rate_std)]

	return FR_early_zscore, FR_late_zscore

def zscore_FR_separate(FR_late):
	'''
	Z-score corresponding arrays.

	Input:
	- FR_late: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_late corresponds to an entry in Q_late.
	Output:
	- FR_late_zscore: same format as FR_late, but with entries z-scored using corresponding means and std from 
				early and late arrays
	'''
	FR_late_zscore = []
	num_units = len(FR_late)
	for i in range(num_units):
		all_rates = FR_late[i]
		rate_mean = np.nanmean(all_rates)
		rate_std = np.nanstd(all_rates)
		FR_late_zscore += [(FR_late[i] - rate_mean)/float(rate_std)]

	return FR_late_zscore

def BinFiringRatesByValue(Q_early, Q_late, Q_bins, FR_early, FR_late):
	'''
	This method sorts FRs by their associated values and bins firing rates accordingly, in order to make a 
	value versus FR curve where value is binned. 

	Input:
	- Q_early: list of arrays of Q-values, where each array corresponds to a particular unit and entry in the 
				array corresponds to a trial. data is taken only from first 100 - 150 "early" trials.
	- Q_late: list of arrays of Q-values, where each array corresponds to a particular unit and entry in the 
				array corresponds to a trial. data is taken only starting from trial 200 or 250 ("late" trials).
	- Q_bins: array defining bin edges for binning the Q-values
	- FR_early: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_early corresponds to an entry in Q_early.
	- FR_late: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_late corresponds to an entry in Q_late.
	'''
	num_units = len(Q_early)
	num_bins = len(Q_bins) - 1
	all_FR_early = np.array([])
	all_FR_late = np.array([])

	# add in z-scoring before flattening
	FR_early_zscore, FR_late_zscore = zscore_FR_together(FR_early, FR_late)

	all_FR_early = [item for sublist in FR_early_zscore for item in sublist]
	all_FR_late = [item for sublist in FR_late_zscore for item in sublist]
	all_Q_early = [item for sublist in Q_early for item in sublist]
	all_Q_late = [item for sublist in Q_late for item in sublist]

	FR_e_means, bin_edges, binnumber_e = stats.binned_statistic(all_Q_early, all_FR_early, statistic= 'mean', bins = Q_bins)
	FR_l_means, bin_edges, binnumber_l = stats.binned_statistic(all_Q_late, all_FR_late, statistic= 'mean', bins = Q_bins)
	
	FR_e_sem = np.zeros(num_bins)
	FR_l_sem = np.zeros(num_bins)

	bin_FR_early_all = []
	bin_FR_late_all = []

	for i in range(1,num_bins+1):
		bin_FR_early = [all_FR_early[ind] for ind in range(len(all_FR_early)) if binnumber_e[ind]==i]
		bin_FR_late = [all_FR_late[ind] for ind in range(len(all_FR_late)) if binnumber_l[ind]==i]
		FR_e_sem[i-1] = np.nanstd(bin_FR_early)/np.sqrt(len(bin_FR_early))
		FR_l_sem[i-1] = np.nanstd(bin_FR_late)/np.sqrt(len(bin_FR_late))
		bin_FR_early_all += [np.array(bin_FR_early)]
		bin_FR_late_all += [np.array(bin_FR_late)]

	return FR_e_means, FR_l_means, FR_e_sem, FR_l_sem, bin_FR_early_all, bin_FR_late_all

def BinFiringRatesByValue_SepReg(Q_early, Q_late, Q_bins, FR_early, FR_late, beta_late):
	'''
	This method sorts FRs by their associated values and bins firing rates accordingly, in order to make a 
	value versus FR curve where value is binned. 

	Results are separated by units that had a positive beta coefficient relating to value and units that 
	had a negative beta coefficient relating to value.

	Input:
	- Q_early: list of arrays of Q-values, where each array corresponds to a particular unit and entry in the 
				array corresponds to a trial. data is taken only from first 100 - 150 "early" trials.
	- Q_late: list of arrays of Q-values, where each array corresponds to a particular unit and entry in the 
				array corresponds to a trial. data is taken only starting from trial 200 or 250 ("late" trials).
	- Q_bins: array defining bin edges for binning the Q-values
	- FR_early: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_early corresponds to an entry in Q_early.
	- FR_late: list of arrays of firing rates, where each array corresponds to a particular unit and entry in the
				array corresponds to a trial. each entry in FR_late corresponds to an entry in Q_late.
	'''
	num_units = len(Q_early)
	num_bins = len(Q_bins) - 1
	all_FR_early = np.array([])
	all_FR_late = np.array([])

	# add in z-scoring before flattening
	FR_early_zscore, FR_late_zscore = zscore_FR_together(FR_early, FR_late)
	FR_late_zscore = zscore_FR_separate(FR_late)

	all_FR_early_posreg = [item for i,sublist in enumerate(FR_early_zscore) for item in sublist if beta_late[i] > 0]
	all_FR_late_posreg = [item for i,sublist in enumerate(FR_late_zscore) for item in sublist if beta_late[i] > 0]
	all_Q_early_posreg = [item for i,sublist in enumerate(Q_early) for item in sublist if beta_late[i] > 0]
	all_Q_late_posreg = [item for i,sublist in enumerate(Q_late) for item in sublist if beta_late[i] > 0]

	all_FR_early_negreg = [item for i,sublist in enumerate(FR_early_zscore) for item in sublist if beta_late[i] < 0]
	all_FR_late_negreg = [item for i,sublist in enumerate(FR_late_zscore) for item in sublist if beta_late[i] < 0]
	all_Q_early_negreg = [item for i,sublist in enumerate(Q_early) for item in sublist if beta_late[i] < 0]
	all_Q_late_negreg = [item for i,sublist in enumerate(Q_late) for item in sublist if beta_late[i] < 0]

	FR_e_means_posreg, bin_edges, binnumber_e_posreg = stats.binned_statistic(all_Q_early_posreg, all_FR_early_posreg, statistic= 'mean', bins = Q_bins)
	FR_l_means_posreg, bin_edges, binnumber_l_posreg = stats.binned_statistic(all_Q_late_posreg, all_FR_late_posreg, statistic= 'mean', bins = Q_bins)
	
	FR_e_means_negreg, bin_edges, binnumber_e_negreg = stats.binned_statistic(all_Q_early_negreg, all_FR_early_negreg, statistic= 'mean', bins = Q_bins)
	FR_l_means_negreg, bin_edges, binnumber_l_negreg = stats.binned_statistic(all_Q_late_negreg, all_FR_late_negreg, statistic= 'mean', bins = Q_bins)
	

	FR_e_sem_posreg = np.zeros(num_bins)
	FR_l_sem_posreg = np.zeros(num_bins)
	FR_e_sem_negreg = np.zeros(num_bins)
	FR_l_sem_negreg = np.zeros(num_bins)

	bin_FR_early_posreg_all = []
	bin_FR_late_posreg_all = []
	bin_FR_early_negreg_all = []
	bin_FR_late_negreg_all = []

	for i in range(1,num_bins+1):
		bin_FR_early_posreg = [all_FR_early_posreg[ind] for ind in range(len(all_FR_early_posreg)) if binnumber_e_posreg[ind]==i]
		bin_FR_late_posreg = [all_FR_late_posreg[ind] for ind in range(len(all_FR_late_posreg)) if binnumber_l_posreg[ind]==i]
		
		FR_e_sem_posreg[i-1] = np.nanstd(bin_FR_early_posreg)/np.sqrt(len(bin_FR_early_posreg))
		FR_l_sem_posreg[i-1] = np.nanstd(bin_FR_late_posreg)/np.sqrt(len(bin_FR_late_posreg))

		bin_FR_early_negreg = [all_FR_early_negreg[ind] for ind in range(len(all_FR_early_negreg)) if binnumber_e_negreg[ind]==i]
		bin_FR_late_negreg = [all_FR_late_negreg[ind] for ind in range(len(all_FR_late_negreg)) if binnumber_l_negreg[ind]==i]
		
		FR_e_sem_negreg[i-1] = np.nanstd(bin_FR_early_negreg)/np.sqrt(len(bin_FR_early_negreg))
		FR_l_sem_negreg[i-1] = np.nanstd(bin_FR_late_negreg)/np.sqrt(len(bin_FR_late_negreg))

		bin_FR_early_posreg_all += [np.array(bin_FR_early_posreg)]
		bin_FR_late_posreg_all += [np.array(bin_FR_late_posreg)]
		bin_FR_early_negreg_all += [np.array(bin_FR_early_negreg)]
		bin_FR_late_negreg_all += [np.array(bin_FR_late_negreg)]

	return FR_e_means_posreg, FR_e_means_negreg, FR_l_means_posreg, FR_l_means_negreg, FR_e_sem_posreg, FR_e_sem_negreg, FR_l_sem_posreg, FR_l_sem_negreg, \
			bin_FR_early_posreg_all, bin_FR_late_posreg_all, bin_FR_early_negreg_all, bin_FR_late_negreg_all

####### Start code ##############


# Define code parameters
num_files = len(hdf_list)
t_before = 0.
t_after = 0.4
smoothed = 1

for i in range(num_files)[10:]:
	hdf = hdf_list[i]
	sync = syncHDF_list[i]
	spike = spike_list[i]

	if spike[0]!= ['']:
		spike_data1 = OfflineSorted_CSVFile(spike[0][0])
		spike_data2 = OfflineSorted_CSVFile(spike[0][1])
		good_channels = np.append(spike_data1.good_channels, spike_data2.good_channels)
		
		channels = np.array([chan for chan in all_units if chan in good_channels])

		for chann in channels:

			ThreeTargetTask_RegressedFiringRatesWithRPE_RewardOnset(dir, hdf, sync, spike, chann, t_before, t_after, smoothed)


"""
Average across all files
- use pre-processed data in picture_onset_fr/ folder generated from ThreeTargetTask_RegressedFiringRatesWithValue_PictureOnset to identify 
  encoding of units during picture onset where encoding is determined via linear regression
    - pre-processed files contain Non-zero firing rates from trials, Q-values, beta values, p-values, and r-squared values
- regression performed only on last 100 trials in Block A and trials where firing rate was 0 Hz were excluded
- Q-values in regression comes from traditional Q-learning mordel and is fit from data from all trials
"""


spike_dir = dir + 'picture_onset_fr/'
filenames = listdir(spike_dir)

rsquared_blockA_cd = np.array([])
rsquared_blocksAB_cd = np.array([])
rsquared_blockA_acc = np.array([])
rsquared_blocksAB_acc = np.array([])
sig_reg_blockA_cd = np.array([])		# array to hold all significant regressors
sig_reg_blocksAB_cd = np.array([])		# array to hold all significant regressors
sig_reg_blockA_acc = np.array([])		# array to hold all significant regressors
sig_reg_blocksAB_acc = np.array([])		# array to hold all significant regressors
num_units = len(filenames)

count_LV_blockA_cd = 0
count_MV_blockA_cd = 0
count_HV_blockA_cd = 0
count_LVMV_blockA_cd = 0
count_LVHV_blockA_cd = 0
count_MVHV_blockA_cd = 0
count_LVMVHV_blockA_cd = 0
count_rt_blockA_cd = 0
count_mt_blockA_cd = 0
count_choice_blockA_cd = 0
count_reward_blockA_cd = 0

count_LV_blocksAB_cd = 0
count_MV_blocksAB_cd = 0
count_HV_blocksAB_cd = 0
count_LVMV_blocksAB_cd = 0
count_LVHV_blocksAB_cd = 0
count_MVHV_blocksAB_cd = 0
count_LVMVHV_blocksAB_cd = 0
count_rt_blocksAB_cd = 0
count_mt_blocksAB_cd = 0
count_choice_blocksAB_cd = 0
count_reward_blocksAB_cd = 0

count_LV_blockA_acc = 0
count_MV_blockA_acc = 0
count_HV_blockA_acc = 0
count_LVMV_blockA_acc = 0
count_LVHV_blockA_acc = 0
count_MVHV_blockA_acc = 0
count_LVMVHV_blockA_acc = 0
count_rt_blockA_acc = 0
count_mt_blockA_acc = 0
count_choice_blockA_acc = 0
count_reward_blockA_acc = 0

count_LV_blocksAB_acc = 0
count_MV_blocksAB_acc = 0
count_HV_blocksAB_acc = 0
count_LVMV_blocksAB_acc = 0
count_LVHV_blocksAB_acc = 0
count_MVHV_blocksAB_acc = 0
count_LVMVHV_blocksAB_acc = 0
count_rt_blocksAB_acc = 0
count_mt_blocksAB_acc = 0
count_choice_blocksAB_acc = 0
count_reward_blocksAB_acc = 0

count_blockA_acc = 0
count_blocksAB_acc = 0
count_blockA_cd = 0
count_blocksAB_cd = 0

channels_blockA_cd = np.array([])
channels_blocksAB_cd = np.array([])
channels_blockA_acc = np.array([])
channels_blocksAB_acc = np.array([])

stim_file = 0

Q_late_mv_cd_sham = []
Q_early_mv_cd_sham = []
FR_late_mv_cd_sham = []
FR_early_mv_cd_sham = []
Q_late_other_cd_sham = []
Q_early_other_cd_sham = []
FR_late_other_cd_sham = []
FR_early_other_cd_sham = []
Q_late_mv_acc_sham = []
Q_early_mv_acc_sham = []
FR_late_mv_acc_sham = []
FR_early_mv_acc_sham = []
Q_late_other_acc_sham = []
Q_early_other_acc_sham = []
FR_late_other_acc_sham = []
FR_early_other_acc_sham = []

Q_late_mv_cd_stim = []
Q_early_mv_cd_stim = []
FR_late_mv_cd_stim = []
FR_early_mv_cd_stim = []
Q_late_other_cd_stim = []
Q_early_other_cd_stim = []
FR_late_other_cd_stim = []
FR_early_other_cd_stim = []
Q_late_mv_acc_stim = []
Q_early_mv_acc_stim = []
FR_late_mv_acc_stim = []
FR_early_mv_acc_stim = []
Q_late_other_acc_stim = []
Q_early_other_acc_stim = []
FR_late_other_acc_stim = []
FR_early_other_acc_stim = []

# recording beta values for mv coding units
beta_late_mv_cd_stim = np.array([])
beta_late_mv_acc_stim = np.array([])
beta_late_mv_cd_sham = np.array([])
beta_late_mv_acc_sham = np.array([])


syncHDF_list_stim_flat = [item for sublist in syncHDF_list_stim for item in sublist]

'''
Loop through pre-processed files to count up the number of neurons that have significant beta values for the various
regressors. 
'''

for filen in filenames:
	data = dict()
	sp.io.loadmat(spike_dir + filen, data)
	chan_ind = filen.index('Channel')
	unit_ind = filen.index('Unit')
	channel_num = float(filen[chan_ind + 8:unit_ind - 3])

	sync_name = dir + filen[:chan_ind-3] + '_syncHDF.mat'
	#print sync_name
	if sync_name in syncHDF_list_stim_flat:
		stim_file = 1
	else:
		stim_file = 0
	#print stim_file
	
	regression_labels = data['regression_labels']

	if channel_num in cd_units:

		if ('beta_values_blockA' in data.keys()):
			beta_blockA = np.ravel(data['beta_values_blockA'])
			pvalues_blockA = np.ravel(data['pvalues_blockA'])
			sig_beta_blockA = beta_blockA*np.ravel(np.less(pvalues_blockA, 0.05))
			if count_blockA_cd==0:
				sig_reg_blockA_cd = sig_beta_blockA
			else:
				sig_reg_blockA_cd = np.vstack([sig_reg_blockA_cd, sig_beta_blockA])
			count_blockA_cd += 1

			val_check = [sig_beta_blockA[0], sig_beta_blockA[1], sig_beta_blockA[2], sig_beta_blockA[7], sig_beta_blockA[8], sig_beta_blockA[9]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			MV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1 or val_check[3]==1:
				LV_enc = 1
			if val_check[1]==1 or val_check[4]==1:
				MV_enc = 1
				if val_check[1]==1:
					beta_early_mv = sig_beta_blockA[1]
				else:
					beta_early_mv = sig_beta_blockA[8]
			if val_check[2]==1 or val_check[5]==1:
				HV_enc = 1

			if ((LV_enc==1) and (MV_enc==0) and (HV_enc==0)):
				count_LV_blockA_cd += 1
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==0)):
				count_MV_blockA_cd += 1
			elif ((LV_enc==0) and (MV_enc==0) and (HV_enc==1)):
				count_HV_blockA_cd += 1
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==0)):
				count_LVMV_blockA_cd += 1
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==1)):
				count_LVHV_blockA_cd += 1
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==1)):
				count_MVHV_blockA_cd += 1
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==1)):
				count_LVMVHV_blockA_cd += 1
			else:
				no_val_enc = 1

			if no_val_enc:
				max_beta = np.argmax(np.abs(sig_beta_blockA))
				if max_beta==3:
					count_rt_blockA_cd += 1
				if max_beta==4:
					count_mt_blockA_cd += 1
				if max_beta==5:
					count_choice_blockA_cd +=1
				if max_beta==6:
					count_reward_blockA_cd +=1

		if ('beta_values_blocksAB' in data.keys()):
			beta_blocksAB = np.ravel(data['beta_values_blocksAB'])
			pvalues_blocksAB = np.ravel(data['pvalues_blocksAB'])
			sig_beta_blocksAB = beta_blocksAB*np.ravel(np.less(pvalues_blocksAB, 0.05))
			if count_blocksAB_cd == 0:
				sig_reg_blocksAB_cd = sig_beta_blocksAB
			else:
				sig_reg_blocksAB_cd = np.vstack([sig_reg_blocksAB_cd, sig_beta_blocksAB])
			count_blocksAB_cd += 1

			val_check = [sig_beta_blocksAB[0], sig_beta_blocksAB[1], sig_beta_blocksAB[2], sig_beta_blocksAB[7], sig_beta_blocksAB[8], sig_beta_blocksAB[9]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			MV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1 or val_check[3]==1:
				LV_enc = 1
			if val_check[1]==1 or val_check[4]==1:
				MV_enc = 1
				if val_check[1]==1:
					beta_late_mv = sig_beta_blocksAB[1]
				else:
					beta_late_mv = sig_beta_blocksAB[8]
			if val_check[2]==1 or val_check[5]==1:
				HV_enc = 1

			Q_late = np.ravel(data['Q_mid_late'])
			FR_late = np.ravel(data['FR_late'])
			Q_early = np.ravel(data['Q_mid_early'])
			FR_early = np.ravel(data['FR_early'])
			if ((LV_enc==0) and (MV_enc==1) and (HV_enc==0)):
				count_MV_blocksAB_cd += 1
				if stim_file:
					Q_late_mv_cd_stim += [Q_late]
					Q_early_mv_cd_stim += [Q_early]
					FR_late_mv_cd_stim += [FR_late]
					FR_early_mv_cd_stim += [FR_early]
					beta_late_mv_cd_stim = np.append(beta_late_mv_cd_stim, beta_late_mv)
					#beta_early_mv_cd_stim = np.append(beta_early_mv_cd_stim, beta_early_mv)
				else:
					Q_late_mv_cd_sham += [Q_late]
					Q_early_mv_cd_sham += [Q_early]
					FR_late_mv_cd_sham += [FR_late]
					FR_early_mv_cd_sham += [FR_early]
					beta_late_mv_cd_sham = np.append(beta_late_mv_cd_sham, beta_late_mv)
					#beta_early_mv_cd_sham = np.append(beta_early_mv_cd_sham, beta_early_mv)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==0)):
				count_LV_blocksAB_cd += 1
				if stim_file:
					Q_late_other_cd_stim += [Q_late]
					Q_early_other_cd_stim += [Q_early]
					FR_late_other_cd_stim += [FR_late]
					FR_early_other_cd_stim += [FR_early]
				else:
					Q_late_other_cd_sham += [Q_late]
					Q_early_other_cd_sham += [Q_early]
					FR_late_other_cd_sham += [FR_late]
					FR_early_other_cd_sham += [FR_early]
			elif ((LV_enc==0) and (MV_enc==0) and (HV_enc==1)):
				count_HV_blocksAB_cd += 1
				if stim_file:
					Q_late_other_cd_stim += [Q_late]
					Q_early_other_cd_stim += [Q_early]
					FR_late_other_cd_stim += [FR_late]
					FR_early_other_cd_stim += [FR_early]
				else:
					Q_late_other_cd_sham += [Q_late]
					Q_early_other_cd_sham += [Q_early]
					FR_late_other_cd_sham += [FR_late]
					FR_early_other_cd_sham += [FR_early]
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==0)):
				count_LVMV_blocksAB_cd += 1
				if stim_file:
					Q_late_mv_cd_stim += [Q_late]
					Q_early_mv_cd_stim += [Q_early]
					FR_late_mv_cd_stim += [FR_late]
					FR_early_mv_cd_stim += [FR_early]
					beta_late_mv_cd_stim = np.append(beta_late_mv_cd_stim, beta_late_mv)
					#beta_early_mv_cd_stim = np.append(beta_early_mv_cd_stim, beta_early_mv)
				else:
					Q_late_mv_cd_sham += [Q_late]
					Q_early_mv_cd_sham += [Q_early]
					FR_late_mv_cd_sham += [FR_late]
					FR_early_mv_cd_sham += [FR_early]
					beta_late_mv_cd_sham = np.append(beta_late_mv_cd_sham, beta_late_mv)
					#beta_early_mv_cd_sham = np.append(beta_early_mv_cd_sham, beta_early_mv)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==1)):
				count_LVHV_blocksAB_cd += 1
				if stim_file:
					Q_late_other_cd_stim += [Q_late]
					Q_early_other_cd_stim += [Q_early]
					FR_late_other_cd_stim += [FR_late]
					FR_early_other_cd_stim += [FR_early]
				else:
					Q_late_other_cd_sham += [Q_late]
					Q_early_other_cd_sham += [Q_early]
					FR_late_other_cd_sham += [FR_late]
					FR_early_other_cd_sham += [FR_early]
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==1)):
				count_MVHV_blocksAB_cd += 1
				if stim_file:
					Q_late_mv_cd_stim += [Q_late]
					Q_early_mv_cd_stim += [Q_early]
					FR_late_mv_cd_stim += [FR_late]
					FR_early_mv_cd_stim += [FR_early]
					beta_late_mv_cd_stim = np.append(beta_late_mv_cd_stim, beta_late_mv)
					#beta_early_mv_cd_stim = np.append(beta_early_mv_cd_stim, beta_early_mv)
				else:
					Q_late_mv_cd_sham += [Q_late]
					Q_early_mv_cd_sham += [Q_early]
					FR_late_mv_cd_sham += [FR_late]
					FR_early_mv_cd_sham += [FR_early]
					beta_late_mv_cd_sham = np.append(beta_late_mv_cd_sham, beta_late_mv)
					#beta_early_mv_cd_sham = np.append(beta_early_mv_cd_sham, beta_early_mv)
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==1)):
				count_LVMVHV_blocksAB_cd += 1
				if stim_file:
					Q_late_mv_cd_stim += [Q_late]
					Q_early_mv_cd_stim += [Q_early]
					FR_late_mv_cd_stim += [FR_late]
					FR_early_mv_cd_stim += [FR_early]
					beta_late_mv_cd_stim = np.append(beta_late_mv_cd_stim, beta_late_mv)
					#beta_early_mv_cd_stim = np.append(beta_early_mv_cd_stim, beta_early_mv)
				else:
					Q_late_mv_cd_sham += [Q_late]
					Q_early_mv_cd_sham += [Q_early]
					FR_late_mv_cd_sham += [FR_late]
					FR_early_mv_cd_sham += [FR_early]
					beta_late_mv_cd_sham = np.append(beta_late_mv_cd_sham, beta_late_mv)
					#beta_early_mv_cd_sham = np.append(beta_early_mv_cd_sham, beta_early_mv)
			else:
				no_val_enc = 1
				if stim_file:
					Q_late_other_cd_stim += [Q_late]
					Q_early_other_cd_stim += [Q_early]
					FR_late_other_cd_stim += [FR_late]
					FR_early_other_cd_stim += [FR_early]
				else:
					Q_late_other_cd_sham += [Q_late]
					Q_early_other_cd_sham += [Q_early]
					FR_late_other_cd_sham += [FR_late]
					FR_early_other_cd_sham += [FR_early]

			if no_val_enc:
				max_beta = np.argmax(np.abs(sig_beta_blocksAB))
				if max_beta==3:
					count_rt_blocksAB_cd += 1
				if max_beta==4:
					count_mt_blocksAB_cd += 1
				if max_beta==5:
					count_choice_blocksAB_cd +=1
				if max_beta==6:
					count_reward_blocksAB_cd +=1

		if ('rsquared_blockA' in data.keys()):
			rsquared_blockA_cd = np.append(rsquared_blockA_cd, data['rsquared_blockA'])
		if ('rsquared_blocksAB' in data.keys()):
			rsquared_blocksAB_cd = np.append(rsquared_blocksAB_cd, data['rsquared_blocksAB'])

	else:
		if ('beta_values_blockA' in data.keys()):
			beta_blockA = np.ravel(data['beta_values_blockA'])
			pvalues_blockA = np.ravel(data['pvalues_blockA'])
			sig_beta_blockA = beta_blockA*np.ravel(np.less(pvalues_blockA, 0.05))
			if count_blockA_acc==0:
				sig_reg_blockA_acc = sig_beta_blockA
			else:
				sig_reg_blockA_acc = np.vstack([sig_reg_blockA_acc, sig_beta_blockA])
			count_blockA_acc += 1

			val_check = [sig_beta_blockA[0], sig_beta_blockA[1], sig_beta_blockA[2], sig_beta_blockA[7], sig_beta_blockA[8], sig_beta_blockA[9]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			MV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1 or val_check[3]==1:
				LV_enc = 1
			if val_check[1]==1 or val_check[4]==1:
				MV_enc = 1
				if val_check[1]==1:
					beta_early_mv = sig_beta_blockA[1]
				else:
					beta_early_mv = sig_beta_blockA[8]
			if val_check[2]==1 or val_check[5]==1:
				HV_enc = 1

			if ((LV_enc==1) and (MV_enc==0) and (HV_enc==0)):
				count_LV_blockA_acc += 1
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==0)):
				count_MV_blockA_acc += 1
			elif ((LV_enc==0) and (MV_enc==0) and (HV_enc==1)):
				count_HV_blockA_acc += 1
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==0)):
				count_LVMV_blockA_acc += 1
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==1)):
				count_LVHV_blockA_acc += 1
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==1)):
				count_MVHV_blockA_acc += 1
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==1)):
				count_LVMVHV_blockA_acc += 1
			else:
				no_val_enc = 1

			if no_val_enc:
				max_beta = np.argmax(np.abs(sig_beta_blockA))
				if max_beta==3:
					count_rt_blockA_acc += 1
				if max_beta==4:
					count_mt_blockA_acc += 1
				if max_beta==5:
					count_choice_blockA_acc +=1
				if max_beta==6:
					count_reward_blockA_acc +=1

		if ('beta_values_blocksAB' in data.keys()):
			beta_blocksAB = np.ravel(data['beta_values_blocksAB'])
			pvalues_blocksAB = np.ravel(data['pvalues_blocksAB'])
			sig_beta_blocksAB = beta_blocksAB*np.ravel(np.less(pvalues_blocksAB, 0.05))
			if count_blocksAB_acc==0:
				sig_reg_blocksAB_acc = sig_beta_blocksAB
			else:
				sig_reg_blocksAB_acc = np.vstack([sig_reg_blocksAB_acc, sig_beta_blocksAB])
			count_blocksAB_acc += 1

			val_check = [sig_beta_blocksAB[0], sig_beta_blocksAB[1], sig_beta_blocksAB[2], sig_beta_blocksAB[7], sig_beta_blocksAB[8], sig_beta_blocksAB[9]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			MV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1 or val_check[3]==1:
				LV_enc = 1
			if val_check[1]==1 or val_check[4]==1:
				MV_enc = 1
				if val_check[1]==1:
					beta_late_mv = sig_beta_blocksAB[1]
				else:
					beta_late_mv = sig_beta_blocksAB[8]
			if val_check[2]==1 or val_check[5]==1:
				HV_enc = 1

			Q_late = np.ravel(data['Q_mid_late'])
			FR_late = np.ravel(data['FR_late'])
			Q_early = np.ravel(data['Q_mid_early'])
			FR_early = np.ravel(data['FR_early'])
			if ((LV_enc==0) and (MV_enc==1) and (HV_enc==0)):
				count_MV_blocksAB_acc += 1
				if stim_file:
					Q_late_mv_acc_stim += [Q_late]
					Q_early_mv_acc_stim += [Q_early]
					FR_late_mv_acc_stim += [FR_late]
					FR_early_mv_acc_stim += [FR_early]
					beta_late_mv_acc_stim = np.append(beta_late_mv_acc_stim, beta_late_mv)
					#beta_early_mv_acc_stim = np.append(beta_early_mv_acc_stim, beta_early_mv)
				else:
					Q_late_mv_acc_sham += [Q_late]
					Q_early_mv_acc_sham += [Q_early]
					FR_late_mv_acc_sham += [FR_late]
					FR_early_mv_acc_sham += [FR_early]
					beta_late_mv_acc_sham = np.append(beta_late_mv_acc_sham, beta_late_mv)
					#beta_early_mv_acc_sham = np.append(beta_early_mv_acc_sham, beta_early_mv)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==0)):
				count_LV_blocksAB_acc += 1
				if stim_file:
					Q_late_other_acc_stim += [Q_late]
					Q_early_other_acc_stim += [Q_early]
					FR_late_other_acc_stim += [FR_late]
					FR_early_other_acc_stim += [FR_early]
				else:
					Q_late_other_acc_sham += [Q_late]
					Q_early_other_acc_sham += [Q_early]
					FR_late_other_acc_sham += [FR_late]
					FR_early_other_acc_sham += [FR_early]
			elif ((LV_enc==0) and (MV_enc==0) and (HV_enc==1)):
				count_HV_blocksAB_acc += 1
				if stim_file:
					Q_late_other_acc_stim += [Q_late]
					Q_early_other_acc_stim += [Q_early]
					FR_late_other_acc_stim += [FR_late]
					FR_early_other_acc_stim += [FR_early]
				else:
					Q_late_other_acc_sham += [Q_late]
					Q_early_other_acc_sham += [Q_early]
					FR_late_other_acc_sham += [FR_late]
					FR_early_other_acc_sham += [FR_early]
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==0)):
				count_LVMV_blocksAB_acc += 1
				if stim_file:
					Q_late_mv_acc_stim += [Q_late]
					Q_early_mv_acc_stim += [Q_early]
					FR_late_mv_acc_stim += [FR_late]
					FR_early_mv_acc_stim += [FR_early]
					beta_late_mv_acc_stim = np.append(beta_late_mv_acc_stim, beta_late_mv)
					#beta_early_mv_acc_stim = np.append(beta_early_mv_acc_stim, beta_early_mv)
				else:
					Q_late_mv_acc_sham += [Q_late]
					Q_early_mv_acc_sham += [Q_early]
					FR_late_mv_acc_sham += [FR_late]
					FR_early_mv_acc_sham += [FR_early]
					beta_late_mv_acc_sham = np.append(beta_late_mv_acc_sham, beta_late_mv)
					#beta_early_mv_acc_sham = np.append(beta_early_mv_acc_sham, beta_early_mv)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==1)):
				count_LVHV_blocksAB_acc += 1
				if stim_file:
					Q_late_other_acc_stim += [Q_late]
					Q_early_other_acc_stim += [Q_early]
					FR_late_other_acc_stim += [FR_late]
					FR_early_other_acc_stim += [FR_early]
				else:
					Q_late_other_acc_sham += [Q_late]
					Q_early_other_acc_sham += [Q_early]
					FR_late_other_acc_sham += [FR_late]
					FR_early_other_acc_sham += [FR_early]
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==1)):
				count_MVHV_blocksAB_acc += 1
				if stim_file:
					Q_late_mv_acc_stim += [Q_late]
					Q_early_mv_acc_stim += [Q_early]
					FR_late_mv_acc_stim += [FR_late]
					FR_early_mv_acc_stim += [FR_early]
					beta_late_mv_acc_stim = np.append(beta_late_mv_acc_stim, beta_late_mv)
					#beta_early_mv_acc_stim = np.append(beta_early_mv_acc_stim, beta_early_mv)
				else:
					Q_late_mv_acc_sham += [Q_late]
					Q_early_mv_acc_sham += [Q_early]
					FR_late_mv_acc_sham += [FR_late]
					FR_early_mv_acc_sham += [FR_early]
					beta_late_mv_acc_sham = np.append(beta_late_mv_acc_sham, beta_late_mv)
					#beta_early_mv_acc_sham = np.append(beta_early_mv_acc_sham, beta_early_mv)
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==1)):
				count_LVMVHV_blocksAB_acc += 1
				if stim_file:
					Q_late_mv_acc_stim += [Q_late]
					Q_early_mv_acc_stim += [Q_early]
					FR_late_mv_acc_stim += [FR_late]
					FR_early_mv_acc_stim += [FR_early]
					beta_late_mv_acc_stim = np.append(beta_late_mv_acc_stim, beta_late_mv)
					#beta_early_mv_acc_stim = np.append(beta_early_mv_acc_stim, beta_early_mv)
				else:
					Q_late_mv_acc_sham += [Q_late]
					Q_early_mv_acc_sham += [Q_early]
					FR_late_mv_acc_sham += [FR_late]
					FR_early_mv_acc_sham += [FR_early]
					beta_late_mv_acc_sham = np.append(beta_late_mv_acc_sham, beta_late_mv)
					#beta_early_mv_acc_sham = np.append(beta_early_mv_acc_sham, beta_early_mv)
			else:
				no_val_enc = 1
				if stim_file:
					Q_late_other_acc_stim += [Q_late]
					Q_early_other_acc_stim += [Q_early]
					FR_late_other_acc_stim += [FR_late]
					FR_early_other_acc_stim += [FR_early]
				else:
					Q_late_other_acc_sham += [Q_late]
					Q_early_other_acc_sham += [Q_early]
					FR_late_other_acc_sham += [FR_late]
					FR_early_other_acc_sham += [FR_early]

			if no_val_enc:
				max_beta = np.argmax(np.abs(sig_beta_blocksAB))
				if max_beta==3:
					count_rt_blocksAB_acc += 1
				if max_beta==4:
					count_mt_blocksAB_acc += 1
				if max_beta==5:
					count_choice_blocksAB_acc +=1
				if max_beta==6:
					count_reward_blocksAB_acc +=1

		if ('rsquared_blockA' in data.keys()):
			rsquared_blockA_acc = np.append(rsquared_blockA_acc, data['rsquared_blockA'])
		if ('rsquared_blocksAB' in data.keys()):
			rsquared_blocksAB_acc = np.append(rsquared_blocksAB_acc, data['rsquared_blocksAB'])

	"""
	regression_labels = ['Q_low', 'Q_mid', 'Q_high','RT', 'MT', 'Choice', 'Reward', 'Q_low_on', 'Q_mid_on', 'Q_high_on']
	data['beta_values_blockA'] = fit_glm.params
	data['pvalues_blockA'] = fit_glm.pvalues
	data['beta_values_blocksAB'] = fit_glm_late.params
	data['pvalues_blocksAB'] = fit_glm_late.pvalues
	data['Q_mid_early'] = Q_mid_BlockA
	data['Q_mid_late'] = Q_late
	data['FR_early'] = FR_BlockA
	data['FR_late'] = FR_late
	"""
true_blockA_cd = np.greater(sig_reg_blockA_cd, 0)
true_blocksAB_cd = np.greater(sig_reg_blocksAB_cd, 0)
true_blockA_acc = np.greater(sig_reg_blockA_acc, 0)
true_blocksAB_acc = np.greater(sig_reg_blocksAB_acc, 0)

count_all_blockA_cd = np.sum(true_blockA_cd, axis = 1)
count_all_blocksAB_cd = np.sum(true_blocksAB_cd, axis = 1)
count_all_blockA_acc = np.sum(true_blockA_acc, axis = 1)
count_all_blocksAB_acc = np.sum(true_blocksAB_acc, axis = 1)

# R-squared: model fit plots
avg_rsquared_blockA_cd = np.nanmean(rsquared_blockA_cd)
sem_rsquared_blockA_cd = np.nanstd(rsquared_blockA_cd)/np.sqrt(len(rsquared_blockA_cd))
avg_rsquared_blocksAB_cd = np.nanmean(rsquared_blocksAB_cd)
sem_rsquared_blocksAB_cd = np.nanstd(rsquared_blocksAB_cd)/np.sqrt(len(rsquared_blocksAB_cd))
avg_rsquared_blockA_acc = np.nanmean(rsquared_blockA_acc)
sem_rsquared_blockA_acc = np.nanstd(rsquared_blockA_acc)/np.sqrt(len(rsquared_blockA_acc))
avg_rsquared_blocksAB_acc = np.nanmean(rsquared_blocksAB_acc)
sem_rsquared_blocksAB_acc = np.nanstd(rsquared_blocksAB_acc)/np.sqrt(len(rsquared_blocksAB_acc))
avg_rsquared_cd = [avg_rsquared_blockA_cd, avg_rsquared_blocksAB_cd]
sem_rsquared_cd = [sem_rsquared_blockA_cd, sem_rsquared_blocksAB_cd]
avg_rsquared_acc = [avg_rsquared_blockA_acc, avg_rsquared_blocksAB_acc]
sem_rsquared_acc = [sem_rsquared_blockA_acc, sem_rsquared_blocksAB_acc]
width = 0.35
ind = np.arange(2)
plt.figure()
plt.bar(ind, avg_rsquared_cd, width, color = 'y', yerr = sem_rsquared_cd, label = 'Cd')
plt.bar(ind+width, avg_rsquared_acc, width, color = 'g', yerr = sem_rsquared_acc, label = 'ACC')
plt.text(ind[0]+0.1,0.35,'r2=%0.2f' % (avg_rsquared_cd[0]))
plt.text(ind[1]+0.1,0.30,'r2=%0.2f' % (avg_rsquared_cd[1]))
plt.text(ind[0]+0.1+width,0.30,'r2=%0.2f' % (avg_rsquared_acc[0]))
plt.text(ind[1]+0.1+width,0.25,'r2=%0.2f' % (avg_rsquared_acc[1]))
xticklabels = ['Block A', 'Blocks A and B']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Data used for fit')
plt.ylabel('R-squared')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# Pie charts of overall encoding and then value-coding 
all_value_blockA_cd = float(count_LV_blockA_cd + count_MV_blockA_cd + count_HV_blockA_cd + count_LVMV_blockA_cd + \
						count_LVHV_blockA_cd + count_MVHV_blockA_cd + count_LVMVHV_blockA_cd)
all_value_blockA_acc = float(count_LV_blockA_acc + count_MV_blockA_acc + count_HV_blockA_acc + count_LVMV_blockA_acc + \
						count_LVHV_blockA_acc + count_MVHV_blockA_acc + count_LVMVHV_blockA_acc)
all_value_blocksAB_cd = float(count_LV_blocksAB_cd + count_MV_blocksAB_cd + count_HV_blocksAB_cd + count_LVMV_blocksAB_cd + \
						count_LVHV_blocksAB_cd + count_MVHV_blocksAB_cd + count_LVMVHV_blocksAB_cd)
all_value_blocksAB_acc = float(count_LV_blocksAB_acc + count_MV_blocksAB_acc + count_HV_blocksAB_acc + count_LVMV_blocksAB_acc + \
						count_LVHV_blocksAB_acc + count_MVHV_blocksAB_acc + count_LVMVHV_blocksAB_acc)

labels = ['Value', 'RT', 'MT', 'Choice', 'Reward', 'Non-responsive']
noncoding_blockA_cd = count_blockA_cd - all_value_blockA_cd - count_rt_blockA_cd - count_mt_blockA_cd - \
						count_choice_blockA_cd - count_reward_blockA_cd
noncoding_blockA_acc = count_blockA_acc - all_value_blockA_acc - count_rt_blockA_acc - count_mt_blockA_acc - \
						count_choice_blockA_acc - count_reward_blockA_acc
noncoding_blocksAB_cd = count_blocksAB_cd - all_value_blocksAB_cd - count_rt_blocksAB_cd - count_mt_blocksAB_cd - \
						count_choice_blocksAB_cd - count_reward_blocksAB_cd
noncoding_blocksAB_acc = count_blocksAB_acc - all_value_blocksAB_acc - count_rt_blocksAB_acc - count_mt_blocksAB_acc - \
						count_choice_blocksAB_acc - count_reward_blocksAB_acc

count_blockA_cd = float(count_blockA_cd)
count_blocksAB_cd = float(count_blocksAB_cd)
count_blockA_acc = float(count_blockA_acc)
count_blocksAB_acc = float(count_blocksAB_acc)
fracs_blockA_cd = [all_value_blockA_cd/count_blockA_cd, count_rt_blockA_cd/count_blockA_cd, count_mt_blockA_cd/count_blockA_cd,\
					count_choice_blockA_cd/count_blockA_cd, count_reward_blockA_cd/count_blockA_cd, noncoding_blockA_cd/count_blockA_cd]
fracs_blockA_acc = [all_value_blockA_acc/count_blockA_acc, count_rt_blockA_acc/count_blockA_acc, count_mt_blockA_acc/count_blockA_acc,\
					count_choice_blockA_acc/count_blockA_acc, count_reward_blockA_acc/count_blockA_acc, noncoding_blockA_acc/count_blockA_acc]
fracs_blocksAB_cd = [all_value_blocksAB_cd/count_blocksAB_cd, count_rt_blocksAB_cd/count_blocksAB_cd, count_mt_blocksAB_cd/count_blocksAB_cd,\
					count_choice_blocksAB_cd/count_blocksAB_cd, count_reward_blocksAB_cd/count_blocksAB_cd, noncoding_blocksAB_cd/count_blocksAB_cd]
fracs_blocksAB_acc = [all_value_blocksAB_acc/count_blocksAB_acc, count_rt_blocksAB_acc/count_blocksAB_acc, count_mt_blocksAB_acc/count_blocksAB_acc,\
					count_choice_blocksAB_acc/count_blocksAB_acc, count_reward_blocksAB_acc/count_blocksAB_acc, noncoding_blocksAB_acc/count_blocksAB_acc]

plt.figure()
plt.subplot(2,2,1)
plt.pie(fracs_blockA_cd, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('Cd - Block A - n = %0.f' % (count_blockA_cd))
plt.subplot(2,2,2)
plt.pie(fracs_blocksAB_cd, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('Cd - Blocks A and B - n = %0.f' % (count_blocksAB_cd))
plt.subplot(2,2,3)
plt.pie(fracs_blockA_acc, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('ACC - Block A - n = %0.f' % (count_blockA_acc))
plt.subplot(2,2,4)
plt.pie(fracs_blocksAB_acc, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('ACC - Blocks A and B - n = %0.f' % (count_blocksAB_acc))
plt.show()


labels_value = ['LV', 'MV', 'HV', 'LV&MV', 'LV&HV', 'MV&HV', 'LV,MV,&HV']
fracs_value_blockA_cd = [count_LV_blockA_cd/all_value_blockA_cd, count_MV_blockA_cd/all_value_blockA_cd, count_HV_blockA_cd/all_value_blockA_cd, \
							count_LVMV_blockA_cd/all_value_blockA_cd, count_LVHV_blockA_cd/all_value_blockA_cd, count_MVHV_blockA_cd/all_value_blockA_cd, \
							count_LVMVHV_blockA_cd/all_value_blockA_cd]
fracs_value_blockA_acc = [count_LV_blockA_acc/all_value_blockA_acc, count_MV_blockA_acc/all_value_blockA_acc, count_HV_blockA_acc/all_value_blockA_acc, \
							count_LVMV_blockA_acc/all_value_blockA_acc, count_LVHV_blockA_acc/all_value_blockA_acc, count_MVHV_blockA_acc/all_value_blockA_acc, \
							count_LVMVHV_blockA_acc/all_value_blockA_acc]
fracs_value_blocksAB_cd = [count_LV_blocksAB_cd/all_value_blocksAB_cd, count_MV_blocksAB_cd/all_value_blocksAB_cd, count_HV_blocksAB_cd/all_value_blocksAB_cd, \
							count_LVMV_blocksAB_cd/all_value_blocksAB_cd, count_LVHV_blocksAB_cd/all_value_blocksAB_cd, count_MVHV_blocksAB_cd/all_value_blocksAB_cd, \
							count_LVMVHV_blocksAB_cd/all_value_blocksAB_cd]
fracs_value_blocksAB_acc = [count_LV_blocksAB_acc/all_value_blocksAB_acc, count_MV_blocksAB_acc/all_value_blocksAB_acc, count_HV_blocksAB_acc/all_value_blocksAB_acc, \
							count_LVMV_blocksAB_acc/all_value_blocksAB_acc, count_LVHV_blocksAB_acc/all_value_blocksAB_acc, count_MVHV_blocksAB_acc/all_value_blocksAB_acc, \
							count_LVMVHV_blocksAB_acc/all_value_blocksAB_acc]

plt.figure()
plt.subplot(2,2,1)
plt.pie(fracs_value_blockA_cd, labels = labels_value, autopct='%.2f%%', shadow = False)
plt.title('Cd - Block A')
plt.subplot(2,2,2)
plt.pie(fracs_value_blocksAB_cd, labels = labels_value, autopct='%.2f%%', shadow = False)
plt.title('Cd - Blocks A and B')
plt.subplot(2,2,3)
plt.pie(fracs_value_blockA_acc, labels = labels_value, autopct='%.2f%%', shadow = False)
plt.title('ACC - Block A')
plt.subplot(2,2,4)
plt.pie(fracs_value_blocksAB_acc, labels = labels_value, autopct='%.2f%%', shadow = False)
plt.title('ACC - Blocks A and B')
plt.show()

# Fraction of units with positive and negative correlation with FR

# Change in firing rates: Q_mid_late, FR_late vs Q_mid_early_FR_late, with two-way ANOVA statistics, 
# done separately for Cd and ACC, done separately for non-value coding units and then MV-value-coding only units 
# do the above first, then re-run code with regression changed for first one done with (50,250)
'''
Q_late_mv_cd = np.array([])
Q_early_mv_cd = np.array([])
FR_late_mv_cd = np.array([])
FR_early_mv_cd = np.array([])
Q_late_other_cd = np.array([])
Q_early_other_cd = np.array([])
FR_late_other_cd = np.array([])
FR_early_other_cd = np.array([])
Q_late_mv_acc = np.array([])
Q_early_mv_acc = np.array([])
FR_late_mv_acc = np.array([])
FR_early_mv_acc = np.array([])
Q_late_other_acc = np.array([])
Q_early_other_acc = np.array([])
FR_late_other_acc = np.array([])
FR_early_other_acc = np.array([])
'''

all_Q = Q_late_mv_cd_stim + Q_early_mv_cd_stim + Q_late_other_cd_stim + Q_early_other_cd_stim + \
		Q_late_mv_acc_stim + Q_early_mv_acc_stim + Q_late_other_acc_stim + Q_early_other_acc_stim + \
		Q_late_mv_cd_sham + Q_early_mv_cd_sham + Q_late_other_cd_sham + Q_early_other_cd_sham + \
		Q_late_mv_acc_sham + Q_early_mv_acc_sham + Q_late_other_acc_sham + Q_early_other_acc_sham
all_Q_flat = [item for sublist in all_Q for item in sublist]
min_Q = np.min(all_Q_flat)
max_Q = np.max(all_Q_flat)
min_Q = 0.5
max_Q = 0.7
print(min_Q, max_Q)
Q_bins = np.arange(min_Q, max_Q + (max_Q - min_Q)/6., (max_Q - min_Q)/6.)
print(Q_bins)
# bin Qs and sort FRs accordingly for all conditions

# Cd - value coding - stim
Q_bins_cd_mv_stim, delta_FR_cd_mv_stim, norm_delta_FR_cd_mv_stim, avg_delta_FR_cd_mv_stim, \
	sem_delta_FR_cd_mv_stim, avg_norm_delta_FR_cd_mv_stim, sem_norm_delta_FR_cd_mv_stim = \
	BinChangeInFiringRatesByValue(Q_early_mv_cd_stim, Q_late_mv_cd_stim, Q_bins, FR_early_mv_cd_stim, FR_late_mv_cd_stim)
# Cd - value coding - sham
Q_bins_cd_mv_sham, delta_FR_cd_mv_sham, norm_delta_FR_cd_mv_sham, avg_delta_FR_cd_mv_sham, \
	sem_delta_FR_cd_mv_sham, avg_norm_delta_FR_cd_mv_sham, sem_norm_delta_FR_cd_mv_sham = \
	BinChangeInFiringRatesByValue(Q_early_mv_cd_sham, Q_late_mv_cd_sham, Q_bins, FR_early_mv_cd_sham, FR_late_mv_cd_sham)
# Cd - nonvalue coding - stim
Q_bins_cd_other_stim, delta_FR_cd_other_stim, norm_delta_FR_cd_other_stim, avg_delta_FR_cd_other_stim, \
	sem_delta_FR_cd_other_stim, avg_norm_delta_FR_cd_other_stim, sem_norm_delta_FR_cd_other_stim = \
	BinChangeInFiringRatesByValue(Q_early_other_cd_stim, Q_late_other_cd_stim, Q_bins, FR_early_other_cd_stim, FR_late_other_cd_stim)
# Cd - nonvalue coding - sham
Q_bins_cd_other_sham, delta_FR_cd_other_sham, norm_delta_FR_cd_other_sham, avg_delta_FR_cd_other_sham, \
	sem_delta_FR_cd_other_sham, avg_norm_delta_FR_cd_other_sham, sem_norm_delta_FR_cd_other_sham = \
	BinChangeInFiringRatesByValue(Q_early_other_cd_sham, Q_late_other_cd_sham, Q_bins, FR_early_other_cd_sham, FR_late_other_cd_sham)

# ACC - value coding - stim
Q_bins_acc_mv_stim, delta_FR_acc_mv_stim, norm_delta_FR_acc_mv_stim, avg_delta_FR_acc_mv_stim, \
	sem_delta_FR_acc_mv_stim, avg_norm_delta_FR_acc_mv_stim, sem_norm_delta_FR_acc_mv_stim = \
	BinChangeInFiringRatesByValue(Q_early_mv_acc_stim, Q_late_mv_acc_stim, Q_bins, FR_early_mv_acc_stim, FR_late_mv_acc_stim)
# ACC - value coding - sham
Q_bins_acc_mv_sham, delta_FR_acc_mv_sham, norm_delta_FR_acc_mv_sham, avg_delta_FR_acc_mv_sham, \
	sem_delta_FR_acc_mv_sham, avg_norm_delta_FR_acc_mv_sham, sem_norm_delta_FR_acc_mv_sham = \
	BinChangeInFiringRatesByValue(Q_early_mv_acc_sham, Q_late_mv_acc_sham, Q_bins, FR_early_mv_acc_sham, FR_late_mv_acc_sham)
# ACC - nonvalue coding - stim
Q_bins_acc_other_stim, delta_FR_acc_other_stim, norm_delta_FR_acc_other_stim, avg_delta_FR_acc_other_stim, \
	sem_delta_FR_acc_other_stim, avg_norm_delta_FR_acc_other_stim, sem_norm_delta_FR_acc_other_stim = \
	BinChangeInFiringRatesByValue(Q_early_other_acc_stim, Q_late_other_acc_stim, Q_bins, FR_early_other_acc_stim, FR_late_other_acc_stim)
# ACC - nonvalue coding - sham
Q_bins_acc_other_sham, delta_FR_acc_other_sham, norm_delta_FR_acc_other_sham, avg_delta_FR_acc_other_sham, \
	sem_delta_FR_acc_other_sham, avg_norm_delta_FR_acc_other_sham, sem_norm_delta_FR_acc_other_sham = \
	BinChangeInFiringRatesByValue(Q_early_other_acc_sham, Q_late_other_acc_sham, Q_bins, FR_early_other_acc_sham, FR_late_other_acc_sham)


# Plot Cd results
Q_bin_centers = (Q_bins[1:] + Q_bins[:-1])/2.
plt.figure()
plt.subplot(2,2,1)
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_mv_stim, yerr = sem_delta_FR_cd_mv_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - value coding')
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_mv_sham, yerr = sem_delta_FR_cd_mv_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Delta FR (Block A' - Block A) (Hz)")
plt.legend()
plt.title('Value-coding Cd Units')

plt.subplot(2,2,2)
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_mv_stim, yerr = sem_norm_delta_FR_cd_mv_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - value coding')
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_mv_sham, yerr = sem_norm_delta_FR_cd_mv_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Normalized Delta FR (Block A' - Block A) (A.U.)")
plt.legend()
plt.title('Value-coding Cd Units')

plt.subplot(2,2,3)
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_other_stim, yerr = sem_delta_FR_cd_other_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - non value coding')
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_other_sham, yerr = sem_delta_FR_cd_other_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - non value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Delta FR (Block A' - Block A) (Hz)")
plt.legend()
plt.title('Non-value-coding Cd Units')

plt.subplot(2,2,4)
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_other_stim, yerr = sem_norm_delta_FR_cd_other_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - non value coding')
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_other_sham, yerr = sem_norm_delta_FR_cd_other_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - non value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Normalized Delta FR (Block A' - Block A) (A.U.)")
plt.legend()
plt.title('Non-value-coding Cd Units')
plt.show()

# Plot ACC results
plt.figure()
plt.subplot(2,2,1)
plt.errorbar(Q_bin_centers, avg_delta_FR_acc_mv_stim, yerr = sem_delta_FR_acc_mv_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - value coding')
plt.errorbar(Q_bin_centers, avg_delta_FR_acc_mv_sham, yerr = sem_delta_FR_acc_mv_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Delta FR (Block A' - Block A) (Hz)")
plt.legend()
plt.title('Value-coding ACC Units')

plt.subplot(2,2,2)
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_acc_mv_stim, yerr = sem_norm_delta_FR_acc_mv_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - value coding')
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_acc_mv_sham, yerr = sem_norm_delta_FR_acc_mv_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Normalized Delta FR (Block A' - Block A) (A.U.)")
plt.legend()
plt.title('Value-coding ACC Units')

plt.subplot(2,2,3)
plt.errorbar(Q_bin_centers, avg_delta_FR_acc_other_stim, yerr = sem_delta_FR_acc_other_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - non value coding')
plt.errorbar(Q_bin_centers, avg_delta_FR_acc_other_sham, yerr = sem_delta_FR_acc_other_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - non value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Delta FR (Block A' - Block A) (Hz)")
plt.legend()
plt.title('Non-value-coding ACC Units')

plt.subplot(2,2,4)
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_acc_other_stim, yerr = sem_norm_delta_FR_acc_other_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - non value coding')
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_acc_other_sham, yerr = sem_norm_delta_FR_acc_other_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - non value coding')
plt.xlabel('MV Q-value')
plt.ylabel("Normalized Delta FR (Block A' - Block A) (A.U.)")
plt.legend()
plt.title('Non-value-coding ACC Units')
plt.show()


# Aggragating all data
FR_e_means_cd_stim, FR_l_means_cd_stim, FR_e_sem_cd_stim, FR_l_sem_cd_stim, bin_FR_e_cd_stim, bin_FR_l_cd_stim = BinFiringRatesByValue(Q_early_mv_cd_stim, Q_late_mv_cd_stim, Q_bins, FR_early_mv_cd_stim, FR_late_mv_cd_stim)
FR_e_means_cd_sham, FR_l_means_cd_sham, FR_e_sem_cd_sham, FR_l_sem_cd_sham, bin_FR_e_cd_sham, bin_FR_l_cd_sham = BinFiringRatesByValue(Q_early_mv_cd_sham, Q_late_mv_cd_sham, Q_bins, FR_early_mv_cd_sham, FR_late_mv_cd_sham)
FR_e_means_acc_stim, FR_l_means_acc_stim, FR_e_sem_acc_stim, FR_l_sem_acc_stim, bin_FR_e_acc_stim, bin_FR_l_acc_stim = BinFiringRatesByValue(Q_early_mv_acc_stim, Q_late_mv_acc_stim, Q_bins, FR_early_mv_acc_stim, FR_late_mv_acc_stim)
FR_e_means_acc_sham, FR_l_means_acc_sham, FR_e_sem_acc_sham, FR_l_sem_acc_sham, bin_FR_e_acc_sham, bin_FR_l_acc_sham = BinFiringRatesByValue(Q_early_mv_acc_sham, Q_late_mv_acc_sham, Q_bins, FR_early_mv_acc_sham, FR_late_mv_acc_sham)


dta_cd = []
for i in range(len(bin_FR_l_cd_stim)):
	for data in bin_FR_l_cd_stim[i]:
		dta_cd += [(1,i, data)]
for i in range(len(bin_FR_l_cd_sham)):
	for data in bin_FR_l_cd_sham[i]:
		dta_cd += [(0,i, data)]
dta_acc = []
for i in range(len(bin_FR_l_acc_stim)):
	for data in bin_FR_l_acc_stim[i]:
		dta_acc += [(1,i, data)]
for i in range(len(bin_FR_l_acc_sham)):
	for data in bin_FR_l_acc_sham[i]:
		dta_acc += [(0,i, data)]
		
dta_cd = pd.DataFrame(dta_cd, columns=['Stim_condition', 'Bin', 'fr'])
dta_acc = pd.DataFrame(dta_acc, columns=['Stim_condition', 'Bin', 'fr'])

formula = 'fr ~ C(Stim_condition) + C(Bin) + C(Stim_condition):C(Bin)'
model = ols(formula, dta_cd).fit()
aov_table = anova_lm(model, typ=2)
model_acc = ols(formula, dta_acc).fit()
aov_table_acc = anova_lm(model_acc, typ=2)

plt.figure()
plt.subplot(1,3,1)
plt.errorbar(Q_bin_centers,FR_l_means_cd_stim,yerr=FR_l_sem_cd_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.errorbar(Q_bin_centers,FR_e_means_cd_stim,yerr=FR_e_sem_cd_stim,color = 'c', ecolor = 'c', fmt = 'o--', label='Stim -early')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title('Cd - Stim - Firing Rate over Blocks')
plt.legend()
plt.subplot(1,3,2)
plt.errorbar(Q_bin_centers,FR_l_means_cd_sham,yerr=FR_l_sem_cd_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_e_means_cd_sham,yerr=FR_e_sem_cd_sham,color = 'm', ecolor = 'm', fmt = 'o--', label='Sham -early')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title('Cd - Sham - Firing Rate over Blocks')
plt.legend()
plt.subplot(1,3,3)
plt.errorbar(Q_bin_centers,FR_l_means_cd_sham,yerr=FR_l_sem_cd_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_cd_stim,yerr=FR_l_sem_cd_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("Cd - Stim vs Sham - Block A'")
plt.legend()
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.errorbar(Q_bin_centers,FR_l_means_acc_stim,yerr=FR_l_sem_acc_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.errorbar(Q_bin_centers,FR_e_means_acc_stim,yerr=FR_e_sem_acc_stim,color = 'c', ecolor = 'c', fmt = 'o--', label='Stim -early')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title('ACC - Stim - Firing Rate over Blocks')
plt.legend()
plt.subplot(1,3,2)
plt.errorbar(Q_bin_centers,FR_l_means_acc_sham,yerr=FR_l_sem_acc_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_e_means_acc_sham,yerr=FR_e_sem_acc_sham,color = 'm', ecolor = 'm', fmt = 'o--', label='Sham -early')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title('ACC - Sham - Firing Rate over Blocks')
plt.legend()
plt.subplot(1,3,3)
plt.errorbar(Q_bin_centers,FR_l_means_acc_sham,yerr=FR_l_sem_acc_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_acc_stim,yerr=FR_l_sem_acc_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("ACC - Stim vs Sham - Block A'")
plt.legend()
plt.show()

# Aggragating all data but separating by units with positive and negative correlation with value
FR_e_means_posreg_cd_stim, FR_e_means_negreg_cd_stim, FR_l_means_posreg_cd_stim, FR_l_means_negreg_cd_stim, FR_e_sem_posreg_cd_stim, \
	FR_e_sem_negreg_cd_stim, FR_l_sem_posreg_cd_stim, FR_l_sem_negreg_cd_stim, \
	bin_FR_e_posreg_cd_stim, bin_FR_l_posreg_cd_stim, bin_FR_e_negreg_cd_stim, bin_FR_l_negreg_cd_stim \
	= BinFiringRatesByValue_SepReg(Q_early_mv_cd_stim, Q_late_mv_cd_stim, Q_bins, FR_early_mv_cd_stim, \
		FR_late_mv_cd_stim, beta_late_mv_cd_stim)
FR_e_means_posreg_cd_sham, FR_e_means_negreg_cd_sham, FR_l_means_posreg_cd_sham, FR_l_means_negreg_cd_sham, FR_e_sem_posreg_cd_sham, \
	FR_e_sem_negreg_cd_sham, FR_l_sem_posreg_cd_sham, FR_l_sem_negreg_cd_sham, \
	bin_FR_e_posreg_cd_sham, bin_FR_l_posreg_cd_sham, bin_FR_e_negreg_cd_sham, bin_FR_l_negreg_cd_sham \
	 = BinFiringRatesByValue_SepReg(Q_early_mv_cd_sham, \
		Q_late_mv_cd_sham, Q_bins, FR_early_mv_cd_sham, FR_late_mv_cd_sham, beta_late_mv_cd_sham)

# Same for ACC
FR_e_means_posreg_acc_stim, FR_e_means_negreg_acc_stim, FR_l_means_posreg_acc_stim, FR_l_means_negreg_acc_stim, FR_e_sem_posreg_acc_stim, \
	FR_e_sem_negreg_acc_stim, FR_l_sem_posreg_acc_stim, FR_l_sem_negreg_acc_stim, \
	bin_FR_e_posreg_acc_stim, bin_FR_l_posreg_acc_stim, bin_FR_e_negreg_acc_stim, bin_FR_l_negreg_acc_stim \
	= BinFiringRatesByValue_SepReg(Q_early_mv_acc_stim, Q_late_mv_acc_stim, Q_bins, FR_early_mv_acc_stim, \
		FR_late_mv_acc_stim, beta_late_mv_acc_stim)
FR_e_means_posreg_acc_sham, FR_e_means_negreg_acc_sham, FR_l_means_posreg_acc_sham, FR_l_means_negreg_acc_sham, FR_e_sem_posreg_acc_sham, \
	FR_e_sem_negreg_acc_sham, FR_l_sem_posreg_acc_sham, FR_l_sem_negreg_acc_sham, \
	bin_FR_e_posreg_acc_sham, bin_FR_l_posreg_acc_sham, bin_FR_e_negreg_acc_sham, bin_FR_l_negreg_acc_sham \
	 = BinFiringRatesByValue_SepReg(Q_early_mv_acc_sham, \
		Q_late_mv_acc_sham, Q_bins, FR_early_mv_acc_sham, FR_late_mv_acc_sham, beta_late_mv_acc_sham)


x_lin = np.linspace(min_Q, max_Q, num = len(FR_l_means_cd_sham), endpoint = True)
m_cd_sham,b_cd_sham = np.polyfit(x_lin, FR_l_means_cd_sham, 1)
m_cd_stim,b_cd_stim = np.polyfit(x_lin, FR_l_means_cd_stim, 1)
m_posreg_cd_sham, b_posreg_cd_sham = np.polyfit(x_lin[1:], FR_l_means_posreg_cd_sham[1:], 1)
m_posreg_cd_stim, b_posreg_cd_stim = np.polyfit(x_lin, FR_l_means_posreg_cd_stim, 1)
m_negreg_cd_sham, b_negreg_cd_sham = np.polyfit(x_lin, FR_l_means_negreg_cd_sham, 1)
m_negreg_cd_stim, b_negreg_cd_stim = np.polyfit(x_lin, FR_l_means_negreg_cd_stim, 1)

m_acc_sham,b_acc_sham = np.polyfit(x_lin, FR_l_means_acc_sham, 1)
m_acc_stim,b_acc_stim = np.polyfit(x_lin, FR_l_means_acc_stim, 1)
m_posreg_acc_sham, b_posreg_acc_sham = np.polyfit(x_lin[1:], FR_l_means_posreg_acc_sham[1:], 1)
m_posreg_acc_stim, b_posreg_acc_stim = np.polyfit(x_lin, FR_l_means_posreg_acc_stim, 1)
m_negreg_acc_sham, b_negreg_acc_sham = np.polyfit(x_lin[1:], FR_l_means_negreg_acc_sham[1:], 1)
m_negreg_acc_stim, b_negreg_acc_stim = np.polyfit(x_lin, FR_l_means_negreg_acc_stim, 1)


dta_cd_posreg = []
for i in range(len(bin_FR_l_posreg_cd_stim)):
	for data in bin_FR_l_posreg_cd_stim[i]:
		dta_cd_posreg += [(1,i, data)]
for i in range(len(bin_FR_l_posreg_cd_sham)):
	for data in bin_FR_l_posreg_cd_sham[i]:
		dta_cd_posreg += [(0,i, data)]

dta_cd_negreg = []
for i in range(len(bin_FR_l_negreg_cd_stim)):
	for data in bin_FR_l_negreg_cd_stim[i]:
		dta_cd_negreg += [(1,i, data)]
for i in range(len(bin_FR_l_negreg_cd_sham)):
	for data in bin_FR_l_negreg_cd_sham[i]:
		dta_cd_negreg += [(0,i, data)]

dta_acc_posreg = []
for i in range(len(bin_FR_l_posreg_acc_stim)):
	for data in bin_FR_l_posreg_acc_stim[i]:
		dta_acc_posreg += [(1,i, data)]
for i in range(len(bin_FR_l_posreg_acc_sham)):
	for data in bin_FR_l_posreg_acc_sham[i]:
		dta_acc_posreg += [(0,i, data)]

dta_acc_negreg = []
for i in range(len(bin_FR_l_negreg_acc_stim)):
	for data in bin_FR_l_negreg_acc_stim[i]:
		dta_acc_negreg += [(1,i, data)]
for i in range(len(bin_FR_l_negreg_acc_sham)):
	for data in bin_FR_l_negreg_acc_sham[i]:
		dta_acc_negreg += [(0,i, data)]

dta_cd_posreg = pd.DataFrame(dta_cd_posreg, columns=['Stim_condition', 'Bin', 'fr'])
dta_cd_negreg = pd.DataFrame(dta_cd_negreg, columns=['Stim_condition', 'Bin', 'fr'])
dta_acc_posreg = pd.DataFrame(dta_acc_posreg, columns=['Stim_condition', 'Bin', 'fr'])
dta_acc_negreg = pd.DataFrame(dta_acc_negreg, columns=['Stim_condition', 'Bin', 'fr'])

formula = 'fr ~ C(Stim_condition) + C(Bin) + C(Stim_condition):C(Bin)'
model_posreg = ols(formula, dta_cd_posreg).fit()
model_negreg = ols(formula, dta_cd_negreg).fit()
aov_table_posreg = anova_lm(model_posreg, typ=2)
aov_table_negreg = anova_lm(model_negreg, typ=2)

model_posreg_acc = ols(formula, dta_acc_posreg).fit()
model_negreg_acc = ols(formula, dta_acc_negreg).fit()
aov_table_posreg_acc = anova_lm(model_posreg_acc, typ=2)
aov_table_negreg_acc = anova_lm(model_negreg_acc, typ=2)

print("Two-way ANOVA analysis: Cd - late - Stim vs Sham")
print(aov_table)
print("Two-way ANOVA analysis: Cd - Pos Reg - Stim vs Sham")
print(aov_table_posreg)
print("Two-way ANOVA analysis: Cd - Neg Reg - Stim vs Sham")
print(aov_table_negreg)

print("Two-way ANOVA analysis: ACC - late - Stim vs Sham")
print(aov_table_acc)
print("Two-way ANOVA analysis: ACC - Pos Reg - Stim vs Sham")
print(aov_table_posreg_acc)
print("Two-way ANOVA analysis: ACC - Neg Reg - Stim vs Sham")
print(aov_table_negreg_acc)

res_stim = pairwise_tukeyhsd(dta_cd_posreg['fr'], dta_cd_posreg['Bin'],alpha=0.01)
print(res_stim)

plt.figure()
plt.subplot(1,3,1)
plt.errorbar(Q_bin_centers,FR_l_means_cd_sham,yerr=FR_l_sem_cd_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_cd_stim,yerr=FR_l_sem_cd_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.plot(x_lin, m_cd_sham*x_lin + b_cd_sham, c = 'm')
plt.plot(x_lin, m_cd_stim*x_lin + b_cd_stim, c = 'c')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("Cd - Stim vs Sham - Block A'")
plt.ylim((-1,1.5))
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.legend()
plt.subplot(1,3,2)
plt.errorbar(Q_bin_centers,FR_l_means_posreg_cd_sham,yerr=FR_l_sem_posreg_cd_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_posreg_cd_stim,yerr=FR_l_sem_posreg_cd_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.plot(x_lin, m_posreg_cd_sham*x_lin + b_posreg_cd_sham, c = 'm')
plt.plot(x_lin, m_posreg_cd_stim*x_lin + b_posreg_cd_stim, c = 'c')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("Cd - Stim vs Sham - Block A' - Postiive Beta")
plt.ylim((-1,1.5))
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.legend()
plt.subplot(1,3,3)
plt.errorbar(Q_bin_centers,FR_l_means_negreg_cd_sham,yerr=FR_l_sem_negreg_cd_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_negreg_cd_stim,yerr=FR_l_sem_negreg_cd_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.plot(x_lin, m_negreg_cd_sham*x_lin + b_negreg_cd_sham, c = 'm')
plt.plot(x_lin, m_negreg_cd_stim*x_lin + b_negreg_cd_stim, c = 'c')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("Cd - Stim vs Sham - Block A' - Negative Beta")
plt.ylim((-1,1.5))
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.legend()
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.errorbar(Q_bin_centers,FR_l_means_acc_sham,yerr=FR_l_sem_acc_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_acc_stim,yerr=FR_l_sem_acc_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.plot(x_lin, m_acc_sham*x_lin + b_acc_sham, c = 'm')
plt.plot(x_lin, m_acc_stim*x_lin + b_acc_stim, c = 'c')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("ACC - Stim vs Sham - Block A'")
#plt.ylim((-1,1.5))
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.legend()
plt.subplot(1,3,2)
plt.errorbar(Q_bin_centers,FR_l_means_posreg_acc_sham,yerr=FR_l_sem_posreg_acc_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_posreg_acc_stim,yerr=FR_l_sem_posreg_acc_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.plot(x_lin, m_posreg_acc_sham*x_lin + b_posreg_acc_sham, c = 'm')
plt.plot(x_lin, m_posreg_acc_stim*x_lin + b_posreg_acc_stim, c = 'c')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("ACC - Stim vs Sham - Block A' - Postiive Beta")
#plt.ylim((-1,1.5))
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.legend()
plt.subplot(1,3,3)
plt.errorbar(Q_bin_centers,FR_l_means_negreg_acc_sham,yerr=FR_l_sem_negreg_acc_sham,color = 'm', ecolor = 'm', fmt = 'o-', label='Sham -late')
plt.errorbar(Q_bin_centers,FR_l_means_negreg_acc_stim,yerr=FR_l_sem_negreg_acc_stim,color = 'c', ecolor = 'c', fmt = 'o-', label='Stim -late')
plt.plot(x_lin, m_negreg_acc_sham*x_lin + b_negreg_acc_sham, c = 'm')
plt.plot(x_lin, m_negreg_acc_stim*x_lin + b_negreg_acc_stim, c = 'c')
plt.xlabel('MV Q-value')
plt.ylabel('Z-scored Firing Rate (A.U.)')
plt.title("ACC - Stim vs Sham - Block A' - Negative Beta")
#plt.ylim((-1,1.5))
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.legend()
plt.show()

FR_late_mv_cd_stim_zscore = zscore_FR_separate(FR_late_mv_cd_stim)
FR_late_mv_cd_sham_zscore = zscore_FR_separate(FR_late_mv_cd_sham)

# Make scatter plots
all_FR_cd_stim_posreg = [item for i,sublist in enumerate(FR_late_mv_cd_stim_zscore) for item in sublist if beta_late_mv_cd_stim[i] > 0]
all_FR_cd_sham_posreg = [item for i,sublist in enumerate(FR_late_mv_cd_sham_zscore) for item in sublist if beta_late_mv_cd_sham[i] > 0]
all_Q_cd_stim_posreg = [item for i,sublist in enumerate(Q_late_mv_cd_stim) for item in sublist if beta_late_mv_cd_stim[i] > 0]
all_Q_cd_sham_posreg = [item for i,sublist in enumerate(Q_late_mv_cd_sham) for item in sublist if beta_late_mv_cd_sham[i] > 0]

plt.figure()
plt.scatter(all_Q_cd_stim_posreg, all_FR_cd_stim_posreg, c = 'c', marker = 'o', label = 'Stim-late')
plt.scatter(all_Q_cd_sham_posreg, all_FR_cd_sham_posreg, c = 'm', marker = 'o', label = 'Sham-late')
plt.xlim((Q_bins[0], Q_bins[-1]))
plt.xlabel('MV Q-value')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.show()

"""
## firing rate differences
FR_late_mv_cd_stim_mean = np.array([np.nanmean(item) for item in FR_late_mv_cd_stim])
FR_early_mv_cd_stim_mean = np.array([np.nanmean(item) for item in FR_early_mv_cd_stim])
FR_diff_mv_cd_stim = FR_late_mv_cd_stim_mean - FR_early_mv_cd_stim_mean

FR_late_mv_cd_sham_mean = np.array([np.nanmean(item) for item in FR_late_mv_cd_sham])
FR_early_mv_cd_sham_mean = np.array([np.nanmean(item) for item in FR_early_mv_cd_sham])
FR_diff_mv_cd_sham = FR_late_mv_cd_sham_mean - FR_early_mv_cd_sham_mean

FR_late_mv_acc_stim_mean = np.array([np.nanmean(item) for item in FR_late_mv_acc_stim])
FR_early_mv_acc_stim_mean = np.array([np.nanmean(item) for item in FR_early_mv_acc_stim])
FR_diff_mv_acc_stim = FR_late_mv_acc_stim_mean - FR_early_mv_acc_stim_mean

FR_late_mv_acc_sham_mean = np.array([np.nanmean(item) for item in FR_late_mv_acc_sham])
FR_early_mv_acc_sham_mean = np.array([np.nanmean(item) for item in FR_early_mv_acc_sham])
FR_diff_mv_acc_sham = FR_late_mv_acc_sham_mean - FR_early_mv_acc_sham_mean

FR_late_other_cd_stim_mean = np.array([np.nanmean(item) for item in FR_late_other_cd_stim])
FR_early_other_cd_stim_mean = np.array([np.nanmean(item) for item in FR_early_other_cd_stim])
FR_diff_other_cd_stim = FR_late_other_cd_stim_mean - FR_early_other_cd_stim_mean

FR_late_other_cd_sham_mean = np.array([np.nanmean(item) for item in FR_late_other_cd_sham])
FR_early_other_cd_sham_mean = np.array([np.nanmean(item) for item in FR_early_other_cd_sham])
FR_diff_other_cd_sham = FR_late_other_cd_sham_mean - FR_early_other_cd_sham_mean

FR_late_other_acc_stim_mean = np.array([np.nanmean(item) for item in FR_late_other_acc_stim])
FR_early_other_acc_stim_mean = np.array([np.nanmean(item) for item in FR_early_other_acc_stim])
FR_diff_other_acc_stim = FR_late_other_acc_stim_mean - FR_early_other_acc_stim_mean

FR_late_other_acc_sham_mean = np.array([np.nanmean(item) for item in FR_late_other_acc_sham])
FR_early_other_acc_sham_mean = np.array([np.nanmean(item) for item in FR_early_other_acc_sham])
FR_diff_other_acc_sham = FR_late_other_acc_sham_mean - FR_early_other_acc_sham_mean

FR_diff_cd_stim = np.append(FR_diff_mv_cd_stim, FR_diff_other_cd_stim)
FR_diff_acc_stim = np.append(FR_diff_mv_acc_stim, FR_diff_other_acc_stim)
FR_diff_cd_sham = np.append(FR_diff_mv_cd_sham, FR_diff_other_cd_sham)
FR_diff_acc_sham = np.append(FR_diff_mv_acc_sham, FR_diff_other_acc_sham)

FR_diff_stim = np.append(FR_diff_cd_stim, FR_diff_acc_stim)
FR_diff_sham = np.append(FR_diff_cd_sham, FR_diff_acc_sham)

avg_diff_cd_stim = np.nanmean(FR_diff_cd_stim)
sem_diff_cd_stim = np.nanstd(FR_diff_cd_stim)/np.sqrt(len(FR_diff_cd_stim))
avg_diff_acc_stim = np.nanmean(FR_diff_acc_stim)
sem_diff_acc_stim = np.nanstd(FR_diff_acc_stim)/np.sqrt(len(FR_diff_acc_stim))
avg_diff_cd_sham = np.nanmean(FR_diff_cd_sham)
sem_diff_cd_sham = np.nanstd(FR_diff_cd_sham)/np.sqrt(len(FR_diff_cd_sham))
avg_diff_acc_sham = np.nanmean(FR_diff_acc_sham)
sem_diff_acc_sham = np.nanstd(FR_diff_acc_sham)/np.sqrt(len(FR_diff_acc_sham))

t, p = stats.ttest_ind(FR_diff_cd_sham, FR_diff_acc_sham)
print t, p

avg_diff_cd = [avg_diff_cd_stim, avg_diff_cd_sham]
avg_diff_acc = [avg_diff_acc_stim, avg_diff_acc_sham]
sem_diff_cd = [sem_diff_cd_stim, sem_diff_cd_sham]
sem_diff_acc = [sem_diff_acc_stim, sem_diff_acc_sham]

avg_diff_stim = np.nanmean(FR_diff_stim)
sem_diff_stim = np.nanstd(FR_diff_stim)/np.sqrt(len(FR_diff_stim))
avg_diff_sham = np.nanmean(FR_diff_sham)
sem_diff_sham = np.nanstd(FR_diff_sham)/np.sqrt(len(FR_diff_sham))

avg_diff = [avg_diff_stim, avg_diff_sham]
sem_diff = [sem_diff_stim, sem_diff_sham]

ind = np.arange(2)
plt.figure()
plt.subplot(1,2,1)
plt.bar(ind[1], avg_diff_cd[1], width, color = 'y', yerr = sem_diff_cd[1], label = 'Cd')
plt.bar(ind[1]+width, avg_diff_acc[1], width, color = 'g', yerr = sem_diff_acc[1], label = 'ACC')
xticklabels = ['Stim', 'Sham']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Condition')
plt.ylabel('Change in Firing Rate (Hz)')
plt.title('Change in FR between first and late block')
plt.legend()

plt.subplot(1,2,2)
plt.bar(ind, avg_diff, width, color = 'y', yerr = sem_diff, label = 'Cd + ACC')
xticklabels = ['Stim', 'Sham']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Condition')
plt.ylabel('Change in Firing Rate (Hz)')
plt.title('Change in FR between first and late block')
plt.legend()
plt.show()
"""