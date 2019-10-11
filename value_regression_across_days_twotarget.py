from DecisionMakingBehavior import TwoTargetTask_RegressedFiringRatesWithValue_PictureOnset
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
#from value_regression_across_days import BinChangeInFiringRatesByValue

# Define units of interest
cd_units = list(range(1,65))
acc_units = list(range(65,129))
all_units = np.append(cd_units, acc_units)
# List data

dir = "C:/Users/ss45436/Box/UC Berkeley/Cd Stim/Neural Correlates/Luigi/spike_data/"

hdf_list_sham = [[dir + 'luig20170822_07_te133.hdf'], \
			[dir + 'luig20170824_02_te139.hdf'], \
			[dir + 'luig20170831_11_te183.hdf'], \
			[dir + 'luig20170902_08_te197.hdf'], \
			[dir + 'luig20170907_04_te212.hdf'], \
			[dir + 'luig20170915_13_te262.hdf'], \
			[dir + 'luig20170929_11_te376.hdf', dir + 'luig20170929_13_te378.hdf'], \
			[dir + 'luig20171003_08_te399.hdf'], \
			[dir + 'luig20171015_03_te447.hdf'], \
			[dir + 'luig20171019_07_te467.hdf'], \
			[dir + 'luig20171024_10_te483.hdf'], \
			[dir + 'luig20171028_08_te500.hdf'], \
			]

hdf_list_stim = [[dir + 'luig20170909_08_te226.hdf'], \
			[dir + 'luig20170915_15_te264.hdf'], \
			[dir + 'luig20170927_07_te361.hdf'], \
			[dir + 'luig20171001_06_te385.hdf'], \
			[dir + 'luig20171005_02_te408.hdf'], \
			[dir + 'luig20171017_04_te454.hdf', dir + 'luig20171017_05_te455.hdf'], \
			[dir + 'luig20171031_04_te509.hdf'], \
			[dir + 'luig20171108_03_te543.hdf'], \
			]


hdf_list = hdf_list_sham + hdf_list_stim

syncHDF_list_sham = [[dir + 'Luigi20170822_b2_syncHDF.mat'], \
				[dir + 'Luigi20170824_b1_syncHDF.mat'], \
				[dir + 'Luigi20170831_b2_syncHDF.mat'], \
				[dir + 'Luigi20170902_b1_syncHDF.mat'], \
				[dir + 'Luigi20170907_b1_syncHDF.mat'], \
				[dir + 'Luigi20170915_b1_syncHDF.mat'], \
				[dir + 'Luigi20170929_b3_syncHDF.mat', dir + 'Luigi20170929_b4_syncHDF.mat'], \
				[dir + 'Luigi20171003_b1_syncHDF.mat'], \
				[dir + 'Luigi20171015_b1_syncHDF.mat'], \
				[dir + 'Luigi20171019_b1_syncHDF.mat'], \
				[dir + 'Luigi20171024_b1_syncHDF.mat'], \
				[dir + 'Luigi20171028_b1_syncHDF.mat'], \
				]
syncHDF_list_stim = [[dir + 'Luigi20170909_b3_syncHDF.mat'], \
				[dir + 'Luigi20170915-2_b1_syncHDF.mat'], \
				[dir + 'Luigi20170927_b3_syncHDF.mat'], \
				[dir + 'Luigi20171001_b1_syncHDF.mat'], \
				[dir + 'Luigi20171005_b1_syncHDF.mat'], \
				[dir + 'Luigi20171017_b1_syncHDF.mat', dir + 'Luigi20171017_b2_syncHDF.mat'], \
				[dir + 'Luigi20171031_b1_syncHDF.mat'], \
				[dir + 'Luigi20171108_b1_syncHDF.mat'], \
				]

syncHDF_list = syncHDF_list_sham + syncHDF_list_stim

spike_list_sham = [[[dir + 'Luigi20170822_Block-2_eNe1_Offline.csv', '']], \
			  [[dir + 'Luigi20170824_Block-1_eNe1_Offline.csv', '']], \
			  [[dir + 'Luigi20170831_Block-2_eNe1_Offline.csv', dir + 'Luigi20170831_Block-2_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20170902_Block-1_eNe1_Offline.csv', '']], \
			  [[dir + 'Luigi20170907_Block-1_eNe1_Offline.csv', dir + 'Luigi20170907_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20170915_Block-1_eNe1_Offline.csv', dir + 'Luigi20170915_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20170929_Block-3_eNe1_Offline.csv', dir + 'Luigi20170929_Block-3_eNe2_Offline.csv'], [dir + 'Luigi20170929_Block-4_eNe1_Offline.csv', dir + 'Luigi20170929_Block-4_eNe2_Offline.csv']],\
			  [[dir + 'Luigi20171003_Block-1_eNe1_Offline.csv', dir + 'Luigi20171003_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171015_Block-1_eNe1_Offline.csv', dir + 'Luigi20171015_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171019_Block-1_eNe1_Offline.csv', dir + 'Luigi20171019_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171024_Block-1_eNe1_Offline.csv', dir + 'Luigi20171024_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171028_Block-1_eNe1_Offline.csv', dir + 'Luigi20171028_Block-1_eNe2_Offline.csv']], \
			  
			  ]
spike_list_stim = [[[dir + 'Luigi20170909_Block-3_eNe1_Offline.csv', dir + 'Luigi20170909_Block-3_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20170915-2_Block-1_eNe1_Offline.csv', dir + 'Luigi20170915-2_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20170927_Block-3_eNe1_Offline.csv', dir + 'Luigi20170927_Block-3_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171001_Block-1_eNe1_Offline.csv', dir + 'Luigi20171001_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171005_Block-1_eNe1_Offline.csv', dir + 'Luigi20171005_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171017_Block-1_eNe1_Offline.csv', dir + 'Luigi20171017_Block-1_eNe2_Offline.csv'], [dir + 'Luigi20171017_Block-2_eNe1_Offline.csv', dir + 'Luigi20171017_Block-2_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171031_Block-1_eNe1_Offline.csv', dir + 'Luigi20171031_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171108_Block-1_eNe1_Offline.csv', dir + 'Luigi20171108_Block-1_eNe2_Offline.csv']], \
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
		if (len(Q_early[i]) > 0):
			FR_e_means, bin_edges, binnumber = stats.binned_statistic(Q_early[i], FR_early[i], statistic= 'mean', bins = Q_bins)
			FR_l_means, bin_edges, binnumber = stats.binned_statistic(Q_late[i], FR_late[i], statistic= 'mean', bins = Q_bins)
			delta_FR[i,:] = FR_l_means - FR_e_means
			norm_delta_FR[i,:] = delta_FR[i,:]/FR_e_means
		else:
			delta_FR[i,:] = np.nan
			norm_delta_FR[i,:] = np.nan
		
		total_nonnans_delta_FR += np.greater(np.abs(delta_FR[i,:]),0)
		total_nonnans_norm_delta_FR += np.greater(np.abs(norm_delta_FR[i,:]), 0)
	
	avg_delta_FR = np.nanmean(delta_FR, axis = 0)
	sem_delta_FR = np.nanstd(delta_FR, axis = 0)/np.sqrt(total_nonnans_delta_FR)
	avg_norm_delta_FR = np.nanmean(norm_delta_FR, axis = 0)
	sem_norm_delta_FR = np.nanstd(norm_delta_FR, axis = 0)/np.sqrt(total_nonnans_norm_delta_FR)

	return Q_bins, delta_FR, norm_delta_FR, avg_delta_FR, sem_delta_FR, avg_norm_delta_FR, sem_norm_delta_FR


num_files = len(hdf_list)
t_before = 0.
t_after = 0.4
smoothed = 1

###RUNNING WITH SLIDING VALUES BUT BACK TO PICTURE ONSET
for i in range(num_files):
	for channel in cd_units:
		hdf = hdf_list[i]
		sync = syncHDF_list[i]
		spike = spike_list[i]

		Qs = TwoTargetTask_RegressedFiringRatesWithValue_PictureOnset(dir, hdf, sync, spike, channel, t_before, t_after, smoothed)


'''

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
count_HV_blockA_cd = 0
count_LVHV_blockA_cd = 0
count_rt_blockA_cd = 0
count_mt_blockA_cd = 0
count_choice_blockA_cd = 0
count_reward_blockA_cd = 0

count_LV_blocksAB_cd = 0
count_HV_blocksAB_cd = 0
count_LVHV_blocksAB_cd = 0
count_rt_blocksAB_cd = 0
count_mt_blocksAB_cd = 0
count_choice_blocksAB_cd = 0
count_reward_blocksAB_cd = 0

count_LV_blockA_acc = 0
count_HV_blockA_acc = 0
count_LVHV_blockA_acc = 0
count_rt_blockA_acc = 0
count_mt_blockA_acc = 0
count_choice_blockA_acc = 0
count_reward_blockA_acc = 0

count_LV_blocksAB_acc = 0
count_HV_blocksAB_acc = 0
count_LVHV_blocksAB_acc = 0
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

Q_late_lv_cd_sham = []
Q_early_lv_cd_sham = []
FR_late_lv_cd_sham = []
FR_early_lv_cd_sham = []
Q_late_other_cd_sham = []
Q_early_other_cd_sham = []
FR_late_other_cd_sham = []
FR_early_other_cd_sham = []
Q_late_lv_acc_sham = []
Q_early_lv_acc_sham = []
FR_late_lv_acc_sham = []
FR_early_lv_acc_sham = []
Q_late_other_acc_sham = []
Q_early_other_acc_sham = []
FR_late_other_acc_sham = []
FR_early_other_acc_sham = []

Q_late_lv_cd_stim = []
Q_early_lv_cd_stim = []
FR_late_lv_cd_stim = []
FR_early_lv_cd_stim = []
Q_late_other_cd_stim = []
Q_early_other_cd_stim = []
FR_late_other_cd_stim = []
FR_early_other_cd_stim = []
Q_late_lv_acc_stim = []
Q_early_lv_acc_stim = []
FR_late_lv_acc_stim = []
FR_early_lv_acc_stim = []
Q_late_other_acc_stim = []
Q_early_other_acc_stim = []
FR_late_other_acc_stim = []
FR_early_other_acc_stim = []

# recording beta values for mv coding units
beta_late_lv_cd_stim = np.array([])
beta_late_lv_acc_stim = np.array([])
beta_late_lv_cd_sham = np.array([])
beta_late_lv_acc_sham = np.array([])


syncHDF_list_stim_flat = [item for sublist in syncHDF_list_stim for item in sublist]


"""
Loop through pre-processed files to count up the number of neurons that have significant beta values for the various
regressors. 
"""


for filen in filenames:
	print(filen)
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

			"""
			For Luigi, only Q-values were put into regression along with RT, MT, choice, and reward. Mario additionally had regressors
			indicating whether a target was shown and a value-coding neuron was counted if these regressors were significant as well.
			"""
			val_check = [sig_beta_blockA[0], sig_beta_blockA[1]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1:
				LV_enc = 1
				beta_early_lv = sig_beta_blockA[0]
			
			if val_check[1]==1:
				HV_enc = 1

			if ((LV_enc==1) and (HV_enc==0)):
				count_LV_blockA_cd += 1
			elif ((LV_enc==0) and (HV_enc==1)):
				count_HV_blockA_cd += 1
			elif ((LV_enc==1) and (HV_enc==1)):
				count_LVHV_blockA_cd += 1
			else:
				no_val_enc = 1

			if no_val_enc:
				max_beta = np.argmax(np.abs(sig_beta_blockA))
				if max_beta==2:
					count_rt_blockA_cd += 1
				if max_beta==3:
					count_mt_blockA_cd += 1
				if max_beta==4:
					count_choice_blockA_cd +=1
				if max_beta==5:
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

			"""
			For Luigi, only Q-values were put into regression along with RT, MT, choice, and reward. Mario additionally had regressors
			indicating whether a target was shown and a value-coding neuron was counted if these regressors were significant as well.
			"""
			val_check = [sig_beta_blocksAB[0], sig_beta_blocksAB[1]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1:
				LV_enc = 1
				beta_late_lv = sig_beta_blocksAB[0]

			if val_check[1]==1:
				HV_enc = 1

			Q_late = np.ravel(data['Q_low_late'])
			FR_late = np.ravel(data['FR_late'])
			if 'Q_low_early' in data.keys():
				Q_early = np.ravel(data['Q_low_early'])
			else:
				Q_early = np.array([])
			if 'FR_early' in data.keys():
				FR_early = np.ravel(data['FR_early'])
			else:
				FR_early = np.array([])
			
			if ((LV_enc==1) and (HV_enc==0)):
				count_LV_blocksAB_cd += 1
				if stim_file:
					Q_late_lv_cd_stim += [Q_late]
					Q_early_lv_cd_stim += [Q_early]
					FR_late_lv_cd_stim += [FR_late]
					FR_early_lv_cd_stim += [FR_early]
					beta_late_lv_cd_stim = np.append(beta_late_lv_cd_stim, beta_late_lv)
				else:
					Q_late_lv_cd_sham += [Q_late]
					Q_early_lv_cd_sham += [Q_early]
					FR_late_lv_cd_sham += [FR_late]
					FR_early_lv_cd_sham += [FR_early]
					beta_late_lv_cd_sham = np.append(beta_late_lv_cd_sham, beta_late_lv)
			elif ((LV_enc==0) and (HV_enc==1)):
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
			
			elif ((LV_enc==1) and (HV_enc==1)):
				count_LVHV_blocksAB_cd += 1
				if stim_file:
					Q_late_lv_cd_stim += [Q_late]
					Q_early_lv_cd_stim += [Q_early]
					FR_late_lv_cd_stim += [FR_late]
					FR_early_lv_cd_stim += [FR_early]
					beta_late_lv_cd_stim = np.append(beta_late_lv_cd_stim, beta_late_lv)
				else:
					Q_late_lv_cd_sham += [Q_late]
					Q_early_lv_cd_sham += [Q_early]
					FR_late_lv_cd_sham += [FR_late]
					FR_early_lv_cd_sham += [FR_early]
					beta_late_lv_cd_sham = np.append(beta_late_lv_cd_sham, beta_late_lv)
			
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
				if max_beta==2:
					count_rt_blocksAB_cd += 1
				if max_beta==3:
					count_mt_blocksAB_cd += 1
				if max_beta==4:
					count_choice_blocksAB_cd +=1
				if max_beta==5:
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

			val_check = [sig_beta_blocksAB[0], sig_beta_blocksAB[1]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1:
				LV_enc = 1
				beta_late_lv = sig_beta_blocksAB[0]

			if val_check[1]==1:
				HV_enc = 1

			if ((LV_enc==1) and (HV_enc==0)):
				count_LV_blockA_acc += 1
			elif ((LV_enc==0) and (HV_enc==1)):
				count_HV_blockA_acc += 1
			elif ((LV_enc==1) and (HV_enc==1)):
				count_LVHV_blockA_acc += 1
			else:
				no_val_enc = 1

			if no_val_enc:
				max_beta = np.argmax(np.abs(sig_beta_blockA))
				if max_beta==2:
					count_rt_blockA_acc += 1
				if max_beta==3:
					count_mt_blockA_acc += 1
				if max_beta==4:
					count_choice_blockA_acc +=1
				if max_beta==5:
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


			val_check = [sig_beta_blocksAB[0], sig_beta_blocksAB[1]]
			val_check = np.greater(np.abs(val_check),0)

			LV_enc = 0
			HV_enc = 0
			no_val_enc = 0
			if val_check[0]==1:
				LV_enc = 1
				beta_late_lv = sig_beta_blocksAB[0]

			if val_check[1]==1:
				HV_enc = 1

			if ((LV_enc==1) and (HV_enc==0)):
				count_LV_blockA_acc += 1
			elif ((LV_enc==0) and (HV_enc==1)):
				count_HV_blockA_acc += 1
			elif ((LV_enc==1) and (HV_enc==1)):
				count_LVHV_blockA_acc += 1
			else:
				no_val_enc = 1

			Q_late = np.ravel(data['Q_low_late'])
			FR_late = np.ravel(data['FR_late'])
			if 'Q_low_early' in data.keys():
				Q_early = np.ravel(data['Q_low_early'])
			else:
				Q_early = np.array([])
			if 'FR_early' in data.keys():
				FR_early = np.ravel(data['FR_early'])
			else:
				FR_early = np.array([])
			
			if ((LV_enc==1) and (HV_enc==0)):
				count_LV_blocksAB_acc += 1
				if stim_file:
					Q_late_lv_acc_stim += [Q_late]
					Q_early_lv_acc_stim += [Q_early]
					FR_late_lv_acc_stim += [FR_late]
					FR_early_lv_acc_stim += [FR_early]
					beta_late_lv_acc_stim = np.append(beta_late_lv_acc_stim, beta_late_lv)
				else:
					Q_late_lv_acc_sham += [Q_late]
					Q_early_lv_acc_sham += [Q_early]
					FR_late_lv_acc_sham += [FR_late]
					FR_early_lv_acc_sham += [FR_early]
					beta_late_lv_acc_sham = np.append(beta_late_lv_acc_sham, beta_late_lv)
			elif ((LV_enc==0) and (HV_enc==1)):
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
			
			elif ((LV_enc==1) and (HV_enc==1)):
				count_LVHV_blocksAB_acc += 1
				if stim_file:
					Q_late_lv_acc_stim += [Q_late]
					Q_early_lv_acc_stim += [Q_early]
					FR_late_lv_acc_stim += [FR_late]
					FR_early_lv_acc_stim += [FR_early]
					beta_late_lv_acc_stim = np.append(beta_late_lv_acc_stim, beta_late_lv)
				else:
					Q_late_lv_acc_sham += [Q_late]
					Q_early_lv_acc_sham += [Q_early]
					FR_late_lv_acc_sham += [FR_late]
					FR_early_lv_acc_sham += [FR_early]
					beta_late_lv_acc_sham = np.append(beta_late_lv_acc_sham, beta_late_lv)
			
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
				if max_beta==2:
					count_rt_blocksAB_acc += 1
				if max_beta==3:
					count_mt_blocksAB_acc += 1
				if max_beta==4:
					count_choice_blocksAB_acc +=1
				if max_beta==5:
					count_reward_blocksAB_acc +=1

		if ('rsquared_blockA' in data.keys()):
			rsquared_blockA_acc = np.append(rsquared_blockA_acc, data['rsquared_blockA'])
		if ('rsquared_blocksAB' in data.keys()):
			rsquared_blocksAB_acc = np.append(rsquared_blocksAB_acc, data['rsquared_blocksAB'])


true_blockA_cd = np.greater(sig_reg_blockA_cd, 0)
true_blocksAB_cd = np.greater(sig_reg_blocksAB_cd, 0)
true_blockA_acc = np.greater(sig_reg_blockA_acc, 0)
true_blocksAB_acc = np.greater(sig_reg_blocksAB_acc, 0)

#count_all_blockA_cd = np.sum(true_blockA_cd, axis = 1)
#count_all_blocksAB_cd = np.sum(true_blocksAB_cd, axis = 1)
#count_all_blockA_acc = np.sum(true_blockA_acc, axis = 1)
#count_all_blocksAB_acc = np.sum(true_blocksAB_acc, axis = 1)

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
all_value_blockA_cd = float(count_LV_blockA_cd + count_HV_blockA_cd +  \
						count_LVHV_blockA_cd )
all_value_blockA_acc = float(count_LV_blockA_acc + count_HV_blockA_acc  + \
						count_LVHV_blockA_acc )
all_value_blocksAB_cd = float(count_LV_blocksAB_cd + count_HV_blocksAB_cd + \
						count_LVHV_blocksAB_cd )
all_value_blocksAB_acc = float(count_LV_blocksAB_acc  + count_HV_blocksAB_acc  + \
						count_LVHV_blocksAB_acc )

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


labels_value = ['LV', 'HV', 'LV&HV']
fracs_value_blockA_cd = [count_LV_blockA_cd/all_value_blockA_cd, count_HV_blockA_cd/all_value_blockA_cd, \
							count_LVHV_blockA_cd/all_value_blockA_cd]
fracs_value_blockA_acc = [count_LV_blockA_acc/all_value_blockA_acc, count_HV_blockA_acc/all_value_blockA_acc, \
							count_LVHV_blockA_acc/all_value_blockA_acc]
fracs_value_blocksAB_cd = [count_LV_blocksAB_cd/all_value_blocksAB_cd, count_HV_blocksAB_cd/all_value_blocksAB_cd, \
							count_LVHV_blocksAB_cd/all_value_blocksAB_cd]
fracs_value_blocksAB_acc = [count_LV_blocksAB_acc/all_value_blocksAB_acc, count_HV_blocksAB_acc/all_value_blocksAB_acc, \
							count_LVHV_blocksAB_acc/all_value_blocksAB_acc]

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


all_Q = Q_late_lv_cd_stim + Q_early_lv_cd_stim + Q_late_other_cd_stim + Q_early_other_cd_stim + \
		Q_late_lv_acc_stim + Q_early_lv_acc_stim + Q_late_other_acc_stim + Q_early_other_acc_stim + \
		Q_late_lv_cd_sham + Q_early_lv_cd_sham + Q_late_other_cd_sham + Q_early_other_cd_sham + \
		Q_late_lv_acc_sham + Q_early_lv_acc_sham + Q_late_other_acc_sham + Q_early_other_acc_sham
all_Q_flat = [item for sublist in all_Q for item in sublist]
min_Q = np.min(all_Q_flat)
max_Q = np.max(all_Q_flat)
min_Q = 0.2
max_Q = 0.6
print(min_Q, max_Q)
Q_bins = np.arange(min_Q, max_Q + (max_Q - min_Q)/6., (max_Q - min_Q)/6.)
print(Q_bins)
# bin Qs and sort FRs accordingly for all conditions

# Cd - value coding - stim
Q_bins_cd_lv_stim, delta_FR_cd_lv_stim, norm_delta_FR_cd_lv_stim, avg_delta_FR_cd_lv_stim, \
	sem_delta_FR_cd_lv_stim, avg_norm_delta_FR_cd_lv_stim, sem_norm_delta_FR_cd_lv_stim = \
	BinChangeInFiringRatesByValue(Q_early_lv_cd_stim, Q_late_lv_cd_stim, Q_bins, FR_early_lv_cd_stim, FR_late_lv_cd_stim)
# Cd - value coding - sham
Q_bins_cd_lv_sham, delta_FR_cd_lv_sham, norm_delta_FR_cd_lv_sham, avg_delta_FR_cd_lv_sham, \
	sem_delta_FR_cd_lv_sham, avg_norm_delta_FR_cd_lv_sham, sem_norm_delta_FR_cd_lv_sham = \
	BinChangeInFiringRatesByValue(Q_early_lv_cd_sham, Q_late_lv_cd_sham, Q_bins, FR_early_lv_cd_sham, FR_late_lv_cd_sham)
# Cd - nonvalue coding - stim
Q_bins_cd_other_stim, delta_FR_cd_other_stim, norm_delta_FR_cd_other_stim, avg_delta_FR_cd_other_stim, \
	sem_delta_FR_cd_other_stim, avg_norm_delta_FR_cd_other_stim, sem_norm_delta_FR_cd_other_stim = \
	BinChangeInFiringRatesByValue(Q_early_other_cd_stim, Q_late_other_cd_stim, Q_bins, FR_early_other_cd_stim, FR_late_other_cd_stim)


# Cd - nonvalue coding - sham
Q_bins_cd_other_sham, delta_FR_cd_other_sham, norm_delta_FR_cd_other_sham, avg_delta_FR_cd_other_sham, \
	sem_delta_FR_cd_other_sham, avg_norm_delta_FR_cd_other_sham, sem_norm_delta_FR_cd_other_sham = \
	BinChangeInFiringRatesByValue(Q_early_other_cd_sham, Q_late_other_cd_sham, Q_bins, FR_early_other_cd_sham, FR_late_other_cd_sham)


# ACC - value coding - stim
Q_bins_acc_lv_stim, delta_FR_acc_lv_stim, norm_delta_FR_acc_lv_stim, avg_delta_FR_acc_lv_stim, \
	sem_delta_FR_acc_lv_stim, avg_norm_delta_FR_acc_lv_stim, sem_norm_delta_FR_acc_lv_stim = \
	BinChangeInFiringRatesByValue(Q_early_lv_acc_stim, Q_late_lv_acc_stim, Q_bins, FR_early_lv_acc_stim, FR_late_lv_acc_stim)
# ACC - value coding - sham
Q_bins_acc_lv_sham, delta_FR_acc_lv_sham, norm_delta_FR_acc_lv_sham, avg_delta_FR_acc_lv_sham, \
	sem_delta_FR_acc_lv_sham, avg_norm_delta_FR_acc_lv_sham, sem_norm_delta_FR_acc_lv_sham = \
	BinChangeInFiringRatesByValue(Q_early_lv_acc_sham, Q_late_lv_acc_sham, Q_bins, FR_early_lv_acc_sham, FR_late_lv_acc_sham)
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
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_lv_stim, yerr = sem_delta_FR_cd_lv_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - value coding')
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_lv_sham, yerr = sem_delta_FR_cd_lv_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - value coding')
plt.xlabel('LV Q-value')
plt.ylabel("Delta FR (Block A' - Block A) (Hz)")
plt.legend()
plt.title('Value-coding Cd Units')

plt.subplot(2,2,2)
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_lv_stim, yerr = sem_norm_delta_FR_cd_lv_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - value coding')
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_lv_sham, yerr = sem_norm_delta_FR_cd_lv_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - value coding')
plt.xlabel('LV Q-value')
plt.ylabel("Normalized Delta FR (Block A' - Block A) (A.U.)")
plt.legend()
plt.title('Value-coding Cd Units')

plt.subplot(2,2,3)
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_other_stim, yerr = sem_delta_FR_cd_other_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - non value coding')
plt.errorbar(Q_bin_centers, avg_delta_FR_cd_other_sham, yerr = sem_delta_FR_cd_other_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - non value coding')
plt.xlabel('LV Q-value')
plt.ylabel("Delta FR (Block A' - Block A) (Hz)")
plt.legend()
plt.title('Non-value-coding Cd Units')

plt.subplot(2,2,4)
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_other_stim, yerr = sem_norm_delta_FR_cd_other_stim, fmt = 'o', color = 'c', ecolor = 'c', label = 'Stim - non value coding')
plt.errorbar(Q_bin_centers, avg_norm_delta_FR_cd_other_sham, yerr = sem_norm_delta_FR_cd_other_sham, fmt = 'o', color = 'm', ecolor = 'm', label = 'Sham - non value coding')
plt.xlabel('LV Q-value')
plt.ylabel("Normalized Delta FR (Block A' - Block A) (A.U.)")
plt.legend()
plt.title('Non-value-coding Cd Units')
plt.show()
'''