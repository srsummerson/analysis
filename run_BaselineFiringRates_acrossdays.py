import numpy as np 
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tables
import sys
import os.path

from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile
from DecisionMakingBehavior import MultiTargetTask_FiringRates_DifferenceBetweenBlocks, MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan

# change MultiTargetTask_FiringRates_DifferenceBetweenBlocks to work over multiple channels rather than one

# write code to compute average firing rate changes (normalized and not normalized) for Mario and Luigi
mario_cd_channels = [1, 3, 4, 17, 18, 20, 35, 37, 38, 40, 41, 51, 53, 54, 56, 57, 63, 64, 65, 67, 70, 72, 73, 75, 81, 83, 86, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
mario_acc_channels = [5, 6, 19, 22, 30, 39, 42, 43, 55, 58, 59, 69, 74, 77, 85, 90, 91, 102, 105, 121, 128]
luigi_cd_channels = np.arange(1,65)
luigi_acc_channels = np.arange(65,129)



###############################
# Code for Mario for Sham days
###############################
print "\nMario - Sham days \n"

behavior_sheet = 'Mario_Behavior_Log.xlsx'
session_type_sheet = 1 		# 1 for sham days, 2 for stim days
data_folder = "C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/spike_data/"

xls = pd.ExcelFile(behavior_sheet)
sheetX = xls.parse(session_type_sheet)
var1 = sheetX['Session']
var2 = sheetX['Hdf filename']
var3 = sheetX['TDT filename']
var4 = sheetX['Use for neural']

session_renumber = var1
#session_renumber = np.array([(var1[n-1] if np.isnan(var1[n]) else var1[n]) for n in range(len(var1))])
sessions, len_sessions = np.unique(session_renumber, return_counts = True) 		# extract session numbers

start_session_num = 1
last_session_num = 12

tot_sessions = last_session_num - start_session_num + 1
num_used_sessions = 0

for k in range(start_session_num, last_session_num+1):
	curr_session = k
	print "On session %i" % (k)
	len_curr_session = len_sessions[k-1]
	row_num = np.ravel(np.argwhere(var1==curr_session))
	if var4[row_num[0]]==1:
		num_used_sessions += 1
		hdf_filename = np.ravel(var2[row_num])
		filename = [np.ravel(var3[row_num])[l][:-7] for l in range(len(row_num))]
		block_num = [np.ravel(var3[row_num])[l][-1] for l in range(len(row_num))]

		hdf_files = [(data_folder + hdf_file) for hdf_file in hdf_filename]
		syncHDF_files = [(data_folder + file + '_b' + str(block_num[m]) +'_syncHDF.mat') for m,file in enumerate(filename)]
		spike_files = [[(data_folder + file + '_Block-' + str(block_num[m]) + '_eNe1_Offline.csv'), (data_folder + file + '_Block-' + str(block_num[m]) + '_eNe2_Offline.csv')] for m,file in enumerate(filename)]
		
		num_spike_files = len(spike_files)
		for i in range(num_spike_files):
			if os.path.isfile(spike_files[i][0])==False:
				spike_files[i][0] = ''
			if os.path.isfile(spike_files[i][1])==False:
				spike_files[i][1] = ''

		num_targets = 3

		# Find good channels that are either Cd or ACC
		spike1 = OfflineSorted_CSVFile(spike_files[0][0])
		spike2 = OfflineSorted_CSVFile(spike_files[0][1])
		all_channels = np.hstack([spike1.good_channels, spike2.good_channels])
		print all_channels
		cd_channels = [chan for chan in all_channels if chan in mario_cd_channels]
		acc_channels = [chan for chan in all_channels if chan in mario_acc_channels]
		all_good_channels = cd_channels + acc_channels

		# compute firing rates across blocks
		print "Beginning Cd data"
		# results are arrays with firing rate data for all units on all indicated channels
		cd_avg_fr_diff, cd_avg_fr_blockA, cd_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, cd_channels)
		
		print "Beginning ACC data"
		# results are arrays with firing rate data for all units on all indicated channels
		acc_avg_fr_diff, acc_avg_fr_blockA, acc_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, acc_channels)
		
		if num_used_sessions==1:
			cd_diff_m_sham = cd_avg_fr_diff
			cd_blockA_m_sham = cd_avg_fr_blockA
			cd_blockAprime_m_sham = cd_avg_fr_blockAprime
			acc_diff_m_sham = acc_avg_fr_diff
			acc_blockA_m_sham = acc_avg_fr_blockA
			acc_blockAprime_m_sham = acc_avg_fr_blockAprime
		else:
			cd_diff_m_sham = np.hstack([cd_diff_m_sham, cd_avg_fr_diff])
			cd_blockA_m_sham = np.hstack([cd_blockA_m_sham, cd_avg_fr_blockA])
			cd_blockAprime_m_sham = np.hstack([cd_blockAprime_m_sham, cd_avg_fr_blockAprime])
			acc_diff_m_sham = np.hstack([acc_diff_m_sham, acc_avg_fr_diff])
			acc_blockA_m_sham = np.hstack([acc_blockA_m_sham, acc_avg_fr_blockA])
			acc_blockAprime_m_sham = np.hstack([acc_blockAprime_m_sham, acc_avg_fr_blockAprime])
		
###############################
# Code for Mario for Stim days
###############################
print "\nMario - Stim days \n"

behavior_sheet = 'Mario_Behavior_Log.xlsx'
session_type_sheet = 2 		# 1 for sham days, 2 for stim days
data_folder = "C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/spike_data/"

xls = pd.ExcelFile(behavior_sheet)
sheetX = xls.parse(session_type_sheet)
var1 = sheetX['Session']
var2 = sheetX['Hdf filename']
var3 = sheetX['TDT filename']
var4 = sheetX['Use for neural']

session_renumber = var1
#session_renumber = np.array([(var1[n-1] if np.isnan(var1[n]) else var1[n]) for n in range(len(var1))])
sessions, len_sessions = np.unique(session_renumber, return_counts = True) 		# extract session numbers

start_session_num = 1
last_session_num = 12

tot_sessions = last_session_num - start_session_num + 1
num_used_sessions = 0

for k in range(start_session_num, last_session_num+1):
	curr_session = k
	print "On session %i" % (k)
	len_curr_session = len_sessions[k-1]
	row_num = np.ravel(np.argwhere(var1==curr_session))
	if var4[row_num[0]]==1:
		num_used_sessions += 1
		hdf_filename = np.ravel(var2[row_num])
		filename = [np.ravel(var3[row_num])[l][:-7] for l in range(len(row_num))]
		block_num = [np.ravel(var3[row_num])[l][-1] for l in range(len(row_num))]

		hdf_files = [(data_folder + hdf_file) for hdf_file in hdf_filename]
		syncHDF_files = [(data_folder + file + '_b' + str(block_num[m]) +'_syncHDF.mat') for m,file in enumerate(filename)]
		spike_files = [[(data_folder + file + '_Block-' + str(block_num[m]) + '_eNe1_Offline.csv'), (data_folder + file + '_Block-' + str(block_num[m]) + '_eNe2_Offline.csv')] for m,file in enumerate(filename)]
		
		num_spike_files = len(spike_files)
		for i in range(num_spike_files):
			if os.path.isfile(spike_files[i][0])==False:
				spike_files[i][0] = ''
			if os.path.isfile(spike_files[i][1])==False:
				spike_files[i][1] = ''

		num_targets = 3

		# Find good channels that are either Cd or ACC
		spike1 = OfflineSorted_CSVFile(spike_files[0][0])
		spike2 = OfflineSorted_CSVFile(spike_files[0][1])
		all_channels = np.hstack([spike1.good_channels, spike2.good_channels])
		print all_channels
		cd_channels = [chan for chan in all_channels if chan in mario_cd_channels]
		acc_channels = [chan for chan in all_channels if chan in mario_acc_channels]
		all_good_channels = cd_channels + acc_channels

		# compute firing rates across blocks
		print "Beginning Cd data"
		# results are arrays with firing rate data for all units on all indicated channels
		cd_avg_fr_diff, cd_avg_fr_blockA, cd_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, cd_channels)
		
		print "Beginning ACC data"
		# results are arrays with firing rate data for all units on all indicated channels
		acc_avg_fr_diff, acc_avg_fr_blockA, acc_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, acc_channels)
		
		if num_used_sessions==1:
			cd_diff_m_stim = cd_avg_fr_diff
			cd_blockA_m_stim = cd_avg_fr_blockA
			cd_blockAprime_m_stim = cd_avg_fr_blockAprime
			acc_diff_m_stim = acc_avg_fr_diff
			acc_blockA_m_stim = acc_avg_fr_blockA
			acc_blockAprime_m_stim = acc_avg_fr_blockAprime
		else:
			cd_diff_m_stim = np.hstack([cd_diff_m_stim, cd_avg_fr_diff])
			cd_blockA_m_stim = np.hstack([cd_blockA_m_stim, cd_avg_fr_blockA])
			cd_blockAprime_m_stim = np.hstack([cd_blockAprime_m_stim, cd_avg_fr_blockAprime])
			acc_diff_m_stim = np.hstack([acc_diff_m_stim, acc_avg_fr_diff])
			acc_blockA_m_stim = np.hstack([acc_blockA_m_stim, acc_avg_fr_blockA])
			acc_blockAprime_m_stim = np.hstack([acc_blockAprime_m_stim, acc_avg_fr_blockAprime])

###############################
# Code for Luigi for Sham days
###############################
print "\nLuigi - Sham days \n"

behavior_sheet = 'Luigi_Behavior_Log.xlsx'
session_type_sheet = 1 		# 1 for sham days, 2 for stim days
data_folder = "C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Luigi/spike_data/"

xls = pd.ExcelFile(behavior_sheet)
sheetX = xls.parse(session_type_sheet)
var1 = sheetX['Session']
var2 = sheetX['Hdf filename']
var3 = sheetX['TDT filename']
var4 = sheetX['Use for neural']

session_renumber = var1
#session_renumber = np.array([(var1[n-1] if np.isnan(var1[n]) else var1[n]) for n in range(len(var1))])
sessions, len_sessions = np.unique(session_renumber, return_counts = True) 		# extract session numbers

start_session_num = 1
last_session_num = 12

tot_sessions = last_session_num - start_session_num + 1
num_used_sessions = 0

for k in range(start_session_num, last_session_num+1):
	curr_session = k
	print "On session %i" % (k)
	len_curr_session = len_sessions[k-1]
	row_num = np.ravel(np.argwhere(var1==curr_session))
	if var4[row_num[0]]==1:
		num_used_sessions += 1
		hdf_filename = np.ravel(var2[row_num])
		filename = [np.ravel(var3[row_num])[l][:-7] for l in range(len(row_num))]
		block_num = [np.ravel(var3[row_num])[l][-1] for l in range(len(row_num))]

		hdf_files = [(data_folder + hdf_file) for hdf_file in hdf_filename]
		syncHDF_files = [(data_folder + file + '_b' + str(block_num[m]) +'_syncHDF.mat') for m,file in enumerate(filename)]
		spike_files = [[(data_folder + file + '_Block-' + str(block_num[m]) + '_eNe1_Offline.csv'), (data_folder + file + '_Block-' + str(block_num[m]) + '_eNe2_Offline.csv')] for m,file in enumerate(filename)]
		
		num_spike_files = len(spike_files)
		for i in range(num_spike_files):
			if os.path.isfile(spike_files[i][0])==False:
				spike_files[i][0] = ''
				spike1_goodchannels = np.array([])
			else:
				spike1 = OfflineSorted_CSVFile(spike_files[0][0])
				spike1_goodchannels = spike1.good_channels
			if os.path.isfile(spike_files[i][1])==False:
				spike_files[i][1] = ''
				spike2_goodchannels = np.array([])
			else:
				spike2 = OfflineSorted_CSVFile(spike_files[0][1])
				spike2_goodchannels = spike2.good_channels

		num_targets = 2

		# Find good channels that are either Cd or ACC
		all_channels = np.hstack([spike1_goodchannels, spike2_goodchannels])
		print all_channels
		cd_channels = [chan for chan in all_channels if chan in luigi_cd_channels]
		acc_channels = [chan for chan in all_channels if chan in luigi_acc_channels]
		all_good_channels = cd_channels + acc_channels

		# compute firing rates across blocks
		print "Beginning Cd data"
		# results are arrays with firing rate data for all units on all indicated channels
		cd_avg_fr_diff, cd_avg_fr_blockA, cd_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, cd_channels)
		
		print "Beginning ACC data"
		# results are arrays with firing rate data for all units on all indicated channels
		acc_avg_fr_diff, acc_avg_fr_blockA, acc_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, acc_channels)
		
		if num_used_sessions==1:
			cd_diff_l_sham = cd_avg_fr_diff
			cd_blockA_l_sham = cd_avg_fr_blockA
			cd_blockAprime_l_sham = cd_avg_fr_blockAprime
			acc_diff_l_sham = acc_avg_fr_diff
			acc_blockA_l_sham = acc_avg_fr_blockA
			acc_blockAprime_l_sham = acc_avg_fr_blockAprime
		else:
			cd_diff_l_sham = np.hstack([cd_diff_l_sham, cd_avg_fr_diff])
			cd_blockA_l_sham = np.hstack([cd_blockA_l_sham, cd_avg_fr_blockA])
			cd_blockAprime_l_sham = np.hstack([cd_blockAprime_l_sham, cd_avg_fr_blockAprime])
			acc_diff_l_sham = np.hstack([acc_diff_l_sham, acc_avg_fr_diff])
			acc_blockA_l_sham = np.hstack([acc_blockA_l_sham, acc_avg_fr_blockA])
			acc_blockAprime_l_sham = np.hstack([acc_blockAprime_l_sham, acc_avg_fr_blockAprime])


###############################
# Code for Luigi for Stim days
###############################
print "\nLuigi - Stim days \n"

behavior_sheet = 'Luigi_Behavior_Log.xlsx'
session_type_sheet = 2 		# 1 for sham days, 2 for stim days
data_folder = "C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Luigi/spike_data/"

xls = pd.ExcelFile(behavior_sheet)
sheetX = xls.parse(session_type_sheet)
var1 = sheetX['Session']
var2 = sheetX['Hdf filename']
var3 = sheetX['TDT filename']
var4 = sheetX['Use for neural']

session_renumber = var1
#session_renumber = np.array([(var1[n-1] if np.isnan(var1[n]) else var1[n]) for n in range(len(var1))])
sessions, len_sessions = np.unique(session_renumber, return_counts = True) 		# extract session numbers

start_session_num = 1
last_session_num = 3

tot_sessions = last_session_num - start_session_num + 1
num_used_sessions = 0

for k in range(start_session_num, last_session_num+1):
	curr_session = k
	print "On session %i" % (k)
	len_curr_session = len_sessions[k-1]
	row_num = np.ravel(np.argwhere(var1==curr_session))
	if var4[row_num[0]]==1:
		num_used_sessions += 1
		hdf_filename = np.ravel(var2[row_num])
		filename = [np.ravel(var3[row_num])[l][:-7] for l in range(len(row_num))]
		block_num = [np.ravel(var3[row_num])[l][-1] for l in range(len(row_num))]

		hdf_files = [(data_folder + hdf_file) for hdf_file in hdf_filename]
		syncHDF_files = [(data_folder + file + '_b' + str(block_num[m]) +'_syncHDF.mat') for m,file in enumerate(filename)]
		spike_files = [[(data_folder + file + '_Block-' + str(block_num[m]) + '_eNe1_Offline.csv'), (data_folder + file + '_Block-' + str(block_num[m]) + '_eNe2_Offline.csv')] for m,file in enumerate(filename)]
		
		num_spike_files = len(spike_files)
		for i in range(num_spike_files):
			if os.path.isfile(spike_files[i][0])==False:
				spike_files[i][0] = ''
			if os.path.isfile(spike_files[i][1])==False:
				spike_files[i][1] = ''

		num_targets = 2

		# Find good channels that are either Cd or ACC
		spike1 = OfflineSorted_CSVFile(spike_files[0][0])
		spike2 = OfflineSorted_CSVFile(spike_files[0][1])
		all_channels = np.hstack([spike1.good_channels, spike2.good_channels])
		print all_channels
		cd_channels = [chan for chan in all_channels if chan in luigi_cd_channels]
		acc_channels = [chan for chan in all_channels if chan in luigi_acc_channels]
		all_good_channels = cd_channels + acc_channels

		# compute firing rates across blocks
		print "Beginning Cd data"
		# results are arrays with firing rate data for all units on all indicated channels
		cd_avg_fr_diff, cd_avg_fr_blockA, cd_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, cd_channels)
		
		print "Beginning ACC data"
		# results are arrays with firing rate data for all units on all indicated channels
		acc_avg_fr_diff, acc_avg_fr_blockA, acc_avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks_multichan(hdf_files, syncHDF_files, spike_files, num_targets, acc_channels)
		
		if num_used_sessions==1:
			cd_diff_l_stim = cd_avg_fr_diff
			cd_blockA_l_stim = cd_avg_fr_blockA
			cd_blockAprime_l_stim = cd_avg_fr_blockAprime
			acc_diff_l_stim = acc_avg_fr_diff
			acc_blockA_l_stim = acc_avg_fr_blockA
			acc_blockAprime_l_stim = acc_avg_fr_blockAprime
		else:
			cd_diff_l_stim = np.hstack([cd_diff_l_stim, cd_avg_fr_diff])
			cd_blockA_l_stim = np.hstack([cd_blockA_l_stim, cd_avg_fr_blockA])
			cd_blockAprime_l_stim = np.hstack([cd_blockAprime_l_stim, cd_avg_fr_blockAprime])
			acc_diff_l_stim = np.hstack([acc_diff_l_stim, acc_avg_fr_diff])
			acc_blockA_l_stim = np.hstack([acc_blockA_l_stim, acc_avg_fr_blockA])
			acc_blockAprime_l_stim = np.hstack([acc_blockAprime_l_stim, acc_avg_fr_blockAprime])


#### Compute values
fr_thres = 0.1  # minimum firing rate to be included in analysis
all_cd_diff_m_sham = np.array([(cd_blockAprime_m_sham[i] - cd_blockA_m_sham[i]) for i in range(len(cd_blockA_m_sham))
							if (cd_blockAprime_m_sham[i] > fr_thres) and (cd_blockA_m_sham[i] > fr_thres)])
all_cd_diff_m_sham_ind = np.array([i for i in range(len(cd_blockA_m_sham))
							if (cd_blockAprime_m_sham[i] > fr_thres) and (cd_blockA_m_sham[i] > fr_thres)])
all_cd_diff_m_sham_norm = all_cd_diff_m_sham/cd_blockA_m_sham[all_cd_diff_m_sham_ind]


all_cd_diff_m_stim = np.array([(cd_blockAprime_m_stim[i] - cd_blockA_m_stim[i]) for i in range(len(cd_blockA_m_stim))
							if (cd_blockAprime_m_stim[i] > fr_thres) and (cd_blockA_m_stim[i] > fr_thres)])
all_cd_diff_m_stim_ind = np.array([i for i in range(len(cd_blockA_m_stim))
							if (cd_blockAprime_m_stim[i] > fr_thres) and (cd_blockA_m_stim[i] > fr_thres)])
all_cd_diff_m_stim_norm = all_cd_diff_m_stim/cd_blockA_m_stim[all_cd_diff_m_stim_ind]


all_acc_diff_m_sham = np.array([(acc_blockAprime_m_sham[i] - acc_blockA_m_sham[i]) for i in range(len(acc_blockA_m_sham))
							if ((acc_blockAprime_m_sham[i] > fr_thres) and (acc_blockA_m_sham[i] > fr_thres))])
all_acc_diff_m_sham_ind = np.array([i for i in range(len(acc_blockA_m_sham))
							if ((acc_blockAprime_m_sham[i] > fr_thres) and (acc_blockA_m_sham[i] > fr_thres))])
all_acc_diff_m_sham_norm = all_acc_diff_m_sham/acc_blockA_m_sham[all_acc_diff_m_sham_ind]


all_acc_diff_m_stim = np.array([(acc_blockAprime_m_stim[i] - acc_blockA_m_stim[i]) for i in range(len(acc_blockA_m_stim))
							if ((acc_blockAprime_m_stim[i] > fr_thres) and (acc_blockA_m_stim[i] > fr_thres))])
all_acc_diff_m_stim_ind = np.array([i for i in range(len(acc_blockA_m_stim))
							if (acc_blockAprime_m_stim[i] > fr_thres) and (acc_blockA_m_stim[i] > fr_thres)])
all_acc_diff_m_stim_norm = all_acc_diff_m_stim/acc_blockA_m_stim[all_acc_diff_m_stim_ind]


all_cd_diff_l_sham = np.array([(cd_blockAprime_l_sham[i] - cd_blockA_l_sham[i]) for i in range(len(cd_blockA_l_sham))
							if (cd_blockAprime_l_sham[i] > fr_thres) and (cd_blockA_l_sham[i] > fr_thres)])
all_cd_diff_l_sham_ind = np.array([i for i in range(len(cd_blockA_l_sham))
							if (cd_blockAprime_l_sham[i] > fr_thres) and (cd_blockA_l_sham[i] > fr_thres)])
all_cd_diff_l_sham_norm = all_cd_diff_l_sham/cd_blockA_l_sham[all_cd_diff_l_sham_ind]


all_cd_diff_l_stim = np.array([(cd_blockAprime_l_stim[i] - cd_blockA_l_stim[i]) for i in range(len(cd_blockA_l_stim))
							if (cd_blockAprime_l_stim[i] > fr_thres) and (cd_blockA_l_stim[i] > fr_thres)])
all_cd_diff_l_stim_ind = np.array([i for i in range(len(cd_blockA_l_stim))
							if (cd_blockAprime_l_stim[i] > fr_thres) and (cd_blockA_l_stim[i] > fr_thres)])
all_cd_diff_l_stim_norm = all_cd_diff_l_stim/cd_blockA_l_stim[all_cd_diff_l_stim_ind]


all_acc_diff_l_sham = np.array([(acc_blockAprime_l_sham[i] - acc_blockA_l_sham[i]) for i in range(len(acc_blockA_l_sham))
							if (acc_blockAprime_l_sham[i] > fr_thres) and (acc_blockA_l_sham[i] > fr_thres)])
all_acc_diff_l_sham_ind = np.array([i for i in range(len(acc_blockA_l_sham))
							if (acc_blockAprime_l_sham[i] > fr_thres) and (acc_blockA_l_sham[i] > fr_thres)])
all_acc_diff_l_sham_norm = all_acc_diff_l_sham/acc_blockA_l_sham[all_acc_diff_l_sham_ind]


all_acc_diff_l_stim = np.array([(acc_blockAprime_l_stim[i] - acc_blockA_l_stim[i]) for i in range(len(acc_blockA_l_stim))
							if (acc_blockAprime_l_stim[i] > fr_thres) and (acc_blockA_l_stim[i] > fr_thres)])
all_acc_diff_l_stim_ind = np.array([i for i in range(len(acc_blockA_l_stim))
							if (acc_blockAprime_l_stim[i] > fr_thres) and (acc_blockA_l_stim[i] > fr_thres)])
all_acc_diff_l_stim_norm = all_acc_diff_l_stim/acc_blockA_l_stim[all_acc_diff_l_stim_ind]


#### Plots

### Add plots of histograms: hist, bins = np.histogram(data,bins=nbins)
bin_range = np.arange(-0.5, 1.7, 0.1)
hist_cd_diff_m_sham_norm, bins_cd_diff_m_sham_norm = np.histogram(all_cd_diff_m_sham_norm, bins = bin_range)
hist_cd_diff_m_stim_norm, bins_cd_diff_m_stim_norm = np.histogram(all_cd_diff_m_stim_norm, bins = bin_range)
hist_acc_diff_m_sham_norm, bins_acc_diff_m_sham_norm = np.histogram(all_acc_diff_m_sham_norm, bins = bin_range)
hist_acc_diff_m_stim_norm, bins_acc_diff_m_stim_norm = np.histogram(all_acc_diff_m_stim_norm, bins = bin_range)
hist_cd_diff_l_sham_norm, bins_cd_diff_l_sham_norm = np.histogram(all_cd_diff_l_sham_norm, bins = bin_range)
hist_cd_diff_l_stim_norm, bins_cd_diff_l_stim_norm = np.histogram(all_cd_diff_l_stim_norm, bins = bin_range)
hist_acc_diff_l_sham_norm, bins_acc_diff_l_sham_norm = np.histogram(all_acc_diff_l_sham_norm, bins = bin_range)
hist_acc_diff_l_stim_norm, bins_acc_diff_l_stim_norm = np.histogram(all_acc_diff_l_stim_norm, bins = bin_range)

chisq_cd_m, p_cd_m = stats.chisquare(hist_cd_diff_m_sham_norm, hist_cd_diff_m_stim_norm)
chisq_acc_m, p_acc_m = stats.chisquare(hist_acc_diff_m_sham_norm, hist_acc_diff_m_stim_norm)
chisq_cd_l, p_cd_l = stats.chisquare(hist_cd_diff_l_sham_norm, hist_cd_diff_l_stim_norm)
chisq_acc_l, p_acc_l = stats.chisquare(hist_acc_diff_l_sham_norm, hist_acc_diff_l_stim_norm)

hist_cd_diff_m_sham_norm = hist_cd_diff_m_sham_norm/(len(all_cd_diff_m_sham_norm)*1.0)
hist_cd_diff_m_stim_norm = hist_cd_diff_m_stim_norm/(len(all_cd_diff_m_stim_norm)*1.0)
hist_acc_diff_m_sham_norm = hist_acc_diff_m_sham_norm/(len(all_acc_diff_m_sham_norm)*1.0)
hist_acc_diff_m_stim_norm = hist_acc_diff_m_stim_norm/(len(all_acc_diff_m_stim_norm)*1.0)
hist_cd_diff_l_sham_norm = hist_cd_diff_l_sham_norm/(len(all_cd_diff_l_sham_norm)*1.0)
hist_cd_diff_l_stim_norm = hist_cd_diff_l_stim_norm/(len(all_cd_diff_l_stim_norm)*1.0)
hist_acc_diff_l_sham_norm = hist_acc_diff_l_sham_norm/(len(all_acc_diff_l_sham_norm)*1.0)
hist_acc_diff_l_stim_norm = hist_acc_diff_l_stim_norm/(len(all_acc_diff_l_stim_norm)*1.0)

width = 0.1

plt.figure()
plt.subplot(221)
plt.bar(bins_cd_diff_m_sham_norm[1:], hist_cd_diff_m_sham_norm, width, color = 'b', label = 'Sham')
plt.ylim(0,0.25)
plt.subplot(222)
plt.bar(bins_cd_diff_m_stim_norm[1:], hist_cd_diff_m_stim_norm, width, color = 'r', label = 'Stim')
plt.ylim(0,0.25)
plt.text(1.0, 0.3, 'p = %0.2f (Chi-square)' % (p_cd_m))
plt.subplot(223)
plt.bar(bins_acc_diff_m_sham_norm[1:], hist_acc_diff_m_sham_norm, width, color = 'b', label = 'Sham')
plt.ylim(0,0.25)
plt.subplot(224)
plt.bar(bins_acc_diff_m_stim_norm[1:], hist_acc_diff_m_stim_norm, width, color = 'r', label = 'Stim')
plt.ylim(0,0.25)
plt.text(1.0, 0.3, 'p = %0.2f (Chi-square)' % (p_acc_m))

plt.figure()
plt.subplot(221)
plt.bar(bins_cd_diff_l_sham_norm[1:], hist_cd_diff_l_sham_norm, width, color = 'b', label = 'Sham')
plt.ylim(0,0.20)
plt.subplot(222)
plt.bar(bins_cd_diff_l_stim_norm[1:], hist_cd_diff_l_stim_norm, width, color = 'r', label = 'Stim')
plt.text(1.0, 0.3, 'p = %0.2f (Chi-square)' % (p_cd_l))
plt.ylim(0,0.20)
plt.subplot(223)
plt.bar(bins_acc_diff_l_sham_norm[1:], hist_acc_diff_l_sham_norm, width, color = 'b', label = 'Sham')
plt.ylim(0,0.20)
plt.subplot(224)
plt.bar(bins_acc_diff_l_stim_norm[1:], hist_acc_diff_l_stim_norm, width, color = 'r', label = 'Stim')
plt.text(1.0, 0.3, 'p = %0.2f (Chi-square)' % (p_acc_l))
plt.ylim(0,0.20)

plt.show()


ind_sham = [0, 2]
ind_stim = [1, 3]
m_means_sham = [np.nanmean(all_cd_diff_m_sham_norm), 
			 	np.nanmean(all_acc_diff_m_sham_norm)]
m_sems_sham = [np.nanstd(all_cd_diff_m_sham_norm)/np.sqrt(len(all_cd_diff_m_sham_norm)), 
			 	np.nanstd(all_acc_diff_m_sham_norm)/np.sqrt(len(all_acc_diff_m_sham_norm))]
m_means_stim = [np.nanmean(all_cd_diff_m_stim_norm), 
			 	np.nanmean(all_acc_diff_m_stim_norm)]
m_sems_stim = [np.nanstd(all_cd_diff_m_stim_norm)/np.sqrt(len(all_cd_diff_m_stim_norm)), 
			  	np.nanstd(all_acc_diff_m_stim_norm)/np.sqrt(len(all_acc_diff_m_stim_norm))]
l_means_sham = [np.nanmean(all_cd_diff_l_sham_norm), 
			 	np.nanmean(all_acc_diff_l_sham_norm)]
l_sems_sham = [np.nanstd(all_cd_diff_l_sham_norm)/np.sqrt(len(all_cd_diff_l_sham_norm)), 
			 	np.nanstd(all_acc_diff_l_sham_norm)/np.sqrt(len(all_acc_diff_l_sham_norm))]
l_means_stim = [np.nanmean(all_cd_diff_l_stim_norm), 
			 	np.nanmean(all_acc_diff_l_stim_norm)]
l_sems_stim = [np.nanstd(all_cd_diff_l_stim_norm)/np.sqrt(len(all_cd_diff_l_stim_norm)), 
			  	np.nanstd(all_acc_diff_l_stim_norm)/np.sqrt(len(all_acc_diff_l_stim_norm))]

t, p_cd_m = stats.ttest_ind(all_cd_diff_m_sham_norm, all_cd_diff_m_stim_norm)
t, p_acc_m = stats.ttest_ind(all_acc_diff_m_sham_norm, all_acc_diff_m_stim_norm)
t, p_cd_l = stats.ttest_ind(all_cd_diff_l_sham_norm, all_cd_diff_l_stim_norm)
t, p_acc_l = stats.ttest_ind(all_acc_diff_l_sham_norm, all_acc_diff_l_stim_norm)

width = 0.35
plt.figure()
plt.subplot(121)
plt.bar(ind_sham, m_means_sham, width, yerr=m_sems_sham, color = 'b', label='Sham')
plt.bar(ind_stim, m_means_stim, width, yerr=m_sems_stim, color = 'r', label='Stim')
plt.legend()
plt.xticks([0.5, 2.5], ['Cd', 'ACC'])
plt.ylabel("Normalized change in Firing Rate: (A' - A)/A")
plt.title('Mario - Change in Baseline Firing')
plt.subplot(122)
plt.bar(ind_sham, l_means_sham, width, yerr=l_sems_sham, color = 'b', label = 'Sham')
plt.bar(ind_stim, l_means_stim, width, yerr=l_sems_stim, color = 'r', label = 'Stim')
plt.legend()
plt.xticks([0.5, 2.5], ['Cd', 'ACC'])
plt.ylabel("Normalized change in Firing Rate: (A' - A)/A")
plt.title('Luigi - Change in Baseline Firing')
plt.show()

data_m = [all_cd_diff_m_sham_norm, all_cd_diff_m_stim_norm, all_acc_diff_m_sham_norm, all_acc_diff_m_stim_norm]
data_l = [all_cd_diff_l_sham_norm, all_cd_diff_l_stim_norm, all_acc_diff_l_sham_norm, all_acc_diff_l_stim_norm]

plt.figure()
plt.subplot(121)
plt.boxplot(data_m, 0, '')
plt.title('Mario')
plt.subplot(122)
plt.boxplot(data_l, 0, '')
plt.title('Luigi')

#### Add box-and-whisker plot amenities and mann-whitney u-test/kruskal-wallis test

#### Two way ANOVA:
#### One dependent var: change in firing rate
#### Two independent vars: (1) recording location (cd or acc), (2) stim condition (stim or sham)