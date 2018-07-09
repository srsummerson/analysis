import numpy as np 
import scipy as sp
import pandas as pd
import matplotlib as mpl
import tables
import sys
import os.path

from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile
from DecisionMakingBehavior import MultiTargetTask_FiringRates_DifferenceBetweenBlocks

# write code to compute average firing rate changes (normalized and not normalized) for Mario and Luigi
mario_cd_channels = [1, 3, 4, 17, 18, 20, 35, 37, 38, 40, 41, 51, 53, 54, 56, 57, 63, 64, 65, 67, 70, 72, 73, 75, 81, 83, 86, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
mario_acc_channels = [5, 6, 19, 22, 30, 39, 42, 43, 55, 58, 59, 69, 74, 77, 85, 90, 91, 102, 105, 121, 128]
luigi_cd_channels = np.arange(1,65)
luigi_acc_channels = np.arange(65,129)

# Code for Mario for Sham days

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

start_session_num = 3
last_session_num = 5

for k in range(start_session_num, last_session_num+1):
	curr_session = k
	len_curr_session = len_sessions[k-1]
	row_num = np.ravel(np.argwhere(var1==curr_session))
	if var4[row_num[0]]==1:
	
		hdf_filename = np.ravel(var2[row_num])
		filename = [np.ravel(var3[row_num])[l][:-7] for l in range(len(row_num))]
		block_num = [np.ravel(var3[row_num])[l][-1] for l in range(len(row_num))]

		print hdf_filename
		print filename

		hdf_files = [(data_folder + hdf_file) for hdf_file in hdf_filename]
		syncHDF_files = [(data_folder + file + '_b' + str(block_num[m]) +'_syncHDF.mat') for m,file in enumerate(filename)]
		spike_files = [[(data_folder + file + '_Block-' + str(block_num[m]) + '_eNe1_Offline.csv'), (data_folder + file + '_Block-' + str(block_num[m]) + '_eNe2_Offline.csv')] for m,file in enumerate(filename)]
		print spike_files

		num_spike_files = len(spike_files)
		for i in range(num_spike_files):
			if os.path.isfile(spike_files[i][0])==False:
				spike_files[i][0] = ''
			if os.path.isfile(spike_files[i][1])==False:
				spike_files[i][1] = ''

		num_targets = 3

		
		print hdf_files
		print syncHDF_files
		print spike_files
		

		# Find good channels that are either Cd or ACC
		spike1 = OfflineSorted_CSVFile(spike_files[0][0])
		spike2 = OfflineSorted_CSVFile(spike_files[0][1])
		all_channels = np.ravel([spike1.good_channels, spike2.good_channels])
		print all_channels
		cd_channels = [chan for chan in all_channels if chan in mario_cd_channels]
		acc_channels = [chan for chan in all_channels if chan in mario_acc_channels]

		print cd_channels
		print acc_channels
		print '/n'

		"""
		# Build list of firing rate differences
		cd_fr_diff = []
		cd_fr_blockA = []
		cd_fr_blockAprime = []
		for m,chann in enumerate(cd_channels):
			avg_fr_diff, avg_fr_blockA, avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks(hdf_files, syncHDF_files, spike_files, num_targets, chann)
			cd_fr_diff += avg_fr_diff
			cd_fr_blockA += avg_fr_blockA
			cd_fr_blockAprime += avg_fr_blockAprime

		acc_fr_diff = []
		acc_fr_blockA = []
		acc_fr_blockAprime = []
		for m,chann in enumerate(cd_channels):
			avg_fr_diff, avg_fr_blockA, avg_fr_blockAprime = MultiTargetTask_FiringRates_DifferenceBetweenBlocks(hdf_files, syncHDF_files, spike_files, num_targets, chann)
			cd_fr_diff += avg_fr_diff
			cd_fr_blockA += avg_fr_blockA
			cd_fr_blockAprime += avg_fr_blockAprime
		"""