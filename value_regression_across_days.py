from DecisionMakingBehavior import ThreeTargetTask_RegressedFiringRatesWithValue_PictureOnset
from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile
import numpy as np
from os import listdir
from scipy import io 
import scipy as sp
import matplotlib.pyplot as plt

# Define units of interest
cd_units = [1, 3, 4, 17, 18, 20, 40, 41, 54, 56, 57, 63, 64, 72, 75, 81, 83, 88, 89, 96, 100, 112, 114, 126, 130, 140, 143, 146, 156, 157, 159]
acc_units = [5, 6, 19, 22, 30, 39, 42, 43, 55, 58, 59, 69, 74, 77, 85, 90, 91, 102, 105, 121, 128]
all_units = np.append(cd_units, acc_units)
# List data

dir = "C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/spike_data/"


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

# Define code parameters
t_before = 0.
t_after = 0.4
smoothed = 1

# Loop through sessions to compute regressions per session
for j in range(len(hdf_list))[15:16]:
	# Pull out the relevant session's data
	hdf_files = hdf_list[j]
	syncHDF_files = syncHDF_list[j]
	spike_files = spike_list[j]
	print "Working on file:", hdf_files[0]

	# Find which Cd and ACC channels have good spiking data
	if (spike_files[0] != ['']):
		spike1 = OfflineSorted_CSVFile(spike_files[0][0])
		fr1 = spike1.get_avg_firing_rates(all_units)
		units1 = np.array([unit for unit in all_units if fr1[unit][0]>0])
		spike2 = OfflineSorted_CSVFile(spike_files[0][1])
		fr2 = spike2.get_avg_firing_rates(all_units)
		units2 = np.array([unit for unit in all_units if fr2[unit][0]>0])
		good_units = np.append(units1, units2)
	elif (spike_files[1] != ['']):
		spike1 = OfflineSorted_CSVFile(spike_files[1][0])
		fr1 = spike1.get_avg_firing_rates(all_units)
		units1 = np.array([unit for unit in all_units if fr1[unit][0]>0])
		spike2 = OfflineSorted_CSVFile(spike_files[1][1])
		fr2 = spike2.get_avg_firing_rates(all_units)
		units2 = np.array([unit for unit in all_units if fr2[unit][0]>0])
		good_units = np.append(units1, units2)
	else:
		good_units = np.array([])

	# Perform analysis for all Cd and ACC channels with spiking data
	for k,item in enumerate(good_units):
		output = ThreeTargetTask_RegressedFiringRatesWithValue_PictureOnset(hdf_files, syncHDF_files, spike_files, item, t_before, t_after, smoothed)


'''
Average across all files
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

for filen in filenames:
	data = dict()
	sp.io.loadmat(spike_dir + filen, data)
	chan_ind = filen.index('Channel')
	unit_ind = filen.index('Unit')
	channel_num = float(filen[chan_ind + 8:unit_ind - 3])

	sync_name = filen[:chan_ind-3] + '_syncHDF.mat'
	if sync_name in syncHDF_list_stim:
		stim_file = 1
	else:
		stim_file = 0

	
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
			if val_check[2]==1 or val_check[5]==1:
				HV_enc = 1

			Q_late = np.ravel(data['Q_mid_late'])
			FR_late = np.ravel(data['FR_late'])
			Q_early = np.ravel(data['Q_mid_early'])
			FR_early = np.ravel(data['FR_early'])
			if ((LV_enc==0) and (MV_enc==1) and (HV_enc==0)):
				count_MV_blocksAB_cd += 1
				Q_late_mv_cd = np.append(Q_late_mv_cd, Q_late)
				Q_early_mv_cd = np.append(Q_early_mv_cd, Q_early)
				FR_late_mv_cd = np.append(FR_late_mv_cd, FR_late)
				FR_early_mv_cd = np.append(FR_early_mv_cd, FR_early)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==0)):
				count_LV_blocksAB_cd += 1
				Q_late_other_cd = np.append(Q_late_other_cd, Q_late)
				Q_early_other_cd = np.append(Q_early_other_cd, Q_early)
				FR_late_other_cd = np.append(FR_late_other_cd, FR_late)
				FR_early_other_cd = np.append(FR_early_other_cd, FR_early)
			elif ((LV_enc==0) and (MV_enc==0) and (HV_enc==1)):
				count_HV_blocksAB_cd += 1
				Q_late_other_cd = np.append(Q_late_other_cd, Q_late)
				Q_early_other_cd = np.append(Q_early_other_cd, Q_early)
				FR_late_other_cd = np.append(FR_late_other_cd, FR_late)
				FR_early_other_cd = np.append(FR_early_other_cd, FR_early)
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==0)):
				count_LVMV_blocksAB_cd += 1
				Q_late_mv_cd = np.append(Q_late_mv_cd, Q_late)
				Q_early_mv_cd = np.append(Q_early_mv_cd, Q_early)
				FR_late_mv_cd = np.append(FR_late_mv_cd, FR_late)
				FR_early_mv_cd = np.append(FR_early_mv_cd, FR_early)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==1)):
				count_LVHV_blocksAB_cd += 1
				Q_late_other_cd = np.append(Q_late_other_cd, Q_late)
				Q_early_other_cd = np.append(Q_early_other_cd, Q_early)
				FR_late_other_cd = np.append(FR_late_other_cd, FR_late)
				FR_early_other_cd = np.append(FR_early_other_cd, FR_early)
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==1)):
				count_MVHV_blocksAB_cd += 1
				Q_late_mv_cd = np.append(Q_late_mv_cd, Q_late)
				Q_early_mv_cd = np.append(Q_early_mv_cd, Q_early)
				FR_late_mv_cd = np.append(FR_late_mv_cd, FR_late)
				FR_early_mv_cd = np.append(FR_early_mv_cd, FR_early)
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==1)):
				count_LVMVHV_blocksAB_cd += 1
				Q_late_mv_cd = np.append(Q_late_mv_cd, Q_late)
				Q_early_mv_cd = np.append(Q_early_mv_cd, Q_early)
				FR_late_mv_cd = np.append(FR_late_mv_cd, FR_late)
				FR_early_mv_cd = np.append(FR_early_mv_cd, FR_early)
			else:
				no_val_enc = 1
				Q_late_other_cd = np.append(Q_late_other_cd, Q_late)
				Q_early_other_cd = np.append(Q_early_other_cd, Q_early)
				FR_late_other_cd = np.append(FR_late_other_cd, FR_late)
				FR_early_other_cd = np.append(FR_early_other_cd, FR_early)

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
			if val_check[2]==1 or val_check[5]==1:
				HV_enc = 1

			Q_late = np.ravel(data['Q_mid_late'])
			FR_late = np.ravel(data['FR_late'])
			Q_early = np.ravel(data['Q_mid_early'])
			FR_early = np.ravel(data['FR_early'])
			if ((LV_enc==0) and (MV_enc==1) and (HV_enc==0)):
				count_MV_blocksAB_cd += 1
				Q_late_mv_acc = np.append(Q_late_mv_acc, Q_late)
				Q_early_mv_acc = np.append(Q_early_mv_acc, Q_early)
				FR_late_mv_acc = np.append(FR_late_mv_acc, FR_late)
				FR_early_mv_acc = np.append(FR_early_mv_acc, FR_early)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==0)):
				count_LV_blocksAB_cd += 1
				Q_late_other_acc = np.append(Q_late_other_acc, Q_late)
				Q_early_other_acc = np.append(Q_early_other_acc, Q_early)
				FR_late_other_acc = np.append(FR_late_other_acc, FR_late)
				FR_early_other_acc = np.append(FR_early_other_acc, FR_early)
			elif ((LV_enc==0) and (MV_enc==0) and (HV_enc==1)):
				count_HV_blocksAB_cd += 1
				Q_late_other_acc = np.append(Q_late_other_acc, Q_late)
				Q_early_other_acc = np.append(Q_early_other_acc, Q_early)
				FR_late_other_acc = np.append(FR_late_other_acc, FR_late)
				FR_early_other_acc = np.append(FR_early_other_acc, FR_early)
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==0)):
				count_LVMV_blocksAB_cd += 1
				Q_late_mv_acc = np.append(Q_late_mv_acc, Q_late)
				Q_early_mv_acc = np.append(Q_early_mv_acc, Q_early)
				FR_late_mv_acc = np.append(FR_late_mv_acc, FR_late)
				FR_early_mv_acc = np.append(FR_early_mv_acc, FR_early)
			elif ((LV_enc==1) and (MV_enc==0) and (HV_enc==1)):
				count_LVHV_blocksAB_cd += 1
				Q_late_other_acc = np.append(Q_late_other_acc, Q_late)
				Q_early_other_acc = np.append(Q_early_other_acc, Q_early)
				FR_late_other_acc = np.append(FR_late_other_acc, FR_late)
				FR_early_other_acc = np.append(FR_early_other_acc, FR_early)
			elif ((LV_enc==0) and (MV_enc==1) and (HV_enc==1)):
				count_MVHV_blocksAB_cd += 1
				Q_late_mv_acc = np.append(Q_late_mv_acc, Q_late)
				Q_early_mv_acc = np.append(Q_early_mv_acc, Q_early)
				FR_late_mv_acc = np.append(FR_late_mv_acc, FR_late)
				FR_early_mv_acc = np.append(FR_early_mv_acc, FR_early)
			elif ((LV_enc==1) and (MV_enc==1) and (HV_enc==1)):
				count_LVMVHV_blocksAB_cd += 1
				Q_late_mv_acc = np.append(Q_late_mv_acc, Q_late)
				Q_early_mv_acc = np.append(Q_early_mv_acc, Q_early)
				FR_late_mv_acc = np.append(FR_late_mv_acc, FR_late)
				FR_early_mv_acc = np.append(FR_early_mv_acc, FR_early)
			else:
				no_val_enc = 1
				Q_late_other_acc = np.append(Q_late_other_acc, Q_late)
				Q_early_other_acc = np.append(Q_early_other_acc, Q_early)
				FR_late_other_acc = np.append(FR_late_other_acc, FR_late)
				FR_early_other_acc = np.append(FR_early_other_acc, FR_early)

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
plt.title('Cd - Block A')
plt.subplot(2,2,2)
plt.pie(fracs_blocksAB_cd, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('Cd - Blocks A and B')
plt.subplot(2,2,3)
plt.pie(fracs_blockA_acc, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('ACC - Block A')
plt.subplot(2,2,4)
plt.pie(fracs_blocksAB_acc, labels = labels, autopct='%.2f%%', shadow = False)
plt.title('ACC - Blocks A and B')
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

all_Q = Q_late_mv_cd + Q_early_mv_cd + Q_late_other_cd + Q_early_other_cd + \
		Q_late_mv_acc + Q_early_mv_acc + Q_late_other_acc + Q_early_other_acc
min_Q = np.min(all_Q)
max_Q = np.max(all_Q)
Q_bins = np.arange(min_Q, max_Q + (max_Q - min_Q)/10., 10)
# bin Qs and sort FRs accordingly for all conditions

