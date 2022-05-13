from DecisionMakingBehavior import TwoTargetTask_RegressFiringRates_Value, TwoTargetTask_FiringRates_OverTime
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
import seaborn as sns
import time
import xlsxwriter

t = time.time()
	


# Define units of interest
cd_units = list(range(1,65))
acc_units = list(range(65,129))
all_units = np.append(cd_units, acc_units)
dir_figs = "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Paper/Figures/"	
# List data

dir = "C:/Users/ss45436/Box Sync/UC Berkeley/Cd Stim/Neural Correlates/Luigi/spike_data/"
tdt_data_dir = "E:/UC Berkeley/Caudate Stim/Luigi/"

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
			  [[dir + 'Luigi20171005_Block-1_eNe1_Offline.csv', '']], \
			  [[dir + 'Luigi20171017_Block-1_eNe1_Offline.csv', dir + 'Luigi20171017_Block-1_eNe2_Offline.csv'], [dir + 'Luigi20171017_Block-2_eNe1_Offline.csv', dir + 'Luigi20171017_Block-2_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171031_Block-1_eNe1_Offline.csv', dir + 'Luigi20171031_Block-1_eNe2_Offline.csv']], \
			  [[dir + 'Luigi20171108_Block-1_eNe1_Offline.csv', dir + 'Luigi20171108_Block-1_eNe2_Offline.csv']], \
			  ]
spike_list = spike_list_sham + spike_list_stim

###########################################################################
# Below code looks at the fraction of neurons encoding value over time for 
# each day, and then takes the average across both stim days and sham days
# (separately). This tells us an average fraction across days, but this 
# potentially differs from the typical behavior since a different number of
# neurons are recorded each day. We should also consider just counting the
# total fraction of neurons that encode value over time for stim and sham 
# days (separately).
###########################################################################

t_before = 0.5
t_after = 2
t_resolution = 0.2
t_overlap = 0.150
smoothed = 1
align_to = 1
trial_first = 0
trial_last = 100

for i in range(len(hdf_list_sham)):
	print('Processing file ',i)
	hdf_files = hdf_list_sham[i]
	syncHDF_files = syncHDF_list_sham[i]
	spike_files = spike_list_sham[i]
	
	# Cd units and ACC units separately
	time_values_cd, frac_Q_low_cd, frac_Q_high_cd, Q_low_sig_cd, Q_high_sig_cd, beta_Q_low_sig_cd, beta_Q_high_sig_cd = TwoTargetTask_RegressFiringRates_Value(hdf_files, syncHDF_files, spike_files, cd_units, 'Cd', t_before, t_after, t_resolution, t_overlap, smoothed, align_to, trial_first, trial_last)
	time_values_acc, frac_Q_low_acc, frac_Q_high_acc, Q_low_sig_acc, Q_high_sig_acc, beta_Q_low_sig_acc, beta_Q_high_sig_acc = TwoTargetTask_RegressFiringRates_Value(hdf_files, syncHDF_files, spike_files, acc_units, 'ACC', t_before, t_after, t_resolution, t_overlap, smoothed, align_to, trial_first, trial_last)

	if i == 0:
		frac_low_sham_cd = frac_Q_low_cd
		frac_high_sham_cd = frac_Q_high_cd
		Q_low_sig_sham_cd = Q_low_sig_cd	
		Q_high_sig_sham_cd = Q_high_sig_cd

		frac_low_sham_acc = frac_Q_low_acc
		frac_high_sham_acc = frac_Q_high_acc
		Q_low_sig_sham_acc = Q_low_sig_acc	
		Q_high_sig_sham_acc = Q_high_sig_acc		
	else:
		frac_low_sham_cd = np.vstack([frac_low_sham_cd, frac_Q_low_cd])
		frac_high_sham_cd = np.vstack([frac_high_sham_cd, frac_Q_high_cd])
		Q_low_sig_sham_cd = np.vstack([Q_low_sig_sham_cd, Q_low_sig_cd])
		Q_high_sig_sham_cd	= np.vstack([Q_high_sig_sham_cd, Q_high_sig_cd])

		frac_low_sham_acc = np.vstack([frac_low_sham_acc, frac_Q_low_acc])
		frac_high_sham_acc = np.vstack([frac_high_sham_acc, frac_Q_high_acc])
		Q_low_sig_sham_acc = np.vstack([Q_low_sig_sham_acc, Q_low_sig_acc])
		Q_high_sig_sham_acc	= np.vstack([Q_high_sig_sham_acc, Q_high_sig_acc])

all_frac_Q_low_sham_cd = np.sum(Q_low_sig_sham_cd, axis = 0)/Q_low_sig_sham_cd.shape[0]
all_frac_Q_high_sham_cd = np.sum(Q_high_sig_sham_cd, axis = 0)/Q_high_sig_sham_cd.shape[0]

all_frac_Q_low_sham_acc = np.sum(Q_low_sig_sham_acc, axis = 0)/Q_low_sig_sham_acc.shape[0]
all_frac_Q_high_sham_acc = np.sum(Q_high_sig_sham_acc, axis = 0)/Q_high_sig_sham_acc.shape[0]

frac_avg_low_sham_cd = np.nanmean(frac_low_sham_cd, axis = 0)
frac_sem_low_sham_cd = np.nanstd(frac_low_sham_cd	, axis = 0)/np.sqrt(frac_low_sham_cd.shape[0])
frac_avg_high_sham_cd = np.nanmean(frac_high_sham_cd, axis = 0)
frac_sem_high_sham_cd = np.nanstd(frac_high_sham_cd, axis = 0)/np.sqrt(frac_high_sham_cd.shape[0])

frac_avg_low_sham_acc = np.nanmean(frac_low_sham_acc, axis = 0)
frac_sem_low_sham_acc = np.nanstd(frac_low_sham_acc	, axis = 0)/np.sqrt(frac_low_sham_acc.shape[0])
frac_avg_high_sham_acc = np.nanmean(frac_high_sham_acc, axis = 0)
frac_sem_high_sham_acc = np.nanstd(frac_high_sham_acc, axis = 0)/np.sqrt(frac_high_sham_acc.shape[0])

# Adding in averaging of beta, separating positive and negative vaues - STOPPED HERE
#beta_Q_low_sig_pos_cd = (beta_Q_low_sig_cd > 0)
#mean_beta_low_cd = np.nanmean(beta_Q_low_sig_cd,axis = 0)
#sem_beta_low_cd = np.nanstd(beta_Q_low_sig_cd, axis = 0)/np.sqrt(sem_denom_low)


for i in range(len(hdf_list_stim)):
	print('Processing file ',i)
	hdf_files = hdf_list_stim[i]
	syncHDF_files = syncHDF_list_stim[i]
	spike_files = spike_list_stim[i]

	# Cd units and ACC units separately
	time_values_cd, frac_Q_low_cd, frac_Q_high_cd, Q_low_sig_cd, Q_high_sig_cd, beta_Q_low_sig_cd, beta_Q_high_sig_cd = TwoTargetTask_RegressFiringRates_Value(hdf_files, syncHDF_files, spike_files, cd_units, 'Cd',t_before, t_after, t_resolution, t_overlap, smoothed, align_to, trial_first, trial_last)
	time_values_acc, frac_Q_low_acc, frac_Q_high_acc, Q_low_sig_acc, Q_high_sig_acc, beta_Q_low_sig_acc, beta_Q_high_sig_acc = TwoTargetTask_RegressFiringRates_Value(hdf_files, syncHDF_files, spike_files, acc_units, 'ACC', t_before, t_after, t_resolution, t_overlap, smoothed, align_to, trial_first, trial_last)

	if i == 0:
		frac_low_stim_cd = frac_Q_low_cd
		frac_high_stim_cd = frac_Q_high_cd
		Q_low_sig_stim_cd = Q_low_sig_cd	
		Q_high_sig_stim_cd = Q_high_sig_cd

		frac_low_stim_acc = frac_Q_low_acc
		frac_high_stim_acc = frac_Q_high_acc
		Q_low_sig_stim_acc = Q_low_sig_acc	
		Q_high_sig_stim_acc = Q_high_sig_acc
	else:
		frac_low_stim_cd = np.vstack([frac_low_stim_cd, frac_Q_low_cd])
		frac_high_stim_cd = np.vstack([frac_high_stim_cd, frac_Q_high_cd])
		Q_low_sig_stim_cd = np.vstack([Q_low_sig_stim_cd, Q_low_sig_cd])
		Q_high_sig_stim_cd	= np.vstack([Q_high_sig_stim_cd, Q_high_sig_cd])

		frac_low_stim_acc = np.vstack([frac_low_stim_acc, frac_Q_low_acc])
		frac_high_stim_acc = np.vstack([frac_high_stim_acc, frac_Q_high_acc])
		Q_low_sig_stim_acc = np.vstack([Q_low_sig_stim_acc, Q_low_sig_acc])
		Q_high_sig_stim_acc	= np.vstack([Q_high_sig_stim_acc, Q_high_sig_acc])


all_frac_Q_low_stim_cd = np.sum(Q_low_sig_stim_cd, axis = 0)/Q_low_sig_stim_cd.shape[0]
all_frac_Q_high_stim_cd = np.sum(Q_high_sig_stim_cd, axis = 0)/Q_high_sig_stim_cd.shape[0]

all_frac_Q_low_stim_acc = np.sum(Q_low_sig_stim_acc, axis = 0)/Q_low_sig_stim_acc.shape[0]
all_frac_Q_high_stim_acc = np.sum(Q_high_sig_stim_acc, axis = 0)/Q_high_sig_stim_acc.shape[0]

frac_avg_low_stim_cd = np.nanmean(frac_low_stim_cd, axis = 0)
frac_sem_low_stim_cd = np.nanstd(frac_low_stim_cd	, axis = 0)/np.sqrt(frac_low_stim_cd.shape[0])
frac_avg_high_stim_cd = np.nanmean(frac_high_stim_cd, axis = 0)
frac_sem_high_stim_cd = np.nanstd(frac_high_stim_cd, axis = 0)/np.sqrt(frac_high_stim_cd.shape[0])

frac_avg_low_stim_acc = np.nanmean(frac_low_stim_acc, axis = 0)
frac_sem_low_stim_acc = np.nanstd(frac_low_stim_acc	, axis = 0)/np.sqrt(frac_low_stim_acc.shape[0])
frac_avg_high_stim_acc = np.nanmean(frac_high_stim_acc, axis = 0)
frac_sem_high_stim_acc = np.nanstd(frac_high_stim_acc, axis = 0)/np.sqrt(frac_high_stim_acc.shape[0])

plt.figure()
ax = plt.subplot(121)
plt.plot(time_values_cd, frac_avg_low_sham_cd	, 'r', label = 'LV')
plt.fill_between(time_values_cd,frac_avg_low_sham_cd - frac_sem_low_sham_cd, frac_avg_low_sham_cd + frac_sem_low_sham_cd, facecolor = 'r', alpha = 0.25)
plt.plot(time_values_cd, frac_avg_high_sham_cd, 'b', label = 'HV')
plt.fill_between(time_values_cd,frac_avg_high_sham_cd - frac_sem_high_sham_cd, frac_avg_high_sham_cd + frac_sem_high_sham_cd, facecolor = 'b', alpha = 0.25)
plt.xlabel('Time from Center Hold (s) ')
plt.ylabel('Fraction of Cd Neurons')
plt.title('Cd Encoding of Value over Time - Avg over Sham Days')
plt.ylim((0,0.6))
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.legend()
ax2 = plt.subplot(122)
plt.plot(time_values_cd, frac_avg_low_stim_cd	, 'r', label = 'LV')
plt.fill_between(time_values_cd,frac_avg_low_stim_cd - frac_sem_low_stim_cd, frac_avg_low_stim_cd + frac_sem_low_stim_cd, facecolor = 'r', alpha = 0.25)
plt.plot(time_values_cd, frac_avg_high_stim_cd, 'b', label = 'HV')
plt.fill_between(time_values_cd,frac_avg_high_stim_cd - frac_sem_high_stim_cd, frac_avg_high_stim_cd + frac_sem_high_stim_cd, facecolor = 'b', alpha = 0.25)
plt.xlabel('Time from Center Hold (s) ')
plt.ylabel('Fraction of Cd Neurons')
plt.title('Cd Encoding of Value over Time - Avg Stim Days')
plt.ylim((0,0.6))
ax2.get_yaxis().set_tick_params(direction='out')
ax2.get_xaxis().set_tick_params(direction='out')
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
plt.savefig(dir_figs + 'Luigi_AverageOverDays_FractionCdEncodingValOverTime_trials%i-%i.png' % (trial_first, trial_last))
plt.savefig(dir_figs + 'Luigi_AverageOverDays_FractionCdEncodingValOverTime_trials%i-%i.svg' % (trial_first, trial_last))

plt.figure()
ax = plt.subplot(121)
plt.plot(time_values_acc, frac_avg_low_sham_acc	, 'r', label = 'LV')
plt.fill_between(time_values_acc,frac_avg_low_sham_acc - frac_sem_low_sham_acc, frac_avg_low_sham_acc + frac_sem_low_sham_acc, facecolor = 'r', alpha = 0.25)
plt.plot(time_values_acc, frac_avg_high_sham_acc, 'b', label = 'HV')
plt.fill_between(time_values_acc,frac_avg_high_sham_acc - frac_sem_high_sham_acc, frac_avg_high_sham_acc + frac_sem_high_sham_acc, facecolor = 'b', alpha = 0.25)
plt.xlabel('Time from Center Hold (s) ')
plt.ylabel('Fraction of ACC Neurons')
plt.title('ACC Encoding of Value over Time - Avg over Sham Days')
plt.ylim((0,0.6))
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.legend()
ax2 = plt.subplot(122)
plt.plot(time_values_acc, frac_avg_low_stim_acc	, 'r', label = 'LV')
plt.fill_between(time_values_acc,frac_avg_low_stim_acc - frac_sem_low_stim_acc, frac_avg_low_stim_acc + frac_sem_low_stim_acc, facecolor = 'r', alpha = 0.25)
plt.plot(time_values_acc, frac_avg_high_stim_acc, 'b', label = 'HV')
plt.fill_between(time_values_acc,frac_avg_high_stim_acc - frac_sem_high_stim_acc, frac_avg_high_stim_acc + frac_sem_high_stim_acc, facecolor = 'b', alpha = 0.25)
plt.xlabel('Time from Center Hold (s) ')
plt.ylabel('Fraction of ACC Neurons')
plt.title('ACC Encoding of Value over Time - Avg Stim Days')
plt.ylim((0,0.6))
ax2.get_yaxis().set_tick_params(direction='out')
ax2.get_xaxis().set_tick_params(direction='out')
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
plt.savefig(dir_figs + 'Luigi_AverageOverDays_FractionACCEncodingValOverTime_trials%i-%i.png' % (trial_first, trial_last))
plt.savefig(dir_figs + 'Luigi_AverageOverDays_FractionACCEncodingValOverTime_trials%i-%i.svg' % (trial_first, trial_last))

###STOPPED HERE: Y-AXIS LOOKS WEIRD FOR SOME REASON
plt.figure()
ax3 = plt.subplot(111)
plt.plot(time_values_cd, all_frac_Q_low_sham_cd, 'r', label = 'LV - sham')
plt.plot(time_values_cd, all_frac_Q_high_sham_cd, 'b', label = 'HV - sham')
plt.plot(time_values_cd, all_frac_Q_low_stim_cd, 'm', label = 'LV - stim')
plt.plot(time_values_cd, all_frac_Q_high_stim_cd, 'k', label = 'HV - stim')
plt.xlabel('Time from Center Hold (s) ')
plt.ylabel('Fraction of All Cd Neurons')
plt.title('Cd Encoding of Value over Time - All Days Combined (not averaged)')
plt.ylim((0,0.6))
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.legend()
plt.savefig(dir_figs + 'Luigi_AllDaysCombined_FractionCdEncodingValOverTime_trials%i-%i.png' % (trial_first, trial_last))
plt.savefig(dir_figs + 'Luigi_AllDaysCombined_FractionCdEncodingValOverTime_trials%i-%i.svg' % (trial_first, trial_last))
plt.clf()

plt.figure()
ax3 = plt.subplot(111)
plt.plot(time_values_acc, all_frac_Q_low_sham_acc, 'r', label = 'LV - sham')
plt.plot(time_values_acc, all_frac_Q_high_sham_acc, 'b', label = 'HV - sham')
plt.plot(time_values_acc, all_frac_Q_low_stim_acc, 'm', label = 'LV - stim')
plt.plot(time_values_acc, all_frac_Q_high_stim_acc, 'k', label = 'HV - stim')
plt.xlabel('Time from Center Hold (s) ')
plt.ylabel('Fraction of All ACC Neurons')
plt.title('ACC Encoding of Value over Time - All Days Combined (not averaged)')
plt.ylim((0,0.6))
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.legend()
plt.savefig(dir_figs + 'Luigi_AllDaysCombined_FractionACCEncodingValOverTime_trials%i-%i.png' % (trial_first, trial_last))
plt.savefig(dir_figs + 'Luigi_AllDaysCombined_FractionACCEncodingValOverTime_trials%i-%i.svg' % (trial_first, trial_last))
plt.clf()

workbook = xlsxwriter.Workbook('Luigi_AverageOverDays_FractionCdEncodingValOverTime_trials%i-%i.xlsx' % (trial_first, trial_last))
sheet1 = workbook.add_worksheet()
sheet1.write(0,0, 'Time Values')
sheet1.write(0,1, 'Fraction of low-value coding neurons - sham')
sheet1.write(0,2, 'Fraction of low-value coding neurons - stim')
sheet1.write(0,3, 'Fraction of high-value coding neurons - sham')
sheet1.write(0,4, 'Fraction of high-value coding neurons - stim')
sheet1.write_column(1,0, time_values_cd)
sheet1.write_column(1,1, all_frac_Q_low_sham_cd)
sheet1.write_column(1,2, all_frac_Q_low_stim_cd)
sheet1.write_column(1,3, all_frac_Q_high_sham_cd)
sheet1.write_column(1,4, all_frac_Q_high_stim_cd)
workbook.close()

workbook = xlsxwriter.Workbook('Luigi_AverageOverDays_FractionACCEncodingValOverTime_trials%i-%i.xlsx' % (trial_first, trial_last))
sheet1 = workbook.add_worksheet()
sheet1.write(0,0, 'Time Values')
sheet1.write(0,1, 'Fraction of low-value coding neurons - sham')
sheet1.write(0,2, 'Fraction of low-value coding neurons - stim')
sheet1.write(0,3, 'Fraction of high-value coding neurons - sham')
sheet1.write(0,4, 'Fraction of high-value coding neurons - stim')
sheet1.write_column(1,0, time_values_acc)
sheet1.write_column(1,1, all_frac_Q_low_sham_acc)
sheet1.write_column(1,2, all_frac_Q_low_stim_acc)
sheet1.write_column(1,3, all_frac_Q_high_sham_acc)
sheet1.write_column(1,4, all_frac_Q_high_stim_acc)
workbook.close()


elapsed = (time.time() - t)/60.
print('Took %f mins for processing' % (elapsed))
	
#TwoTargetTask_RegressFiringRates_Value(hdf_files, syncHDF_files, spike_files, channel, sc, t_before, t_after, t_resolution, t_overlap, smoothed, trial_start,trial_end)