from DecisionMakingBehavior import Compare_QValue_Models_ThreeTarget, Compare_Qlearning_across_blocks
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
"""
BIC_single_model = np.zeros(len(hdf_list))
BIC_sep_model = np.zeros(len(hdf_list))
BIC_ind_model = np.zeros(len(hdf_list))
accuracy_single_model = np.zeros(len(hdf_list))
accuracy_sep_model = np.zeros(len(hdf_list))
accuracy_ind_model = np.zeros(len(hdf_list))

for j in range(len(hdf_list)):
	# Pull out the relevant session's data
	hdf_files = hdf_list[j]
	BIC_single_model[j], BIC_sep_model[j], BIC_ind_model[j], accuracy_single_model[j], accuracy_sep_model[j], \
		accuracy_ind_model[j] = Compare_QValue_Models_ThreeTarget(hdf_files)

avg_BIC = [np.nanmean(BIC_single_model), np.nanmean(BIC_sep_model), np.nanmean(BIC_ind_model)]
sem_BIC = [np.nanstd(BIC_single_model)/np.sqrt(len(BIC_single_model)), np.nanstd(BIC_sep_model)/np.sqrt(len(BIC_sep_model)), \
			np.nanstd(BIC_ind_model)/np.sqrt(len(BIC_ind_model))]
avg_accuracy = [np.nanmean(accuracy_single_model), np.nanmean(accuracy_sep_model), np.nanmean(accuracy_ind_model)]
sem_accuracy = [np.nanstd(accuracy_single_model)/np.sqrt(len(accuracy_single_model)), np.nanstd(accuracy_sep_model)/np.sqrt(len(accuracy_sep_model)), \
				np.nanstd(accuracy_ind_model)/np.sqrt(len(accuracy_ind_model))]

ind = np.arange(3)
width = 0.35

plt.figure()
plt.subplot(1,2,1)
plt.bar(ind, avg_BIC, width, color = 'b', yerr = sem_BIC)
xticklabels = ['Single Parameter', 'Separate Parameters', 'Individual Parameters']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Q-learning Model')
plt.ylabel('BIC')

plt.subplot(1,2,2)
plt.bar(ind, avg_accuracy, width, color = 'c', yerr = sem_accuracy)
xticklabels = ['Single Parameter', 'Separate Parameters', 'Individual Parameters']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Q-learning Model')
plt.ylabel('Accuracy')
plt.show()

"""

Q_early_sham = np.array([])
Q_early_stim = np.array([])
Q_late_sham = np.array([])
Q_late_stim = np.array([])
Q_early_sham_list = []
Q_early_stim_list = []
Q_late_sham_list = []
Q_late_stim_list = []
allQ = np.array([])

RPE_early_sham = np.array([])
RPE_early_stim = np.array([])
RPE_late_sham = np.array([])
RPE_late_stim = np.array([])
RPE_early_sham_list = []
RPE_early_stim_list = []
RPE_late_sham_list = []
RPE_late_stim_list = []
allRPE = np.array([])

for j in range(len(hdf_list)):
	hdf_files = hdf_list[j]
	
	Q_mid_block1, Q_mid_block3, RPE, RPE3 = Compare_Qlearning_across_blocks(hdf_files)

	if hdf_files in hdf_list_stim:
		Q_early_stim = np.append(Q_early_stim, Q_mid_block1)
		Q_late_stim = np.append(Q_late_stim, Q_mid_block3)
		Q_early_stim_list += [Q_mid_block1.tolist()]
		Q_late_stim_list += [Q_mid_block3.tolist()]

		RPE_early_stim = np.append(RPE_early_stim, RPE)
		RPE_late_stim = np.append(RPE_late_stim, RPE3)
		RPE_early_stim_list += [RPE.tolist()]
		RPE_late_stim_list += [RPE3.tolist()]
	else:
		Q_early_sham = np.append(Q_early_sham, Q_mid_block1)
		Q_late_sham = np.append(Q_late_sham, Q_mid_block3)
		Q_early_sham_list += [Q_mid_block1.tolist()]
		Q_late_sham_list += [Q_mid_block3.tolist()]

		RPE_early_sham = np.append(RPE_early_sham, RPE)
		RPE_late_sham = np.append(RPE_late_sham, RPE3)
		RPE_early_sham_list += [RPE.tolist()]
		RPE_late_sham_list += [RPE3.tolist()]

	allQ = np.append(allQ, Q_mid_block1)
	allQ = np.append(allQ, Q_mid_block3)
	allRPE = np.append(allRPE, RPE)
	allRPE = np.append(allRPE, RPE3)

minQ = np.min(allQ)
maxQ = np.max(allQ)

minRPE = np.min(allRPE)
maxRPE = np.max(allRPE)

num_bins = 20
Q_bins = np.arange(minQ, maxQ, (maxQ - minQ)/num_bins)
Q_bin_centers = (Q_bins[1:] + Q_bins[:-1])/2.
width = Q_bin_centers[1] - Q_bin_centers[0]

RPE_bins = np.arange(minRPE, maxRPE, (maxRPE - minRPE)/num_bins)
RPE_bin_centers = (RPE_bins[1:] + RPE_bins[:-1])/2.
width_RPE = RPE_bin_centers[1] - RPE_bin_centers[0]

## Bin aggragated value data

hist_early_sham, bins = np.histogram(Q_early_sham, Q_bins)
hist_late_sham, bins = np.histogram(Q_late_sham, Q_bins)
hist_early_stim, bins = np.histogram(Q_early_stim, Q_bins)
hist_late_stim, bins = np.histogram(Q_late_stim, Q_bins)

hist_early_sham = hist_early_sham/float(np.sum(hist_early_sham))
hist_late_sham = hist_late_sham/float(np.sum(hist_late_sham))
hist_early_stim = hist_early_stim/float(np.sum(hist_early_stim))
hist_late_stim = hist_late_stim/float(np.sum(hist_late_stim))

chisq_early, p_early = stats.chisquare(hist_early_sham, hist_early_stim)
chisq_late, p_late = stats.chisquare(hist_late_sham, hist_late_stim)

plt.figure()
plt.subplot(1,2,1)
plt.barh(Q_bin_centers, -hist_early_sham, width, color = 'm', label = 'Sham')
plt.barh(Q_bin_centers, hist_early_stim, width, color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('Medium Value')
plt.title('Block A')
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_early))
plt.xlim((-0.6,0.6))
plt.legend()
plt.subplot(1,2,2)
plt.barh(Q_bin_centers, -hist_late_sham, width, color = 'm', label = 'Sham')
plt.barh(Q_bin_centers, hist_late_stim, width, color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('Medium Value')
plt.title("Block A'")
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_late))
plt.xlim((-0.6,0.6))
plt.legend()
plt.show()

## Find average value distributions across sessions
hist_early_sham_ind = np.array([])
hist_early_stim_ind = np.array([])
hist_late_sham_ind = np.array([])
hist_late_stim_ind = np.array([])

for j in range(len(hdf_list_sham)):
	earlyQ = Q_early_sham_list[j]
	lateQ = Q_late_sham_list[j]
	hist_early_sham, bins = np.histogram(earlyQ, Q_bins)
	hist_late_sham, bins = np.histogram(lateQ, Q_bins)
	hist_early_sham = hist_early_sham/float(len(earlyQ))
	hist_late_sham = hist_late_sham/float(len(lateQ))
	if j==0:
		hist_early_sham_ind = hist_early_sham
		hist_late_sham_ind = hist_late_sham
	else:
		hist_early_sham_ind = np.vstack([hist_early_sham_ind, hist_early_sham])
		hist_late_sham_ind = np.vstack([hist_late_sham_ind, hist_late_sham])
	
for j in range(len(hdf_list_stim)):
	earlyQ = Q_early_stim_list[j]
	lateQ = Q_late_stim_list[j]
	hist_early_stim, bins = np.histogram(earlyQ, Q_bins)
	hist_late_stim, bins = np.histogram(lateQ, Q_bins)
	hist_early_stim = hist_early_stim/float(len(earlyQ))
	hist_late_stim = hist_late_stim/float(len(lateQ))
	if j==0:
		hist_early_stim_ind = hist_early_stim
		hist_late_stim_ind = hist_late_stim
	else:
		hist_early_stim_ind = np.vstack([hist_early_stim_ind, hist_early_stim])
		hist_late_stim_ind = np.vstack([hist_late_stim_ind, hist_late_stim])

hist_early_sham_ind_avg = np.nanmean(hist_early_sham_ind, axis = 0)
hist_late_sham_ind_avg = np.nanmean(hist_late_sham_ind, axis = 0)
hist_early_stim_ind_avg = np.nanmean(hist_early_stim_ind, axis = 0)
hist_late_stim_ind_avg = np.nanmean(hist_late_stim_ind, axis = 0)
hist_early_sham_ind_sem = np.nanstd(hist_early_sham_ind, axis = 0)/np.sqrt(len(hdf_list_sham))
hist_late_sham_ind_sem = np.nanstd(hist_late_sham_ind, axis = 0)/np.sqrt(len(hdf_list_sham))
hist_early_stim_ind_sem = np.nanstd(hist_early_stim_ind, axis = 0)/np.sqrt(len(hdf_list_stim))
hist_late_stim_ind_sem = np.nanstd(hist_late_stim_ind, axis = 0)/np.sqrt(len(hdf_list_stim))

chisq_early, p_early = stats.chisquare(hist_early_sham_ind_avg, hist_early_stim_ind_avg)
chisq_late, p_late = stats.chisquare(hist_late_sham_ind_avg, hist_late_stim_ind_avg)

plt.figure()
plt.subplot(1,2,1)
plt.barh(Q_bin_centers, -hist_early_sham_ind_avg, width, xerr = hist_early_sham_ind_sem,color = 'm', label = 'Sham')
plt.barh(Q_bin_centers, hist_early_stim_ind_avg, width, xerr = hist_early_stim_ind_sem,color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('Medium Value')
plt.title('Block A')
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_early))
plt.xlim((-0.6,0.6))
plt.legend()
plt.subplot(1,2,2)
plt.barh(Q_bin_centers, -hist_late_sham_ind_avg, width, xerr = hist_late_sham_ind_sem, color = 'm', label = 'Sham')
plt.barh(Q_bin_centers, hist_late_stim_ind_avg, width, xerr = hist_late_stim_ind_sem, color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('Medium Value')
plt.title("Block A'")
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_late))
plt.xlim((-0.6,0.6))
plt.legend()
plt.show()

## Bin aggragated RPE data 

hist_early_sham_RPE, bins = np.histogram(RPE_early_sham, RPE_bins)
hist_late_sham_RPE, bins = np.histogram(RPE_late_sham, RPE_bins)
hist_early_stim_RPE, bins = np.histogram(RPE_early_stim, RPE_bins)
hist_late_stim_RPE, bins = np.histogram(RPE_late_stim, RPE_bins)

hist_early_sham_RPE = hist_early_sham_RPE/float(np.sum(hist_early_sham_RPE))
hist_late_sham_RPE = hist_late_sham_RPE/float(np.sum(hist_late_sham_RPE))
hist_early_stim_RPE = hist_early_stim_RPE/float(np.sum(hist_early_stim_RPE))
hist_late_stim_RPE = hist_late_stim_RPE/float(np.sum(hist_late_stim_RPE))

chisq_early, p_early_RPE = stats.chisquare(hist_early_sham_RPE, hist_early_stim_RPE)
chisq_late, p_late_RPE = stats.chisquare(hist_late_sham_RPE, hist_late_stim_RPE)

plt.figure()
plt.subplot(1,2,1)
plt.barh(RPE_bin_centers, -hist_early_sham_RPE, width_RPE, color = 'm', label = 'Sham')
plt.barh(RPE_bin_centers, hist_early_stim_RPE, width_RPE, color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('RPE')
plt.title('Block A')
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_early_RPE))
plt.xlim((-0.6,0.6))
plt.legend()
plt.subplot(1,2,2)
plt.barh(RPE_bin_centers, -hist_late_sham_RPE, width_RPE, color = 'm', label = 'Sham')
plt.barh(RPE_bin_centers, hist_late_stim_RPE, width_RPE, color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('RPE')
plt.title("Block A'")
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_late_RPE))
plt.xlim((-0.6,0.6))
plt.legend()
plt.show()

## Find average value distributions across sessions
hist_early_sham_ind_RPE = np.array([])
hist_early_stim_ind_RPE = np.array([])
hist_late_sham_ind_RPE = np.array([])
hist_late_stim_ind_RPE = np.array([])

for j in range(len(hdf_list_sham)):
	earlyRPE = RPE_early_sham_list[j]
	lateRPE = RPE_late_sham_list[j]
	hist_early_sham_RPE, bins = np.histogram(earlyRPE, RPE_bins)
	hist_late_sham_RPE, bins = np.histogram(lateRPE, RPE_bins)
	hist_early_sham_RPE = hist_early_sham_RPE/float(len(earlyRPE))
	hist_late_sham_RPE = hist_late_sham_RPE/float(len(lateRPE))
	if j==0:
		hist_early_sham_ind_RPE = hist_early_sham_RPE
		hist_late_sham_ind_RPE = hist_late_sham_RPE
	else:
		hist_early_sham_ind_RPE = np.vstack([hist_early_sham_ind_RPE, hist_early_sham_RPE])
		hist_late_sham_ind_RPE = np.vstack([hist_late_sham_ind_RPE, hist_late_sham_RPE])
	
for j in range(len(hdf_list_stim)):
	earlyRPE = RPE_early_stim_list[j]
	lateRPE = RPE_late_stim_list[j]
	hist_early_stim_RPE, bins = np.histogram(earlyRPE, RPE_bins)
	hist_late_stim_RPE, bins = np.histogram(lateRPE, RPE_bins)
	hist_early_stim_RPE = hist_early_stim_RPE/float(len(earlyRPE))
	hist_late_stim_RPE = hist_late_stim_RPE/float(len(lateRPE))
	if j==0:
		hist_early_stim_ind_RPE = hist_early_stim_RPE
		hist_late_stim_ind_RPE = hist_late_stim_RPE
	else:
		hist_early_stim_ind_RPE = np.vstack([hist_early_stim_ind_RPE, hist_early_stim_RPE])
		hist_late_stim_ind_RPE = np.vstack([hist_late_stim_ind_RPE, hist_late_stim_RPE])

hist_early_sham_ind_avg_RPE = np.nanmean(hist_early_sham_ind_RPE, axis = 0)
hist_late_sham_ind_avg_RPE = np.nanmean(hist_late_sham_ind_RPE, axis = 0)
hist_early_stim_ind_avg_RPE = np.nanmean(hist_early_stim_ind_RPE, axis = 0)
hist_late_stim_ind_avg_RPE = np.nanmean(hist_late_stim_ind_RPE, axis = 0)
hist_early_sham_ind_sem_RPE = np.nanstd(hist_early_sham_ind_RPE, axis = 0)/np.sqrt(len(hdf_list_sham))
hist_late_sham_ind_sem_RPE = np.nanstd(hist_late_sham_ind_RPE, axis = 0)/np.sqrt(len(hdf_list_sham))
hist_early_stim_ind_sem_RPE = np.nanstd(hist_early_stim_ind_RPE, axis = 0)/np.sqrt(len(hdf_list_stim))
hist_late_stim_ind_sem_RPE = np.nanstd(hist_late_stim_ind_RPE, axis = 0)/np.sqrt(len(hdf_list_stim))

chisq_early, p_early_RPE = stats.chisquare(hist_early_sham_ind_avg_RPE, hist_early_stim_ind_avg_RPE)
chisq_late, p_late_RPE = stats.chisquare(hist_late_sham_ind_avg_RPE, hist_late_stim_ind_avg_RPE)

plt.figure()
plt.subplot(1,2,1)
plt.barh(RPE_bin_centers, -hist_early_sham_ind_avg, width_RPE, xerr = hist_early_sham_ind_sem,color = 'm', label = 'Sham')
plt.barh(RPE_bin_centers, hist_early_stim_ind_avg, width_RPE, xerr = hist_early_stim_ind_sem,color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('RPE')
plt.title('Block A')
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_early_RPE))
plt.xlim((-0.6,0.6))
plt.legend()
plt.subplot(1,2,2)
plt.barh(RPE_bin_centers, -hist_late_sham_ind_avg_RPE, width_RPE, xerr = hist_late_sham_ind_sem_RPE, color = 'm', label = 'Sham')
plt.barh(RPE_bin_centers, hist_late_stim_ind_avg_RPE, width_RPE, xerr = hist_late_stim_ind_sem_RPE, color = 'c', label = 'Stim')
plt.xlabel('Frequency')
plt.ylabel('RPE')
plt.title("Block A'")
plt.text(-0.5, 0.8, 'p = %0.2f (Chi-square)' % (p_late_RPE))
plt.xlim((-0.6,0.6))
plt.legend()
plt.show()