import numpy as np
import scipy.optimize as op
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from rt_calc import compute_rt_per_trial_FreeChoiceTask
from collections import namedtuple
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys
import pandas as pd
from DecisionMakingBehavior import ChoiceBehavior_ThreeTargets_Stimulation

mario_sham_days = [['\mari20161220_05_te2795.hdf'],
					['\mari20170106_03_te2818.hdf'],
					['\mari20170119_03_te2878.hdf', '\mari20170119_05_te2880.hdf'],
					['\mari20170126_03_te2931.hdf', '\mari20170126_05_te2933.hdf'], 
					['\mari20170126_07_te2935.hdf', '\mari20170126_11_te2939.hdf'],
					['\mari20170204_03_te2996.hdf'],
					['\mari20170207_05_te3018.hdf', '\mari20170207_07_te3020.hdf', '\mari20170207_09_te3022.hdf'],
					['\mari20170207_13_te3026.hdf', '\mari20170207_15_te3028.hdf', '\mari20170207_17_te3030.hdf', '\mari20170207_19_te3032.hdf'],
					['\mari20170214_03_te3085.hdf', '\mari20170214_07_te3089.hdf', '\mari20170214_09_te3091.hdf', '\mari20170214_11_te3093.hdf', '\mari20170214_13_te3095.hdf', '\mari20170214_16_te3098.hdf'],
					['\mari20170215_03_te3101.hdf', '\mari20170215_05_te3103.hdf', '\mari20170215_07_te3105.hdf'],
					['\mari20170220_07.hdf', '\mari20170220_09.hdf', '\mari20170220_11.hdf', '\mari20170220_14.hdf']]

mario_stim_days = [['\mari20161221_03_te2800.hdf'],
					['\mari20161222_03_te2803.hdf'],
					['\mari20170108_03_te2821.hdf'],
					['\mari20170125_10_te2924.hdf', '\mari20170125_12_te2926.hdf','\mari20170125_14_te2928.hdf'],
					['\mari20170131_05_te2972.hdf', '\mari20170131_07_te2974.hdf'],
					['\mari20170201_03_te2977.hdf'],
					['\mari20170202_06_te2985.hdf','\mari20170202_08_te2987.hdf'],
					['\mari20170208_07_te3039.hdf', '\mari20170208_09_te3041.hdf', '\mari20170208_11_te3043.hdf'],
					['\mari20170209_03_te3047.hdf', '\mari20170209_05_te3049.hdf', '\mari20170209_08_te3052.hdf'],
					['\mari20170216_03_te3108.hdf', '\mari20170216_05_te3110.hdf', '\mari20170216_08_te3113.hdf', 'mari20170216_10_te3115.hdf'],
					['\mari20170219_14.hdf', '\mari20170219_16.hdf', '\mari20170219_18.hdf']]

data_dir = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Mario\spike_data'

'''
1. Check if there is a side bias for selecting the HV target when HV+MV targets are shown.
'''
num_sham_days = len(mario_sham_days)
prob_HV_given_left = np.zeros(num_sham_days)
prob_HV_given_right = np.zeros(num_sham_days)
for i, day in enumerate(mario_sham_days):
	print "Session %i of %i" %(i+1,num_sham_days)
	num_hdf_files = len(day)
	hdf_files = [(data_dir + hdf) for hdf in day]
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	prob_HV_given_left[i], prob_HV_given_right[i] = cb.TargetSideSelection_Block3()

mean_given_left = np.nanmean(prob_HV_given_left)
mean_given_right = np.nanmean(prob_HV_given_right)
sem_given_left = np.nanstd(prob_HV_given_left)/np.sqrt(num_sham_days)
sem_given_right = np.nanstd(prob_HV_given_right)/np.sqrt(num_sham_days)

print "P(Choose HV|HV on right) = %f" %(mean_given_right)
print "P(Choose HV|HV on left) = %f" %(mean_given_left)
plt.figure()
plt.errorbar(mean_given_right, mean_given_left, yerr = sem_given_left, xerr = sem_given_right, fmt = '--o')
plt.xlabel('P(Choose HV|HV on right)')
plt.ylabel('P(Choose HV|HV on left)')
plt.xlim((0.6,1.0))
plt.ylim((0.6,1.0))
plt.show()

