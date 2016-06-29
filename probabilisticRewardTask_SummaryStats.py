from probabilisticRewardTaskPerformance import PeriStimulusFreeChoiceBehavior, FreeChoicePilotTask_Behavior, FreeChoiceBehaviorConditionalProbabilities, FreeChoicePilotTask_ChoiceAfterStim, FreeChoiceBehaviorConditionalProbabilitiesAllBlocks, FreeChoiceTask_PathLengths
import numpy as np
import scipy.optimize as op
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from rt_calc import compute_rt_per_trial_FreeChoiceTask
import pyvttbl as pt
from collections import namedtuple
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys
import pandas as pd
from logLikelihoodRLPerformance import RLPerformance, logLikelihoodRLPerformance, RLPerformance_multiplicative_Qstimparameter, logLikelihoodRLPerformance_multiplicative_Qstimparameter, RLPerformance_multiplicative_Qstimparameter_withQstimOutput


luigi_stim_days = ['\luig20160204_15_te1382.hdf','\luig20160208_07_te1401.hdf','\luig20160212_08_te1429.hdf','\luig20160217_06_te1451.hdf',
                '\luig20160229_11_te1565.hdf','\luig20160301_07_te1572.hdf','\luig20160301_09_te1574.hdf', '\luig20160311_08_te1709.hdf',
                '\luig20160313_07_te1722.hdf', '\luig20160315_14_te1739.hdf']

luigi_sham_days = ['\luig20160213_05_te1434.hdf','\luig20160219_04_te1473.hdf','\luig20160221_05_te1478.hdf', '\luig20160305_26_te1617.hdf', \
                 '\luig20160306_11_te1628.hdf', '\luig20160307_13_te1641.hdf', '\luig20160319_23_te1801.hdf','\luig20160320_07_te1809.hdf', \
                 '\luig20160322_08_te1826.hdf']
luigi_sham_days = ['\luig20160213_05_te1434.hdf','\luig20160219_04_te1473.hdf','\luig20160221_05_te1478.hdf', '\luig20160305_26_te1617.hdf', \
                 '\luig20160306_11_te1628.hdf', '\luig20160307_13_te1641.hdf', '\luig20160310_16_te1695.hdf','\luig20160319_23_te1801.hdf', \
                 '\luig20160320_07_te1809.hdf', '\luig20160322_08_te1826.hdf']

luigi_control_days = ['\luig20160218_10_te1469.hdf','\luig20160223_11_te1508.hdf', \
				'\luig20160224_15_te1523.hdf', '\luig20160303_11_te1591.hdf',  \
                '\luig20160308_06_te1647.hdf','\luig20160309_25_te1672.hdf', '\luig20160323_04_te1830.hdf', '\luig20160323_09_te1835.hdf',
                '\luig20160324_10_te1845.hdf', '\luig20160324_12_te1847.hdf','\luig20160324_14_te1849.hdf']
#luigi_control_days = ['\luig20160218_10_te1469.hdf','\luig20160223_09_te1506.hdf','\luig20160223_11_te1508.hdf','\luig20160224_11_te1519.hdf',
#               '\luig20160224_15_te1523.hdf', '\luig20160302_06_te1580.hdf', '\luig20160303_09_te1589.hdf', '\luig20160302_06_te1580.hdf']
'''
control_days = ['\luig20160218_10_te1469.hdf','\luig20160223_11_te1508.hdf','\luig20160224_15_te1523.hdf', \
                '\luig20160303_11_te1591.hdf', '\luig20160308_06_te1647.hdf','\luig20160309_25_te1672.hdf', '\luig20160316_13_te1752.hdf']
'''
sham_days = ['\papa20150217_05.hdf','\papa20150305_02.hdf',
    '\papa20150310_02.hdf',
    '\papa20150519_02.hdf','\papa20150519_04.hdf','\papa20150528_02.hdf']
sham_days = ['\papa20150213_10.hdf','\papa20150217_05.hdf','\papa20150225_02.hdf',
    '\papa20150307_02.hdf','\papa20150308_06.hdf','\papa20150310_02.hdf','\papa20150506_09.hdf','\papa20150506_10.hdf',
    '\papa20150519_02.hdf','\papa20150519_03.hdf','\papa20150519_04.hdf','\papa20150527_01.hdf','\papa20150528_02.hdf']

control_days = ['\papa20150203_10.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf']
"""
control_days = ['\papa20150203_10.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf']
"""
stim_days = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']


Q_initial = [0.5, 0.5]
alpha_true = 0.2
beta_true = 0.2
gamma_true = 0.5

def ReactionTimeVersusLowValue(Qvalue, rt):
	
	# Sort both arrays in terms of ascending value
	qsort = np.argsort(Qvalue)
	Qvalue = Qvalue[qsort]
	rt = rt[qsort]

	# Bin values with bin size = 0.1
	bins = np.arange(0.0, 1.1, 0.1)
	hist, Qbins = np.histogram(Qvalue, bins=bins)

	avg_rt = np.zeros(len(bins)-1)
	sem_rt = np.zeros(len(bins)-1)

	avg_rt[0] = np.nanmean(rt[0:hist[0]])
	sem_rt[0] = np.nanstd(rt[0:hist[0]])/np.sqrt(hist[0])
	hist_start = hist[0]
	for i in range(1, len(bins)-1):
		avg_rt[i] = np.nanmean(rt[hist_start:hist_start+hist[i]])
		sem_rt[i] = np.nanstd(rt[hist_start:hist_start+hist[i]])/np.sqrt(hist[i])
		hist_start += hist[i]

	return Qbins[1:], avg_rt, sem_rt

def make_statsmodel_dtstructure(inputs, treatment_labels, independent_var_label):
	'''
	Method makes data structure like what stats library likes to work with. 
	Input:
		inputs: list of arrays that will be compared
		labels: list of labels for the arrays, should be equal to the number of arrays in inputs
	'''
	
	dta = []
	total = 0

	for data in inputs:
		ind = inputs.index(data)

		for i in range(len(data)):
			dta += [(total, treatmeant_labels[ind], data[i])]
			total += 1

	dta = np.rec.array(dta, dtype=[('idx', '<i4'),
                                ('Treatment', '|S8'),
                                (independent_var_label, '<f4')])
	
	return dta


def probabilisticRewardTaskPerformance_RewardAndStim(sham_days, stim_days, control_days, animal_id):
	num_stim_days = len(stim_days)
	num_sham_days = len(sham_days)
	num_control_days = len(control_days)

	stim_prob_choose_lv_stim_rewarded = np.zeros(num_stim_days)
	stim_prob_choose_lv_stim_unrewarded = np.zeros(num_stim_days)
	sham_prob_choose_lv_stim_rewarded = np.zeros(num_sham_days)
	sham_prob_choose_lv_stim_unrewarded = np.zeros(num_sham_days)
	control_prob_choose_lv_stim_rewarded = np.zeros(num_control_days)
	control_prob_choose_lv_stim_unrewarded = np.zeros(num_control_days)
	

	for i in range(0,num_stim_days):
		name = stim_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		
		stim_prob_choose_lv_stim_rewarded[i], stim_prob_choose_lv_stim_unrewarded[i] = FreeChoicePilotTask_ChoiceAfterStim(reward3, target3, instructed_or_freechoice_block3,stim_trials)

	for i in range(0,num_sham_days):
		name = sham_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		
		sham_prob_choose_lv_stim_rewarded[i], sham_prob_choose_lv_stim_unrewarded[i] = FreeChoicePilotTask_ChoiceAfterStim(reward3, target3, instructed_or_freechoice_block3,stim_trials)

	for i in range(0,num_control_days):
		name = control_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		
		control_prob_choose_lv_stim_rewarded[i], control_prob_choose_lv_stim_unrewarded[i] = FreeChoicePilotTask_ChoiceAfterStim(reward3, target3, instructed_or_freechoice_block3,stim_trials)

	mean_stim_rewarded = np.nanmean(stim_prob_choose_lv_stim_rewarded)
	mean_stim_unrewarded = np.nanmean(stim_prob_choose_lv_stim_unrewarded)
	sem_stim_rewarded = np.nanstd(stim_prob_choose_lv_stim_rewarded)/np.sqrt(len(stim_prob_choose_lv_stim_rewarded))
	sem_stim_unrewarded = np.nanstd(stim_prob_choose_lv_stim_unrewarded)/np.sqrt(len(stim_prob_choose_lv_stim_unrewarded))

	mean_sham_rewarded = np.nanmean(sham_prob_choose_lv_stim_rewarded)
	mean_sham_unrewarded = np.nanmean(sham_prob_choose_lv_stim_unrewarded)
	sem_sham_rewarded = np.nanstd(sham_prob_choose_lv_stim_rewarded)/np.sqrt(len(sham_prob_choose_lv_stim_rewarded))
	sem_sham_unrewarded = np.nanstd(sham_prob_choose_lv_stim_unrewarded)/np.sqrt(len(sham_prob_choose_lv_stim_unrewarded))

	mean_control_rewarded = np.nanmean(control_prob_choose_lv_stim_rewarded)
	mean_control_unrewarded = np.nanmean(control_prob_choose_lv_stim_unrewarded)
	sem_control_rewarded = np.nanstd(control_prob_choose_lv_stim_rewarded)/np.sqrt(len(control_prob_choose_lv_stim_rewarded))
	sem_control_unrewarded = np.nanstd(control_prob_choose_lv_stim_unrewarded)/np.sqrt(len(control_prob_choose_lv_stim_unrewarded))

	'''
	plt.figure()
	plt.plot(stim_prob_choose_lv_stim_rewarded, stim_prob_choose_lv_stim_unrewarded,marker='o')
	plt.xlabel('Choose LV | Rewarded')
	plt.ylabel('Choose LV | Unrewarded')
	plt.show()
	'''
	plt.figure()
	plt.errorbar(np.arange(2),[mean_stim_rewarded, mean_stim_unrewarded],yerr=[sem_stim_rewarded, sem_stim_unrewarded],color='c')
	plt.errorbar(np.arange(2),[mean_sham_rewarded, mean_sham_unrewarded],yerr=[sem_sham_rewarded, sem_sham_unrewarded],color='m')
	plt.errorbar(np.arange(2),[mean_control_rewarded, mean_control_unrewarded],yerr=[sem_control_rewarded, sem_control_unrewarded],color='y')
	plt.xticks(np.arange(2), ('Rewarded', 'Unrewarded'))
	plt.ylim((0,0.35))
	plt.xlim((-0.1,1.1))
	plt.ylabel('Fraction of LV Choices')
	plt.show()

	return stim_prob_choose_lv_stim_rewarded, stim_prob_choose_lv_stim_unrewarded, sham_prob_choose_lv_stim_rewarded, sham_prob_choose_lv_stim_unrewarded, control_prob_choose_lv_stim_rewarded, control_prob_choose_lv_stim_unrewarded

def probabilisticRewardTaskPerformance_RewardAndTargetSide(days, animal_id):
	
	num_days = len(days)

	dta = []

	for i in range(0,num_days):
		name = days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		
		instructed_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),1)))
		free_choice_ind = [(ind+1) for ind in instructed_choice_ind if ((ind + 1) not in instructed_choice_ind)&(ind + 1 < len(instructed_or_freechoice_block3))]

		rewarded_left = [(2 - target3[i]) for i in free_choice_ind if (target_side3[i-1] == 1)&(reward3[i] == 1)]
		rewarded_right = [(2 - target3[i]) for i in free_choice_ind if (target_side3[i-1] == 0)&(reward3[i] == 1)]
		unrewarded_left = [(2 - target3[i]) for i in free_choice_ind if (target_side3[i-1] == 1)&(reward3[i] == 0)]
		unrewarded_right = [(2 - target3[i]) for i in free_choice_ind if (target_side3[i-1] == 0)&(reward3[i] == 0)]

		prob_lv_stim_rewarded_left = np.sum(rewarded_left)/float(len(rewarded_left))
		prob_lv_stim_rewarded_right = np.sum(rewarded_right)/float(len(rewarded_right))
		if len(unrewarded_left)== 0:
			prob_lv_stim_unrewarded_left = 0
		else:
			prob_lv_stim_unrewarded_left = np.sum(unrewarded_left)/float(len(unrewarded_left))
		prob_lv_stim_unrewarded_right = np.sum(unrewarded_right)/float(len(unrewarded_right))

		rewarded_left = [target_side3[i] for i in free_choice_ind if (target_side3[i-1] == 1)&(reward3[i] == 1)]
		rewarded_right = [target_side3[i] for i in free_choice_ind if (target_side3[i-1] == 0)&(reward3[i] == 1)]
		unrewarded_left = [target_side3[i] for i in free_choice_ind if (target_side3[i-1] == 1)&(reward3[i] == 0)]
		unrewarded_right = [target_side3[i] for i in free_choice_ind if (target_side3[i-1] == 0)&(reward3[i] == 0)]

		prob_right_stim_rewarded_left = np.sum(rewarded_left)/float(len(rewarded_left))
		prob_right_stim_rewarded_right = np.sum(rewarded_right)/float(len(rewarded_right))
		if len(unrewarded_left) == 0:
			prob_right_stim_unrewarded_left = 0
		else:
			prob_right_stim_unrewarded_left = np.sum(unrewarded_left)/float(len(unrewarded_left))
		prob_right_stim_unrewarded_right = np.sum(unrewarded_right)/float(len(unrewarded_right))

		dta += [(1,1,prob_lv_stim_rewarded_left,prob_right_stim_rewarded_left)]
		dta += [(1,2,prob_lv_stim_rewarded_right,prob_right_stim_rewarded_right)]
		dta += [(0,1,prob_lv_stim_unrewarded_left,prob_right_stim_unrewarded_left)]
		dta += [(0,2,prob_lv_stim_unrewarded_right,prob_right_stim_unrewarded_right)]

	dta = pd.DataFrame(dta, columns=['reward', 'stim_targ_side', 'lv_choices', 'right_choices'])
		
	return dta

def probabilisticRewardTaskPerformance_TrialAligned(sham_days,stim_days,control_days, animal_id):

	all_days = stim_days + sham_days + control_days
	counter_stim = 0
	counter_sham = 0
	counter_control = 0
	avg_low_stim = 0
	avg_low_sham = 0
	avg_low_control = 0

	num_stim_days = len(stim_days)
	num_sham_days = len(sham_days)
	num_control_days = len(control_days)

	prob_low_aligned_stim = np.zeros((num_stim_days,5))
	prob_low_aligned_rewarded_stim = np.zeros((num_stim_days,5))
	prob_low_aligned_unrewarded_stim = np.zeros((num_stim_days,5))

	prob_low_aligned_sham = np.zeros((num_sham_days,5))
	prob_low_aligned_rewarded_sham = np.zeros((num_sham_days,5))
	prob_low_aligned_unrewarded_sham = np.zeros((num_sham_days,5))

	prob_low_aligned_control = np.zeros((num_control_days,5))
	prob_low_aligned_rewarded_control = np.zeros((num_control_days,5))
	prob_low_aligned_unrewarded_control = np.zeros((num_control_days,5))

	for i in range(0,num_stim_days):
		name = stim_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		prob_low_aligned_stim[i,:], prob_low_aligned_rewarded_stim[i,:], prob_low_aligned_unrewarded_stim[i,:] = PeriStimulusFreeChoiceBehavior(hdf_location)

	for i in range(0,num_sham_days):
		name = sham_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		prob_low_aligned_sham[i,:], prob_low_aligned_rewarded_sham[i,:], prob_low_aligned_unrewarded_sham[i,:] = PeriStimulusFreeChoiceBehavior(hdf_location)

	for i in range(0,num_control_days):
		name = control_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		prob_low_aligned_control[i,:], prob_low_aligned_rewarded_control[i,:], prob_low_aligned_unrewarded_control[i,:] = PeriStimulusFreeChoiceBehavior(hdf_location)


	avg_prob_low_aligned_stim = np.nanmean(prob_low_aligned_stim,axis=0)
	std_prob_low_aligned_stim = np.nanstd(prob_low_aligned_stim,axis=0)
	sem_prob_low_aligned_stim = std_prob_low_aligned_stim/np.sqrt(num_stim_days)
	avg_prob_low_aligned_rewarded_stim = np.nanmean(prob_low_aligned_rewarded_stim,axis=0)
	std_prob_low_aligned_rewarded_stim = np.nanstd(prob_low_aligned_rewarded_stim,axis=0)
	sem_prob_low_aligned_rewarded_stim = std_prob_low_aligned_rewarded_stim/np.sqrt(num_stim_days)
	avg_prob_low_aligned_unrewarded_stim = np.nanmean(prob_low_aligned_unrewarded_stim,axis=0)
	std_prob_low_aligned_unrewarded_stim = np.nanstd(prob_low_aligned_unrewarded_stim,axis=0)
	sem_prob_low_aligned_unrewarded_stim = std_prob_low_aligned_unrewarded_stim/np.sqrt(num_stim_days)

	avg_prob_low_aligned_sham = np.nanmean(prob_low_aligned_sham,axis=0)
	std_prob_low_aligned_sham = np.nanstd(prob_low_aligned_sham,axis=0)
	sem_prob_low_aligned_sham = std_prob_low_aligned_sham/np.sqrt(num_sham_days)
	avg_prob_low_aligned_rewarded_sham = np.nanmean(prob_low_aligned_rewarded_sham,axis=0)
	std_prob_low_aligned_rewarded_sham = np.nanstd(prob_low_aligned_rewarded_sham,axis=0)
	sem_prob_low_aligned_rewarded_sham = std_prob_low_aligned_sham/np.sqrt(num_sham_days)
	avg_prob_low_aligned_unrewarded_sham = np.nanmean(prob_low_aligned_unrewarded_sham,axis=0)
	std_prob_low_aligned_unrewarded_sham = np.nanstd(prob_low_aligned_unrewarded_sham,axis=0)
	sem_prob_low_aligned_unrewarded_sham = std_prob_low_aligned_unrewarded_sham/np.sqrt(num_sham_days)

	avg_prob_low_aligned_control = np.nanmean(prob_low_aligned_control,axis=0)
	std_prob_low_aligned_control = np.nanstd(prob_low_aligned_control,axis=0)
	sem_prob_low_aligned_control = std_prob_low_aligned_control/np.sqrt(num_control_days)
	avg_prob_low_aligned_rewarded_control = np.nanmean(prob_low_aligned_rewarded_control,axis=0)
	std_prob_low_aligned_rewarded_control = np.nanstd(prob_low_aligned_rewarded_control,axis=0)
	sem_prob_low_aligned_rewarded_control = std_prob_low_aligned_rewarded_control/np.sqrt(num_control_days)
	avg_prob_low_aligned_unrewarded_control = np.nanmean(prob_low_aligned_unrewarded_control,axis=0)
	std_prob_low_aligned_unrewarded_control = np.nanstd(prob_low_aligned_unrewarded_control,axis=0)
	sem_prob_low_aligned_unrewarded_control = std_prob_low_aligned_unrewarded_control/np.sqrt(num_control_days)

	width = float(0.35)
	ind = np.arange(0,5)
	# Get linear fit to total prob
	m_stim,b_stim = np.polyfit(ind, avg_prob_low_aligned_stim, 1)
	m_sham,b_sham = np.polyfit(ind, avg_prob_low_aligned_sham, 1)
	m_control,b_control = np.polyfit(ind, avg_prob_low_aligned_control, 1)


	plt.figure()
	plt.bar(ind, avg_prob_low_aligned_rewarded_stim, width/2, color = 'c', hatch = '//', yerr = sem_prob_low_aligned_rewarded_stim/2)
	plt.bar(ind, avg_prob_low_aligned_unrewarded_stim, width/2, color = 'c', bottom=avg_prob_low_aligned_rewarded_stim,yerr = sem_prob_low_aligned_unrewarded_stim/2, label='LV Stim')
	plt.plot(ind,m_stim*ind + b_stim,'c--')
	plt.bar(ind + width/2, avg_prob_low_aligned_rewarded_sham, width/2, color = 'm',hatch = '//', yerr = sem_prob_low_aligned_rewarded_sham/2)
	plt.bar(ind + width/2, avg_prob_low_aligned_unrewarded_sham, width/2, color = 'm',bottom=avg_prob_low_aligned_rewarded_sham,yerr = sem_prob_low_aligned_unrewarded_sham/2, label='Sham')
	plt.plot(ind+width/2, m_sham*ind + b_sham,'m--')
	plt.bar(ind + width, avg_prob_low_aligned_rewarded_control, width/2, color = 'y',hatch = '//', yerr = sem_prob_low_aligned_rewarded_control/2)
	plt.bar(ind + width, avg_prob_low_aligned_unrewarded_control, width/2, color = 'y',bottom=avg_prob_low_aligned_rewarded_control,yerr = sem_prob_low_aligned_unrewarded_control/2, label='Control')
	plt.plot(ind+width, m_control*ind + b_control,'y--')
	plt.ylabel('P(Choose LV Target)')
	plt.title('Target Selection')
	plt.xticks(ind + width/2, ('1', '2', '3', '4','5'))
	plt.xlabel('Trials post-stimulation')
	plt.ylim([0.0,.35])
	plt.xlim([-0.1,5.4])
	plt.legend()
	'''
	plt.subplot(1,2,2)
	plt.plot(range(0,5), avg_prob_low_aligned_stim,'c')
	plt.plot(range(5,10), avg_prob_low_aligned_sham, 'm')
	plt.plot(range(10,15), avg_prob_low_aligned_control, 'y')
	plt.ylim([0.0,0.3])
	'''
	plt.show()
	return prob_low_aligned_sham, prob_low_aligned_stim, prob_low_aligned_control

def probabilisticRewardTaskPerformance_ConditionalProbabilities(days, hdf_folder, subplot_num):

	num_days = len(days)

	prob_high_given_rhs_block1 = np.zeros(num_days)
	prob_high_given_lhs_block1 = np.zeros(num_days)
	prob_high_given_rhs_block3 = np.zeros(num_days)
	prob_high_given_lhs_block3 = np.zeros(num_days)

	prob_low_given_rhs_block1 = np.zeros(num_days)
	prob_low_given_lhs_block1 = np.zeros(num_days)
	prob_low_given_rhs_block3 = np.zeros(num_days)
	prob_low_given_lhs_block3 = np.zeros(num_days)
	
	for i in range(0,num_days):
		name = days[i]
		hdf_location = hdf_folder+name
		prob_high_given_rhs_block1[i], prob_high_given_lhs_block1[i], prob_high_given_rhs_block3[i],prob_high_given_lhs_block3[i], prob_low_given_rhs_block1[i],prob_low_given_lhs_block1[i],prob_low_given_rhs_block3[i],prob_low_given_lhs_block3[i] = FreeChoiceBehaviorConditionalProbabilities(hdf_location)


	avg_prob_high_given_rhs_block1 = np.nanmean(prob_high_given_rhs_block1)
	sem_prob_high_given_rhs_block1 = np.nanstd(prob_high_given_rhs_block1)/np.sqrt(num_days)

	avg_prob_high_given_lhs_block1 = np.nanmean(prob_high_given_lhs_block1)
	sem_prob_high_given_lhs_block1 = np.nanstd(prob_high_given_lhs_block1)/np.sqrt(num_days)

	avg_prob_high_given_rhs_block3 = np.nanmean(prob_high_given_rhs_block3)
	sem_prob_high_given_rhs_block3 = np.nanstd(prob_high_given_rhs_block3)/np.sqrt(num_days)

	avg_prob_high_given_lhs_block3 = np.nanmean(prob_high_given_lhs_block3)
	sem_prob_high_given_lhs_block3 = np.nanstd(prob_high_given_lhs_block3)/np.sqrt(num_days)

	avg_prob_low_given_rhs_block1 = np.nanmean(prob_low_given_rhs_block1)
	sem_prob_low_given_rhs_block1 = np.nanstd(prob_low_given_rhs_block1)/np.sqrt(num_days)

	avg_prob_low_given_lhs_block1 = np.nanmean(prob_low_given_lhs_block1)
	sem_prob_low_given_lhs_block1 = np.nanstd(prob_low_given_lhs_block1)/np.sqrt(num_days)

	avg_prob_low_given_rhs_block3 = np.nanmean(prob_low_given_rhs_block3)
	sem_prob_low_given_rhs_block3 = np.nanstd(prob_low_given_rhs_block3)/np.sqrt(num_days)

	avg_prob_low_given_lhs_block3 = np.nanmean(prob_low_given_lhs_block3)
	sem_prob_low_given_lhs_block3 = np.nanstd(prob_low_given_lhs_block3)/np.sqrt(num_days)	

	plt.subplot(1,2,subplot_num)
	plt.errorbar(avg_prob_high_given_rhs_block1,avg_prob_high_given_lhs_block1,xerr=sem_prob_high_given_rhs_block1, yerr=sem_prob_high_given_lhs_block1,marker='o',color='c',label='HV - Block1')
	plt.errorbar(avg_prob_high_given_rhs_block3,avg_prob_high_given_lhs_block3,xerr=sem_prob_high_given_rhs_block3, yerr=sem_prob_high_given_lhs_block3,marker='s',color='c',label='HV - Block3')
	plt.errorbar(avg_prob_low_given_rhs_block1,avg_prob_low_given_lhs_block1,xerr=sem_prob_low_given_rhs_block1, yerr=sem_prob_low_given_lhs_block1,marker='o',color='m', label='LV - Block1')
	plt.errorbar(avg_prob_low_given_rhs_block3,avg_prob_low_given_lhs_block3,xerr=sem_prob_low_given_rhs_block3, yerr=sem_prob_low_given_lhs_block3,marker='s',color='m', label='LV - Block3')
	plt.plot([0,1.1],[0,1.1],'k--')
	plt.xlim((0,1.1))
	plt.ylim((0,1.1))
	plt.xlabel('Target on Right')
	plt.ylabel('Target on Left')
	plt.title('P(Target Selection | Target Side Presentation)')
	plt.legend()
	#plt.show()


	return

def probabilisticRewardTaskPerformance_SideBiasAnalysis(days, animal_id):

	num_days = len(days)

	prob_low_given_rhs = np.zeros(num_days)
	prob_low_given_lhs = np.zeros(num_days)
	
	for i in range(0,num_days):
		name = days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		prob_low_given_lhs[i], prob_low_given_rhs[i] = FreeChoiceBehaviorConditionalProbabilitiesAllBlocks(hdf_location)

	'''
	Run one-way ANOVA on these probabilities to see if there is a significant difference
	'''
	f_val, p_val = stats.f_oneway(prob_low_given_lhs, prob_low_given_rhs)
	print "One-way ANOVA Results: Prob(LV) Given Side Presentation"
	print "F = ", f_val
	print "p = ", p_val

	return prob_low_given_lhs, prob_low_given_rhs

def ReactionTimeHist(reaction_time):

	'''
	This method takes an array of reaction times as an input and values used to plot a normalized histogram of reaction times.

	Input:
		reaction_time: array of times in seconds
	Outputs:
		mean_reaction_time: average value in input array
		std_reaction_time: standard deviation of values in input array
		nbins_reaction_time: time bins used for the histogram of reaction time values
		reaction_time_hist: normalized counts for each bin in nbins_reaction_time
	'''

	mean_reaction_time = np.nanmean(reaction_time)
	std_reaction_time = np.nanstd(reaction_time)
	nbins_reaction_time = np.arange(mean_reaction_time-10*std_reaction_time,mean_reaction_time+10*std_reaction_time,float(std_reaction_time)/2)
	reaction_time_hist,nbins_reaction_time = np.histogram(reaction_time,bins=nbins_reaction_time)
	nbins_reaction_time = nbins_reaction_time[1:]
	reaction_time_hist = reaction_time_hist/float(len(reaction_time))

	return mean_reaction_time, std_reaction_time, nbins_reaction_time, reaction_time_hist

def probabilisticRewardTaskPerformance_ReactionTimes(stim_days,sham_days,control_days,animal_id):

	num_stim_days = len(stim_days)
	num_sham_days = len(sham_days)
	num_control_days = len(control_days)

	reaction_time_stim_fc_block1 = np.array([])
	reaction_time_stim_ic_block1 = np.array([])
	reaction_time_sham_fc_block1 = np.array([])
	reaction_time_sham_ic_block1 = np.array([])
	reaction_time_control_fc_block1 = np.array([])
	reaction_time_control_ic_block1 = np.array([])

	reaction_time_stim_fc_block3 = np.array([])
	reaction_time_stim_ic_block3 = np.array([])
	reaction_time_sham_fc_block3 = np.array([])
	reaction_time_sham_ic_block3 = np.array([])
	reaction_time_control_fc_block3 = np.array([])
	reaction_time_control_ic_block3 = np.array([])

	stim_reaction_time_block3 = np.array([])
	stim_Qlow_block3 = np.array([])
	stim_Qlow_adjusted_block3 = np.array([])

	sham_reaction_time_block3 = np.array([])
	sham_Qlow_block3 = np.array([])

	control_reaction_time_block3 = np.array([])
	control_Qlow_block3 = np.array([])
	

	for i in range(0,num_stim_days):
		name = stim_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(hdf_location)

		fc_trials_block1 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block1,2))))
		ic_trials_block1 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block1,1))))

		reaction_time_stim_fc_block1 = np.append(reaction_time_stim_fc_block1, reaction_time[fc_trials_block1])
		reaction_time_stim_ic_block1 = np.append(reaction_time_stim_ic_block1, reaction_time[ic_trials_block1])

		fc_trials_block3 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block3,2)))) + 200
		ic_trials_block3 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block3,1)))) + 200
		
		reaction_time_stim_fc_block3 = np.append(reaction_time_stim_fc_block3, reaction_time[fc_trials_block3])
		reaction_time_stim_ic_block3 = np.append(reaction_time_stim_ic_block3, reaction_time[ic_trials_block3])

		'''
		Get soft-max decision fit
		'''

		nll = lambda *args: -logLikelihoodRLPerformance(*args)
		result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward1, target1, instructed_or_freechoice_block1), bounds=[(0,1),(0,None)])
		alpha_ml_block1, beta_ml_block1 = result1["x"]
		Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, instructed_or_freechoice_block1)
		
		result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward3, target3, instructed_or_freechoice_block3), bounds=[(0,1),(0,None)])
		alpha_ml_block3, beta_ml_block3 = result3["x"]
		Qlow_block3, Qhigh_block3, prob_low_block3_regular, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, instructed_or_freechoice_block3)
		
		stim_reaction_time_block3 = np.append(stim_reaction_time_block3, reaction_time[200:])
		
		'''
		Get fit with Multiplicative stimulation parameter in Q-value update equation
		'''
		nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter(*args)
		result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward3, target3, instructed_or_freechoice_block3, stim_trials), bounds=[(0,1),(0,None),(0,None)])
		alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
		Qlow_block3, Qhigh_block3, prob_low_block3_Qmultiplicative, max_loglikelihood3, Qlow_block3_upmodulated = RLPerformance_multiplicative_Qstimparameter_withQstimOutput([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, instructed_or_freechoice_block3, stim_trials)
		
		stim_Qlow_block3 = np.append(stim_Qlow_block3, Qlow_block3[:-1])
		stim_Qlow_adjusted_block3 = np.append(stim_Qlow_adjusted_block3, Qlow_block3_upmodulated[:-1])
		

	for i in range(0,num_sham_days):
		name = sham_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(hdf_location)

		fc_trials_block1 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block1,2))))
		ic_trials_block1 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block1,1))))

		reaction_time_sham_fc_block1 = np.append(reaction_time_sham_fc_block1, reaction_time[fc_trials_block1])
		reaction_time_sham_ic_block1 = np.append(reaction_time_sham_ic_block1, reaction_time[ic_trials_block1])

		fc_trials_block3 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block3,2)))) + 200
		ic_trials_block3 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block3,1)))) + 200
		
		reaction_time_sham_fc_block3 = np.append(reaction_time_sham_fc_block3, reaction_time[fc_trials_block3])
		reaction_time_sham_ic_block3 = np.append(reaction_time_sham_ic_block3, reaction_time[ic_trials_block3])

		'''
		Get soft-max decision fit
		'''

		nll = lambda *args: -logLikelihoodRLPerformance(*args)
		result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward1, target1, instructed_or_freechoice_block1), bounds=[(0,1),(0,None)])
		alpha_ml_block1, beta_ml_block1 = result1["x"]
		Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, instructed_or_freechoice_block1)

		result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward3, target3, instructed_or_freechoice_block3), bounds=[(0,1),(0,None)])
		alpha_ml_block3, beta_ml_block3 = result3["x"]
		Qlow_block3, Qhigh_block3, prob_low_block3_regular, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, instructed_or_freechoice_block3)

		"""
		'''
		Get fit with Multiplicative stimulation parameter in Q-value update equation
		'''
		nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter(*args)
		result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward3, target3, instructed_or_freechoice_block3, stim_trials), bounds=[(0,1),(0,None),(0,None)])
		alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
		Qlow_block3, Qhigh_block3, prob_low_block3_Qmultiplicative, max_loglikelihood3 = RLPerformance_multiplicative_Qstimparameter([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, instructed_or_freechoice_block3, stim_trials)
		"""
		sham_reaction_time_block3 = np.append(sham_reaction_time_block3, reaction_time[200:])
		sham_Qlow_block3 = np.append(sham_Qlow_block3, Qlow_block3[:-1])


	for i in range(0,num_control_days):
		name = control_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(hdf_location)

		fc_trials_block1 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block1,2))))
		ic_trials_block1 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block1,1))))

		reaction_time_control_fc_block1 = np.append(reaction_time_control_fc_block1, reaction_time[fc_trials_block1])
		reaction_time_control_ic_block1 = np.append(reaction_time_control_ic_block1, reaction_time[ic_trials_block1])

		fc_trials_block3 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block3,2)))) + 200
		ic_trials_block3 = np.ravel(np.nonzero(np.ravel(np.equal(instructed_or_freechoice_block3,1)))) + 200
		
		reaction_time_control_fc_block3 = np.append(reaction_time_control_fc_block3, reaction_time[fc_trials_block3])
		reaction_time_control_ic_block3 = np.append(reaction_time_control_ic_block3, reaction_time[ic_trials_block3])

		'''
		Get soft-max decision fit
		'''

		nll = lambda *args: -logLikelihoodRLPerformance(*args)
		result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward1, target1, instructed_or_freechoice_block1), bounds=[(0,1),(0,None)])
		alpha_ml_block1, beta_ml_block1 = result1["x"]
		Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, instructed_or_freechoice_block1)
		"""
		result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward3, target3, instructed_or_freechoice_block3), bounds=[(0,1),(0,None)])
		alpha_ml_block3, beta_ml_block3 = result3["x"]
		Qlow_block3, Qhigh_block3, prob_low_block3_regular, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, instructed_or_freechoice_block3)
		"""
		'''
		Get fit with Multiplicative stimulation parameter in Q-value update equation
		'''
		nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter(*args)
		result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward3, target3, instructed_or_freechoice_block3, stim_trials), bounds=[(0,1),(0,None),(0,None)])
		alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
		Qlow_block3, Qhigh_block3, prob_low_block3_Qmultiplicative, max_loglikelihood3 = RLPerformance_multiplicative_Qstimparameter([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, instructed_or_freechoice_block3, stim_trials)
		
		control_reaction_time_block3 = np.append(control_reaction_time_block3, reaction_time[200:])
		control_Qlow_block3 = np.append(control_Qlow_block3, Qlow_block3[:-1])

	'''
	Reaction-time as a funciton of LV target
	'''
	stim_Qbins, stim_avg_rt, stim_sem_rt = ReactionTimeVersusLowValue(stim_Qlow_adjusted_block3, stim_reaction_time_block3)
	sham_Qbins, sham_avg_rt, sham_sem_rt = ReactionTimeVersusLowValue(sham_Qlow_block3, sham_reaction_time_block3)
	control_Qbins, control_avg_rt, control_sem_rt = ReactionTimeVersusLowValue(control_Qlow_block3, control_reaction_time_block3)

	get_all_not_nan = np.logical_not(np.isnan(stim_avg_rt))
	get_all_not_nan_ind = np.ravel(np.nonzero(get_all_not_nan))
	m,b = np.polyfit(stim_Qbins[get_all_not_nan_ind], stim_avg_rt[get_all_not_nan_ind], 1)
	stim_avg_rt_fit = m*np.arange(0.1,1,0.1) + b
	
	get_all_not_nan = np.logical_not(np.isnan(sham_avg_rt))
	get_all_not_nan_ind = np.ravel(np.nonzero(get_all_not_nan))
	m,b = np.polyfit(sham_Qbins[get_all_not_nan_ind], sham_avg_rt[get_all_not_nan_ind], 1)
	sham_avg_rt_fit = m*np.arange(0.1,1,0.1) + b


	plt.figure()
	plt.plot(stim_Qbins, stim_avg_rt,'m',label='Stim')
	plt.fill_between(stim_Qbins,stim_avg_rt-stim_sem_rt,stim_avg_rt + stim_sem_rt,facecolor='m',alpha=0.5,linewidth=0.0)
	plt.plot(np.arange(0.1,1,0.1), stim_avg_rt_fit,'m-.')
	plt.plot(sham_Qbins, sham_avg_rt,'c', label='Sham')
	plt.fill_between(sham_Qbins,sham_avg_rt-sham_sem_rt,sham_avg_rt + sham_sem_rt,facecolor='c',alpha=0.5,linewidth=0.0)
	plt.plot(np.arange(0.1,1,0.1), sham_avg_rt_fit,'c-.')
	#plt.plot(control_Qbins, control_avg_rt,'y', label='Control')
	#plt.fill_between(control_Qbins,control_avg_rt-control_sem_rt,control_avg_rt + control_sem_rt,facecolor='y',alpha=0.5,linewidth=0.0)
	plt.xlabel('Q-value LV Target')
	plt.ylabel('Reaction time (s)')
	#plt.xlim((0.2, 0.8))
	plt.ylim((0.05,0.3))
	plt.legend()
	plt.show()

	'''
	Make scatter plots of Q-value computed with just reward and Q-value computed with reward and stimulation paramter
	'''
	
	plt.figure()
	plt.scatter(stim_Qlow_block3, stim_Qlow_adjusted_block3,s=80, c='m',marker='o')
	plt.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),color='k',linestyle='--')
	plt.xlabel('Q-value LV Target')
	plt.ylabel('Adjusted Q-value LV Target')
	plt.legend()
	plt.show()


	
	mean_reaction_time_stim_fc_block1, std_reaction_time_stim_fc_block1, nbins_reaction_time_stim_fc_block1, reaction_time_hist_stim_fc_block1 = ReactionTimeHist(reaction_time_stim_fc_block1)
	mean_reaction_time_stim_fc_block3, std_reaction_time_stim_fc_block3, nbins_reaction_time_stim_fc_block3, reaction_time_hist_stim_fc_block3 = ReactionTimeHist(reaction_time_stim_fc_block3)
	mean_reaction_time_stim_ic_block1, std_reaction_time_stim_ic_block1, nbins_reaction_time_stim_ic_block1, reaction_time_hist_stim_ic_block1 = ReactionTimeHist(reaction_time_stim_ic_block1)
	mean_reaction_time_stim_ic_block3, std_reaction_time_stim_ic_block3, nbins_reaction_time_stim_ic_block3, reaction_time_hist_stim_ic_block3 = ReactionTimeHist(reaction_time_stim_ic_block3)

	mean_reaction_time_sham_fc_block1, std_reaction_time_sham_fc_block1, nbins_reaction_time_sham_fc_block1, reaction_time_hist_sham_fc_block1 = ReactionTimeHist(reaction_time_sham_fc_block1)
	mean_reaction_time_sham_fc_block3, std_reaction_time_sham_fc_block3, nbins_reaction_time_sham_fc_block3, reaction_time_hist_sham_fc_block3 = ReactionTimeHist(reaction_time_sham_fc_block3)
	mean_reaction_time_sham_ic_block1, std_reaction_time_sham_ic_block1, nbins_reaction_time_sham_ic_block1, reaction_time_hist_sham_ic_block1 = ReactionTimeHist(reaction_time_sham_ic_block1)
	mean_reaction_time_sham_ic_block3, std_reaction_time_sham_ic_block3, nbins_reaction_time_sham_ic_block3, reaction_time_hist_sham_ic_block3 = ReactionTimeHist(reaction_time_sham_ic_block3)

	mean_reaction_time_control_fc_block1, std_reaction_time_control_fc_block1, nbins_reaction_time_control_fc_block1, reaction_time_hist_control_fc_block1 = ReactionTimeHist(reaction_time_control_fc_block1)
	mean_reaction_time_control_fc_block3, std_reaction_time_control_fc_block3, nbins_reaction_time_control_fc_block3, reaction_time_hist_control_fc_block3 = ReactionTimeHist(reaction_time_control_fc_block3)
	mean_reaction_time_control_ic_block1, std_reaction_time_control_ic_block1, nbins_reaction_time_control_ic_block1, reaction_time_hist_control_ic_block1 = ReactionTimeHist(reaction_time_control_ic_block1)
	mean_reaction_time_control_ic_block3, std_reaction_time_control_ic_block3, nbins_reaction_time_control_ic_block3, reaction_time_hist_control_ic_block3 = ReactionTimeHist(reaction_time_control_ic_block3)

	'''
	Stats on reaction times: write function for one-way (M)ANOVA (f_oneway or MultiComparison) and post-hoc analysis (Tukey's HSD)
	'''
	# Compute significance
	t_stim,p_stim = stats.f_oneway(reaction_time_stim_fc_block1, reaction_time_stim_fc_block3, reaction_time_stim_ic_block1, reaction_time_stim_ic_block3)
	t_sham,p_sham = stats.f_oneway(reaction_time_sham_fc_block1, reaction_time_sham_fc_block3, reaction_time_sham_ic_block1, reaction_time_sham_ic_block3)
	t_control,p_control = stats.f_oneway(reaction_time_control_fc_block1, reaction_time_control_fc_block3, reaction_time_control_ic_block1, reaction_time_control_ic_block3)

	dta_stim = make_statsmodel_dtstructure([reaction_time_stim_fc_block1, reaction_time_stim_fc_block3, reaction_time_stim_ic_block1, reaction_time_stim_ic_block3], ['fcb1', 'fcb3', 'icb1', 'icb3'], 'ReactionTime')
	dta_sham = make_statsmodel_dtstructure([reaction_time_sham_fc_block1, reaction_time_sham_fc_block3, reaction_time_sham_ic_block1, reaction_time_sham_ic_block3], ['fcb1', 'fcb3', 'icb1', 'icb3'], 'ReactionTime')
	dta_control = make_statsmodel_dtstructure([reaction_time_control_fc_block1, reaction_time_control_fc_block3, reaction_time_control_ic_block1, reaction_time_control_ic_block3], ['fcb1', 'fcb3', 'icb1', 'icb3'], 'ReactionTime')
	
	res_stim = pairwise_tukeyhsd(dta_stim['ReactionTime'], dta_stim['Treatment'],alpha=0.05)
	res_sham = pairwise_tukeyhsd(dta_sham['ReactionTime'], dta_sham['Treatment'],alpha=0.05)
	res_control = pairwise_tukeyhsd(dta_control['ReactionTime'], dta_control['Treatment'],alpha=0.05)
	

	orig_stdout = sys.stdout
	f = file('Tukey HSD Results for Reaction Times - '+animal_id[1:] + '.txt', 'w')
	sys.stdout = f


	print "Stimulation Data - Reaction Time Comparisons"
	print res_stim

	print "Sham Data"
	print res_sham

	print "Control Data"
	print res_control

	sys.stdout = orig_stdout
	f.close()
	
	'''
	Plotting average reaction times
	'''

	mean_reaction_time_stim_block1 = [mean_reaction_time_stim_ic_block1, mean_reaction_time_stim_fc_block1]
	mean_reaction_time_stim_block3 = [mean_reaction_time_stim_ic_block3, mean_reaction_time_stim_fc_block3]
	mean_reaction_time_sham_block1 = [mean_reaction_time_sham_ic_block1, mean_reaction_time_sham_fc_block1]
	mean_reaction_time_sham_block3 = [mean_reaction_time_sham_ic_block3, mean_reaction_time_sham_fc_block3]
	mean_reaction_time_control_block1 = [mean_reaction_time_control_ic_block1, mean_reaction_time_control_fc_block1]
	mean_reaction_time_control_block3 = [mean_reaction_time_control_ic_block3, mean_reaction_time_control_fc_block3]

	sem_reaction_time_stim_block1 = [std_reaction_time_stim_ic_block1/np.sqrt(len(reaction_time_stim_ic_block1)), std_reaction_time_stim_fc_block1/np.sqrt(len(reaction_time_stim_fc_block1))]
	sem_reaction_time_stim_block3 = [std_reaction_time_stim_ic_block3/np.sqrt(len(reaction_time_stim_ic_block3)), std_reaction_time_stim_fc_block3/np.sqrt(len(reaction_time_stim_fc_block3))]
	sem_reaction_time_sham_block1 = [std_reaction_time_sham_ic_block1/np.sqrt(len(reaction_time_sham_ic_block1)), std_reaction_time_sham_fc_block1/np.sqrt(len(reaction_time_sham_fc_block1))]
	sem_reaction_time_sham_block3 = [std_reaction_time_sham_ic_block3/np.sqrt(len(reaction_time_sham_ic_block3)), std_reaction_time_sham_fc_block3/np.sqrt(len(reaction_time_sham_fc_block3))]
	sem_reaction_time_control_block1 = [std_reaction_time_control_ic_block1/np.sqrt(len(reaction_time_control_ic_block1)), std_reaction_time_control_fc_block1/np.sqrt(len(reaction_time_control_fc_block1))]
	sem_reaction_time_control_block3 = [std_reaction_time_control_ic_block3/np.sqrt(len(reaction_time_control_ic_block3)), std_reaction_time_control_fc_block3/np.sqrt(len(reaction_time_control_fc_block3))]
	
	
	width = float(0.4)
	plt.figure()
	plt.subplot(1,3,1)
	plt.bar(np.arange(1,3),mean_reaction_time_stim_block1, width/2, color='m',yerr=sem_reaction_time_stim_block1, label='Block 1')
	plt.bar(np.arange(1,3) + width/2,mean_reaction_time_stim_block3,width/2, color='c',yerr=sem_reaction_time_stim_block3,label='Block 3')
	plt.xticks(np.arange(1,3) + width/2, ('Instructed', 'Free-choice'))
	plt.legend()
	plt.ylim((0,0.25))
	plt.title('Stim')

	plt.subplot(1,3,2)
	plt.bar(np.arange(1,3),mean_reaction_time_sham_block1,width/2, color='m',yerr=sem_reaction_time_sham_block1, label='Block 1')
	plt.bar(np.arange(1,3) + width/2,mean_reaction_time_sham_block3,width/2, color='c',yerr=sem_reaction_time_sham_block3,label='Block 3')
	plt.xticks(np.arange(1,3) + width/2, ('Instructed', 'Free-choice'))
	plt.legend()
	plt.title('Sham')
	plt.ylim((0,0.25))
	
	plt.subplot(1,3,3)
	plt.bar(np.arange(1,3),mean_reaction_time_control_block1,width/2, color='m',yerr=sem_reaction_time_control_block1, label='Block 1')
	plt.bar(np.arange(1,3) + width/2,mean_reaction_time_control_block3,width/2, color='c',yerr=sem_reaction_time_control_block3,label='Block 3')
	plt.xticks(np.arange(1,3) + width/2, ('Instructed', 'Free-choice'))
	plt.legend()
	plt.title('Control')
	plt.ylim((0,0.25))
	
	plt.show()

	'''
	Exponential distribution fits for histograms
	'''
	stim_fc_block1_fit = np.exp(-nbins_reaction_time_stim_fc_block1/float(mean_reaction_time_stim_fc_block1))/float(mean_reaction_time_stim_fc_block1)
	stim_ic_block1_fit = np.exp(-nbins_reaction_time_stim_ic_block1/float(mean_reaction_time_stim_ic_block1))/float(mean_reaction_time_stim_ic_block1)
	stim_fc_block3_fit = np.exp(-nbins_reaction_time_stim_fc_block3/float(mean_reaction_time_stim_fc_block3))/float(mean_reaction_time_stim_fc_block3)
	stim_ic_block3_fit = np.exp(-nbins_reaction_time_stim_ic_block3/float(mean_reaction_time_stim_ic_block3))/float(mean_reaction_time_stim_ic_block3)

	sham_fc_block1_fit = np.exp(-nbins_reaction_time_sham_fc_block1/float(mean_reaction_time_sham_fc_block1))/float(mean_reaction_time_sham_fc_block1)
	sham_ic_block1_fit = np.exp(-nbins_reaction_time_sham_ic_block1/float(mean_reaction_time_sham_ic_block1))/float(mean_reaction_time_sham_ic_block1)
	sham_fc_block3_fit = np.exp(-nbins_reaction_time_sham_fc_block3/float(mean_reaction_time_sham_fc_block3))/float(mean_reaction_time_sham_fc_block3)
	sham_ic_block3_fit = np.exp(-nbins_reaction_time_sham_ic_block3/float(mean_reaction_time_sham_ic_block3))/float(mean_reaction_time_sham_ic_block3)

	control_fc_block1_fit = np.exp(-nbins_reaction_time_control_fc_block1/float(mean_reaction_time_control_fc_block1))/float(mean_reaction_time_control_fc_block1)
	control_ic_block1_fit = np.exp(-nbins_reaction_time_control_ic_block1/float(mean_reaction_time_control_ic_block1))/float(mean_reaction_time_control_ic_block1)
	control_fc_block3_fit = np.exp(-nbins_reaction_time_control_fc_block3/float(mean_reaction_time_control_fc_block3))/float(mean_reaction_time_control_fc_block3)
	control_ic_block3_fit = np.exp(-nbins_reaction_time_control_ic_block3/float(mean_reaction_time_control_ic_block3))/float(mean_reaction_time_control_ic_block3)

	
	'''
	Plotting reaction time histograms with exponential distribution fits
	'''

	
	width = float(0.05)
	plt.figure()
	plt.title('Stim')
	plt.subplot(2,2,1)
	plt.bar(nbins_reaction_time_stim_fc_block1, reaction_time_hist_stim_fc_block1, width,color='m')
	#plt.plot(nbins_reaction_time_stim_fc_block1, stim_fc_block1_fit,color='k', linestyle = '-')
	plt.title('Free-choice Trials - Block 1')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	plt.subplot(2,2,2)
	plt.bar(nbins_reaction_time_stim_ic_block1, reaction_time_hist_stim_ic_block1,width,color='m')
	#plt.plot(nbins_reaction_time_stim_ic_block1, stim_ic_block1_fit,'k')
	plt.title('Instructed Trials - Block 1')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))

	plt.subplot(2,2,3)
	plt.bar(nbins_reaction_time_stim_fc_block3, reaction_time_hist_stim_fc_block3, width,color='m')
	#plt.plot(nbins_reaction_time_stim_fc_block3, stim_fc_block3_fit,'k')
	plt.title('Free-choice Trials - Block 3')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	plt.subplot(2,2,4)
	plt.bar(nbins_reaction_time_stim_ic_block3, reaction_time_hist_stim_ic_block3,width,color='m')
	#plt.plot(nbins_reaction_time_stim_ic_block3, stim_ic_block3_fit,'k')
	plt.title('Instructed Trials - Block 3')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	
	plt.show()

	plt.figure()
	plt.title('Sham')
	plt.subplot(2,2,1)
	plt.bar(nbins_reaction_time_sham_fc_block1, reaction_time_hist_sham_fc_block1, width,color='c')
	#plt.plot(nbins_reaction_time_sham_fc_block1, sham_fc_block1_fit, 'k')
	plt.title('Free-choice Trials - Block 1')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	plt.subplot(2,2,2)
	plt.bar(nbins_reaction_time_sham_ic_block1, reaction_time_hist_sham_ic_block1,width,color='c')
	#plt.plot(nbins_reaction_time_sham_ic_block1, sham_ic_block1_fit, 'k')
	plt.title('Instructed Trials - Block 1')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))

	plt.subplot(2,2,3)
	plt.bar(nbins_reaction_time_sham_fc_block3, reaction_time_hist_sham_fc_block3, width,color='c')
	#plt.plot(nbins_reaction_time_sham_fc_block3, sham_fc_block3_fit, 'k')
	plt.title('Free-choice Trials - Block 3')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	plt.subplot(2,2,4)
	plt.bar(nbins_reaction_time_sham_ic_block3, reaction_time_hist_sham_ic_block3,width,color='c')
	#plt.plot(nbins_reaction_time_sham_ic_block3, sham_ic_block3_fit, 'k')
	plt.title('Instructed Trials - Block 3')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))

	plt.show()

	plt.figure()
	plt.title('Control')
	plt.subplot(2,2,1)
	plt.bar(nbins_reaction_time_control_fc_block1, reaction_time_hist_control_fc_block1, width,color='y')
	#plt.plot(nbins_reaction_time_control_fc_block1, control_fc_block1_fit, 'k')
	plt.title('Free-choice Trials - Block 1')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	plt.subplot(2,2,2)
	plt.bar(nbins_reaction_time_control_ic_block1, reaction_time_hist_control_ic_block1,width,color='y')
	#plt.plot(nbins_reaction_time_control_ic_block1, control_ic_block1_fit, 'k')
	plt.title('Instructed Trials - Block 1')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))

	plt.subplot(2,2,3)
	plt.bar(nbins_reaction_time_control_fc_block3, reaction_time_hist_control_fc_block3, width,color='y')
	#plt.plot(nbins_reaction_time_control_fc_block3, control_fc_block3_fit, 'k')
	plt.title('Free-choice Trials - Block 3')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))
	plt.subplot(2,2,4)
	plt.bar(nbins_reaction_time_control_ic_block3, reaction_time_hist_control_ic_block3,width,color='y')
	#plt.plot(nbins_reaction_time_control_ic_block3, control_ic_block3_fit, 'k')
	plt.title('Instructed Trials - Block 3')
	plt.xlabel('Reaction time (s)')
	plt.ylabel('Frequency')
	plt.xlim((0,2.5))
	plt.ylim((0,0.5))

	plt.show()

	return

def probabilisticRewardTaskPerformance_LVChoice(sham_days, stim_days, control_days, animal_id):

	num_sham_days = len(sham_days)
	num_stim_days = len(stim_days)
	num_control_days = len(control_days)

	sham_prob_lv = np.zeros(num_sham_days)
	stim_prob_lv = np.zeros(num_stim_days)
	control_prob_lv = np.zeros(num_control_days)

	for i in range(0,num_sham_days):
		name = sham_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name

		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)

		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
		target_freechoice_block3 = target3[free_choice_ind]
		reward_freechoice_block3 = reward3[free_choice_ind]

		sham_prob_lv[i] = np.sum(target_freechoice_block3 == 1)/float(len(free_choice_ind))

	for i in range(0,num_stim_days):
		name = stim_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name

		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)

		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
		target_freechoice_block3 = target3[free_choice_ind]
		reward_freechoice_block3 = reward3[free_choice_ind]

		stim_prob_lv[i] = np.sum(target_freechoice_block3 == 1)/float(len(free_choice_ind))

	for i in range(0,num_control_days):
		name = control_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name

		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)

		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
		target_freechoice_block3 = target3[free_choice_ind]
		reward_freechoice_block3 = reward3[free_choice_ind]

		control_prob_lv[i] = np.sum(target_freechoice_block3 == 1)/float(len(free_choice_ind))

	return sham_prob_lv, stim_prob_lv, control_prob_lv

def probabilisticRewardTaskPerformance_TargetsAndChoices(sham_days, animal_id):

	num_sham_days = len(sham_days)
	
	f_obs = np.array([0, 0])  # counting frequency of Lv and Hv choices respectively when target positions are LV = left, HV = right
	f_exp = np.array([0, 0])  # counting frequency of LV and HV choices respectively when target positions are Lv = right, Hv = left

	for i in range(0,num_sham_days):
		name = sham_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)

		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block1),2)))
		
		f_obs[0] += len([1 for ind in free_choice_ind if (target_side1[ind]==1)&(target1[ind]==1)])
		f_obs[1] += len([1 for ind in free_choice_ind if (target_side1[ind]==1)&(target1[ind]==2)])
		f_exp[0] += len([1 for ind in free_choice_ind if (target_side1[ind]==0)&(target1[ind]==1)])
		f_exp[1] += len([1 for ind in free_choice_ind if (target_side1[ind]==0)&(target1[ind]==2)])
		
		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
		
		f_obs[0] += len([1 for ind in free_choice_ind if (target_side3[ind]==1)&(target3[ind]==1)])
		f_obs[1] += len([1 for ind in free_choice_ind if (target_side3[ind]==1)&(target3[ind]==2)])
		f_exp[0] += len([1 for ind in free_choice_ind if (target_side3[ind]==0)&(target3[ind]==1)])
		f_exp[1] += len([1 for ind in free_choice_ind if (target_side3[ind]==0)&(target3[ind]==2)])
	
	stats.chisquare(f_obs, f_exp)

	return f_obs, f_exp


def probabilisticRewardTaskPerformance_RegressFreeChoice(days, animal_id):

	num_days = len(days)

	targ_choice = []
	targ_side = []
	stim_dist = []
	stim_reward = []
	stim_targ_side = []

	for i in range(0,num_days):
		name = days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name

		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)

		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
		instructed_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),1)))  # instructed trials in block 3 = stim trials
		target_freechoice_block3 = target3[free_choice_ind]
		reward_freechoice_block3 = reward3[free_choice_ind]

		
		first_stim_trial_ind = instructed_choice_ind[0]
		prev_stim_trial_ind = first_stim_trial_ind

		for i in range(len(instructed_or_freechoice_block3[prev_stim_trial_ind:])):
			# If instructed/stim trial, update this as the most recent stim trial
			if i in instructed_choice_ind:
				prev_stim_trial_ind = i
			# If free-choice trial, record the choice on this trial, the distance from the most recent stim trial, whether that stim trial was rewarded, what side the target was 
			# on during the stim trial, and what side the target was on this time.
			else:
				targ_choice.append(target3[i] - 1)  # subject one so that the variable is binary 
				targ_side.append(target_side3[i])
				stim_dist.append(i - prev_stim_trial_ind)
				stim_reward.append(reward3[prev_stim_trial_ind])
				stim_targ_side.append(target_side3[prev_stim_trial_ind])

	# Convert lists to arrays
	targ_choice = np.array(targ_choice)
	targ_side = np.array(targ_side)
	stim_dist = np.array(stim_dist)
	stim_reward = np.array(stim_reward)
	stim_targ_side = np.array(stim_targ_side)

	'''
	Oraganize data and regress with GLM 
	'''
	x = np.vstack((stim_dist, stim_reward, stim_targ_side))
	x = np.transpose(x)
	x = sm.add_constant(x,prepend='False')

	model_glm = sm.Logit(targ_choice,x)
	fit_glm = model_glm.fit()


	return fit_glm


def probabilisticRewardTaskPerformance_RegressFreeChoiceV2(days, animal_id):

	"""
	Regress choices as function of indicator variables of whether or not a stim trial occured in the previous five trials.
	"""


	num_days = len(days)

	targ_choice = []
	stim_dist1 = []
	stim_dist2 = []
	stim_dist3 = []
	stim_dist4 = []
	stim_dist5 = []

	for i in range(0,num_days):
		name = days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name

		reward1, target1, instructed_or_freechoice_block1, target_side1, reward3, target3, instructed_or_freechoice_block3, target_side3, stim_trials = FreeChoicePilotTask_Behavior(hdf_location)

		free_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),2)))
		instructed_choice_ind = np.ravel(np.nonzero(np.equal(np.ravel(instructed_or_freechoice_block3),1)))  # instructed trials in block 3 = stim trials
		target_freechoice_block3 = target3[free_choice_ind]
		reward_freechoice_block3 = reward3[free_choice_ind]

		
		first_stim_trial_ind = instructed_choice_ind[0]
		prev_stim_trial_ind = first_stim_trial_ind

		find_first_100 = np.nonzero((free_choice_ind < 100))

		for i in range(len(instructed_or_freechoice_block3[prev_stim_trial_ind:len(find_first_100)])):
			# If instructed/stim trial, update this as the most recent stim trial
			if i in instructed_choice_ind:
				prev_stim_trial_ind = i
			# If free-choice trial, record the choice on this trial, the distance from the most recent stim trial, whether that stim trial was rewarded, what side the target was 
			# on during the stim trial, and what side the target was on this time.
			else:
				targ_choice.append(target3[i] - 1)  # subject one so that the variable is binary
				stim_dist1.append((i - prev_stim_trial_ind)==1) 
				stim_dist2.append((i - prev_stim_trial_ind)==2)
				stim_dist3.append((i - prev_stim_trial_ind)==3)
				stim_dist4.append((i - prev_stim_trial_ind)==4)
				stim_dist5.append((i - prev_stim_trial_ind)==5)


	# Convert lists to arrays
	targ_choice = np.array(targ_choice)
	stim_dist1 = np.array(stim_dist1)
	stim_dist2 = np.array(stim_dist2)
	stim_dist3 = np.array(stim_dist3)
	stim_dist4 = np.array(stim_dist4)
	stim_dist5 = np.array(stim_dist5)
	'''
	Oraganize data and regress with GLM 
	'''
	x = np.vstack((stim_dist1, stim_dist2, stim_dist3))
	x = np.transpose(x)
	x = sm.add_constant(x,prepend='False')

	model_glm = sm.Logit(targ_choice,x)
	fit_glm = model_glm.fit()


	return fit_glm

def TwoWayMANOVA(x_dta, factor1, factor2, dv1, dv2):
	'''
	Inputs:
		- x_dta: input of the form pd.DataFrame(dta, columns=['reward', 'stim_targ_side', 'lv_choices', 'right_choices']), 
				 length is equal to number of days
		- factor1: string name for first independent variable
		- factor2: string name for second independent variable
		- dv1: string name for first dependent variable
		- dv2: string name for second dependent variable

	Outputs:

	'''
	factor1_levels = x_dta[factor1].unique()
	factor2_levels = x_dta[factor2].unique()
	g = len(factor1_levels)
	b = len(factor2_levels)
	n = np.sum(x_dta[factor1]==factor1_levels[0])/b
	p = 2

	total_num_obs = len(x_dta)

	factor1_level1_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[0]))
	factor1_level2_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[1]))
	factor2_level1_ind = np.ravel(np.nonzero(x_dta[factor2]==factor2_levels[0]))
	factor2_level2_ind = np.ravel(np.nonzero(x_dta[factor2]==factor2_levels[1]))

	x_mean_factor1_level1 = np.array([np.nanmean(x_dta[dv1][factor1_level1_ind]), np.nanmean(x_dta[dv2][factor1_level1_ind])]) # vector of mean values for dependent metrics averages over groups in factor 1
	x_mean_factor1_level2 = np.array([np.nanmean(x_dta[dv1][factor1_level2_ind]), np.nanmean(x_dta[dv2][factor1_level2_ind])])
	x_mean_factor2_level1 = np.array([np.nanmean(x_dta[dv1][factor2_level1_ind]), np.nanmean(x_dta[dv2][factor2_level1_ind])]) # vector of mean values for dependent metrics averages over groups in factor 1
	x_mean_factor2_level2 = np.array([np.nanmean(x_dta[dv1][factor2_level2_ind]), np.nanmean(x_dta[dv2][factor2_level2_ind])])
	
	x_mean_factor1 = [x_mean_factor1_level1, x_mean_factor1_level2]
	x_mean_factor2 = [x_mean_factor2_level1, x_mean_factor2_level2]

	x_mean = np.array([np.nanmean(x_dta[dv1]), np.nanmean(x_dta[dv2])])

	SSP_interaction = 0
	SSP_residual = 0
	SSP_corrected = 0

	for i in range(g):
		for j in range(b):

			factor1_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[i]))
			factor2_ind = np.ravel(np.nonzero(x_dta[factor2]==factor2_levels[j]))

			factor_ind = [ind for ind in range(total_num_obs) if (ind in factor1_ind)&(ind in factor2_ind)]

			x_mean_cross_terms = np.array([np.nanmean(x_dta[dv1][factor_ind]), np.nanmean(x_dta[dv2][factor_ind])]) 

			SSP_interaction += n*np.outer(x_mean_cross_terms - x_mean_factor1[i] - x_mean_factor2[j] + x_mean, (x_mean_cross_terms - x_mean_factor1[i] - x_mean_factor2[j] + x_mean))
			
			x_dv1_obs = [item for item in x_dta[dv1][factor_ind]]
			x_dv2_obs = [item for item in x_dta[dv2][factor_ind]]
			for r in range(n):
				x_obs = [x_dv1_obs[r], x_dv2_obs[r]]
				SSP_residual += np.outer((x_obs - x_mean_cross_terms), (x_obs - x_mean_cross_terms))
				SSP_corrected += np.outer((x_obs - x_mean),(x_obs - x_mean))

	SSP_factor1 = np.outer(b*n*(x_mean_factor1[0] - x_mean).T,(x_mean_factor1[0] - x_mean)) + np.dot(b*n*(x_mean_factor1[1] - x_mean),(x_mean_factor1[1] - x_mean))
	SSP_factor2 = np.outer(g*n*(x_mean_factor2[0] - x_mean).T,(x_mean_factor2[0] - x_mean)) + np.dot(g*n*(x_mean_factor2[1] - x_mean),(x_mean_factor2[1] - x_mean))

	df_factor1 = g - 1
	df_factor2 = b - 1
	df_interaction = (g - 1)*(b - 1)
	df_residual = g*b*(n - 1)
	df_corrected = g*b*n - 1
	
	F_factor1 = (SSP_factor1/float(df_factor1))/(SSP_residual/float(df_residual))
	F_factor2 = (SSP_factor2/float(df_factor2))/(SSP_residual/float(df_residual))
	F_interaction = (SSP_interaction/float(df_interaction))/(SSP_residual/float(df_residual))

	# Testing for presence of absernce of interaction effect: Wilks statistic
	wilks_lambda_interaction = np.linalg.det(SSP_residual)/float(np.linalg.det(SSP_residual+ SSP_interaction))
	chi_squared_comp_interaction = stats.chi2.cdf(0.05, p*df_factor1*df_factor2)

	reject_interaction = -(b*g*(n-1) - (p + 1 - (g-1)*(b-1))/2.)*np.log(wilks_lambda_interaction)
	interaction_test = (reject_interaction < chi_squared_comp_interaction) 
	print "Significant interaction term:", interaction_test

	wilks_lambda_factor1 = np.linalg.det(SSP_residual)/float(np.linalg.det(SSP_residual+ SSP_factor1))
	chi_squared_comp_factor1 = stats.chi2.cdf(0.05, p*df_factor1)
	reject_factor1 = -(b*g*(n-1) - (p + 1 - df_factor1)/2.)*np.log(wilks_lambda_factor1)
	factor1_test = (reject_factor1 < chi_squared_comp_factor1)

	print "Factor 1 Significant: ", factor1_test

	wilks_lambda_factor2 = np.linalg.det(SSP_residual)/float(np.linalg.det(SSP_residual + SSP_factor2))
	chi_squared_comp_factor2 = stats.chi2.cdf(0.05, p*df_factor2)
	reject_factor2 = -(b*g*(n-1) - (p + 1 - df_factor2)/2.)*np.log(wilks_lambda_factor2)
	factor2_test = (reject_factor2 < chi_squared_comp_factor2)

	print "Factor 2 Significant:", factor2_test
	
	return F_factor1, F_factor2, F_interaction, df_factor1, df_factor2, df_interaction, df_residual

def OneWayMANOVA(x_dta, factor1, dv1, dv2):
	'''
	Inputs:
		- x_dta: input of the form pd.DataFrame(dta, columns=['stim_condtion', 'alpha', 'beta']), 
				 length is equal to number of days
		- factor1: string name for first independent variable
		- dv1: string name for first dependent variable
		- dv2: string name for second dependent variable

	Outputs:

	'''
	factor1_levels = x_dta[factor1].unique()
	g = len(factor1_levels)
	n = np.sum(x_dta[factor1]==factor1_levels[0])/g
	p = 2

	total_num_obs = len(x_dta)

	factor1_level1_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[0]))
	factor1_level2_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[1]))
	factor1_level3_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[2]))

	x_mean_factor1_level1 = np.array([np.nanmean(x_dta[dv1][factor1_level1_ind]), np.nanmean(x_dta[dv2][factor1_level1_ind])]) # vector of mean values for dependent metrics averages over groups in factor 1
	x_mean_factor1_level2 = np.array([np.nanmean(x_dta[dv1][factor1_level2_ind]), np.nanmean(x_dta[dv2][factor1_level2_ind])])
	x_mean_factor1_level3 = np.array([np.nanmean(x_dta[dv1][factor1_level3_ind]), np.nanmean(x_dta[dv2][factor1_level3_ind])])

	x_mean_factor1 = [x_mean_factor1_level1, x_mean_factor1_level2, x_mean_factor1_level3]
	
	x_mean = np.array([np.nanmean(x_dta[dv1]), np.nanmean(x_dta[dv2])])

	SSP_interaction = 0
	SSP_residual = 0
	SSP_corrected = 0

	for i in range(g):

		factor1_ind = np.ravel(np.nonzero(x_dta[factor1]==factor1_levels[i]))
		factor_ind = [ind for ind in range(total_num_obs) if (ind in factor1_ind)]

		x_mean_cross_terms = np.array([np.nanmean(x_dta[dv1][factor_ind]), np.nanmean(x_dta[dv2][factor_ind])]) 
	
		x_dv1_obs = [item for item in x_dta[dv1][factor_ind]]
		x_dv2_obs = [item for item in x_dta[dv2][factor_ind]]
		
		for r in range(n):
			x_obs = [x_dv1_obs[r], x_dv2_obs[r]]
			SSP_residual += np.outer((x_obs - x_mean_cross_terms), (x_obs - x_mean_cross_terms))
			SSP_corrected += np.outer((x_obs - x_mean),(x_obs - x_mean))

	SSP_factor1 = np.outer(g*n*(x_mean_factor1[0] - x_mean).T,(x_mean_factor1[0] - x_mean)) + np.dot(g*n*(x_mean_factor1[1] - x_mean),(x_mean_factor1[1] - x_mean))
	
	df_factor1 = g - 1
	df_residual = g*(n - 1)
	df_corrected = g*n - 1
	
	F_factor1 = (SSP_factor1/float(df_factor1))/(SSP_residual/float(df_residual))
	
	return F_factor1, df_factor1, df_residual


def probabilisticRewardTaskPerformance_TrajectoryLengths(stim_days,sham_days,control_days, animal_id):

	num_stim_days = len(stim_days)
	num_sham_days = len(sham_days)
	num_control_days = len(control_days)

	trial1_stim = []
	target1_stim = []
	traj_length1_stim = []
	trial3_stim = []
	target3_stim = []
	traj_length3_stim = []

	trial1_sham = []
	target1_sham = []
	traj_length1_sham = []
	trial3_sham = []
	target3_sham = []
	traj_length3_sham = []

	trial1_control = []
	target1_control = []
	traj_length1_control = []
	trial3_control = []
	target3_control = []
	traj_length3_control = []

	for i in range(0,num_stim_days):
		name = stim_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		trial1, target1, traj_length1, trial3, target3, traj_length3 = FreeChoiceTask_PathLengths(hdf_location)

		trial1_stim.append(trial1)
		target1_stim.append(target1)
		traj_length1_stim.append(traj_length1)
		trial3_stim.append(trial3)
		target3_stim.append(target3)
		traj_length3_stim.append(traj_length3)

	for i in range(0,num_sham_days):
		name = sham_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		trial1, target1, traj_length1, trial3, target3, traj_length3 = FreeChoiceTask_PathLengths(hdf_location)

		trial1_sham.append(trial1)
		target1_sham.append(target1)
		traj_length1_sham.append(traj_length1)
		trial3_sham.append(trial3)
		target3_sham.append(target3)
		traj_length3_sham.append(traj_length3)
		
	for i in range(0,num_control_days):
		name = control_days[i]
		hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab'+ animal_id + '\hdf'+name
		trial1, target1, traj_length1, trial3, target3, traj_length3 = FreeChoiceTask_PathLengths(hdf_location)

		trial1_control.append(trial1)
		target1_control.append(target1)
		traj_length1_control.append(traj_length1)
		trial3_control.append(trial3)
		target3_control.append(target3)
		traj_length3_control.append(traj_length3)
	
	return trial3_stim, target3_stim, traj_length3_stim

trial3_stim, target3_stim, traj_length3_stim = probabilisticRewardTaskPerformance_TrajectoryLengths(sham_days, stim_days, control_days, '\Papa')

#papa_sham_prob_lv, papa_stim_prob_lv, papa_control_prob_lv = probabilisticRewardTaskPerformance_LVChoice(sham_days, stim_days, control_days, '\Papa')
#luigi_sham_prob_lv, luigi_stim_prob_lv, luigi_control_prob_lv = probabilisticRewardTaskPerformance_LVChoice(luigi_sham_days, luigi_stim_days, luigi_control_days, '\Luigi')

#dta_papa_lv = make_statsmodel_dtstructure([papa_sham_prob_lv, papa_stim_prob_lv, papa_control_prob_lv], ['Sham', 'Stim', 'Control'], 'lv_choices')
"""
dta = []
for data in papa_sham_prob_lv:
	dta += [(0, data)]
for data in papa_stim_prob_lv:
	dta += [(1,data)]
for data in papa_control_prob_lv:
	dta += [(2, data)]

dta_papa_lv = pd.DataFrame(dta, columns=['Stim_condition', 'lv_choices'])

dta = []
for data in luigi_sham_prob_lv:
	dta += [(0, data)]
for data in luigi_stim_prob_lv:
	dta += [(1,data)]
for data in luigi_control_prob_lv:
	dta += [(2, data)]

dta_luigi_lv = pd.DataFrame(dta, columns=['Stim_condition', 'lv_choices'])

formula = 'lv_choices ~ C(Stim_condition)'
model = ols(formula, dta_papa_lv).fit()
aov_table = anova_lm(model, typ=2)
print aov_table

model = ols(formula, dta_luigi_lv).fit()
aov_table = anova_lm(model, typ=2)
print aov_table


papa_hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Papa\hdf'
luigi_hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Luigi\hdf'
probabilisticRewardTaskPerformance_ConditionalProbabilities(sham_days, papa_hdf_location,1)
probabilisticRewardTaskPerformance_ConditionalProbabilities(luigi_sham_days,luigi_hdf_location,2)
plt.show()
"""
#probabilisticRewardTaskPerformance_TrialAligned(luigi_sham_days,luigi_stim_days,luigi_control_days)

#probabilisticRewardTaskPerformance_ReactionTimes(stim_days,sham_days,control_days, '\Papa')
#probabilisticRewardTaskPerformance_ReactionTimes(luigi_stim_days,luigi_sham_days,luigi_control_days,'\Luigi')

"""
papa_stim_rewarded, papa_stim_unrewarded, papa_sham_rewarded, papa_sham_unrewarded, papa_control_rewarded, papa_control_unrewarded = probabilisticRewardTaskPerformance_RewardAndStim(sham_days, stim_days, control_days, '\Papa')
luigi_stim_rewarded, luigi_stim_unrewarded, luigi_sham_rewarded, luigi_sham_unrewarded, luigi_control_rewarded, luigi_control_unrewarded = probabilisticRewardTaskPerformance_RewardAndStim(luigi_sham_days, luigi_stim_days, luigi_control_days, '\Luigi')

dta = []
for data in luigi_sham_rewarded:
	dta += [(0, data)]
for data in luigi_sham_unrewarded:
	dta += [(1, data)]

dta_luigi_sham = pd.DataFrame(dta, columns=['Reward', 'lv_choices'])

dta = []
for data in luigi_control_rewarded:
	dta += [(0, data)]
for data in luigi_control_unrewarded:
	dta += [(1, data)]

dta_luigi_control = pd.DataFrame(dta, columns=['Reward', 'lv_choices'])

formula = 'lv_choices ~ C(Reward)'
model = ols(formula, dta_luigi_sham).fit()
aov_table = anova_lm(model, typ=2)
print "Luigi - sham days"
print aov_table

formula = 'lv_choices ~ C(Reward)'
model = ols(formula, dta_luigi_control).fit()
aov_table = anova_lm(model, typ=2)
print "Luigi - control days"
print aov_table

dta = []
for data in papa_sham_rewarded:
	dta += [(0, data)]
for data in papa_sham_unrewarded:
	dta += [(1, data)]

dta_papa_sham = pd.DataFrame(dta, columns=['Reward', 'lv_choices'])

dta = []
for data in papa_control_rewarded:
	dta += [(0, data)]
for data in papa_control_unrewarded:
	dta += [(1, data)]

dta_papa_control = pd.DataFrame(dta, columns=['Reward', 'lv_choices'])

formula = 'lv_choices ~ C(Reward)'
model = ols(formula, dta_papa_sham).fit()
aov_table = anova_lm(model, typ=2)
print "Papa - sham days"
print aov_table

formula = 'lv_choices ~ C(Reward)'
model = ols(formula, dta_papa_control).fit()
aov_table = anova_lm(model, typ=2)
print "Papa - control days"
print aov_table


Two-way ANOVA: 
	Indendent variables (factors):
		- Stimulation condition: 0 = sham, 1 = stim, 2 = control
		- Reward: 0 = unrewarded stim trials, 1 = rewarded stim trials
	Dependent variable:
		- Probability of choosing a low-value target 

dta = []

for data in papa_sham_rewarded:
	dta += [(0,1, data)]
for data in papa_sham_unrewarded:
	dta += [(0,0,data)]
for data in papa_stim_rewarded:
	dta += [(1, 1, data)]
for data in papa_stim_unrewarded:
	dta += [(1,0, data)]
for data in papa_control_rewarded:
	dta += [(2,1,data)]
for data in papa_control_unrewarded:
	dta += [(2,0,data)]
		
dta_pd = pd.DataFrame(dta, columns=['Stim_condition', 'Reward', 'lv_choices'])

formula = 'lv_choices ~ C(Stim_condition) + C(Reward) + C(Stim_condition):C(Reward)'
model = ols(formula, dta).fit()
aov_table = anova_lm(model, typ=2)

print "Two-way ANOVA analysis: Papa"
print(aov_table)

dta = []

for data in luigi_sham_rewarded:
	dta += [(0,1, data)]
for data in luigi_sham_unrewarded:
	dta += [(0,0,data)]
for data in luigi_stim_rewarded:
	dta += [(1, 1, data)]
for data in luigi_stim_unrewarded:
	dta += [(1,0, data)]
for data in luigi_control_rewarded:
	dta += [(2,1,data)]
for data in luigi_control_unrewarded:
	dta += [(2,0,data)]
		
dta_pd = pd.DataFrame(dta, columns=['Stim_condition', 'Reward', 'lv_choices'])

formula = 'lv_choices ~ C(Stim_condition) + C(Reward) + C(Stim_condition):C(Reward)'
model = ols(formula, dta).fit()
aov_table = anova_lm(model, typ=2)

print "Two-way ANOVA analysis: Luigi"
print(aov_table)
"""

#fit_glm = probabilisticRewardTaskPerformance_RegressFreeChoiceV2(luigi_stim_days, '\Luigi')
#prob_low_given_lhs, prob_low_given_rhs = probabilisticRewardTaskPerformance_SideBiasAnalysis(luigi_stim_days, '\Luigi')
"""
dta_papa = probabilisticRewardTaskPerformance_RewardAndTargetSide(stim_days, '\Papa')
F_factor1, F_factor2, F_interaction, df_factor1, df_factor2, df_interaction, df_residual = TwoWayMANOVA(dta_papa, 'reward', 'stim_targ_side', 'lv_choices', 'right_choices')

print "Main effect reward: F = ", F_factor1, "df = ", df_factor1
print "Main effect of side: F = ", F_factor2, "df = ", df_factor2
print "Interaction effect: F = ", F_interaction, "df = ", df_interaction
print "df error:", df_residual


papa_sham_prob_trialaligned, papa_stim_prob_trialaligned, papa_control_prob_trialaligned = probabilisticRewardTaskPerformance_TrialAligned(sham_days, stim_days, control_days, '\Papa')
luigi_sham_prob_trialaligned, luigi_stim_prob_trialaligned, luigi_control_prob_trialaligned = probabilisticRewardTaskPerformance_TrialAligned(luigi_sham_days, luigi_stim_days, luigi_control_days, '\Luigi')

dta_trialaligned = []
total = 0
for data in papa_sham_prob_trialaligned:
	for i in range(5):
		dta_trialaligned += [(0, i, data[i])]
		total += 1
for data in papa_stim_prob_trialaligned:
	for i in range(5):
		dta_trialaligned += [(1, i, data[i])]
		total += 1
for data in papa_control_prob_trialaligned:
	for i in range(5):
		dta_trialaligned += [(2, i, data[i])]
		total += 1

dta_trialaligned = pd.DataFrame(dta_trialaligned, columns=['Stim_condition','Trial_pos', 'lv_choices'])

formula = 'lv_choices ~ C(Stim_condition) + C(Trial_pos) + C(Stim_condition):C(Trial_pos)'
model = ols(formula, dta_trialaligned).fit()
aov_table_papa = anova_lm(model, typ=2)

dta_trialaligned = []
total = 0
for data in luigi_sham_prob_trialaligned:
	for i in range(5):
		dta_trialaligned += [(0, i, data[i])]
		total += 1
for data in luigi_stim_prob_trialaligned:
	for i in range(5):
		dta_trialaligned += [(1, i, data[i])]
		total += 1
for data in luigi_control_prob_trialaligned:
	for i in range(5):
		dta_trialaligned += [(2, i, data[i])]
		total += 1

dta_trialaligned = pd.DataFrame(dta_trialaligned, columns=['Stim_condition','Trial_pos', 'lv_choices'])

formula = 'lv_choices ~ C(Stim_condition) + C(Trial_pos) + C(Stim_condition):C(Trial_pos)'
model = ols(formula, dta_trialaligned).fit()
aov_table_luigi = anova_lm(model, typ=2)





dta_papa_lv = make_statsmodel_dtstructure([papa_sham_prob_lv, papa_stim_prob_lv, papa_control_prob_lv], ['Sham', 'Stim', 'Control'], 'lv_choices')
dta_luigi_lv = make_statsmodel_dtstructure([luigi_sham_prob_lv, luigi_stim_prob_lv, luigi_control_prob_lv], ['Sham', 'Stim', 'Control'], 'lv_choices')

res_papa_05 = pairwise_tukeyhsd(dta_papa_lv['lv_choices'], dta_papa_lv['Treatment'], alpha= 0.05)
res_luigi_05 = pairwise_tukeyhsd(dta_luigi_lv['lv_choices'], dta_luigi_lv['Treatment'], alpha = 0.05)
res_papa_01 = pairwise_tukeyhsd(dta_papa_lv['lv_choices'], dta_papa_lv['Treatment'], alpha= 0.01)
res_luigi_01 = pairwise_tukeyhsd(dta_luigi_lv['lv_choices'], dta_luigi_lv['Treatment'], alpha = 0.01)
#formula = 'lv_choices ~ C(Stim_condition)'
#model = ols(formula, dta_lv).fit()
#aov_table = anova_lm(model, typ=2)

orig_stdout = sys.stdout
f = file('Prob of LV Choices Aligned to Trials - Fig 2BC.txt', 'w')
sys.stdout = f

print "Two-way ANOVA analysis: Papa"
print(aov_table_papa)

print "Two-way ANOVA analysis: Luigi"
print(aov_table_luigi)

sys.stdout = orig_stdout
f.close()

"""