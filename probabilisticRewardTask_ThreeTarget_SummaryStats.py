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
from DecisionMakingBehavior import loglikelihood_ThreeTargetTask_Qlearning_MVHV, ThreeTargetTask_Qlearning_MVHV, \
									loglikelihood_ThreeTargetTask_Qlearning_QAdditive_MVHV, \
									ThreeTargetTask_Qlearning_QAdditive_MVHV, loglikelihood_ThreeTargetTask_Qlearning_QMultiplicative_MVHV, \
									ThreeTargetTask_Qlearning_QMultiplicative_MVHV, loglikelihood_ThreeTargetTask_Qlearning_PAdditive_MVHV, \
									ThreeTargetTask_Qlearning_PAdditive_MVHV, loglikelihood_ThreeTargetTask_Qlearning_PMultiplicative_MVHV, \
									ThreeTargetTask_Qlearning_PMultiplicative_MVHV, ThreeTargetTask_Qlearning_QMultiplicative_MVLV, \
									loglikelihood_ThreeTargetTask_Qlearning_QMultiplicative_MVLV

mario_sham_days_all = [['\mari20161219_07_te2785.hdf'],
					['\mari20161220_05_te2795.hdf'],
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
mario_stim_days_all = [['\mari20161221_03_te2800.hdf'],
					['\mari20161222_03_te2803.hdf'],
					['\mari20170108_03_te2821.hdf'],
					['\mari20170125_10_te2924.hdf', '\mari20170125_12_te2926.hdf','\mari20170125_14_te2928.hdf'],
					['\mari20170130_12_te2960.hdf', '\mari20170130_13_te2961.hdf'],
					['\mari20170131_05_te2972.hdf', '\mari20170131_07_te2974.hdf'],
					['\mari20170201_03_te2977.hdf'],
					['\mari20170202_06_te2985.hdf','\mari20170202_08_te2987.hdf'],
					['\mari20170208_07_te3039.hdf', '\mari20170208_09_te3041.hdf', '\mari20170208_11_te3043.hdf'],
					['\mari20170209_03_te3047.hdf', '\mari20170209_05_te3049.hdf', '\mari20170209_08_te3052.hdf'],
					['\mari20170216_03_te3108.hdf', '\mari20170216_05_te3110.hdf', '\mari20170216_08_te3113.hdf', '\mari20170216_10_te3115.hdf'],
					['\mari20170219_14.hdf', '\mari20170219_16.hdf', '\mari20170219_18.hdf']]

# days with sorted-cell activity
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
					['\mari20170216_03_te3108.hdf', '\mari20170216_05_te3110.hdf', '\mari20170216_08_te3113.hdf', '\mari20170216_10_te3115.hdf'],
					['\mari20170219_14.hdf', '\mari20170219_16.hdf', '\mari20170219_18.hdf']]

data_dir = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Mario\spike_data'

alpha_true = 0.2
beta_true = 0.2
gamma_true = 0.5
Q_initial = [0.5,0.9]


def make_statsmodel_dtstructure(inputs, treatment_labels, independent_var_label):
	'''
	Method makes data structure like what stats library likes to work with. 
	Input:
		inputs: list of arrays that will be compared
		labels: list of labels for the arrays, should be equal to the number of arrays in inputs
	'''
	dta = []
	total = 0

	for ind, data in enumerate(inputs):
		#ind = inputs.index(data)

		for i in range(len(data)):
			dta += [(total, treatment_labels[ind], data[i])]
			total += 1

	dta = np.rec.array(dta, dtype=[('idx', '<i4'),
                                ('Treatment', '|S8'),
                                (independent_var_label, '<f4')])
	
	return dta

def SideBias(mario_sham_days):
	'''
	This method checks if there is a side bias for selecting the HV target when HV+MV targets are shown.
	Using data from the A' (third) block when probabilities are well-learned, we compute the likelihood of 
	selecting the HV target when HV+MV targets are shown given that the HV is presented on the left versus
	the right side of the screen. 

	Input:
	- mario_sham_days: list of hdf file locations for different sham sessions

	Output:
	- prob_HV_given_left: float indicating the conditional probability of selecting the HV target given
		that it was presented on the LHS of the screen
	- prob_HV_given_right: float indicating the conditional probability of selecting the HV target given
		that it was presented on the RHS of the screen
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

	'''
	Run one-way ANOVA on these probabilities to see if there is a significant difference
	'''
	f_val, p_val = stats.f_oneway(prob_HV_given_left, prob_HV_given_right)
	print "One-way ANOVA Results: Prob(HV) Given Side Presentation"
	print "F = ", f_val
	print "p = ", p_val

	print "P(Choose HV|HV on right) = %f" %(mean_given_right)
	print "P(Choose HV|HV on left) = %f" %(mean_given_left)
	plt.figure()
	plt.errorbar(mean_given_right, mean_given_left, yerr = sem_given_left, xerr = sem_given_right, fmt = '--o')
	plt.xlabel('P(Choose HV|HV on right)')
	plt.ylabel('P(Choose HV|HV on left)')
	plt.xlim((0.6,1.0))
	plt.ylim((0.6,1.0))
	plt.show()

	return prob_HV_given_left, prob_HV_given_right

def EarlyAndLateTargetSelection_HV(mario_sham_days):
	'''
	This method computes the rate of selecting the HV target when it's presented with the MV target under
	two conditions: (1) midway through Block A to assess early learning and (2) at the end of Block A' to 
	assess late learning.

	Input:
	- mario_sham_days: list of hdf file locations for different sham sessions

	Output:
	- prob_HV_early: array of length equal to number of days in mario_sham_days where values indicate the 
		probability of selecting the HV target during early learningwhen presented with the MV target on session i
	- prob_HV_late: array of length equal to number of days in mario_sham_days where values indicate the 
		probability of selecting the HV target during late learning when presented with the MV target on session i

	'''
	num_sham_days = len(mario_sham_days)
	prob_HV_early = np.zeros(num_sham_days)
	prob_HV_late = np.zeros(num_sham_days)
	for i, day in enumerate(mario_sham_days):
		print "Session %i of %i" %(i+1,num_sham_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()

		# Early learning
		counter = 0
		prob_chooseHV = 0
		for j in range(cb.num_trials_A/2):  				# only consider trials in Block A'
			if np.array_equal(targets_on[j],[0,1,1]):  			# only consider when H and M are shown
				choice = chosen_target[j]						# choice = 1 if MV, choice = 2 if HV
				prob_chooseHV += float((choice==2))
				counter += 1
		prob_HV_early[i] = float(prob_chooseHV)/counter

		# Late learning
		counter = 0
		prob_chooseHV = 0
		for j in range(cb.num_trials_A + cb.num_trials_B,len(chosen_target)):  				# only consider trials in Block A'
			if np.array_equal(targets_on[j],[0,1,1]):  			# only consider when H and M are shown
				choice = chosen_target[j]						# choice = 1 if MV, choice = 2 if HV
				prob_chooseHV += float((choice==2))
				counter += 1
		prob_HV_late[i] = float(prob_chooseHV)/counter

	mean_HV_early = np.nanmean(prob_HV_early)
	mean_HV_late = np.nanmean(prob_HV_late)
	sem_HV_early = np.nanstd(prob_HV_early)/np.sqrt(num_sham_days)
	sem_HV_late = np.nanstd(prob_HV_late)/np.sqrt(num_sham_days)

	print "Early learning HV selection is %f +/- %f" % (mean_HV_early, sem_HV_early)
	print "Late learning HV selection is %f +/- %f" % (mean_HV_late, sem_HV_late)

	return prob_HV_early, prob_HV_late

def MV_choice_behavior(mario_sham_days, mario_stim_days):
	'''
	This method computes the likelihood of the subject selecting the MV target under four different conditions:
	(1) when paired with the LV target on a stim day, (2) when paired with the HV target on a stim day,
	(3) when paired with the LV target on a sham day, and (4) when paired with the HV target on a sham day.
	Only the immediate trials following stimulation/sham stimulation are considered.

	Input:
	- mario_sham_days: list of hdf file locations for different sham sessions
	- mario_stim_days: list of hdf file locations for different stim sessions

	Output:
	- prob_MV_with_HV_stim: length n array, where n is the number of stim days, containing the probability of 
		selecting the MV target during Block A' when presented with the HV target on stim days
	- prob_MV_with_HV_sham: length m array, where m is the number of sham days, containing the probability of
		selecting the MV target during Block A' when presented with the HV target on sham days
	- prob_MV_with_LV_stim: length n array, containing the probability of selecting the MV target during Block A'
		when presented with the LV target on stim days
	- prob_MV_with_LV_sham: length m array, containing the probability of selecting the MV target during Block A'
		when presented with the LV target on sham days
	'''
	num_sham_days = len(mario_sham_days)
	num_stim_days = len(mario_stim_days)

	prob_MV_with_LV_sham = np.zeros(num_sham_days)
	prob_MV_with_HV_sham = np.zeros(num_sham_days)
	prob_MV_with_LV_stim = np.zeros(num_stim_days)
	prob_MV_with_HV_stim = np.zeros(num_stim_days)

	for i, day in enumerate(mario_sham_days):
		print "Sham session %i of %i" %(i+1,num_sham_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		targets_on_after, chosen_target, chosen_target_side,stim_reward, target_reward, stim_side, stim_trial_ind = cb.ChoicesAfterStimulation()

		
		counter_HV = 0
		counter_LV = 0
		prob_chooseMV_HV = 0
		prob_chooseMV_LV = 0
		for j in range(len(chosen_target)):  							# only consider trials in Block A'
			# when MV is paired with HV
			if np.array_equal(targets_on_after[j],[0,1,1]):  	# only consider when H and M are shown
				choice = chosen_target[j]						# choice = 1 if MV, choice = 2 if HV
				prob_chooseMV_HV += float((choice==1))
				counter_HV += 1
			# when MV is paired with LV
			if np.array_equal(targets_on_after[j],[1,0,1]):  	# only consider when L and M are shown
				choice = chosen_target[j]						# choice = 1 if MV, choice = 0 if LV
				prob_chooseMV_LV += float((choice==1))
				counter_LV += 1
		
		prob_MV_with_HV_sham[i] = float(prob_chooseMV_HV)/counter_HV	
		prob_MV_with_LV_sham[i] = float(prob_chooseMV_LV)/counter_LV

	for i, day in enumerate(mario_stim_days):
		print "Stim session %i of %i" %(i+1,num_stim_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		targets_on_after, chosen_target, chosen_target_side,stim_reward, target_reward, stim_side, stim_trial_ind = cb.ChoicesAfterStimulation()

		counter_HV = 0
		counter_LV = 0
		prob_chooseMV_HV = 0
		prob_chooseMV_LV = 0
		for j in range(len(chosen_target)):  							# only consider trials in Block A'
			# when MV is paired with HV
			if np.array_equal(targets_on_after[j],[0,1,1]):  	# only consider when H and M are shown
				choice = chosen_target[j]						# choice = 1 if MV, choice = 2 if HV
				prob_chooseMV_HV += float((choice==1))
				counter_HV += 1
			# when MV is paired with LV
			if np.array_equal(targets_on_after[j],[1,0,1]):  	# only consider when L and M are shown
				choice = chosen_target[j]						# choice = 1 if MV, choice = 0 if LV
				prob_chooseMV_LV += float((choice==1))
				counter_LV += 1
		
		prob_MV_with_HV_stim[i] = float(prob_chooseMV_HV)/counter_HV	
		prob_MV_with_LV_stim[i] = float(prob_chooseMV_LV)/counter_LV

	mean_MV_with_HV_sham = np.nanmean(prob_MV_with_HV_sham)
	mean_MV_with_LV_sham = np.nanmean(prob_MV_with_LV_sham)
	mean_MV_with_HV_stim = np.nanmean(prob_MV_with_HV_stim)
	mean_MV_with_LV_stim = np.nanmean(prob_MV_with_LV_stim)

	sem_MV_with_HV_sham = np.nanstd(prob_MV_with_HV_sham)/np.sqrt(num_sham_days)
	sem_MV_with_LV_sham = np.nanstd(prob_MV_with_LV_sham)/np.sqrt(num_sham_days)
	sem_MV_with_HV_stim = np.nanstd(prob_MV_with_HV_stim)/np.sqrt(num_stim_days)
	sem_MV_with_LV_stim = np.nanstd(prob_MV_with_LV_stim)/np.sqrt(num_stim_days)

	'''
	Run one-way ANOVA on these probabilities to see if there is a significant difference
	'''
	f_val, p_val = stats.f_oneway(prob_MV_with_HV_sham, prob_MV_with_LV_sham, prob_MV_with_HV_stim, prob_MV_with_LV_stim)
	print "One-way ANOVA Results: Prob(MV) Given Stim/Sham Conditions"
	print "F = ", f_val
	print "p = ", p_val

	# Do post-hoc LSD testing if ANOVA is significant
	if p_val < 0.051:
		dta_stim = make_statsmodel_dtstructure([prob_MV_with_HV_stim, prob_MV_with_HV_sham, 1.-prob_MV_with_LV_stim, 1.-prob_MV_with_LV_sham], ['Stim:MH', 'Sham:MH', 'Stim:ML', 'Sham:ML'], 'Prob')
		res_stim = pairwise_tukeyhsd(dta_stim['Prob'], dta_stim['Treatment'],alpha=0.01)

	print "Sham control: P(Choose MV) = %f" %(mean_MV_with_LV_sham)
	print "Sham: P(Choose MV) = %f" %(mean_MV_with_HV_sham)
	print "Stim control: P(Choose LV) = %f" %(mean_MV_with_LV_stim)
	print "Stim: P(Choose LV) = %f" %(mean_MV_with_HV_stim)

	print "Post-hoc Analysis - Pairwise Comparisons"
	print res_stim

	width = 0.35
	plt.figure()
	plt.bar(1, mean_MV_with_HV_stim, width/2., color = 'c', ecolor = 'k', yerr = sem_MV_with_HV_stim/2.)
	plt.bar(2, mean_MV_with_HV_sham, width/2., color = 'm', ecolor = 'k', yerr = sem_MV_with_HV_sham/2.)
	plt.bar(3, 1. - mean_MV_with_LV_stim, width/2., color = 'y', ecolor = 'k', yerr = sem_MV_with_LV_stim/2.)
	plt.bar(4, 1. - mean_MV_with_LV_sham, width/2., color = 'b', ecolor = 'k', yerr = sem_MV_with_LV_sham/2.)
	plt.ylabel('P(Choose Less-rewarding target)')
	xticklabels = ['Stim: MV vs HV', 'Sham: MV vs HV', 'Stim: MV vs LV', 'Sham: MV vs LV']
  	plt.xticks(range(1,5), xticklabels)
	plt.xlim((0.5,4.5))
	plt.ylim((0.0,0.35))
	plt.show()

	return prob_MV_with_HV_stim, prob_MV_with_HV_sham, prob_MV_with_LV_stim, prob_MV_with_LV_sham

def MV_aligned_choice_behavior(mario_sham_days, mario_stim_days, ntrials):
	'''
	This method computes the likelihood of the subject selecting the MV target at different latencies
	for the stimulation under four different conditions:
	(1) when paired with the LV target on a stim day, (2) when paired with the HV target on a stim day,
	(3) when paired with the LV target on a sham day, and (4) when paired with the HV target on a sham day.
	Only the immediate trials following stimulation/sham stimulation are considered.

	Input:
	- mario_sham_days: list of hdf file locations for different sham sessions
	- mario_stim_days: list of hdf file locations for different stim sessions

	Output:
	- prob_MV_with_HV_sham: n x ntrials array, where n is the number of sham sessions, where the [i,j]th entry
		contains the probability of selecting the MV when paired with the HV in session i on trials with 
		latency j from a trial with stimulation in Block A'
	- prob_MV_with_LV_sham: n x ntrials array, where n is the number of sham sessions, where the [i,j]th entry
		contains the probability of selecting the MV when paired with the LV in session i on trials with 
		latency j from a trial with stimulation in Block A'
	- prob_MV_with_HV_stim: m x ntrials array, where m is the number of stim sessions, where the [i,j]th entry
		contains the probability of selecting the MV when paired with the HV in session i on trials with 
		latency j from a trial with stimulation in Block A'
	- prob_MV_with_LV_stim: m x ntrials array, where m is the number of stim sessions, where the [i,j]th entry
		contains the probability of selecting the MV when paired with the LV in session i on trials with 
		latency j from a trial with stimulation in Block A'
	'''
	num_sham_days = len(mario_sham_days)
	num_stim_days = len(mario_stim_days)

	prob_MV_with_LV_sham = np.zeros((num_sham_days,ntrials))
	prob_MV_with_HV_sham = np.zeros((num_sham_days,ntrials))
	prob_MV_with_LV_stim = np.zeros((num_stim_days,ntrials))
	prob_MV_with_HV_stim = np.zeros((num_stim_days,ntrials))

	for k, day in enumerate(mario_sham_days):
		print "Sham session %i of %i" %(k+1,num_sham_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		targets_on_after, chosen_target, chosen_target_side,stim_reward, target_reward, stim_side, stim_trial_ind = cb.ChoicesAfterStimulation()
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()

		num_stim_trials = len(stim_trial_ind)
		aligned_mv_hv_choices = np.zeros((num_stim_trials,ntrials))  	# look at ntrials free-choice trials out of from stim trial
		aligned_mv_lv_choices = np.zeros((num_stim_trials,ntrials))
		number_mv_hv_choices = np.zeros(ntrials)						# counter to keep track of number of trials
		number_mv_lv_choices = np.zeros(ntrials)

		for i in range(0,num_stim_trials-1):
			ind_stim = stim_trial_ind[i]
			max_ind_out = int(np.min([ntrials,stim_trial_ind[i+1]-stim_trial_ind[i]-1]))
			if max_ind_out > 0:
				# indicator array of trials when MV+HV shown together
				mv_hv_choices = [np.array_equal([0,1,1],targets_on[stim_trial_ind[i]+j+1]) for j in range(0,max_ind_out)]
				# indicator array of these trials when MV is chosen over HV
				mv_hv_chosen = [(chosen_target[stim_trial_ind[i]+j+1]==1)&(mv_hv_choices[j]==1) for j in range(0,max_ind_out)]
				aligned_mv_hv_choices[i,0:max_ind_out] += mv_hv_chosen
				number_mv_hv_choices[0:max_ind_out] += mv_hv_choices

				# indicator array of trials when MV+LV shown together
				mv_lv_choices = [np.array_equal([1,0,1],targets_on[stim_trial_ind[i]+j+1]) for j in range(0,max_ind_out)]
				# indicator array of these trials when MV is chosen over LV
				mv_lv_chosen = [(chosen_target[stim_trial_ind[i]+j+1]==1)&(mv_lv_choices[j]==1) for j in range(0,max_ind_out)]
				aligned_mv_lv_choices[i,0:max_ind_out] += mv_lv_chosen
				number_mv_lv_choices[0:max_ind_out] += mv_lv_choices
		
		prob_MV_with_HV_sham[k,:] = np.nansum(aligned_mv_hv_choices,axis = 0)/number_mv_hv_choices
		prob_MV_with_LV_sham[k,:] = np.nansum(aligned_mv_lv_choices,axis = 0)/number_mv_lv_choices

	for k, day in enumerate(mario_stim_days):
		print "Stim session %i of %i" %(k+1,num_stim_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		targets_on_after, chosen_target, chosen_target_side,stim_reward, target_reward, stim_side, stim_trial_ind = cb.ChoicesAfterStimulation()
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()

		num_stim_trials = len(stim_trial_ind)
		aligned_mv_hv_choices = np.zeros((num_stim_trials,ntrials))  	# look at ntrials free-choice trials out of from stim trial
		aligned_mv_lv_choices = np.zeros((num_stim_trials,ntrials))
		number_mv_hv_choices = np.zeros(ntrials)						# counter to keep track of number of trials
		number_mv_lv_choices = np.zeros(ntrials)

		for i in range(0,num_stim_trials-1):
			ind_stim = stim_trial_ind[i]
			max_ind_out = int(np.min([ntrials,stim_trial_ind[i+1]-stim_trial_ind[i]-1]))
			if max_ind_out > 0:
				# indicator array of trials when MV+HV shown together
				mv_hv_choices = [np.array_equal([0,1,1],targets_on[stim_trial_ind[i]+j+1]) for j in range(0,max_ind_out)]
				# indicator array of these trials when MV is chosen over HV
				mv_hv_chosen = [(chosen_target[stim_trial_ind[i]+j+1]==1)&(mv_hv_choices[j]==1) for j in range(0,max_ind_out)]
				aligned_mv_hv_choices[i,0:max_ind_out] += mv_hv_chosen
				number_mv_hv_choices[0:max_ind_out] += mv_hv_choices

				# indicator array of trials when MV+LV shown together
				mv_lv_choices = [np.array_equal([1,0,1],targets_on[stim_trial_ind[i]+j+1]) for j in range(0,max_ind_out)]
				# indicator array of these trials when MV is chosen over LV
				mv_lv_chosen = [(chosen_target[stim_trial_ind[i]+j+1]==1)&(mv_lv_choices[j]==1) for j in range(0,max_ind_out)]
				aligned_mv_lv_choices[i,0:max_ind_out] += mv_lv_chosen
				number_mv_lv_choices[0:max_ind_out] += mv_lv_choices
		
		prob_MV_with_HV_stim[k,:] = np.nansum(aligned_mv_hv_choices,axis = 0)/number_mv_hv_choices
		prob_MV_with_LV_stim[k,:] = np.nansum(aligned_mv_lv_choices,axis = 0)/number_mv_lv_choices

	mean_MV_with_HV_sham = np.nanmean(prob_MV_with_HV_sham,axis = 0)
	mean_MV_with_LV_sham = np.nanmean(prob_MV_with_LV_sham,axis = 0)
	mean_MV_with_HV_stim = np.nanmean(prob_MV_with_HV_stim,axis = 0)
	mean_MV_with_LV_stim = np.nanmean(prob_MV_with_LV_stim,axis = 0)
	sem_MV_with_HV_sham = np.nanstd(prob_MV_with_HV_sham,axis = 0)/np.sqrt(num_sham_days)
	sem_MV_with_LV_sham = np.nanstd(prob_MV_with_LV_sham,axis = 0)/np.sqrt(num_sham_days)
	sem_MV_with_HV_stim = np.nanstd(prob_MV_with_HV_stim,axis = 0)/np.sqrt(num_stim_days)
	sem_MV_with_LV_stim = np.nanstd(prob_MV_with_LV_stim,axis = 0)/np.sqrt(num_stim_days)

	width = float(0.35)
	ind = np.arange(0,ntrials)
	# Get linear fit to total prob
	m_stim_hv,b_stim_hv = np.polyfit(ind, mean_MV_with_HV_stim, 1)
	m_stim_lv,b_stim_lv = np.polyfit(ind, 1 - mean_MV_with_LV_stim, 1)
	m_sham_hv,b_sham_hv = np.polyfit(ind, mean_MV_with_HV_sham, 1)
	m_sham_lv,b_sham_lv = np.polyfit(ind, 1 - mean_MV_with_LV_sham, 1)

	plt.figure()
	plt.bar(ind, mean_MV_with_HV_stim, width/2, color = 'c', ecolor = 'k', yerr = sem_MV_with_HV_stim/2.)
	plt.plot(ind,m_stim_hv*ind + b_stim_hv,'c--')
	plt.bar(ind + width/2, mean_MV_with_HV_sham, width/2, color = 'm',ecolor = 'k', yerr = sem_MV_with_HV_sham/2.)
	plt.plot(ind+width/2, m_sham_hv*ind + b_sham_hv,'m--')
	plt.bar(ind + width, 1. - mean_MV_with_LV_stim, width/2, color = 'y',ecolor = 'k', yerr = sem_MV_with_LV_stim/2.)
	plt.plot(ind+width, m_stim_lv*ind + b_stim_lv,'y--')
	#plt.bar(ind + width*1.5, 1. - mean_MV_with_LV_sham, width/2, color = 'b',ecolor = 'k', yerr = sem_MV_with_LV_sham/2.)
	#plt.plot(ind+width*1.5, m_sham_lv*ind + b_sham_lv,'b--')
	plt.ylabel('P(Choose Lesser-Value Target)')
	plt.title('Target Selection')
	xticklabels = [str(num + 1) for num in ind]
	plt.xticks(ind + width/2, xticklabels)
	plt.xlabel('Trials post-stimulation')
	#plt.ylim([0.0,0.32])
	plt.xlim([-0.1,ntrials + 0.4])
	
	plt.show()

	'''
	Do Two-Way ANOVA:
	- independent vars: (1) stim/pair condition [4 levels] (2) Trial latency [ntrials levels]
	'''
	dta_trialaligned = []
	for data in prob_MV_with_HV_sham:
		for i in range(ntrials):
			dta_trialaligned += [(0, i, data[i])]
	for data in prob_MV_with_HV_stim:
		for i in range(ntrials):
			dta_trialaligned += [(1, i, data[i])]
	for data in prob_MV_with_LV_sham:
		for i in range(ntrials):
			dta_trialaligned += [(2, i, 1 - data[i])]
	for data in prob_MV_with_LV_stim:
		for i in range(ntrials):
			dta_trialaligned += [(3, i, 1 - data[i])]
	

	dta_trialaligned = pd.DataFrame(dta_trialaligned, columns=['Stim_condition','Trial_pos', 'lv_choices'])
	# need to get rid of NaNs for code to run
	df2 = dta_trialaligned.dropna(how="any")
	print dta_trialaligned
	formula = 'lv_choices ~ C(Stim_condition) + C(Trial_pos) + C(Stim_condition):C(Trial_pos)'
	#model = ols(formula, dta_trialaligned).fit()
	model = ols(formula, df2).fit()
	aov_table = anova_lm(model, typ=2)
	print aov_table

	return prob_MV_with_HV_sham, prob_MV_with_LV_sham, prob_MV_with_HV_stim, prob_MV_with_LV_stim

def TargetLocationAndReward(mario_days, regular_or_control):
	'''
	This method computes the likelihood of selecting the MV in free-choice trials based on what side it 
	was presented on during the free-choice trial and the prior stimulation trial, and whether or not 
	the stimulation trial was rewarded. Only trials immediately following a stimulation trial is considered.
	The goal is to assess whether the effects of stimulation are lateralized, as well as whether reward 
	modulates these effects or not.

	Input: 
	- mario_days: list of hdf file locations for different sessions
	- regular_or_control: Boolean, indicating if choices are only considered on trials where MV and HV are shown (= 1)
		or if choices are only considered on trials where MV and LV are shown (= 2)
	'''
	num_days = len(mario_days)
	dta = []

	prob_left_given_left = np.zeros(num_days)
	prob_right_given_left = np.zeros(num_days)
	prob_left_given_right = np.zeros(num_days)
	prob_right_given_right = np.zeros(num_days)

	prob_mv_given_rewarded = np.zeros(num_days)
	prob_mv_given_unrewarded = np.zeros(num_days)
	
	for k, day in enumerate(mario_days):
		print "Session %i of %i" %(k+1,num_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		targets_on_after, chosen_target, chosen_target_side, stim_reward, target_reward, stim_side, stim_trial_ind = cb.ChoicesAfterStimulation()
		
		
		if regular_or_control==1:
			# find trials when MV and HV are shown
			trial_inds = [j for j in range(len(chosen_target)) if np.array_equal(targets_on_after[j],[0,1,1])]
		elif regular_or_control==2:
			# or find trials when MV and LV are shown
			trial_inds = [j for j in range(len(chosen_target)) if np.array_equal(targets_on_after[j],[1,0,1])]

		# Compute conditional probabilities of selecting the MV target on the trial following stimulation
		rewarded_left = [(2 - chosen_target[i]) for i in trial_inds if (stim_side[i] == 1)&(stim_reward[i] == 1)]
		rewarded_right = [(2 - chosen_target[i]) for i in trial_inds if (stim_side[i] == -1)&(stim_reward[i] == 1)]
		unrewarded_left = [(2 - chosen_target[i]) for i in trial_inds if (stim_side[i] == 1)&(stim_reward[i] == 0)]
		unrewarded_right = [(2 - chosen_target[i]) for i in trial_inds if (stim_side[i] == -1)&(stim_reward[i] == 0)]

		prob_mv_stim_rewarded_left = np.sum(rewarded_left)/float(len(rewarded_left) + (len(rewarded_left)==0))
		prob_mv_stim_rewarded_right = np.sum(rewarded_right)/float(len(rewarded_right) + (len(rewarded_right)==0))
		prob_mv_stim_unrewarded_left = np.sum(unrewarded_left)/float(len(unrewarded_left) + (len(unrewarded_left)==0))
		prob_mv_stim_unrewarded_right = np.sum(unrewarded_right)/float(len(unrewarded_right) + (len(unrewarded_right)==0))

		# Compute conditional probabilities of selecting the a target on the RHS on the trial following stimulation
		rewarded_left = [(chosen_target_side[i]==-1) for i in trial_inds if (stim_side[i] == 1)&(stim_reward[i] == 1)]
		rewarded_right = [(chosen_target_side[i]==-1) for i in trial_inds if (stim_side[i] == -1)&(stim_reward[i] == 1)]
		unrewarded_left = [(chosen_target_side[i]==-1) for i in trial_inds if (stim_side[i] == 1)&(stim_reward[i] == 0)]
		unrewarded_right = [(chosen_target_side[i]==-1) for i in trial_inds if (stim_side[i] == -1)&(stim_reward[i] == 0)]

		prob_right_stim_rewarded_left = np.sum(rewarded_left)/float(len(rewarded_left) + (len(rewarded_left)==0))
		prob_right_stim_rewarded_right = np.sum(rewarded_right)/float(len(rewarded_right) + (len(rewarded_right)==0))
		prob_right_stim_unrewarded_left = np.sum(unrewarded_left)/float(len(unrewarded_left) + (len(unrewarded_left)==0))
		prob_right_stim_unrewarded_right = np.sum(unrewarded_right)/float(len(unrewarded_right) + (len(unrewarded_right)==0))
		
		# Make DataFrame structure entries to do statistical analysis
		dta += [(1,1,prob_mv_stim_rewarded_left,prob_right_stim_rewarded_left)]
		dta += [(1,2,prob_mv_stim_rewarded_right,prob_right_stim_rewarded_right)]
		dta += [(0,1,prob_mv_stim_unrewarded_left,prob_right_stim_unrewarded_left)]
		dta += [(0,2,prob_mv_stim_unrewarded_right,prob_right_stim_unrewarded_right)]

		# Compute conditional probabilities of stim target side and chosen target side pairs
		# P(choose side | stim side)
		prob_stim_rewarded = np.sum(stim_reward)/float(len(stim_reward))
		prob_stim_unrewarded = 1 - prob_stim_rewarded

		prob_stim_left = np.sum(stim_side == 1)/float(len(stim_side))
		prob_stim_right = np.sum(stim_side == -1)/float(len(stim_side))
		print "P(stim left) = ", prob_stim_left
		print "P(stim right) = ", prob_stim_right

		prob_right_given_left[k] = prob_right_stim_unrewarded_left*prob_stim_unrewarded + prob_right_stim_rewarded_left*prob_stim_rewarded
		prob_right_given_right[k] = prob_right_stim_unrewarded_right*prob_stim_unrewarded + prob_right_stim_rewarded_right*prob_stim_rewarded

		left_given_left = [(chosen_target_side[i]==1) for i in trial_inds if (stim_side[i] == 1)]
		left_given_right = [(chosen_target_side[i]==1) for i in trial_inds if (stim_side[i] == -1)]
		prob_left_given_left[k] = np.sum(left_given_left)/float(len(left_given_left) + (len(left_given_left)==1))
		prob_left_given_right[k] = np.sum(left_given_right)/float(len(left_given_right) + (len(left_given_right)==1))
		
		# Compute joint probabilities
		prob_right_given_left[k] = prob_right_given_left[k]*prob_stim_left
		prob_right_given_right[k] = prob_right_given_right[k]*prob_stim_right
		prob_left_given_left[k] = prob_left_given_left[k]*prob_stim_left
		prob_left_given_right[k] = prob_left_given_right[k]*prob_stim_right

		# Compute joint probabilities of selecting the MV target given that the stim trial was rewarded/unrewarded
		

		prob_mv_given_rewarded[k] = prob_mv_stim_rewarded_right*prob_stim_right + prob_mv_stim_rewarded_left*prob_stim_left
		prob_mv_given_unrewarded[k] = prob_mv_stim_unrewarded_right*prob_stim_right + prob_mv_stim_unrewarded_left*prob_stim_left


	# Make DataFrame structure to do statistical analysis
	dta = pd.DataFrame(dta, columns=['reward', 'stim_targ_side', 'mv_choices', 'right_choices'])

	# Make plots
	mean_right_given_right = np.nanmean(prob_right_given_right)
	mean_right_given_left = np.nanmean(prob_right_given_left)
	mean_left_given_right = np.nanmean(prob_left_given_right)
	mean_left_given_left = np.nanmean(prob_left_given_left)

	sem_right_given_right = np.nanmean(prob_right_given_right)/np.sqrt(len(prob_right_given_right))
	sem_right_given_left = np.nanmean(prob_right_given_left)/np.sqrt(len(prob_right_given_left))
	sem_left_given_right = np.nanmean(prob_left_given_right)/np.sqrt(len(prob_left_given_right))
	sem_left_given_left = np.nanmean(prob_left_given_left)/np.sqrt(len(prob_left_given_left))


	print "Sum = ", mean_right_given_right + mean_right_given_left + mean_left_given_right + mean_left_given_left

	sem_right_given_right = np.nanstd(prob_right_given_right)/np.sqrt(num_days)
	sem_right_given_left = np.nanstd(prob_right_given_left)/np.sqrt(num_days)
	sem_left_given_right = np.nanstd(prob_left_given_right)/np.sqrt(num_days)
	sem_left_given_left = np.nanstd(prob_left_given_left)/np.sqrt(num_days)

	width = float(0.35)

	plt.figure()
	plt.barh(1, 0 - mean_left_given_left, width/2, color = 'c', ecolor = 'k', xerr = sem_left_given_left/2.)
	plt.barh(1, mean_right_given_right, width/2, color = 'g', ecolor = 'k', xerr = sem_right_given_right/2.)
	plt.barh(0, 0 - mean_right_given_left, width/2, color = 'm', ecolor = 'k', xerr = sem_right_given_left/2.)
	plt.barh(0, mean_left_given_right, width/2, color = 'y', ecolor = 'k', xerr = sem_left_given_right/2.)
	plt.ylabel('P(Choose LV Target)')
	plt.title('Fraction of Choices')
	yticklabels = ['Opposite', 'Same']
	plt.yticks(range(2), yticklabels)
	plt.ylabel('Chosen Target Side After Stim')
	plt.xlabel('Target Side During Stimulation')
	#plt.ylim([-0.35,0.35])
	#plt.xlim([-0.1,ntrials + 0.4])
	#plt.legend()
	
	plt.show()

	if regular_or_control==1:
		mean_mv_given_rewarded = np.nanmean(prob_mv_given_rewarded)
		mean_mv_given_unrewarded = np.nanmean(prob_mv_given_unrewarded)
		sem_mv_given_rewarded = np.nanstd(prob_mv_given_rewarded)/np.sqrt(num_days)
		sem_mv_given_unrewarded = np.nanstd(prob_mv_given_unrewarded)/np.sqrt(num_days)
	elif regular_or_control==2:
		mean_mv_given_rewarded = 1 - np.nanmean(prob_mv_given_rewarded)
		mean_mv_given_unrewarded = 1 - np.nanmean(prob_mv_given_unrewarded)
		sem_mv_given_rewarded = np.nanstd(prob_mv_given_rewarded)/np.sqrt(num_days)
		sem_mv_given_unrewarded = np.nanstd(prob_mv_given_unrewarded)/np.sqrt(num_days)

	plt.figure()
	plt.errorbar(range(2), [mean_mv_given_rewarded, mean_mv_given_unrewarded], color = 'k', ecolor = 'k', yerr = [sem_mv_given_rewarded/2., sem_mv_given_unrewarded/2.])
	xticklabels = ['rewarded', 'unrewarded']
	plt.xticks(range(2), xticklabels)
	plt.ylabel('Fraction of lower-value choices')
	plt.xlim((-0.3, 1.3))
	plt.ylim((0,0.4))
	plt.show()
	

	return dta, prob_left_given_right, prob_left_given_left, prob_right_given_right, prob_right_given_left, prob_mv_given_rewarded, prob_mv_given_unrewarded


def RLModel_ThreeTarget_MVHV(mario_days):
	'''
	This method computes the RL model fit for the various proposed models: (1) regular, (2) additive Q paramter,
	(3) multiplicative Q parameter, (4) additive P parameter, and (5) multiplicative P parameter. Only trials from
	Block A' where MV and HV are presented are used.

	Inputs:
	- mario_days: list of hdf file locations for different sessions

	'''
	# Initialize parameters
	num_days = len(mario_days)
	alpha_mat = np.zeros((num_days, 5))		# each column stores alpha values for each of the 5 RL models considered
	beta_mat = np.zeros((num_days, 5))		# each column stores beta values for each of the 5 RL models considered
	lambda_mat = np.zeros((num_days, 4)) 	# each column stores lambda value for the 4 adjusted RL models
	BIC_mat = np.zeros((num_days, 5))		# each column stores the BIC value for the 5 models considered

	for k, day in enumerate(mario_days):
		print "Session %i of %i" %(k+1,num_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]

		# 1. Load behavior data and pull out trial indices for the designated trial case
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		
		# 2. Get Q-values, chosen targets, and rewards
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	
		# 3. Model 1: Regular model
		# Find ML fit of alpha and beta
		
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_MVHV(*args)
		result = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:]), bounds=[(0,1),(1,None)])
		alpha_ml, beta_ml = result["x"]
		# RL model fit for Q values
		Q_mid, Q_high, prob_choice_mid, prob_choice_high, max_log_likelihood, counter = ThreeTargetTask_Qlearning_MVHV([alpha_ml, beta_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:])

		print counter, len(Q_mid)
		# Compute BIC 
		BIC = -2*max_log_likelihood + len(result["x"])*np.log(len(Q_mid))

		# Store values
		alpha_mat[k,0] = alpha_ml
		beta_mat[k,0] = beta_ml
		BIC_mat[k,0] = BIC

		# 4. Model 2: Additive Q parameter
		# Find ML fit of alpha, beta, and gamma
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_QAdditive_MVHV(*args)
		result = op.minimize(nll, [alpha_true, beta_true, gamma_true], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:]), bounds=[(0,1),(1,None),(0.1,None)])
		alpha_ml, beta_ml, lambda_ml = result["x"]
		# RL model fit for Q values
		Q_mid, Q_high, prob_choice_mid, prob_choice_high, max_log_likelihood, counter = ThreeTargetTask_Qlearning_QAdditive_MVHV([alpha_ml, beta_ml, lambda_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:])
		
		# Compute BIC 
		BIC = -2*max_log_likelihood + len(result["x"])*np.log(len(Q_mid))

		# Store values
		alpha_mat[k,1] = alpha_ml
		beta_mat[k,1] = beta_ml
		lambda_mat[k,0] = lambda_ml
		BIC_mat[k,1] = BIC

		# 5. Model 3: Multiplicative Q parameter
		# Find ML fit of alpha, beta, and gamma
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_QMultiplicative_MVHV(*args)
		result = op.minimize(nll, [alpha_true, beta_true, gamma_true], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:]), bounds=[(0,1),(1,None),(0.1,None)])
		alpha_ml, beta_ml, lambda_ml = result["x"]
		# RL model fit for Q values
		Q_mid, Q_high, prob_choice_mid, prob_choice_high, max_log_likelihood, counter = ThreeTargetTask_Qlearning_QMultiplicative_MVHV([alpha_ml, beta_ml, lambda_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:])

		# Compute BIC 
		BIC = -2*max_log_likelihood + len(result["x"])*np.log(len(Q_mid))

		# Store values
		alpha_mat[k,2] = alpha_ml
		beta_mat[k,2] = beta_ml
		lambda_mat[k,1] = lambda_ml
		BIC_mat[k,2] = BIC

		# 6. Model 4: Additive P parameter
		# Find ML fit of alpha, beta, and gamma
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_PAdditive_MVHV(*args)
		result = op.minimize(nll, [alpha_true, beta_true, gamma_true], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:]), bounds=[(0,1),(1,None),(0.1,None)])
		alpha_ml, beta_ml, lambda_ml = result["x"]
		# RL model fit for Q values
		Q_mid, Q_high, prob_choice_mid, prob_choice_high, max_log_likelihood, counter = ThreeTargetTask_Qlearning_PAdditive_MVHV([alpha_ml, beta_ml, lambda_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:])

		# Compute BIC 
		BIC = -2*max_log_likelihood + len(result["x"])*np.log(len(Q_mid))

		# Store values
		alpha_mat[k,3] = alpha_ml
		beta_mat[k,3] = beta_ml
		lambda_mat[k,2] = lambda_ml
		BIC_mat[k,3] = BIC

		# 7. Model 5: Multiplicative P parameter
		# Find ML fit of alpha, beta, and gamma
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_PMultiplicative_MVHV(*args)
		result = op.minimize(nll, [alpha_true, beta_true, gamma_true], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:]), bounds=[(0,1),(1,None),(0.1,None)])
		alpha_ml, beta_ml, lambda_ml = result["x"]
		# RL model fit for Q values
		Q_mid, Q_high, prob_choice_mid, prob_choice_high, max_log_likelihood, counter = ThreeTargetTask_Qlearning_PMultiplicative_MVHV([alpha_ml, beta_ml, lambda_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:])

		# Compute BIC 
		BIC = -2*max_log_likelihood + len(result["x"])*np.log(len(Q_mid))

		# Store values
		alpha_mat[k,4] = alpha_ml
		beta_mat[k,4] = beta_ml
		lambda_mat[k,3] = lambda_ml
		BIC_mat[k,4] = BIC

	return alpha_mat, beta_mat, lambda_mat, BIC_mat

def RLModel_ThreeTarget_MVLV(mario_days):
	'''
	This method computes the RL model fit for the various proposed models: (1) regular, (2) additive Q paramter,
	(3) multiplicative Q parameter, (4) additive P parameter, and (5) multiplicative P parameter. Only trials from
	Block A' where MV and HV are presented are used.

	Inputs:
	- mario_days: list of hdf file locations for different sessions

	'''
	# Initialize parameters
	num_days = len(mario_days)
	alpha_mat = np.zeros((num_days, 1))		# each column stores alpha values for each of the 5 RL models considered
	beta_mat = np.zeros((num_days, 1))		# each column stores beta values for each of the 5 RL models considered
	lambda_mat = np.zeros((num_days, 1)) 	# each column stores lambda value for the 4 adjusted RL models
	BIC_mat = np.zeros((num_days, 1))		# each column stores the BIC value for the 5 models considered

	for k, day in enumerate(mario_days):
		print "Session %i of %i" %(k+1,num_days)
		num_hdf_files = len(day)
		hdf_files = [(data_dir + hdf) for hdf in day]

		# 1. Load behavior data and pull out trial indices for the designated trial case
		cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
		
		# 2. Get Q-values, chosen targets, and rewards
		targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()

		# 3. Model 3: Multiplicative Q parameter
		# Find ML fit of alpha, beta, and gamma
		nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning_QMultiplicative_MVLV(*args)
		result = op.minimize(nll, [alpha_true, beta_true, gamma_true], args=(Q_initial, chosen_target[250:], rewards[250:], targets_on[250:]), bounds=[(0,1),(1,10),(0.1,None)])
		alpha_ml, beta_ml, lambda_ml = result["x"]
		# RL model fit for Q values
		Q_mid, Q_high, prob_choice_mid, prob_choice_high, max_log_likelihood, counter = ThreeTargetTask_Qlearning_QMultiplicative_MVLV([alpha_ml, beta_ml, lambda_ml], Q_initial, chosen_target[250:], rewards[250:], targets_on[250:])

		# Compute BIC 
		BIC = -2*max_log_likelihood + len(result["x"])*np.log(len(Q_mid))

		# Store values
		alpha_mat[k,0] = alpha_ml
		beta_mat[k,0] = beta_ml
		lambda_mat[k,0] = lambda_ml
		BIC_mat[k,0] = BIC

	return alpha_mat, beta_mat, lambda_mat, BIC_mat

def SimulateThreeTargetTask_QmultiplicativeModel(alpha, beta, gamma, num_trials, percent_instructed, reward_probs):
	'''
	This task simulates the choice behavior during the three choice task during Block A' assuming that value 
	scales multiplicatively during stimulation trials. It simulates only choice behavior during stimulation
	sessions, but separaters out the choice behavior from when MV is chosen when paired with the HV target and 
	the choice behavior when MV is chosen when paired with the LV target. A separate function is used for 
	simulating choice behavior during sham sessions, which does not use an explicit paramter to model as value 
	changes when stimulation is applied.

	Input:
	- alpha: the alpha learning rate parameters used to simulate the behavior session
	- beta: the beta temperature parameter used to simulate the behavior session
	- gamma: the lambda stimulation parameter used to simulate the behavior session
	- num_trials: integer representing the number of trials to be simulated
	- percent_instructed: integer representing the percentage trials that are instructed, i.e. have stimulation. For
		example, if the percentage of instructed trials is 30%, percent_instructed = 30
	- reward_probs: array of length 3 containing the reward probabilities for the LV, HV, and MV targets, respectively.
		The probabilities are given as integers. For example, if the reward probabilities are 40%, 90%, and 65% 
		respectively, then reward_probs = [35, 85, 60].
	'''
	# Set initial parameters
	Q_low = 0.3
	Q_mid = 0.5
	Q_high = 0.9

	counter_LV = 0
	counter_HV = 0

	counter_MV_with_HV = 0
	counter_MV_with_LV = 0

	reward = 0
	prob_reward_low = reward_probs[0]
	prob_reward_mid = reward_probs[2]
	prob_reward_high = reward_probs[1]

	# Design possible targets
	targets = np.array([[0,1,1], [1,0,1], [1,1,0]])

	# Make trial indices for instructed trials with stimulation
	stim_trials = np.ravel(np.nonzero(np.greater(np.random.randint(0, 100, num_trials), 100 - percent_instructed)))
	# Make array of targets presented
	targets_shown = np.random.randint(0,3,num_trials)
	targets_on = targets[targets_shown]
	targets_on[stim_trials] = np.array([0,0,1])

	# Simulate choices
	for i in range(1,num_trials):

		if (i in stim_trials):
			Q_mid = Q_mid + gamma*Q_mid
		else:
			# Get current targets
			targ_on = targets_on[i]
			# Look at likelihood of choosing MV when MV and HV are shown
			if np.array_equal(targ_on,[0,1,1]):
				prob_choice_mid = 1./(1 + np.exp(beta*(Q_high - Q_mid)))
				choice = np.greater(np.random.randint(100), 100*(1 - prob_choice_mid)) 		# MV selection: choice = 1, HV selection: choice = 0
				counter_MV_with_HV += choice
				counter_HV += 1
				reward = np.greater(np.random.randint(100),100- prob_reward_high)*(choice==0) + np.greater(np.random.randint(100),100- prob_reward_mid)*(choice==1)
				delta_mid = reward - Q_mid
				delta_high = reward - Q_high
				Q_mid = Q_mid + alpha*delta_mid*(choice==1)
				Q_high = Q_high + alpha*delta_high*(choice==0)
			# Look at the likelihood of choosing MV when MV and LV are shown
			if np.array_equal(targ_on,[1,0,1]):
				prob_choice_mid = 1./(1 + np.exp(beta*(Q_mid - Q_low)))
				choice = np.greater(np.random.randint(100), 100*(1 - prob_choice_mid)) 		# MV selection: choice = 1, LV selection: choice = 0
				counter_MV_with_LV += choice
				counter_LV += 1
				reward = np.greater(np.random.randint(100),100- prob_reward_low)*(choice==0) + np.greater(np.random.randint(100),100- prob_reward_mid)*(choice==1)
				delta_mid = reward - Q_mid
				delta_low = reward - Q_low
				Q_mid = Q_mid + alpha*delta_mid*(choice==1)
				Q_low = Q_low + alpha*delta_low*(choice==0)
			# Look at how values evolve when LV and HV are shown
			if np.array_equal(targ_on,[1,1,0]):
				prob_choice_high = 1./(1 + np.exp(beta*(Q_high - Q_low)))
				choice = np.greater(np.random.randint(100), 100*(1 - prob_choice_high)) 		# HV selection: choice = 1, LV selection: choice = 0
				reward = np.greater(np.random.randint(100),100- prob_reward_high)*(choice==1) + np.greater(np.random.randint(100),100- prob_reward_low)*(choice==0)
				delta_low = reward - Q_low
				delta_high = reward - Q_high
				Q_low = Q_low + alpha*delta_low*(choice==0)
				Q_high = Q_high + alpha*delta_high*(choice==1)

	prob_MV_with_HV = float(counter_MV_with_HV)/counter_HV
	prob_MV_with_LV = float(counter_MV_with_LV)/counter_LV

	return prob_MV_with_HV, prob_MV_with_LV

def SimulateThreeTargetTask_QmultiplicativeModel_AcrossSessions(alpha_mat, beta_mat, gamma_mat, num_trials, percent_instructed, reward_probs):
	'''
	This method simulates choice behavior across multiple sham sessions.

	Input: 
	- alpha_mat: array of length N, where N is the number of sessions, containing the alpha learning rate parameter
	- beta_mat: array of length N containing the beta temperature parameter
	- gamma_mat: array of length N containing the lambda stimulation parameter
	- num_trials: number of trials per session to be simulated
	- percent_instructed: integer representing the percentage of instructed trials
	- reward_probs: array of length 3 containing the reward probabilities for the LV, HV, and MV targets, respectively.
		The probabilities are given as integers. For example, if the reward probabilities are 40%, 90%, and 65% 
		respectively, then reward_probs = [35, 85, 60].
	'''
	num_sessions = len(alpha_mat)
	prob_MV_with_HV = np.zeros(num_sessions)
	prob_MV_with_LV = np.zeros(num_sessions)
	for k in range(num_sessions):
		prob_MV_with_HV[k], prob_MV_with_LV[k] = SimulateThreeTargetTask_QmultiplicativeModel(alpha_mat[k], beta_mat[k], gamma_mat[k], num_trials, percent_instructed, reward_probs)
	
	return prob_MV_with_HV, prob_MV_with_LV

def SimulateThreeTargetTask_ShamModel(alpha, beta, num_trials, percent_instructed, reward_probs):
	'''
	This task simulates the choice behavior during the three choice task during Block A' assuming that value 
	scales multiplicatively during stimulation trials. It simulates only choice behavior during sham sessions, 
	which does not use an explicit paramter to model as value changes when stimulation is applied.

	Input:
	- alpha: the alpha learning rate parameters used to simulate the behavior session
	- beta: the beta temperature parameter used to simulate the behavior session
	- num_trials: integer representing the number of trials to be simulated
	- percent_instructed: integer representing the percentage trials that are instructed, i.e. have stimulation. For
		example, if the percentage of instructed trials is 30%, percent_instructed = 30
	- reward_probs: array of length 3 containing the reward probabilities for the LV, HV, and MV targets, respectively.
		The probabilities are given as integers. For example, if the reward probabilities are 40%, 90%, and 65% 
		respectively, then reward_probs = [35, 85, 60].
	'''
	# Set initial parameters
	Q_low = 0.3
	Q_mid = 0.5
	Q_high = 0.9

	counter_HV = 0

	counter_MV_with_HV = 0
	
	reward = 0
	prob_reward_low = reward_probs[0]
	prob_reward_mid = reward_probs[2]
	prob_reward_high = reward_probs[1]

	# Design possible targets
	targets = np.array([[0,1,1], [1,0,1], [1,1,0]])

	# Make trial indices for instructed trials with stimulation
	stim_trials = np.ravel(np.nonzero(np.greater(np.random.randint(0, 100, num_trials), 100 - percent_instructed)))
	# Make array of targets presented
	targets_shown = np.random.randint(0,3,num_trials)
	targets_on = targets[targets_shown]
	targets_on[stim_trials] = np.array([0,0,1])

	# Simulate choices
	for i in range(1,num_trials):

		if (i in stim_trials):
			reward = np.greater(np.random.randint(100),100- prob_reward_mid)
			delta_mid = reward - Q_mid
			Q_mid = Q_mid + alpha*delta_mid
		else:
			# Get current targets
			targ_on = targets_on[i]
			# Look at likelihood of choosing MV when MV and HV are shown
			if np.array_equal(targ_on,[0,1,1]):
				prob_choice_mid = 1./(1 + np.exp(beta*(Q_high - Q_mid)))
				choice = np.greater(np.random.randint(100), 100*(1 - prob_choice_mid)) 		# MV selection: choice = 1, HV selection: choice = 0
				counter_MV_with_HV += choice
				counter_HV += 1
				reward = np.greater(np.random.randint(100),100- prob_reward_high)*(choice==0) + np.greater(np.random.randint(100),100- prob_reward_mid)*(choice==1)
				delta_mid = reward - Q_mid
				delta_high = reward - Q_high
				Q_mid = Q_mid + alpha*delta_mid*(choice==1)
				Q_high = Q_high + alpha*delta_high*(choice==0)
			# Look at the likelihood of choosing MV when MV and LV are shown
			if np.array_equal(targ_on,[1,0,1]):
				prob_choice_mid = 1./(1 + np.exp(beta*(Q_mid - Q_low)))
				choice = np.greater(np.random.randint(100), 100*(1 - prob_choice_mid)) 		# MV selection: choice = 1, LV selection: choice = 0
				reward = np.greater(np.random.randint(100),100- prob_reward_low)*(choice==0) + np.greater(np.random.randint(100),100- prob_reward_mid)*(choice==1)
				delta_mid = reward - Q_mid
				delta_low = reward - Q_low
				Q_mid = Q_mid + alpha*delta_mid*(choice==1)
				Q_low = Q_low + alpha*delta_low*(choice==0)
			# Look at how values evolve when LV and HV are shown
			if np.array_equal(targ_on,[1,1,0]):
				prob_choice_high = 1./(1 + np.exp(beta*(Q_high - Q_low)))
				choice = np.greater(np.random.randint(100), 100*(1 - prob_choice_high)) 		# HV selection: choice = 1, LV selection: choice = 0
				reward = np.greater(np.random.randint(100),100- prob_reward_high)*(choice==1) + np.greater(np.random.randint(100),100- prob_reward_low)*(choice==0)
				delta_low = reward - Q_low
				delta_high = reward - Q_high
				Q_low = Q_low + alpha*delta_low*(choice==0)
				Q_high = Q_high + alpha*delta_high*(choice==1)

	prob_MV_with_HV = float(counter_MV_with_HV)/counter_HV
	
	return prob_MV_with_HV

def SimulateThreeTargetTask_ShamModel_AcrossSessions(alpha_mat, beta_mat, num_trials, percent_instructed, reward_probs):
	'''
	This method simulates choice behavior across multiple sham sessions.

	Input: 
	- alpha_mat: array of length N, where N is the number of sessions, containing the alpha learning rate parameter
	- beta_mat: array of length N containing the beta temperature parameter
	- num_trials: number of trials per session to be simulated
	- percent_instructed: integer representing the percentage of instructed trials
	- reward_probs: array of length 3 containing the reward probabilities for the LV, HV, and MV targets, respectively.
		The probabilities are given as integers. For example, if the reward probabilities are 40%, 90%, and 65% 
		respectively, then reward_probs = [35, 85, 60].
	'''
	num_sessions = len(alpha_mat)
	prob_MV = np.zeros(num_sessions)
	for k in range(num_sessions):
		prob_MV[k] = SimulateThreeTargetTask_ShamModel(alpha_mat[k], beta_mat[k], num_trials, percent_instructed, reward_probs)
	
	return prob_MV

def Simulate_ChoiceBehavior_ThreeTargetTask(mario_stim_days, mario_sham_days, num_trials, percent_instructed, reward_probs):
	'''
	This method simulates choice behavior across the behavior sessions on sham and stim days.

	Inputs:
	- mario_sham_days: list of hdf file locations for different sham sessions
	- mario_stim_days: list of hdf file locations for different stim sessions
	- num_trials: number of trials per session to be simulated
	- percent_instructed: integer representing the percentage of instructed trials
	- reward_probs: array of length 3 containing the reward probabilities for the LV, HV, and MV targets, respectively.
		The probabilities are given as integers. For example, if the reward probabilities are 40%, 90%, and 65% 
		respectively, then reward_probs = [35, 85, 60].
	'''

	alpha_mat_sham, beta_mat_sham, lambda_mat_sham, BIC_mat_sham = RLModel_ThreeTarget_MVHV(mario_sham_days)
	alpha_mat, beta_mat, lambda_mat, BIC_mat = RLModel_ThreeTarget_MVHV(mario_stim_days)

	prob_MV_with_HV, prob_MV_with_LV = SimulateThreeTargetTask_QmultiplicativeModel_AcrossSessions(alpha_mat[:,2], beta_mat[:,2], lambda_mat[:,2], num_trials, percent_instructed, reward_probs)
	prob_MV_with_HV_sham = SimulateThreeTargetTask_ShamModel_AcrossSessions(alpha_mat_sham[:,0], beta_mat_sham[:,0], num_trials, percent_instructed, reward_probs)

	choice_mean = [np.nanmean(prob_MV_with_HV), np.nanmean(prob_MV_with_HV_sham), 1 - np.nanmean(prob_MV_with_LV)]
	choice_sem = [np.nanstd(prob_MV_with_HV)/np.sqrt(len(mario_stim_days)), np.nanstd(prob_MV_with_HV_sham)/np.sqrt(len(mario_sham_days)), np.nanstd(prob_MV_with_LV)/np.sqrt(len(mario_stim_days))]

	width = float(0.35)
	ind = range(1,4)

	plt.figure()
	plt.bar(ind[0], choice_mean[0], width/2., color = 'c', ecolor = 'k', yerr = choice_sem[0]/2.)
	plt.bar(ind[1], choice_mean[1], width/2., color = 'm', ecolor = 'k', yerr = choice_sem[1]/2.)
	plt.bar(ind[2], choice_mean[2], width/2., color = 'y', ecolor = 'k', yerr = choice_sem[2]/2.)
	plt.ylabel('P(Choose Less-rewarding target)')
	xticklabels = ['Stim: MV vs HV', 'Sham: MV vs HV', 'Stim: MV vs LV']
  	plt.xticks(range(1,4), xticklabels)
	plt.xlim((0.5,4.5))
	#plt.ylim((0.0,0.35))
	plt.show()

	return prob_MV_with_HV, prob_MV_with_LV, prob_MV_with_HV_sham

##### NEED TO ADD METHODS FOR CONTROL ANALYSIS WHERE ONLY LV AND MV TARGETS ARE SHOWN

# Figure 1C
# prob_HV_given_left, prob_HV_given_right = SideBias(mario_sham_days_all)
# Stats reported in paper
#prob_HV_early, prob_HV_late = EarlyAndLateTargetSelection_HV(mario_sham_days_all)
# Figure 2A
#prob_MV_choices = MV_choice_behavior(mario_sham_days_all, mario_stim_days_all)
# Figure 2E
#prob_MV_with_HV_sham, prob_MV_with_LV_sham, prob_MV_with_HV_stim, prob_MV_with_LV_stim = MV_aligned_choice_behavior(mario_sham_days_all, mario_stim_days_all, 5)
# Figure 2B,C
#output = TargetLocationAndReward(mario_stim_days_all, 1)
# Figure 3 b,c,d,e,f
alpha_mat_sham, beta_mat_sham, lambda_mat_sham, BIC_mat_sham = RLModel_ThreeTarget_MVHV(mario_sham_days_all)
#alpha_mat, beta_mat, lambda_mat, BIC_mat = RLModel_ThreeTarget_MVHV(mario_stim_days_all)
#alpha_mat, beta_mat, lambda_mat, BIC_mat = RLModel_ThreeTarget_MVLV(mario_stim_days_all)
# Figure 4
#reward_probs = [35, 85, 60]
#percent_instructed = 30
#num_trials = 100
#prob_MV_with_HV, prob_MV_with_LV, prob_MV_with_HV_sham = Simulate_ChoiceBehavior_ThreeTargetTask(mario_stim_days_all, mario_sham_days_all, num_trials, percent_instructed, reward_probs)