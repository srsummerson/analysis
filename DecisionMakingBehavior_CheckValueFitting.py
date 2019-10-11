from DecisionMakingBehavior import ChoiceBehavior_TwoTargets_Stimulation, ChoiceBehavior_ThreeTargets_Stimulation, loglikelihood_ThreeTargetTask_Qlearning, ThreeTargetTask_Qlearning
import numpy as np 
import scipy as sp
from scipy import stats
from statsmodels.formula.api import ols
from scipy.interpolate import spline
from scipy import signal
from scipy.ndimage import filters
import scipy.optimize as op
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import pandas as pd
from scipy import io
from scipy import stats
import matplotlib as mpl
from matplotlib import mlab
import tables
from rt_calc import compute_rt_per_trial_FreeChoiceTask
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from neo import io
from PulseMonitorData import findIBIs
from offlineSortedSpikeAnalysis import OfflineSorted_CSVFile
from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance


dir = "C:/Users/ss45436/Box/UC Berkeley/Cd Stim/Neural Correlates/Luigi/spike_data/"

luigi_hdf_list_sham = [[dir + 'luig20170822_07_te133.hdf'], \
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

luigi_hdf_list_stim = [[dir + 'luig20170909_08_te226.hdf'], \
			[dir + 'luig20170915_15_te264.hdf'], \
			[dir + 'luig20170927_07_te361.hdf'], \
			[dir + 'luig20171001_06_te385.hdf'], \
			[dir + 'luig20171005_02_te408.hdf'], \
			[dir + 'luig20171017_04_te454.hdf', dir + 'luig20171017_05_te455.hdf'], \
			[dir + 'luig20171031_04_te509.hdf'], \
			[dir + 'luig20171108_03_te543.hdf'], \
			]

luigi_hdf_list = luigi_hdf_list_sham + luigi_hdf_list_stim

dir = "C:/Users/ss45436/Box/UC Berkeley/Cd Stim/Neural Correlates/Mario/spike_data/"

mario_hdf_list_sham = [[dir + 'mari20161220_05_te2795.hdf'], \
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
mario_hdf_list_stim = [[dir + 'mari20161221_03_te2800.hdf'], \
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

mario_hdf_list = mario_hdf_list_sham + mario_hdf_list_stim


def trial_sliding_avg(trial_array, num_trials_slide):

	num_trials = len(trial_array)
	slide_avg = np.zeros(num_trials)

	for i in range(num_trials):
		if i < num_trials_slide:
			slide_avg[i] = np.sum(trial_array[:i+1])/float(i+1)
		else:
			slide_avg[i] = np.sum(trial_array[i-num_trials_slide+1:i+1])/float(num_trials_slide)

	return slide_avg

def LR_ValueEstimates_TwoTargetTask(hdf_files):

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 3. Get Q-values, chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	freechoice_inds = np.array([i for i in range(5,len(instructed_or_freechoice)) if instructed_or_freechoice[i]==2]) 		# =2 is freechoice trial

	choice_array = chosen_target[freechoice_inds] - 1 		# 0 for LV choice, 1 for HV choice
	y_hv = np.zeros((len(freechoice_inds),5))
	y_lv = np.zeros((len(freechoice_inds),5))
	n_hv = np.zeros((len(freechoice_inds),5))
	n_lv = np.zeros((len(freechoice_inds),5))

	

	for i,ind in enumerate(freechoice_inds):
		for j in range(5):
			choice = chosen_target[ind-j]
			reward = rewards[ind-j]
			if (choice==1):
				if (reward==0):
					n_lv[i,j] = 1
				else:
					y_lv[i,j] = 1
			else:
				if (reward==0):
					n_hv[i,j] = 1
				else:
					y_hv[i,j] = 1

	x = np.hstack((y_hv, -y_lv))
	x = np.hstack((x, np.ones([len(freechoice_inds),1]))) 	
		
	# 4. Do regression for each bin 
	y = choice_array

	print("Regression for choices using 5 trail history ")
	model_glm = sm.OLS(y,x, family = sm.families.Binomial())
	#model_glm = sm.GLM(y,x, family = sm.families.Binomial())
	fit_glm = model_glm.fit()
	beta_params = fit_glm.params
	print(fit_glm.summary())

	prob_comp = np.matmul(x,beta_params)
	prob_hv = np.exp(prob_comp)/(1 + np.exp(prob_comp))
	q_hv = np.matmul(x[:,:2],beta_params[:2])
	q_lv = np.matmul(x[:,5:7],beta_params[5:7])


	return beta_params, prob_hv, q_hv, q_lv


def Value_from_reward_history_TwoTargetTask(hdf_files):

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 2. Get chosen targets and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	chose_hv_inds = np.ravel(np.nonzero(chosen_target[:-1]-1))
	chose_lv_inds = np.ravel(np.nonzero(2-chosen_target[:-1]))

	# 3. Extract reward information for each target
	q_hv = np.zeros(len(chosen_target))
	q_lv = np.zeros(len(chosen_target))
	q_hv[0] = 0.5
	q_lv[0] = 0.5

	reward_hv = rewards[chose_hv_inds]
	reward_lv = rewards[chose_lv_inds]

	# 4. Determine values as sliding average of reward history
	reward_hv_avg = trial_sliding_avg(reward_hv,10)
	reward_lv_avg = trial_sliding_avg(reward_lv,10)

	q_hv[chose_hv_inds+1] = reward_hv_avg
	q_lv[chose_lv_inds+1] = reward_lv_avg

	# 4a. Fill in values for trials where target was not selected
	for i in range(len(q_hv)-1):
		if i not in chose_hv_inds:
			q_hv[i+1] = q_hv[i]

	for i in range(len(q_lv)-1):
		if i not in chose_lv_inds:
			q_lv[i+1] = q_lv[i]

	# 5. Smooth value information
	b = signal.gaussian(20, 1)
	q_hv_smooth = filters.convolve1d(q_hv, b/b.sum())
	q_lv_smooth = filters.convolve1d(q_lv, b/b.sum())

	'''
	plt.figure()
	plt.plot(q_hv,'r')
	plt.plot(q_hv_smooth,'m')
	plt.plot(q_lv,'b')
	plt.plot(q_lv_smooth,'c')
	plt.show()
	'''

	return q_hv_smooth, q_lv_smooth

def Value_from_reward_history_ThreeTargetTask(hdf_files):

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	
	# 2. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	chose_hv_inds = np.ravel(np.nonzero(np.floor(0.5*chosen_target[:-1])))
	chose_mv_inds = np.ravel(np.nonzero(1 - np.abs(chosen_target[:-1] -1)))
	chose_lv_inds = np.ravel(np.nonzero(np.floor(1 - 0.5*chosen_target[:-1])))

	# 3. Extract reward information for each target
	q_hv = np.zeros(len(chosen_target))
	q_lv = np.zeros(len(chosen_target))
	q_mv = np.zeros(len(chosen_target))
	q_hv[0] = 0.5
	q_lv[0] = 0.5
	q_mv[0] = 0.5

	reward_hv = rewards[chose_hv_inds]
	reward_lv = rewards[chose_lv_inds]
	reward_mv = rewards[chose_mv_inds]

	# 4. Determine values as sliding average of reward history
	reward_hv_avg = trial_sliding_avg(reward_hv,10)
	reward_lv_avg = trial_sliding_avg(reward_lv,10)
	reward_mv_avg = trial_sliding_avg(reward_mv,10)

	q_hv[chose_hv_inds+1] = reward_hv_avg
	q_lv[chose_lv_inds+1] = reward_lv_avg
	q_mv[chose_mv_inds+1] = reward_mv_avg

	# 4a. Fill in values for trials where target was not selected
	for i in range(len(q_hv)-1):
		if i not in chose_hv_inds:
			q_hv[i+1] = q_hv[i]

	for i in range(len(q_lv)-1):
		if i not in chose_lv_inds:
			q_lv[i+1] = q_lv[i]

	for i in range(len(q_mv)-1):
		if i not in chose_mv_inds:
			q_mv[i+1] = q_mv[i]

	# 5. Smooth value information
	b = signal.gaussian(20, 1)
	q_hv_smooth = filters.convolve1d(q_hv, b/b.sum())
	q_lv_smooth = filters.convolve1d(q_lv, b/b.sum())
	q_mv_smooth = filters.convolve1d(q_mv, b/b.sum())

	'''
	plt.figure()
	plt.plot(q_hv,'r', label = 'HV')
	plt.plot(q_hv_smooth,'m')
	plt.plot(q_lv,'b', label = 'LV')
	plt.plot(q_lv_smooth,'c')
	plt.plot(q_mv,'g', label = 'MV')
	plt.plot(q_mv_smooth,'y')
	plt.plot(0.5*chosen_target,'k')
	plt.show()
	'''

	return q_hv_smooth, q_mv_smooth, q_lv_smooth


def RL_TwoTargetTask(hdf_files):

	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, 100, 100)
	total_trials = cb.num_successful_trials
	#targets_on = cb.targets_on[cb.state_time[cb.ind_check_reward_states]]

	# 1a. Get reaction time information
	rt = np.array([])
	for file in hdf_files:
		print(file)
		reaction_time, velocity = compute_rt_per_trial_FreeChoiceTask(file)
		rt = np.append(rt, reaction_time)

	# 1b. Get movementment time information
	mt = (cb.state_time[cb.ind_target_states + 1] - cb.state_time[cb.ind_target_states])/60.

	# 3. Get Q-values, chosen targets, and rewards
	chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	freechoice_inds = np.array([i for i in range(len(instructed_or_freechoice)) if instructed_or_freechoice[i]==2]) 		# =2 is freechoice trial
	sliding_avg_LH_choices = trial_sliding_avg(chosen_target[freechoice_inds]-1, 25)						# chosen target = 1 for LV, = 2 for HV
	
	# Varying Q-values
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(2)
	nll = lambda *args: -logLikelihoodRLPerformance(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, rewards, chosen_target, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	alpha_ml_avg = 0.0546
	beta_ml_avg = 5.7695
	beta_ml_avg = 4.9571

	Q_low, Q_high, prob_choice_low, log_likelihood = RLPerformance([alpha_ml_avg, beta_ml_avg], Q_initial, rewards, chosen_target, instructed_or_freechoice)

	fig = plt.figure()
	ax = plt.subplot(121)
	plt.title(hdf_files[0])
	#plt.plot(sliding_avg_LH_choices, c = 'b', label = 'High')
	plt.plot(1-sliding_avg_LH_choices, c = 'r', label = 'Empirical - LV')
	plt.plot(prob_choice_low[:-1], c = 'c', label = 'RL Model - LV')
	plt.xlabel('Trials')
	plt.ylabel('Choice Probability')
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.ylim((0,1))
	plt.legend()

	ax2 = plt.subplot(122)
	plt.plot(1-sliding_avg_LH_choices, c = 'r', label = 'Empirical - LV', alpha = 0.2)
	plt.plot(Q_low[freechoice_inds],'r', label = 'Q LV')
	plt.plot(Q_high[freechoice_inds],'b', label = 'Q HV')
	plt.xlabel('Trials')
	plt.ylabel('Choice Probability')
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.ylim((0,1))
	plt.legend()
	plt.show()

	return alpha_ml, beta_ml


def RL_ThreeTargetTask(hdf_files):
	# 1. Load behavior data and pull out trial indices for the designated trial case
	cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)
	

	# 3. Get Q-values, chosen targets, and rewards
	targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
	all_choices_A, LM_choices_A, LH_choices_A, MH_choices_A, all_choices_Aprime, LM_choices_Aprime, LH_choices_Aprime, MH_choices_Aprime = cb.TrialChoices(10, False)
	sliding_avg_LM_choices = trial_sliding_avg(LM_choices_A, 25)
	sliding_avg_LH_choices = trial_sliding_avg(LH_choices_A, 25)
	sliding_avg_MH_choices = trial_sliding_avg(MH_choices_A, 25)
	
	# Varying Q-values
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_ThreeTargetTask_Qlearning(*args)
	result = op.minimize(nll, [0.2, 1], args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,None)])
	alpha_ml, beta_ml = result["x"]
	print("Best fitting alpha and beta are: ", alpha_ml, beta_ml)
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_opt_lvmv, prob_choice_opt_lvhv, prob_choice_opt_mvhv, accuracy, log_prob_total = ThreeTargetTask_Qlearning([alpha_ml, beta_ml], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	fig = plt.figure()
	ax11 = plt.subplot(131)
	plt.plot(sliding_avg_LM_choices, c = 'b', label = 'Mid')
	plt.xlabel('Trials')
	plt.ylabel('Probability Best Choice')
	plt.title('Low vs. Mid')
	ax11.get_yaxis().set_tick_params(direction='out')
	ax11.get_xaxis().set_tick_params(direction='out')
	ax11.get_xaxis().tick_bottom()
	ax11.get_yaxis().tick_left()
	plt.legend()

	ax12 = plt.subplot(132)
	plt.plot(sliding_avg_LH_choices, c = 'b', label = 'High')
	plt.xlabel('Trials')
	plt.ylabel('Probability Best Choice')
	plt.title('Low vs. High')
	ax12.get_yaxis().set_tick_params(direction='out')
	ax12.get_xaxis().set_tick_params(direction='out')
	ax12.get_xaxis().tick_bottom()
	ax12.get_yaxis().tick_left()
	plt.legend()

	return alpha_ml, beta_ml

###########################
# Begin running analysis
###########################

alphas = np.array([])
betas = np.array([])
for file in mario_hdf_list:
	output1 = RL_ThreeTargetTask(file)
	alphas = np.append(alphas, output1[0])
	betas = np.append(betas, output1[1])
	#output2 = LR_ValueEstimates_TwoTargetTask(file)
	#output2 = Value_from_reward_history_TwoTargetTask(file)

'''
q_low = output1[2]
q_high = output1[3]
prob_choice_low = output1[4]
prob_choice_high = 1 - prob_choice_low
q_hv = output2[0]
q_lv = output2[1]
q_hv_smooth = output2[2]
q_lv_smooth = output2[3]

plt.figure()
plt.subplot(121)
plt.title('HV')
plt.plot(q_high,'r',label = 'RL')
plt.plot(q_hv,'m', label = 'Avg')
plt.plot(q_hv_smooth, 'b', label = 'Smooth Avg')
plt.plot(prob_choice_high, 'k', label = 'Prob Choice')
plt.subplot(122)
plt.title('LV')
plt.plot(q_low,'r',label = 'RL')
plt.plot(q_lv,'m', label = 'Avg')
plt.plot(q_lv_smooth, 'b', label = 'Smooth Avg')
plt.plot(prob_choice_low, 'k', label = 'Prob Choice')
plt.show()
'''