from scipy import stats
import statsmodels.api as sm
import scipy
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
import scipy.optimize as op
import tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from logLikelihoodRLPerformance import RLPerformance, logLikelihoodRLPerformance, RLPerformance_multiplicative_Qstimparameter, logLikelihoodRLPerformance_multiplicative_Qstimparameter, \
                                        RLPerformance_additive_Qstimparameter, logLikelihoodRLPerformance_additive_Qstimparameter, RLPerformance_multiplicative_Pstimparameter, \
                                        logLikelihoodRLPerformance_multiplicative_Pstimparameter, RLPerformance_additive_Pstimparameter, logLikelihoodRLPerformance_additive_Pstimparameter, \
                                        logLikelihoodRLPerformance_multiplicative_Qstimparameter_HVTarget, RLPerformance_multiplicative_Qstimparameter_HVTarget
from probabilisticRewardTaskPerformance import FreeChoicePilotTask_Behavior
from probabilisticRewardTask_SummaryStats import OneWayMANOVA
from basicAnalysis import ComputeRSquared


hdf_list_sham_papa = ['\papa20150213_10.hdf','\papa20150217_05.hdf','\papa20150225_02.hdf','\papa20150305_02.hdf',
    '\papa20150307_02.hdf','\papa20150308_06.hdf','\papa20150310_02.hdf','\papa20150506_09.hdf','\papa20150506_10.hdf',
    '\papa20150519_02.hdf','\papa20150519_03.hdf','\papa20150519_04.hdf','\papa20150527_01.hdf','\papa20150528_02.hdf']
"""
hdf_list_sham = ['\papa20150217_05.hdf','\papa20150305_02.hdf',
    '\papa20150310_02.hdf',
    '\papa20150519_02.hdf','\papa20150519_04.hdf','\papa20150528_02.hdf']
"""
# sessions with good RL fit only
# 9 sessions contant voltage, 10 sessions constant current, 12 sessions sham
# 
"""
hdf_list_stim = ['\papa20150203_10.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_01.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf']

hdf_list_sham = ['\papa20150213_10.hdf','\papa20150217_05.hdf','\papa20150225_02.hdf',
    '\papa20150307_02.hdf','\papa20150308_06.hdf','\papa20150310_02.hdf','\papa20150506_09.hdf','\papa20150506_10.hdf',
    '\papa20150519_03.hdf','\papa20150519_04.hdf','\papa20150527_01.hdf','\papa20150528_02.hdf']
"""

# constant voltage: 11 good stim sessions
hdf_list_control_papa = ['\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf']
"""
hdf_list_stim = ['\papa20150211_11.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf']
# constant current: 18 good stim sessions
# hdf_list_stim2 = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
#    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
#    '\papa20150524_03.hdf','\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
#    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_01.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
#    '\papa20150602_04.hdf']
"""
hdf_list_stim2 = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']



hdf_list_stim = ['\luig20160204_15_te1382.hdf','\luig20160208_07_te1401.hdf','\luig20160212_08_te1429.hdf','\luig20160217_06_te1451.hdf',
                '\luig20160229_11_te1565.hdf','\luig20160301_07_te1572.hdf','\luig20160301_09_te1574.hdf', '\luig20160311_08_te1709.hdf',
                '\luig20160313_07_te1722.hdf', '\luig20160315_14_te1739.hdf']
hdf_list_sham = ['\luig20160213_05_te1434.hdf','\luig20160219_04_te1473.hdf','\luig20160221_05_te1478.hdf', '\luig20160305_26_te1617.hdf', \
                 '\luig20160306_11_te1628.hdf', '\luig20160307_13_te1641.hdf', '\luig20160310_16_te1695.hdf','\luig20160319_23_te1801.hdf', \
                 '\luig20160320_07_te1809.hdf', '\luig20160322_08_te1826.hdf']
hdf_list_hv = ['\luig20160218_10_te1469.hdf','\luig20160223_11_te1508.hdf','\luig20160224_15_te1523.hdf', \
                '\luig20160303_11_te1591.hdf', '\luig20160308_06_te1647.hdf','\luig20160309_25_te1672.hdf']
hdf_list_hv = ['\luig20160218_10_te1469.hdf','\luig20160223_11_te1508.hdf', \
                '\luig20160224_15_te1523.hdf', '\luig20160303_11_te1591.hdf',  \
                '\luig20160308_06_te1647.hdf','\luig20160309_25_te1672.hdf', '\luig20160323_04_te1830.hdf', '\luig20160323_09_te1835.hdf',
                '\luig20160324_10_te1845.hdf', '\luig20160324_12_te1847.hdf','\luig20160324_14_te1849.hdf']


hdf_prefix = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Luigi\hdf'
stim_hdf_list = hdf_list_stim2
sham_hdf_list = hdf_list_sham_papa
hv_hdf_list = hdf_list_control_papa

stim_hdf_list = hdf_list_stim
sham_hdf_list = hdf_list_sham
hv_hdf_list = hdf_list_hv

global_max_trial_dist = 0
Q_initial = [0.5, 0.5]
alpha_true = 0.2
beta_true = 0.2
gamma_true = 0.2

def FirstChoiceAfterStim(target3,trial3,stim_trials):
    choice = [target3[i] for i in range(1,len(target3[0:100])) if (trial3[i]==2)&(stim_trials[i-1]==1)]
    choice = np.array(choice)
    choose_low = np.sum(-choice + 2)

    if len(choice) > 0:
        prob_choose_low = float(choose_low)/len(choice)
    else:
        prob_choose_low = np.nan

    return prob_choose_low

def computeProbabilityChoiceWithRegressors(params_block1, params_block3,reward1, target1, trial1, target_side1, reward3, target3, trial3, target_side3, stim_trials):


    '''
    Previous rewards and no rewards
    '''
    relative_action_value_block1 = []
    prob_choice_block1 = []
    relative_action_value_block3 = []
    prob_choice_block3 = []


    for i in range(5,len(trial1)):
        if trial1[i] == 2:
            #fc_target_low_block1.append(2 -target1[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block1 = (target1[i] - 1)  # = 1 if selected high-value, =  0 if selected low-value
            fc_target_high_side_block1 = target_side1[i]
            prev_reward1_block1= (2*target1[i-1] - 3)*reward1[i-1]  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block1 = ((2*target1[i-2] - 3)*reward1[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block1 = ((2*target1[i-3] - 3)*reward1[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block1 = ((2*target1[i-4] - 3)*reward1[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block1 = ((2*target1[i-5] - 3)*reward1[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block1 = ((2*target1[i-1] - 3)*(1 - reward1[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block1 = ((2*target1[i-2] - 3)*(1 - reward1[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block1 = ((2*target1[i-3] - 3)*(1 - reward1[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block1 = ((2*target1[i-4] - 3)*(1 - reward1[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block1 = ((2*target1[i-5] - 3)*(1 - reward1[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
    
            relative_action_value_block1.append(params_block1[1]*prev_reward1_block1 + params_block1[2]*prev_reward2_block1 + params_block1[3]*prev_reward3_block1 + \
                            params_block1[4]*prev_reward4_block1 + params_block1[5]*prev_reward5_block1 + params_block1[6]*prev_noreward1_block1 + \
                           params_block1[7]*prev_noreward2_block1 + params_block1[8]*prev_noreward3_block1 + params_block1[9]*prev_noreward4_block1 + \
                           params_block1[10]*prev_noreward5_block1)

    log_prob = relative_action_value_block1 + params_block1[0]  # add intercept
    prob_choice_block1 = (float(1)/(np.exp(-log_prob) + 1))

    num_block3 = len(trial3)
    for i in range(5,num_block3):
        if (trial3[i] == 2):
            #fc_target_low_block3.append(2 - target3[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block3 = (target3[i] - 1)
            fc_target_high_side_block3 = target_side3[i]
            prev_reward1_block3 = ((2*target3[i-1] - 3)*reward3[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block3 = ((2*target3[i-2] - 3)*reward3[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block3 = ((2*target3[i-3] - 3)*reward3[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block3 = ((2*target3[i-4] - 3)*reward3[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block3 = ((2*target3[i-5] - 3)*reward3[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block3 = ((2*target3[i-1] - 3)*(1 - reward3[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block3 = ((2*target3[i-2] - 3)*(1 - reward3[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block3 = ((2*target3[i-3] - 3)*(1 - reward3[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block3 = ((2*target3[i-4] - 3)*(1 - reward3[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block3 = ((2*target3[i-5] - 3)*(1 - reward3[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim1_block3 = (stim_trials[i - 1]*(2*target3[i-1]- 1))  # = 1 if stim was delivered with HV target and = -1 if stim was delivered with the LV target
            prev_stim2_block3 = (stim_trials[i - 2]*(2*target3[i-2]- 1))
            prev_stim3_block3 = (stim_trials[i - 3]*(2*target3[i-3]- 1))
            prev_stim4_block3 = (stim_trials[i - 4]*(2*target3[i-4]- 1))
            prev_stim5_block3 = (stim_trials[i - 5]*(2*target3[i-5]- 1))

            relative_action_value_block3.append( params_block3[1]*prev_reward1_block3 + params_block3[2]*prev_reward2_block3 + params_block3[3]*prev_reward3_block3 + \
                                            params_block3[4]*prev_reward4_block3 + params_block3[5]*prev_reward5_block3 + params_block3[6]*prev_noreward1_block3 + \
                                            params_block3[7]*prev_noreward2_block3 + \
                                            params_block3[8]*prev_noreward3_block3 + \
                                            params_block3[9]*prev_noreward4_block3 + params_block3[10]*prev_noreward5_block3 + \
                                            params_block3[11]*prev_stim1_block3 + \
                                            params_block3[12]*prev_stim2_block3 + params_block3[13]*prev_stim3_block3 + params_block3[14]*prev_stim4_block3 + \
                                            params_block3[15]*prev_stim5_block3)

    log_prob = relative_action_value_block3 + params_block3[0]  
    prob_choice_block3 = (float(1)/(np.exp(-log_prob) + 1))


    return relative_action_value_block1, relative_action_value_block3, prob_choice_block1, prob_choice_block3

def computeProbabilityChoiceWithRegressors_StimHist(params_block1, params_block3,reward1, target1, trial1, reward3, target3, trial3, stim_trials,stim_dist):


    '''
    Previous rewards and no rewards
    '''
    relative_action_value_block1 = []
    prob_choice_block1 = []
    relative_action_value_block3 = []
    prob_choice_block3 = []


    for i in range(5,len(trial1)):
        if trial1[i] == 2:
            #fc_target_low_block1.append(2 -target1[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block1 = (target1[i] - 1)  # = 1 if selected high-value, =  0 if selected low-value
            prev_reward1_block1= (2*target1[i-1] - 3)*reward1[i-1]  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block1 = ((2*target1[i-2] - 3)*reward1[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block1 = ((2*target1[i-3] - 3)*reward1[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block1 = ((2*target1[i-4] - 3)*reward1[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block1 = ((2*target1[i-5] - 3)*reward1[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block1 = ((2*target1[i-1] - 3)*(1 - reward1[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block1 = ((2*target1[i-2] - 3)*(1 - reward1[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block1 = ((2*target1[i-3] - 3)*(1 - reward1[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block1 = ((2*target1[i-4] - 3)*(1 - reward1[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block1 = ((2*target1[i-5] - 3)*(1 - reward1[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
    
            relative_action_value_block1.append(params_block1[1]*prev_reward1_block1 + params_block1[2]*prev_reward2_block1 + params_block1[3]*prev_reward3_block1 + \
                            params_block1[4]*prev_reward4_block1 + params_block1[5]*prev_reward5_block1 + params_block1[6]*prev_noreward1_block1 + \
                           params_block1[7]*prev_noreward2_block1 + params_block1[8]*prev_noreward3_block1 + params_block1[9]*prev_noreward4_block1 + \
                           params_block1[10]*prev_noreward5_block1)

    log_prob = relative_action_value_block1 + params_block1[0]  # add intercept
    prob_choice_block1 = (float(1)/(np.exp(-log_prob) + 1))

    num_block3 = len(trial3)
    for i in range(5,num_block3):
        latest_stim = np.sum(stim_trials[i-stim_dist:i])
        if (trial3[i] == 2)&(stim_trials[i - stim_dist]==1)&(latest_stim==1):
            #fc_target_low_block3.append(2 - target3[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block3 = (target3[i] - 1)
            prev_reward1_block3 = ((2*target3[i-1] - 3)*reward3[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block3 = ((2*target3[i-2] - 3)*reward3[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block3 = ((2*target3[i-3] - 3)*reward3[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block3 = ((2*target3[i-4] - 3)*reward3[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block3 = ((2*target3[i-5] - 3)*reward3[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block3 = ((2*target3[i-1] - 3)*(1 - reward3[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block3 = ((2*target3[i-2] - 3)*(1 - reward3[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block3 = ((2*target3[i-3] - 3)*(1 - reward3[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block3 = ((2*target3[i-4] - 3)*(1 - reward3[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block3 = ((2*target3[i-5] - 3)*(1 - reward3[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim1_block3 = (stim_trials[i - 1]*(2*target3[i-1]- 1))  # = 1 if stim was delivered and = -1 if stim was not delivered
            prev_stim2_block3 = (stim_trials[i - 2]*(2*target3[i-2]- 1))
            prev_stim3_block3 = (stim_trials[i - 3]*(2*target3[i-3]- 1))
            prev_stim4_block3 = (stim_trials[i - 4]*(2*target3[i-4]- 1))
            prev_stim5_block3 = (stim_trials[i - 5]*(2*target3[i-5]- 1))

            relative_action_value_block3.append( params_block3[1]*prev_reward1_block3 + params_block3[2]*prev_reward2_block3 + params_block3[3]*prev_reward3_block3 + \
                                            params_block3[4]*prev_reward4_block3 + params_block3[5]*prev_reward5_block3 + params_block3[6]*prev_noreward1_block3 + \
                                            params_block3[7]*prev_noreward2_block3 + \
                                            params_block3[8]*prev_noreward3_block3 + \
                                            params_block3[9]*prev_noreward4_block3 + params_block3[10]*prev_noreward5_block3 + \
                                            params_block3[11]*prev_stim2_block3 + params_block3[12]*prev_stim3_block3 + params_block3[13]*prev_stim4_block3 + \
                                            params_block3[14]*prev_stim5_block3)

    log_prob = relative_action_value_block3 + params_block3[0]  
    prob_choice_block3 = (float(1)/(np.exp(-log_prob) + 1))


    return relative_action_value_block1, relative_action_value_block3, prob_choice_block1, prob_choice_block3

def probabilisticFreeChoicePilotTask_logisticRegression_StimHist(reward1, target1, trial1, reward3, target3, trial3, stim_trials, stim_dist):

    '''
    Stim_dist indicates the distance of the stimulation trial from the trials considered. For example, if we want to fit data only for trials that had 
    stimulation on the previous trial, stim_dist = 1. If we want to fit data for trials that had stimulation two trials ago, stim_dist = 2
    '''
    
    '''
    Previous rewards and no rewards
    '''
    fc_target_low_block1 = []
    fc_target_high_block1 = []
    fc_prob_low_block1 = []
    prev_reward1_block1 = []
    prev_reward2_block1 = []
    prev_reward3_block1 = []
    prev_reward4_block1 = []
    prev_reward5_block1 = []
    prev_noreward1_block1 = []
    prev_noreward2_block1 = []
    prev_noreward3_block1 = []
    prev_noreward4_block1 = []
    prev_noreward5_block1 = []
    prev_stim_block1 = []

    fc_target_low_block3 = []
    fc_target_high_block3 = []
    fc_prob_low_block3 = []
    prev_reward1_block3 = []
    prev_reward2_block3 = []
    prev_reward3_block3 = []
    prev_reward4_block3 = []
    prev_reward5_block3 = []
    prev_noreward1_block3 = []
    prev_noreward2_block3 = []
    prev_noreward3_block3 = []
    prev_noreward4_block3 = []
    prev_noreward5_block3 = []
    prev_stim1_block3 = []
    prev_stim2_block3 = []
    prev_stim3_block3 = []
    prev_stim4_block3 = []
    prev_stim5_block3 = []

    for i in range(5,len(trial1)):
        if trial1[i] == 2:
            fc_target_low_block1.append(2 -target1[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block1.append(target1[i] - 1)  # = 1 if selected high-value, =  0 if selected low-value
            prev_reward1_block1.append((2*target1[i-1] - 3)*reward1[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block1.append((2*target1[i-2] - 3)*reward1[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block1.append((2*target1[i-3] - 3)*reward1[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block1.append((2*target1[i-4] - 3)*reward1[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block1.append((2*target1[i-5] - 3)*reward1[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block1.append((2*target1[i-1] - 3)*(1 - reward1[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block1.append((2*target1[i-2] - 3)*(1 - reward1[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block1.append((2*target1[i-3] - 3)*(1 - reward1[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block1.append((2*target1[i-4] - 3)*(1 - reward1[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block1.append((2*target1[i-5] - 3)*(1 - reward1[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim_block1.append(0)
    num_block3 = len(trial3)
    for i in range(5,num_block3):
        latest_stim = np.sum(stim_trials[i-stim_dist:i])
        if (trial3[i] == 2)&(stim_trials[i - stim_dist]==1)&(latest_stim==1):
            fc_target_low_block3.append(2 - target3[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block3.append(target3[i] - 1)
            prev_reward1_block3.append((2*target3[i-1] - 3)*reward3[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block3.append((2*target3[i-2] - 3)*reward3[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block3.append((2*target3[i-3] - 3)*reward3[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block3.append((2*target3[i-4] - 3)*reward3[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block3.append((2*target3[i-5] - 3)*reward3[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block3.append((2*target3[i-1] - 3)*(1 - reward3[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block3.append((2*target3[i-2] - 3)*(1 - reward3[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block3.append((2*target3[i-3] - 3)*(1 - reward3[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block3.append((2*target3[i-4] - 3)*(1 - reward3[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block3.append((2*target3[i-5] - 3)*(1 - reward3[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim1_block3.append(stim_trials[i - 1]*(2*target3[i-1]- 1))  # = 1 if stim was delivered and = -1 if stim was not delivered
            prev_stim2_block3.append(stim_trials[i - 2]*(2*target3[i-2]- 1))
            prev_stim3_block3.append(stim_trials[i - 3]*(2*target3[i-3]- 1))
            prev_stim4_block3.append(stim_trials[i - 4]*(2*target3[i-4]- 1))
            prev_stim5_block3.append(stim_trials[i - 5]*(2*target3[i-5]- 1))


    '''
    Turn everything into an array
    '''
    fc_target_low_block1 = np.array(fc_target_low_block1)
    fc_target_high_block1 = np.array(fc_target_high_block1)
    prev_reward1_block1 = np.array(prev_reward1_block1)
    prev_reward2_block1 = np.array(prev_reward2_block1)
    prev_reward3_block1 = np.array(prev_reward3_block1)
    prev_reward4_block1 = np.array(prev_reward4_block1)
    prev_reward5_block1 = np.array(prev_reward5_block1)
    prev_noreward1_block1 = np.array(prev_noreward1_block1)
    prev_noreward2_block1 = np.array(prev_noreward2_block1)
    prev_noreward3_block1 = np.array(prev_noreward3_block1)
    prev_noreward4_block1 = np.array(prev_noreward4_block1)
    prev_noreward5_block1 = np.array(prev_noreward5_block1)
    prev_stim_block1 = np.array(prev_stim_block1)

    fc_target_low_block3 = np.array(fc_target_low_block3)
    fc_target_high_block3 = np.array(fc_target_high_block3)
    prev_reward1_block3 = np.array(prev_reward1_block3)
    prev_reward2_block3 = np.array(prev_reward2_block3)
    prev_reward3_block3 = np.array(prev_reward3_block3)
    prev_reward4_block3 = np.array(prev_reward4_block3)
    prev_reward5_block3 = np.array(prev_reward5_block3)
    prev_noreward1_block3 = np.array(prev_noreward1_block3)
    prev_noreward2_block3 = np.array(prev_noreward2_block3)
    prev_noreward3_block3 = np.array(prev_noreward3_block3)
    prev_noreward4_block3 = np.array(prev_noreward4_block3)
    prev_noreward5_block3 = np.array(prev_noreward5_block3)
    prev_stim1_block3 = np.array(prev_stim1_block3)
    prev_stim2_block3 = np.array(prev_stim2_block3)
    prev_stim3_block3 = np.array(prev_stim3_block3)
    prev_stim4_block3 = np.array(prev_stim4_block3)
    prev_stim5_block3 = np.array(prev_stim5_block3)

    const_logit_block1 = np.ones(fc_target_low_block1.size)
    const_logit_block3 = np.ones(fc_target_low_block3.size)

    
    '''
    Oraganize data and regress with GLM 
    '''
    x = np.vstack((prev_reward1_block1,prev_reward2_block1,prev_reward3_block1,prev_reward4_block1,prev_reward5_block1,
        prev_noreward1_block1,prev_noreward2_block1,prev_noreward3_block1,prev_noreward4_block1,prev_noreward5_block1))
    x = np.transpose(x)
    x = sm.add_constant(x,prepend='False')

    y = np.vstack((prev_reward1_block3,prev_reward2_block3,prev_reward3_block3,prev_reward4_block3,prev_reward5_block3,
        prev_noreward1_block3,prev_noreward2_block3,prev_noreward3_block3,prev_noreward4_block3,prev_noreward5_block3,
        prev_stim1_block3, prev_stim2_block3, prev_stim3_block3, prev_stim4_block3, prev_stim5_block3))
    y = np.transpose(y)
    y = sm.add_constant(y,prepend='False')

    model_glm_block1 = sm.Logit(fc_target_high_block1,x)
    model_glm_block3 = sm.Logit(fc_target_high_block3,y)
    fit_glm_block1 = model_glm_block1.fit()
    fit_glm_block3 = model_glm_block3.fit()
    #print fit_glm_block1.predict()
    
    return fit_glm_block1, fit_glm_block3



def probabilisticFreeChoicePilotTask_logisticRegression(reward1, target1, trial1, target_side1, reward3, target3, trial3, target_side3, stim_trials):

    
    '''
    Previous rewards and no rewards
    '''
    fc_target_low_block1 = []
    fc_target_high_block1 = []
    fc_prob_low_block1 = []
    fc_target_high_side_block1 = []
    prev_reward1_block1 = []
    prev_reward2_block1 = []
    prev_reward3_block1 = []
    prev_reward4_block1 = []
    prev_reward5_block1 = []
    prev_noreward1_block1 = []
    prev_noreward2_block1 = []
    prev_noreward3_block1 = []
    prev_noreward4_block1 = []
    prev_noreward5_block1 = []
    prev_stim_block1 = []

    fc_target_low_block3 = []
    fc_target_high_block3 = []
    fc_prob_low_block3 = []
    fc_target_high_side_block3 = []
    prev_reward1_block3 = []
    prev_reward2_block3 = []
    prev_reward3_block3 = []
    prev_reward4_block3 = []
    prev_reward5_block3 = []
    prev_noreward1_block3 = []
    prev_noreward2_block3 = []
    prev_noreward3_block3 = []
    prev_noreward4_block3 = []
    prev_noreward5_block3 = []
    prev_stim1_block3 = []
    prev_stim2_block3 = []
    prev_stim3_block3 = []
    prev_stim4_block3 = []
    prev_stim5_block3 = []

    for i in range(5,len(trial1)):
        if trial1[i] == 2:
            fc_target_low_block1.append(2 -target1[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block1.append(target1[i] - 1)  # = 1 if selected high-value, =  0 if selected low-value
            fc_target_high_side_block1.append(target_side1[i])  # current side HV target is on
            prev_reward1_block1.append((2*target1[i-1] - 3)*reward1[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block1.append((2*target1[i-2] - 3)*reward1[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block1.append((2*target1[i-3] - 3)*reward1[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block1.append((2*target1[i-4] - 3)*reward1[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block1.append((2*target1[i-5] - 3)*reward1[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block1.append((2*target1[i-1] - 3)*(1 - reward1[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block1.append((2*target1[i-2] - 3)*(1 - reward1[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block1.append((2*target1[i-3] - 3)*(1 - reward1[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block1.append((2*target1[i-4] - 3)*(1 - reward1[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block1.append((2*target1[i-5] - 3)*(1 - reward1[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim_block1.append(0)
    num_block3 = len(trial3)
    for i in range(5,num_block3):
        if (trial3[i] == 2):
            fc_target_low_block3.append(2 - target3[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block3.append(target3[i] - 1)
            fc_target_high_side_block3.append(target_side3[i])  # current side HV target is on
            prev_reward1_block3.append((2*target3[i-1] - 3)*reward3[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2_block3.append((2*target3[i-2] - 3)*reward3[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3_block3.append((2*target3[i-3] - 3)*reward3[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4_block3.append((2*target3[i-4] - 3)*reward3[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5_block3.append((2*target3[i-5] - 3)*reward3[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1_block3.append((2*target3[i-1] - 3)*(1 - reward3[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2_block3.append((2*target3[i-2] - 3)*(1 - reward3[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3_block3.append((2*target3[i-3] - 3)*(1 - reward3[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4_block3.append((2*target3[i-4] - 3)*(1 - reward3[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5_block3.append((2*target3[i-5] - 3)*(1 - reward3[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim1_block3.append(stim_trials[i - 1]*(2*target3[i-1]- 1))  # = 1 if stim was delivered and = -1 if stim was not delivered
            prev_stim2_block3.append(stim_trials[i - 2]*(2*target3[i-2]- 1))
            prev_stim3_block3.append(stim_trials[i - 3]*(2*target3[i-3]- 1))
            prev_stim4_block3.append(stim_trials[i - 4]*(2*target3[i-4]- 1))
            prev_stim5_block3.append(stim_trials[i - 5]*(2*target3[i-5]- 1))


    '''
    Turn everything into an array
    '''
    fc_target_low_block1 = np.array(fc_target_low_block1)
    fc_target_high_block1 = np.array(fc_target_high_block1)
    fc_target_high_side_block1 = np.array(fc_target_high_side_block1)
    prev_reward1_block1 = np.array(prev_reward1_block1)
    prev_reward2_block1 = np.array(prev_reward2_block1)
    prev_reward3_block1 = np.array(prev_reward3_block1)
    prev_reward4_block1 = np.array(prev_reward4_block1)
    prev_reward5_block1 = np.array(prev_reward5_block1)
    prev_noreward1_block1 = np.array(prev_noreward1_block1)
    prev_noreward2_block1 = np.array(prev_noreward2_block1)
    prev_noreward3_block1 = np.array(prev_noreward3_block1)
    prev_noreward4_block1 = np.array(prev_noreward4_block1)
    prev_noreward5_block1 = np.array(prev_noreward5_block1)
    prev_stim_block1 = np.array(prev_stim_block1)

    fc_target_low_block3 = np.array(fc_target_low_block3)
    fc_target_high_block3 = np.array(fc_target_high_block3)
    fc_target_high_side_block3 = np.array(fc_target_high_side_block3)
    prev_reward1_block3 = np.array(prev_reward1_block3)
    prev_reward2_block3 = np.array(prev_reward2_block3)
    prev_reward3_block3 = np.array(prev_reward3_block3)
    prev_reward4_block3 = np.array(prev_reward4_block3)
    prev_reward5_block3 = np.array(prev_reward5_block3)
    prev_noreward1_block3 = np.array(prev_noreward1_block3)
    prev_noreward2_block3 = np.array(prev_noreward2_block3)
    prev_noreward3_block3 = np.array(prev_noreward3_block3)
    prev_noreward4_block3 = np.array(prev_noreward4_block3)
    prev_noreward5_block3 = np.array(prev_noreward5_block3)
    prev_stim1_block3 = np.array(prev_stim1_block3)
    prev_stim2_block3 = np.array(prev_stim2_block3)
    prev_stim3_block3 = np.array(prev_stim3_block3)
    prev_stim4_block3 = np.array(prev_stim4_block3)
    prev_stim5_block3 = np.array(prev_stim5_block3)

    const_logit_block1 = np.ones(fc_target_low_block1.size)
    const_logit_block3 = np.ones(fc_target_low_block3.size)

    
    '''
    Oraganize data and regress with GLM 
    '''
    x = np.vstack((prev_reward1_block1,prev_reward2_block1,prev_reward3_block1,prev_reward4_block1,prev_reward5_block1,
        prev_noreward1_block1,prev_noreward2_block1,prev_noreward3_block1,prev_noreward4_block1,prev_noreward5_block1,fc_target_high_side_block1))
    x = np.transpose(x)
    x = sm.add_constant(x,prepend='False')

    y = np.vstack((prev_reward1_block3,prev_reward2_block3,prev_reward3_block3,prev_reward4_block3,prev_reward5_block3,
        prev_noreward1_block3,prev_noreward2_block3,prev_noreward3_block3,prev_noreward4_block3,prev_noreward5_block3,
        prev_stim1_block3, prev_stim2_block3, prev_stim3_block3, prev_stim4_block3, prev_stim5_block3))
    y = np.transpose(y)
    y = sm.add_constant(y,prepend='False')

    model_glm_block1 = sm.Logit(fc_target_high_block1,x)
    model_glm_block3 = sm.Logit(fc_target_high_block3,y)
    fit_glm_block1 = model_glm_block1.fit()
    fit_glm_block3 = model_glm_block3.fit()
    #print fit_glm_block1.predict()
    
    '''
    Oraganize data and regress with LogisticRegression
    '''
    """
    d_block1 = {'target_selection': fc_target_high_block1, 
            'prev_reward1': prev_reward1_block1, 
            'prev_reward2': prev_reward2_block1, 
            'prev_reward3': prev_reward3_block1, 
            'prev_reward4': prev_reward4_block1, 
            'prev_reward5': prev_reward5_block1, 
            'prev_noreward1': prev_noreward1_block1, 
            'prev_noreward2': prev_noreward2_block1,
            'prev_noreward3': prev_noreward3_block1, 
            'prev_noreward4': prev_noreward4_block1, 
            'prev_noreward5': prev_noreward5_block1}

    df_block1 = pd.DataFrame(d_block1)

    y_block1, X_block1 = dmatrices('target_selection ~ prev_reward1 + prev_reward2 + prev_reward3 + \
                                    prev_reward4 + prev_reward5 + prev_noreward1 + prev_noreward2 + \
                                    prev_noreward3 + prev_noreward4 + prev_noreward5', df_block1,
                                    return_type = "dataframe")
    
    #print X_block1.columns
    # flatten y_block1 into 1-D array
    y_block1 = np.ravel(y_block1)
    
    d_block3 = {'target_selection': fc_target_high_block3, 
            'prev_reward1': prev_reward1_block3, 
            'prev_reward2': prev_reward2_block3, 
            'prev_reward3': prev_reward3_block3, 
            'prev_reward4': prev_reward4_block3, 
            'prev_reward5': prev_reward5_block3, 
            'prev_noreward1': prev_noreward1_block3, 
            'prev_noreward2': prev_noreward2_block3,
            'prev_noreward3': prev_noreward3_block3, 
            'prev_noreward4': prev_noreward4_block3, 
            'prev_noreward5': prev_noreward5_block3, 
            'prev_stim1': prev_stim1_block3,
            'prev_stim2': prev_stim2_block3,
            'prev_stim3': prev_stim3_block3,
            'prev_stim4': prev_stim4_block3,
            'prev_stim5': prev_stim5_block3}
    df_block3 = pd.DataFrame(d_block3)

    y_block3, X_block3 = dmatrices('target_selection ~ prev_reward1 + prev_reward2 + prev_reward3 + \
                                    prev_reward4 + prev_reward5 + prev_noreward1 + prev_noreward2 + \
                                    prev_noreward3 + prev_noreward4 + prev_noreward5 + prev_stim1 + \
                                    prev_stim2 + prev_stim3 + prev_stim4 + prev_stim5', df_block3,
                                    return_type = "dataframe")
    
    # flatten y_block3 into 1-D array
    y_block3 = np.ravel(y_block3)

    # Split data into train and test sets
    X_block1_train, X_block1_test, y_block1_train, y_block1_test = train_test_split(X_block1,y_block1,test_size = 0.3, random_state = 0)
    X_block3_train, X_block3_test, y_block3_train, y_block3_test = train_test_split(X_block3,y_block3,test_size = 0.3, random_state = 0)

    # instantiate a logistic regression model, and fit with X and y training sets
    model_block1 = LogisticRegression()
    model_block3 = LogisticRegression()
    model_block1 = model_block1.fit(X_block1_train, y_block1_train)
    model_block3 = model_block3.fit(X_block3_train, y_block3_train)
    y_block1_score = model_block1.decision_function(X_block1_test)
    y_block3_score = model_block3.decision_function(X_block3_test)

    y_block1_nullscore = np.ones(len(y_block1_score))
    y_block3_nullscore = np.ones(len(y_block3_score))


    # Compute ROC curve and ROC area for each class (low value and high value)
    '''
    fpr_block1 = dict()
    tpr_block1 = dict()
    fpr_block3 = dict()
    tpr_block3 = dict()
    roc_auc_block1 = dict()
    roc_auc_block3 = dict()
    '''
    
    
    fpr_block1, tpr_block1, thresholds_block1 = roc_curve(y_block1_test,y_block1_score)
    roc_auc_block1 = auc(fpr_block1,tpr_block1)
    fpr_block3, tpr_block3, thresholds_block3 = roc_curve(y_block3_test,y_block3_score)
    roc_auc_block3 = auc(fpr_block3,tpr_block3)
    fpr_null_block1, tpr_null_block1, thresholds_null_block1 = roc_curve(y_block1_test,y_block1_nullscore)
    roc_nullauc_block1 = auc(fpr_null_block1,tpr_null_block1)
    fpr_null_block3, tpr_null_block3, thresholds_null_block3 = roc_curve(y_block3_test,y_block3_nullscore)
    roc_nullauc_block3 = auc(fpr_null_block3,tpr_null_block3)

    plt.figure()
    plt.plot(fpr_block1,tpr_block1,'r',label="Block 1 (area = %0.2f)" % roc_auc_block1)
    plt.plot(fpr_null_block1,tpr_null_block1,'r--',label="Block 1 - Null (area = %0.2f)" % roc_nullauc_block1)
    plt.plot(fpr_block3,tpr_block3,'m',label="Block 3 (area = %0.2f)" % roc_auc_block3)
    plt.plot(fpr_null_block3,tpr_null_block3,'m--',label="Block 3 - Null (area = %0.2f)" % roc_nullauc_block3)
    plt.plot([0,1],[0,1],'b--')
    #plt.plot(fpr_block1[1],tpr_block1[1],label="Class HV (area = %0.2f)" % roc_auc_block1[1])
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc=4)
    plt.show()

    # Predict class labels for the test set
    predicted_block1 = model_block1.predict(X_block1_test)
    probs_block1 = model_block1.predict_proba(X_block1_test)
    predicted_block3 = model_block3.predict(X_block3_test)
    probs_block3 = model_block3.predict_proba(X_block3_test)

    # Generate evaluation metrics
    print "Block 1 accuracy:", metrics.accuracy_score(y_block1_test, predicted_block1)
    print "Block 1 ROC area under curve:", metrics.roc_auc_score(y_block1_test, probs_block1[:,1])
    print 'Null accuracy rate for Block 1:',np.max([y_block1_test.mean(),1 - y_block1_test.mean()])
    
    print "Block 3 accuracy:", metrics.accuracy_score(y_block3_test, predicted_block3)
    print "Block 3 ROC area under curve:", metrics.roc_auc_score(y_block3_test, probs_block3[:,1])
    print 'Null accuracy rate for Block 3:',np.max([y_block3_test.mean(),1 - y_block3_test.mean()])
    
    
    # Model evaluation using 10-fold cross-validation
    scores_block1 = cross_val_score(LogisticRegression(),X_block1,y_block1,scoring='accuracy',cv=10)
    scores_block3 = cross_val_score(LogisticRegression(),X_block3,y_block3,scoring='accuracy',cv=10)
    print "Block 1 CV scores:", scores_block1
    print "Block 1 Avg CV score:", scores_block1.mean()
    print "Block 3 CV scores:", scores_block3
    print "Block 3 Avg CV score:", scores_block3.mean()
    """

    """
    # check the accuracy on the training set
    print 'Model accuracy for Block1:',fit_glm_block1.score(X_block1, y_block1)
    print 'Null accuracy rate for Block1:',np.max([y_block1.mean(),1 - y_block1.mean()])

    print 'Model accuracy for Block3:',fit_glm_block3.score(X_block3, y_block3)
    print 'Null accuracy rate for Block3:',np.max([y_block3.mean(),1 - y_block3.mean()])
    
    
    # examine the coefficients
    print pd.DataFrame(zip(X_block1.columns, np.transpose(model_block1.coef_)))
    print pd.DataFrame(zip(X_block3.columns, np.transpose(model_block3.coef_)))
    """
    
    return fit_glm_block1, fit_glm_block3
    #return model_block1, model_block3, predicted_block1, predicted_block3

def probabilisticFreeChoicePilotTask_logisticRegression_sepRegressors(reward1, target1, trial1, reward3, target3, trial3, stim_trials):

    
    '''
    Previous rewards and no rewards
    '''
    fc_target_low_block1 = []
    fc_target_high_block1 = []
    fc_prob_low_block1 = []
    prev_hv_reward1_block1 = []
    prev_hv_reward2_block1 = []
    prev_hv_reward3_block1 = []
    prev_hv_reward4_block1 = []
    prev_hv_reward5_block1 = []
    prev_lv_reward1_block1 = []
    prev_lv_reward2_block1 = []
    prev_lv_reward3_block1 = []
    prev_lv_reward4_block1 = []
    prev_lv_reward5_block1 = []


    fc_target_low_block3 = []
    fc_target_high_block3 = []
    fc_prob_low_block3 = []
    prev_hv_reward1_block3 = []
    prev_hv_reward2_block3 = []
    prev_hv_reward3_block3 = []
    prev_hv_reward4_block3 = []
    prev_hv_reward5_block3 = []
    prev_lv_reward1_block3 = []
    prev_lv_reward2_block3 = []
    prev_lv_reward3_block3 = []
    prev_lv_reward4_block3 = []
    prev_lv_reward5_block3 = []
    prev_stim1_block3 = []
    prev_stim2_block3 = []
    prev_stim3_block3 = []
    prev_stim4_block3 = []
    prev_stim5_block3 = []

    for i in range(5,len(trial1)):
        if (trial1[i] == 2):
            fc_target_low_block1.append(2 -target1[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block1.append(target1[i] - 1)  # = 1 if selected high-value, =  0 if selected low-value
            prev_hv_reward1_block1.append((target1[i-1] - 1)*(2*reward1[i-1]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward2_block1.append((target1[i-2] - 1)*(2*reward1[i-2]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward3_block1.append((target1[i-3] - 1)*(2*reward1[i-3]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward4_block1.append((target1[i-4] - 1)*(2*reward1[i-4]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward5_block1.append((target1[i-5] - 1)*(2*reward1[i-5]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_lv_reward1_block1.append((2 - target1[i-1])*(2*reward1[i-1]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward2_block1.append((2 - target1[i-2])*(2*reward1[i-2]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward3_block1.append((2 - target1[i-3])*(2*reward1[i-3]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward4_block1.append((2 - target1[i-4])*(2*reward1[i-4]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward5_block1.append((2 - target1[i-5])*(2*reward1[i-5]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
    num_block3 = len(trial3)
    for i in range(5,num_block3):
        if (trial3[i] == 2):  # only free-choice trials following an instructed trial with stim
            fc_target_low_block3.append(2 - target3[i])   # = 1 if selected low-value, = 0 if selected high-value
            fc_target_high_block3.append(target3[i] - 1)
            prev_hv_reward1_block3.append((target3[i-1] - 1)*(2*reward3[i-1]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward2_block3.append((target3[i-2] - 1)*(2*reward3[i-2]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward3_block3.append((target3[i-3] - 1)*(2*reward3[i-3]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward4_block3.append((target3[i-4] - 1)*(2*reward3[i-4]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_hv_reward5_block3.append((target3[i-5] - 1)*(2*reward3[i-5]-1))  # = -1 if selected high-value and not rewarded, = 1 if selected high-value and rewarded
            prev_lv_reward1_block3.append((2 - target3[i-1])*(2*reward3[i-1]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward2_block3.append((2 - target3[i-2])*(2*reward3[i-2]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward3_block3.append((2 - target3[i-3])*(2*reward3[i-3]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward4_block3.append((2 - target3[i-4])*(2*reward3[i-4]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_lv_reward5_block3.append((2 - target3[i-5])*(2*reward3[i-5]-1))  # = -1 if selected low-value and not rewarded, = 1 if selected low-value and rewarded
            prev_stim1_block3.append(stim_trials[i - 1]*(2*target3[i-1]- 1))
            prev_stim2_block3.append(stim_trials[i - 2]*(2*target3[i-2]- 1))
            prev_stim3_block3.append(stim_trials[i - 3]*(2*target3[i-3]- 1))
            prev_stim4_block3.append(stim_trials[i - 4]*(2*target3[i-4]- 1))
            prev_stim5_block3.append(stim_trials[i - 5]*(2*target3[i-5]- 1))


    '''
    Turn everything into an array
    '''
    fc_target_low_block1 = np.array(fc_target_low_block1)
    fc_target_high_block1 = np.array(fc_target_high_block1)
    prev_hv_reward1_block1 = np.array(prev_hv_reward1_block1)
    prev_hv_reward2_block1 = np.array(prev_hv_reward2_block1)
    prev_hv_reward3_block1 = np.array(prev_hv_reward3_block1)
    prev_hv_reward4_block1 = np.array(prev_hv_reward4_block1)
    prev_hv_reward5_block1 = np.array(prev_hv_reward5_block1)
    prev_lv_reward1_block1 = np.array(prev_lv_reward1_block1)
    prev_lv_reward2_block1 = np.array(prev_lv_reward2_block1)
    prev_lv_reward3_block1 = np.array(prev_lv_reward3_block1)
    prev_lv_reward4_block1 = np.array(prev_lv_reward4_block1)
    prev_lv_reward5_block1 = np.array(prev_lv_reward5_block1)

    fc_target_low_block3 = np.array(fc_target_low_block3)
    fc_target_high_block3 = np.array(fc_target_high_block3)
    prev_hv_reward1_block3 = np.array(prev_hv_reward1_block3)
    prev_hv_reward2_block3 = np.array(prev_hv_reward2_block3)
    prev_hv_reward3_block3 = np.array(prev_hv_reward3_block3)
    prev_hv_reward4_block3 = np.array(prev_hv_reward4_block3)
    prev_hv_reward5_block3 = np.array(prev_hv_reward5_block3)
    prev_lv_reward1_block3 = np.array(prev_lv_reward1_block3)
    prev_lv_reward2_block3 = np.array(prev_lv_reward2_block3)
    prev_lv_reward3_block3 = np.array(prev_lv_reward3_block3)
    prev_lv_reward4_block3 = np.array(prev_lv_reward4_block3)
    prev_lv_reward5_block3 = np.array(prev_lv_reward5_block3)
    prev_stim1_block3 = np.array(prev_stim1_block3)
    prev_stim2_block3 = np.array(prev_stim2_block3)
    prev_stim3_block3 = np.array(prev_stim3_block3)
    prev_stim4_block3 = np.array(prev_stim4_block3)
    prev_stim5_block3 = np.array(prev_stim5_block3)

    const_logit_block1 = np.ones(fc_target_low_block1.size)
    const_logit_block3 = np.ones(fc_target_low_block3.size)

    
    '''
    Oraganize data and regress with GLM 
    '''
    x = np.vstack((prev_hv_reward1_block1,prev_hv_reward2_block1,prev_hv_reward3_block1,prev_hv_reward4_block1,prev_hv_reward5_block1,
        prev_lv_reward1_block1,prev_lv_reward2_block1,prev_lv_reward3_block1,prev_lv_reward4_block1,prev_lv_reward5_block1))
    x = np.transpose(x)
    x = sm.add_constant(x,prepend='False')

    y = np.vstack((prev_hv_reward1_block3,prev_hv_reward2_block3,prev_hv_reward3_block3,prev_hv_reward4_block3,prev_hv_reward5_block3,
        prev_lv_reward1_block3,prev_lv_reward2_block3,prev_lv_reward3_block3,prev_lv_reward4_block3,prev_lv_reward5_block3, 
        prev_stim1_block3, prev_stim2_block3, prev_stim3_block3, prev_stim4_block3, prev_stim5_block3))
    y = np.transpose(y)
    y = sm.add_constant(y,prepend='False')

    model_glm_block1 = sm.Logit(fc_target_high_block1,x)
    model_glm_block3 = sm.Logit(fc_target_high_block3,y)
    fit_glm_block1 = model_glm_block1.fit()
    fit_glm_block3 = model_glm_block3.fit()
    """

    d_block1 = {'target_selection': fc_target_low_block1, 
            'prev_hv_reward1': prev_hv_reward1_block1, 
            'prev_hv_reward2': prev_hv_reward2_block1, 
            'prev_hv_reward3': prev_hv_reward3_block1, 
            'prev_hv_reward4': prev_hv_reward4_block1, 
            'prev_hv_reward5': prev_hv_reward5_block1, 
            'prev_lv_reward1': prev_lv_reward1_block1, 
            'prev_lv_reward2': prev_lv_reward2_block1, 
            'prev_lv_reward3': prev_lv_reward3_block1, 
            'prev_lv_reward4': prev_lv_reward4_block1, 
            'prev_lv_reward5': prev_lv_reward5_block1}
    df_block1 = pd.DataFrame(d_block1)

    y_block1, X_block1 = dmatrices('target_selection ~ prev_hv_reward1 + prev_hv_reward2 + prev_hv_reward3 + \
                                    prev_hv_reward4 + prev_hv_reward5 + prev_lv_reward1 + prev_lv_reward2 + \
                                    prev_lv_reward3 + prev_lv_reward4 + prev_lv_reward5', df_block1,
                                    return_type = "dataframe")

    #print X_block1.columns
    # flatten y_block1 into 1-D array
    y_block1 = np.ravel(y_block1)

    d_block3 = {'target_selection': fc_target_low_block3, 
            'prev_hv_reward1': prev_hv_reward1_block3, 
            'prev_hv_reward2': prev_hv_reward2_block3, 
            'prev_hv_reward3': prev_hv_reward3_block3, 
            'prev_hv_reward4': prev_hv_reward4_block3, 
            'prev_hv_reward5': prev_hv_reward5_block3, 
            'prev_lv_reward1': prev_lv_reward1_block3, 
            'prev_lv_reward2': prev_lv_reward2_block3, 
            'prev_lv_reward3': prev_lv_reward3_block3, 
            'prev_lv_reward4': prev_lv_reward4_block3, 
            'prev_lv_reward5': prev_lv_reward5_block3,
            'prev_stim1': prev_stim1_block3,
            'prev_stim2': prev_stim2_block3,
            'prev_stim3': prev_stim3_block3,
            'prev_stim4': prev_stim4_block3,
            'prev_stim5': prev_stim5_block3}
    df_block3 = pd.DataFrame(d_block3)

    y_block3, X_block3 = dmatrices('target_selection ~ prev_hv_reward1 + prev_hv_reward2 + prev_hv_reward3 + \
                                    prev_hv_reward4 + prev_hv_reward5 + prev_lv_reward1 + prev_lv_reward2 + \
                                    prev_lv_reward3 + prev_lv_reward4 + prev_lv_reward5 + prev_stim1 + \
                                    prev_stim2 + prev_stim3 + prev_stim4 + prev_stim5', df_block3,
                                    return_type = "dataframe")
    
    # flatten y_block3 into 1-D array
    y_block3 = np.ravel(y_block3)

    # Split data into train and test sets
    X_block1_train, X_block1_test, y_block1_train, y_block1_test = train_test_split(X_block1,y_block1,test_size = 0.3, random_state = 0)
    X_block3_train, X_block3_test, y_block3_train, y_block3_test = train_test_split(X_block3,y_block3,test_size = 0.3, random_state = 0)

    # instantiate a logistic regression model, and fit with X and y training sets
    model_block1 = LogisticRegression()
    model_block3 = LogisticRegression()
    model_block1 = model_block1.fit(X_block1_train, y_block1_train)
    model_block3 = model_block3.fit(X_block3_train, y_block3_train)
    y_block1_score = model_block1.decision_function(X_block1_test)
    y_block3_score = model_block3.decision_function(X_block3_test)


    # Compute ROC curve and ROC area for each class (low value and high value)
    '''
    fpr_block1 = dict()
    tpr_block1 = dict()
    fpr_block3 = dict()
    tpr_block3 = dict()
    roc_auc_block1 = dict()
    roc_auc_block3 = dict()
    '''
    
    
    fpr_block1, tpr_block1, thresholds_block1 = roc_curve(y_block1_test,y_block1_score)
    roc_auc_block1 = auc(fpr_block1,tpr_block1)
    fpr_block3, tpr_block3, thresholds_block3 = roc_curve(y_block3_test,y_block3_score)
    roc_auc_block3 = auc(fpr_block3,tpr_block3)

    plt.figure()
    plt.plot(fpr_block1,tpr_block1,'r',label="Block 1 (area = %0.2f)" % roc_auc_block1)
    plt.plot(fpr_block3,tpr_block3,'m',label="Block 3 (area = %0.2f)" % roc_auc_block3)
    plt.plot([0,1],[0,1],'b--')
    #plt.plot(fpr_block1[1],tpr_block1[1],label="Class HV (area = %0.2f)" % roc_auc_block1[1])
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend()
    plt.show()
    

    # Predict class labels for the test set
    predicted_block1 = model_block1.predict(X_block1_test)
    probs_block1 = model_block1.predict_proba(X_block1_test)
    predicted_block3 = model_block3.predict(X_block3_test)
    probs_block3 = model_block3.predict_proba(X_block3_test)

    # Generate evaluation metrics
    print "Block 1 accuracy:", metrics.accuracy_score(y_block1_test, predicted_block1)
    print "Block 1 ROC area under curve:", metrics.roc_auc_score(y_block1_test, probs_block1[:,1])
    print 'Null accuracy rate for Block 1:',np.max([y_block1_test.mean(),1 - y_block1_test.mean()])
    
    print "Block 3 accuracy:", metrics.accuracy_score(y_block3_test, predicted_block3)
    print "Block 3 ROC area under curve:", metrics.roc_auc_score(y_block3_test, probs_block3[:,1])
    print 'Null accuracy rate for Block 3:',np.max([y_block3_test.mean(),1 - y_block3_test.mean()])
    
    
    # Model evaluation using 10-fold cross-validation
    scores_block1 = cross_val_score(LogisticRegression(),X_block1,y_block1,scoring='accuracy',cv=10)
    scores_block3 = cross_val_score(LogisticRegression(),X_block3,y_block3,scoring='accuracy',cv=10)
    print "Block 1 CV scores:", scores_block1
    print "Block 1 Avg CV score:", scores_block1.mean()
    print "Block 3 CV scores:", scores_block3
    print "Block 3 Avg CV score:", scores_block3.mean()
    """

    """
    # check the accuracy on the training set
    print 'Model accuracy for Block1:',model_block1.score(X_block1, y_block1)
    print 'Null accuracy rate for Block1:',np.max([y_block1.mean(),1 - y_block1.mean()])

    print 'Model accuracy for Block3:',model_block3.score(X_block3, y_block3)
    print 'Null accuracy rate for Block3:',np.max([y_block3.mean(),1 - y_block3.mean()])
    
    
    # examine the coefficients
    print pd.DataFrame(zip(X_block1.columns, np.transpose(model_block1.coef_)))
    print pd.DataFrame(zip(X_block3.columns, np.transpose(model_block3.coef_)))
    """
    
    return fit_glm_block1, fit_glm_block3
    #return model_block1, model_block3, predicted_block1, predicted_block3

stim_num_days = len(stim_hdf_list)
sham_num_days = len(sham_hdf_list)
hv_num_days = len(hv_hdf_list)

stim_counter_hdf = 0
stim_reward1 = []
stim_target1 = []
stim_trial1 = []
stim_target_side1 = []
stim_reward3 = []
stim_target3 = []
stim_trial3 = []
stim_target_side3 = []
stim_stim_trials = []

sham_counter_hdf = 0
sham_reward1 = []
sham_target1 = []
sham_trial1 = []
sham_target_side1 = []
sham_reward3 = []
sham_target3 = []
sham_trial3 = []
sham_target_side3 = []
sham_stim_trials = []

hv_counter_hdf = 0
hv_reward1 = []
hv_target1 = []
hv_trial1 = []
hv_target_side1 = []
hv_reward3 = []
hv_target3 = []
hv_trial3 = []
hv_target_side3 = []
hv_stim_trials = []

color_stim =iter(cm.rainbow(np.linspace(0,1,stim_num_days)))
color_sham = iter(cm.rainbow(np.linspace(0,1,sham_num_days)))
color_hv = iter(cm.rainbow(np.linspace(0,1,hv_num_days)))

stim_learning_ratio = np.zeros(stim_num_days)
sham_learning_ratio = np.zeros(sham_num_days)
hv_learning_ratio = np.zeros(hv_num_days)
stim_alpha_block1 = np.zeros(stim_num_days)
stim_alpha_block3 = np.zeros(stim_num_days)
stim_alpha_block3_Qmultiplicative = np.zeros(stim_num_days)
sham_alpha_block1 = np.zeros(sham_num_days)
sham_alpha_block3 = np.zeros(sham_num_days)
hv_alpha_block1 = np.zeros(hv_num_days)
hv_alpha_block3 = np.zeros(hv_num_days)
hv_alpha_block3_Qmultiplicative = np.zeros(hv_num_days)

stim_beta_block1 = np.zeros(stim_num_days)
stim_beta_block3 = np.zeros(stim_num_days)
stim_beta_block3_Qmultiplicative = np.zeros(stim_num_days)
sham_beta_block1 = np.zeros(sham_num_days)
sham_beta_block3 = np.zeros(sham_num_days)
hv_beta_block1 = np.zeros(hv_num_days)
hv_beta_block3 = np.zeros(hv_num_days)
hv_beta_block3_Qmultiplicative = np.zeros(hv_num_days)

stim_gamma_block3_Qmultiplicative = np.zeros(stim_num_days)
hv_gamma_block3_Qmultiplicative = np.zeros(hv_num_days)

stim_prob_choose_low = np.zeros(stim_num_days)
sham_prob_choose_low = np.zeros(sham_num_days)
hv_prob_choose_low = np.zeros(hv_num_days)
stim_prob_choose_left_block1 = np.zeros(stim_num_days)
sham_prob_choose_left_block1 = np.zeros(sham_num_days)
hv_prob_choose_left_block1 = np.zeros(hv_num_days)
stim_prob_choose_left_block3 = np.zeros(stim_num_days)
sham_prob_choose_left_block3 = np.zeros(sham_num_days)
hv_prob_choose_left_block3 = np.zeros(hv_num_days)

stim_prob_choose_left_and_low_block1 = np.zeros(stim_num_days)
stim_prob_choose_left_and_high_block1 = np.zeros(stim_num_days)
stim_prob_choose_left_and_low_block3 = np.zeros(stim_num_days)
stim_prob_choose_left_and_high_block3 = np.zeros(stim_num_days)

sham_prob_choose_left_and_low_block1 = np.zeros(sham_num_days)
sham_prob_choose_left_and_high_block1 = np.zeros(sham_num_days)
sham_prob_choose_left_and_low_block3 = np.zeros(sham_num_days)
sham_prob_choose_left_and_high_block3 = np.zeros(sham_num_days)

hv_prob_choose_left_and_low_block1 = np.zeros(hv_num_days)
hv_prob_choose_left_and_high_block1 = np.zeros(hv_num_days)
hv_prob_choose_left_and_low_block3 = np.zeros(hv_num_days)
hv_prob_choose_left_and_high_block3 = np.zeros(hv_num_days)

stim_counter = 0
sham_counter = 0
hv_counter = 0

for name in stim_hdf_list:
    
    print name
    full_name = hdf_prefix + name

    '''
    Compute task performance.
    '''
    reward_block1, target_block1, trial_block1, target_side1, reward_block3, target_block3, trial_block3, target_side3, stim_trials_block = FreeChoicePilotTask_Behavior(full_name)
    prob_choose_low = FirstChoiceAfterStim(target_block3,trial_block3, stim_trials_block)

    fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
    fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))
    
    #prob_choose_low = 1 - np.sum(target_block3[fc_trial_ind] - 1)/len(target_block3[fc_trial_ind])  # for prob of choosing low value target over all trials

    prob_choose_left_block3 = 1 - np.sum(target_side3[fc_trial_ind_block3])/len(target_side3[fc_trial_ind_block3])  # for prob of choosing low value target over all trials
    choose_left_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],1))
    prob_choose_left_and_low_block3 = float(np.sum(choose_left_and_low_block3))/len(choose_left_and_low_block3)
    choose_left_and_high_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],2))
    prob_choose_left_and_high_block3 = float(np.sum(choose_left_and_high_block3))/len(choose_left_and_high_block3)

    
    prob_choose_left_block1 = 1 - np.sum(target_side1[fc_trial_ind_block1])/len(target_side1[fc_trial_ind_block1])  # for prob of choosing low value target over all trials
    choose_left_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],1))
    prob_choose_left_and_low_block1 = float(np.sum(choose_left_and_low_block1))/len(choose_left_and_low_block1)
    choose_left_and_high_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],2))
    prob_choose_left_and_high_block1 = float(np.sum(choose_left_and_high_block1))/len(choose_left_and_high_block1)

    stim_prob_choose_left_block3[stim_counter] = prob_choose_left_block3
    stim_prob_choose_left_block1[stim_counter] = prob_choose_left_block1

    stim_prob_choose_left_and_low_block1[stim_counter] = prob_choose_left_and_low_block1
    stim_prob_choose_left_and_high_block1[stim_counter] = prob_choose_left_and_high_block1
    stim_prob_choose_left_and_low_block3[stim_counter] = prob_choose_left_and_low_block3
    stim_prob_choose_left_and_high_block3[stim_counter] = prob_choose_left_and_high_block3
    stim_prob_choose_low[stim_counter] = prob_choose_low

    '''
    Build vector for cumulative regression
    '''

    stim_reward1.extend(reward_block1.tolist())
    stim_target1.extend(target_block1.tolist())
    stim_trial1.extend(trial_block1.tolist())
    stim_target_side1.extend(target_side1.tolist())
    stim_reward3.extend(reward_block3.tolist())
    stim_target3.extend(target_block3.tolist())
    stim_trial3.extend(trial_block3.tolist())
    stim_target_side3.extend(target_side3.tolist())
    stim_stim_trials.extend(stim_trials_block.tolist()) 

    """
    '''
    Do per day regression
    '''
    c = next(color_stim)

    fit_glm_block1, fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(reward_block1, target_block1, trial_block1, reward_block3, target_block3, trial_block3, stim_trials_block)
    params_block1 = fit_glm_block1.params
    params_block3 = fit_glm_block3.params

    relative_action_value_block1, relative_action_value_block3, prob_choice_block1, prob_choice_block3 = computeProbabilityChoiceWithRegressors(params_block1, params_block3, reward_block1, target_block1, trial_block1, reward_block3, target_block3, trial_block3, stim_trials_block)

    sorted_ind = np.argsort(relative_action_value_block3)
    relative_action_value_block3 = np.array(relative_action_value_block3)

    plt.figure(0)
    plt.plot(relative_action_value_block3[sorted_ind],prob_choice_block3[sorted_ind],color=c,marker='*',label='Stim - %s' % name)
    plt.legend(loc=4)
    plt.xlabel('Relative Action Value')
    plt.ylabel('P(Choose High-Value Target)')
    plt.xlim([-5,5])
    plt.ylim([0.0,1.05])
    
    """
    
    '''
    Get soft-max decision fit
    '''
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block1, target_block1, trial_block1), bounds=[(0,1),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward_block1,target_block1, trial_block1)
    
    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block3, target_block3, trial_block3), bounds=[(0,1),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3)

    stim_alpha_block1[stim_counter] = alpha_ml_block1
    stim_alpha_block3[stim_counter] = alpha_ml_block3
    stim_beta_block1[stim_counter] = beta_ml_block1
    stim_beta_block3[stim_counter] = beta_ml_block3
    stim_learning_ratio[stim_counter] = float(alpha_ml_block3)/alpha_ml_block1

    '''
    Get fit with Multiplicative stimulation parameter in Q-value update equation
    '''
    nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter(*args)
    result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(0,None)])
    alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3_Qmultiplicative, max_loglikelihood3 = RLPerformance_multiplicative_Qstimparameter([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    
    stim_alpha_block3_Qmultiplicative[stim_counter] = alpha_ml_block3_Qmultiplicative
    stim_beta_block3_Qmultiplicative[stim_counter] = beta_ml_block3_Qmultiplicative
    stim_gamma_block3_Qmultiplicative[stim_counter] = gamma_ml_block3_Qmultiplicative
    


    stim_counter += 1
    
for name in sham_hdf_list:
    
    print name
    full_name = hdf_prefix + name

    '''
    Compute task performance.
    '''
    reward_block1, target_block1, trial_block1, target_side1, reward_block3, target_block3, trial_block3, target_side3, stim_trials_block = FreeChoicePilotTask_Behavior(full_name)
    prob_choose_low = FirstChoiceAfterStim(target_block3,trial_block3, stim_trials_block)

    fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
    fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))
    #prob_choose_low = 1 - np.sum(target_block3[fc_trial_ind] - 1)/len(target_block3[fc_trial_ind])  # for prob of choosing low value target over all trials
    prob_choose_left_block3 = 1 - np.sum(target_side3[fc_trial_ind_block3])/len(target_side3[fc_trial_ind_block3])  # for prob of choosing low value target over all trials
    choose_left_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],1))
    prob_choose_left_and_low_block3 = float(np.sum(choose_left_and_low_block3))/len(choose_left_and_low_block3)
    choose_left_and_high_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],2))
    prob_choose_left_and_high_block3 = float(np.sum(choose_left_and_high_block3))/len(choose_left_and_high_block3)

    
    prob_choose_left_block1 = 1 - np.sum(target_side1[fc_trial_ind_block1])/len(target_side1[fc_trial_ind_block1])  # for prob of choosing low value target over all trials
    choose_left_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],1))
    prob_choose_left_and_low_block1 = float(np.sum(choose_left_and_low_block1))/len(choose_left_and_low_block1)
    choose_left_and_high_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],2))
    prob_choose_left_and_high_block1 = float(np.sum(choose_left_and_high_block1))/len(choose_left_and_high_block1)

    sham_prob_choose_left_block3[sham_counter] = prob_choose_left_block3
    sham_prob_choose_left_block1[sham_counter] = prob_choose_left_block1

    sham_prob_choose_left_and_low_block1[sham_counter] = prob_choose_left_and_low_block1
    sham_prob_choose_left_and_high_block1[sham_counter] = prob_choose_left_and_high_block1
    sham_prob_choose_left_and_low_block3[sham_counter] = prob_choose_left_and_low_block3
    sham_prob_choose_left_and_high_block3[sham_counter] = prob_choose_left_and_high_block3
    sham_prob_choose_low[sham_counter] = prob_choose_low

    sham_reward1.extend(reward_block1.tolist())
    sham_target1.extend(target_block1.tolist())
    sham_trial1.extend(trial_block1.tolist())
    sham_target_side1.extend(target_side1.tolist())
    sham_reward3.extend(reward_block3.tolist())
    sham_target3.extend(target_block3.tolist())
    sham_trial3.extend(trial_block3.tolist())
    sham_target_side3.extend(target_side3.tolist())
    sham_stim_trials.extend(stim_trials_block.tolist()) 

    """
    '''
    Do per day regression
    '''
    c = next(color_sham)

    fit_glm_block1, fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(reward_block1, target_block1, trial_block1, reward_block3, target_block3, trial_block3, stim_trials_block)
    params_block1 = fit_glm_block1.params
    params_block3 = fit_glm_block3.params

    relative_action_value_block1, relative_action_value_block3, prob_choice_block1, prob_choice_block3 = computeProbabilityChoiceWithRegressors(params_block1, params_block3, reward_block1, target_block1, trial_block1, reward_block3, target_block3, trial_block3, stim_trials_block)

    sorted_ind = np.argsort(relative_action_value_block3)
    relative_action_value_block3 = np.array(relative_action_value_block3)

    plt.figure(0)
    plt.plot(relative_action_value_block3[sorted_ind],prob_choice_block3[sorted_ind],color=c,marker='o',label='Sham - %s' % name)
    plt.legend(loc=4)
    plt.xlabel('Relative Action Value')
    plt.ylabel('P(Choose High-Value Target)')
    plt.xlim([-5,5])
    plt.ylim([0.0,1.05])

    """
    '''
    Get soft-max decision fit
    '''
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block1, target_block1, trial_block1), bounds=[(0,1),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward_block1,target_block1, trial_block1)
    
    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block3, target_block3, trial_block3), bounds=[(0,1),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3)
    

    sham_alpha_block1[sham_counter] = alpha_ml_block1
    sham_alpha_block3[sham_counter] = alpha_ml_block3
    sham_beta_block1[sham_counter] = beta_ml_block1
    sham_beta_block3[sham_counter] = beta_ml_block3
    sham_learning_ratio[sham_counter] = float(alpha_ml_block3)/alpha_ml_block1
    sham_counter += 1

for name in hv_hdf_list:
    
    print name
    full_name = hdf_prefix + name

    '''
    Compute task performance.
    '''
    reward_block1, target_block1, trial_block1, target_side1, reward_block3, target_block3, trial_block3, target_side3, stim_trials_block = FreeChoicePilotTask_Behavior(full_name)
    prob_choose_low = FirstChoiceAfterStim(target_block3,trial_block3, stim_trials_block)

    fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
    fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))
    
    #prob_choose_low = 1 - np.sum(target_block3[fc_trial_ind] - 1)/len(target_block3[fc_trial_ind])  # for prob of choosing low value target over all trials

    prob_choose_left_block3 = 1 - np.sum(target_side3[fc_trial_ind_block3])/len(target_side3[fc_trial_ind_block3])  # for prob of choosing low value target over all trials
    choose_left_and_low_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],1))
    prob_choose_left_and_low_block3 = float(np.sum(choose_left_and_low_block3))/len(choose_left_and_low_block3)
    choose_left_and_high_block3 = (np.equal(target_side3[fc_trial_ind_block3],0)&np.equal(target_block3[fc_trial_ind_block3],2))
    prob_choose_left_and_high_block3 = float(np.sum(choose_left_and_high_block3))/len(choose_left_and_high_block3)

    
    prob_choose_left_block1 = 1 - np.sum(target_side1[fc_trial_ind_block1])/len(target_side1[fc_trial_ind_block1])  # for prob of choosing low value target over all trials
    choose_left_and_low_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],1))
    prob_choose_left_and_low_block1 = float(np.sum(choose_left_and_low_block1))/len(choose_left_and_low_block1)
    choose_left_and_high_block1 = (np.equal(target_side1[fc_trial_ind_block1],0)&np.equal(target_block1[fc_trial_ind_block1],2))
    prob_choose_left_and_high_block1 = float(np.sum(choose_left_and_high_block1))/len(choose_left_and_high_block1)

    hv_prob_choose_left_block3[hv_counter] = prob_choose_left_block3
    hv_prob_choose_left_block1[hv_counter] = prob_choose_left_block1

    hv_prob_choose_left_and_low_block1[hv_counter] = prob_choose_left_and_low_block1
    hv_prob_choose_left_and_high_block1[hv_counter] = prob_choose_left_and_high_block1
    hv_prob_choose_left_and_low_block3[hv_counter] = prob_choose_left_and_low_block3
    hv_prob_choose_left_and_high_block3[hv_counter] = prob_choose_left_and_high_block3
    hv_prob_choose_low[hv_counter] = prob_choose_low

    hv_reward1.extend(reward_block1.tolist())
    hv_target1.extend(target_block1.tolist())
    hv_trial1.extend(trial_block1.tolist())
    hv_target_side1.extend(target_side1.tolist())
    hv_reward3.extend(reward_block3.tolist())
    hv_target3.extend(target_block3.tolist())
    hv_trial3.extend(trial_block3.tolist())
    hv_target_side3.extend(target_side3.tolist())
    hv_stim_trials.extend(stim_trials_block.tolist()) 

    '''
    Get soft-max decision fit
    '''
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block1, target_block1, trial_block1), bounds=[(0,1),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward_block1,target_block1, trial_block1)
    
    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block3, target_block3, trial_block3), bounds=[(0,1),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3)
    
    '''
    Get fit with Multiplicative stimulation parameter in Q-value update equation
    '''
    nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter_HVTarget(*args)
    result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0, None),(1,None)])
    alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3_Qmultiplicative, max_loglikelihood3 = RLPerformance_multiplicative_Qstimparameter_HVTarget([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    
    hv_alpha_block3_Qmultiplicative[hv_counter] = alpha_ml_block3_Qmultiplicative
    hv_beta_block3_Qmultiplicative[hv_counter] = beta_ml_block3_Qmultiplicative
    hv_gamma_block3_Qmultiplicative[hv_counter] = gamma_ml_block3_Qmultiplicative


    hv_alpha_block1[hv_counter] = alpha_ml_block1
    hv_alpha_block3[hv_counter] = alpha_ml_block3
    hv_beta_block1[hv_counter] = beta_ml_block1
    hv_beta_block3[hv_counter] = beta_ml_block3
    hv_learning_ratio[hv_counter] = float(alpha_ml_block3)/alpha_ml_block1
    hv_counter += 1


'''
Perform logistic regression
'''

stim_reward1 = np.ravel(stim_reward1)
stim_target1 = np.ravel(stim_target1)
stim_trial1 = np.ravel(stim_trial1)
stim_target_side1 = np.ravel(stim_target_side1)
stim_reward3 = np.ravel(stim_reward3)
stim_target3 = np.ravel(stim_target3)
stim_trial3 = np.ravel(stim_trial3)
stim_target_side3 = np.ravel(stim_target_side3)
stim_stim_trials = np.ravel(stim_stim_trials)

sham_reward1 = np.ravel(sham_reward1)
sham_target1 = np.ravel(sham_target1)
sham_trial1 = np.ravel(sham_trial1)
sham_target_side1 = np.ravel(sham_target_side1)
sham_reward3 = np.ravel(sham_reward3)
sham_target3 = np.ravel(sham_target3)
sham_trial3 = np.ravel(sham_trial3)
sham_target_side3 = np.ravel(sham_target_side3)
sham_stim_trials = np.ravel(sham_stim_trials)

hv_reward1 = np.ravel(hv_reward1)
hv_target1 = np.ravel(hv_target1)
hv_trial1 = np.ravel(hv_trial1)
hv_target_side1 = np.ravel(hv_target_side1)
hv_reward3 = np.ravel(hv_reward3)
hv_target3 = np.ravel(hv_target3)
hv_trial3 = np.ravel(hv_trial3)
hv_target_side3 = np.ravel(hv_target_side3)
hv_stim_trials = np.ravel(hv_stim_trials)


#len_regress = np.min([len(reward3),100])
stim_len_regress = len(stim_reward3)
sham_len_regress = len(sham_reward3)
hv_len_regress = len(hv_reward3)

stim_fit_glm_block1, stim_fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(stim_reward1, stim_target1, stim_trial1, stim_target_side1,stim_reward3, stim_target3, stim_trial3, stim_target_side3,stim_stim_trials)
sham_fit_glm_block1, sham_fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(sham_reward1, sham_target1, sham_trial1, sham_target_side1,sham_reward3, sham_target3, sham_trial3, sham_target_side3,sham_stim_trials)
hv_fit_glm_block1, hv_fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(hv_reward1, hv_target1, hv_trial1, hv_target_side1, hv_reward3, hv_target3, hv_trial3, hv_target_side3, hv_stim_trials)

stim_fit_glm_block1, stim_fit_glm_block3_hist1 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 1)
sham_fit_glm_block1, sham_fit_glm_block3_hist1 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(sham_reward1, sham_target1, sham_trial1, sham_reward3, sham_target3, sham_trial3, sham_stim_trials, 1)

#fit_glm_block1, fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression_sepRegressors(reward1, target1, trial1, reward3, target3, trial3, stim_trials)

print stim_fit_glm_block1.summary()
print sham_fit_glm_block1.summary()
#print hv_fit_glm_block1.summary()
print stim_fit_glm_block3.summary()
print sham_fit_glm_block3.summary()
print hv_fit_glm_block3.summary()

stim_params_block1 = stim_fit_glm_block1.params
stim_params_block3 = stim_fit_glm_block3.params
sham_params_block1 = sham_fit_glm_block1.params
sham_params_block3 = sham_fit_glm_block3.params
hv_params_block1 = hv_fit_glm_block1.params
hv_params_block3 = hv_fit_glm_block3.params

stim_params_block3_hist1 = stim_fit_glm_block3_hist1.params
sham_params_block3_hist1 = sham_fit_glm_block3_hist1.params

stim_relative_action_value_block1, stim_relative_action_value_block3, stim_prob_choice_block1, stim_prob_choice_block3 = computeProbabilityChoiceWithRegressors(stim_params_block1, stim_params_block3,stim_reward1, stim_target1, stim_trial1, stim_target_side1,stim_reward3, stim_target3, stim_trial3, stim_target_side3,stim_stim_trials)
sham_relative_action_value_block1, sham_relative_action_value_block3, sham_prob_choice_block1, sham_prob_choice_block3 = computeProbabilityChoiceWithRegressors(sham_params_block1, sham_params_block3,sham_reward1, sham_target1, sham_trial1, sham_target_side1,sham_reward3, sham_target3, sham_trial3, sham_target_side3,sham_stim_trials)
hv_relative_action_value_block1, hv_relative_action_value_block3, hv_prob_choice_block1, hv_prob_choice_block3 = computeProbabilityChoiceWithRegressors(hv_params_block1, hv_params_block3,hv_reward1, hv_target1, hv_trial1, hv_target_side1,hv_reward3, hv_target3, hv_trial3, hv_target_side3, hv_stim_trials)

stim_relative_action_value_block1, stim_relative_action_value_block3_hist1, stim_prob_choice_block1, stim_prob_choice_block3_hist1 = computeProbabilityChoiceWithRegressors_StimHist(stim_params_block1, stim_params_block3_hist1,stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 1)
sham_relative_action_value_block1, sham_relative_action_value_block3_hist1, sham_prob_choice_block1, sham_prob_choice_block3_hist1 = computeProbabilityChoiceWithRegressors_StimHist(sham_params_block1, sham_params_block3_hist1,sham_reward1, sham_target1, sham_trial1, sham_reward3, sham_target3, sham_trial3, sham_stim_trials, 1)


stim_sorted_ind = np.argsort(stim_relative_action_value_block3)
sham_sorted_ind = np.argsort(sham_relative_action_value_block3)
hv_sorted_ind = np.argsort(hv_relative_action_value_block3)

stim_sorted_ind_hist1 = np.argsort(stim_relative_action_value_block3_hist1)
sham_sorted_ind_hist1 = np.argsort(sham_relative_action_value_block3_hist1)

stim_relative_action_value_block3 = np.array(stim_relative_action_value_block3)
sham_relative_action_value_block3 = np.array(sham_relative_action_value_block3)
hv_relative_action_value_block3 = np.array(hv_relative_action_value_block3)

stim_relative_action_value_block3_hist1 = np.array(stim_relative_action_value_block3_hist1)
sham_relative_action_value_block3_hist1 = np.array(sham_relative_action_value_block3_hist1)


xval = np.arange(-5,5,0.1)
stim_fit_block3 = (float(1)/(np.exp(-stim_params_block3[0] - xval) + 1))
sham_fit_block3 = (float(1)/(np.exp(-sham_params_block3[0] - xval) + 1))
hv_fit_block3 = (float(1)/(np.exp(-hv_params_block3[0] - xval) + 1))
stim_fit_block3_hist1 = (float(1)/(np.exp(-stim_params_block3_hist1[0] - xval) + 1))
sham_fit_block3_hist1 = (float(1)/(np.exp(-sham_params_block3_hist1[0] - xval) + 1))

plt.figure(1)
plt.plot(stim_relative_action_value_block3[stim_sorted_ind],stim_prob_choice_block3[stim_sorted_ind],'r-*',label='LV Stim')
plt.plot(xval,stim_fit_block3,'r')
plt.plot(sham_relative_action_value_block3[sham_sorted_ind],sham_prob_choice_block3[sham_sorted_ind],'b-*',label='Sham')
plt.plot(xval,sham_fit_block3,'b')
plt.plot(hv_relative_action_value_block3[hv_sorted_ind],hv_prob_choice_block3[hv_sorted_ind],'m*',label='HV stim')
plt.plot(xval,hv_fit_block3,'m')
#plt.plot(stim_relative_action_value_block3_hist1[stim_sorted_ind_hist1],stim_prob_choice_block3_hist1[stim_sorted_ind_hist1],'k-*',label='LV Stim - 1st trial')
#plt.plot(xval,stim_fit_block3_hist1,'k')
#plt.plot(sham_relative_action_value_block3_hist1[sham_sorted_ind_hist1],sham_prob_choice_block3_hist1[sham_sorted_ind_hist1],'c-*',label='Sham - 1st trial')
#plt.plot(xval,sham_fit_block3_hist1,'c')
plt.legend(loc=4)
plt.xlabel('Relative Action Value')
plt.ylabel('P(Choose High-Value Target)')
plt.xlim([-5,5])
plt.ylim([0.0,1.05])
plt.show()


'''
RL Model Parameters Plots
'''

alpha_means = np.array([np.mean(stim_alpha_block1 - stim_alpha_block3), np.mean(sham_alpha_block1 - sham_alpha_block3), np.mean(hv_alpha_block1 - hv_alpha_block3), np.mean(stim_alpha_block3), np.mean(sham_alpha_block3), np.mean(hv_alpha_block3)])
alpha_sem = np.array([float(np.std(stim_alpha_block1 - stim_alpha_block3))/stim_num_days, float(np.std(sham_alpha_block1 - sham_alpha_block3))/sham_num_days, float(np.std(hv_alpha_block1 - hv_alpha_block3))/hv_num_days,float(np.std(stim_alpha_block3))/stim_num_days, float(np.std(sham_alpha_block3))/sham_num_days, float(np.std(hv_alpha_block3))/hv_num_days])
width = float(0.35)
ind = np.arange(6)
plt.figure()
plt.bar(ind, alpha_means, width, color='g', yerr=alpha_sem/2.)
plt.ylabel('Alpha')
plt.title('Average Learning Rate')
plt.xticks(ind + width/2., ('Stim: B1 - B3', 'Sham: B1 - B3','HV: B1 - B3', 'Stim: B3', 'Sham: B3', 'HV: B3'))
plt.show()


#Beta indices for Luigi

stim_beta_ind = range(0,stim_num_days)
sham_beta_ind = range(0,sham_num_days)
stim_gamma_ind = range(0,stim_num_days)


#Indices for Luigi
sham_beta_ind.pop(3)
stim_gamma_ind.pop(2)
stim_gamma_ind.pop(2)

#Indices for papa
#stim_beta_ind.pop(2)
#stim_beta_ind.pop(3)
#sham_beta_ind.pop(12)
#sham_beta_ind.pop(12)
#sham_beta_ind.pop(8)


print sham_beta_block3[sham_beta_ind]
print stim_beta_block3_Qmultiplicative[stim_beta_ind]
print stim_gamma_block3_Qmultiplicative[stim_gamma_ind]

"""
One-way MANOVA for RL parameters
"""

dta = []
for i, data in enumerate(sham_alpha_block3):
    dta += [(0, data, sham_beta_block3[i])]
for i, data in enumerate(stim_alpha_block3_Qmultiplicative):
    dta += [(1, data, stim_beta_block3_Qmultiplicative[i])]
for i, data in enumerate(hv_alpha_block3_Qmultiplicative):
    dta += [(2, data, hv_beta_block3_Qmultiplicative[i])]

x_dta = pd.DataFrame(dta, columns=['Stim_condition', 'alpha', 'beta'])
F_stat, df_factor, df_error = OneWayMANOVA(x_dta, 'Stim_condition', 'alpha', 'beta')
print "One-way MANOVA: Stim Condition on Alpha + Beta"
print "F = ", F_stat, "df_factor = ", df_factor, "df_err = ", df_error


alpha_means = np.array([np.nanmean(sham_alpha_block3), np.nanmean(stim_alpha_block3), np.nanmean(stim_alpha_block3_Qmultiplicative), np.nanmean(hv_alpha_block3), np.nanmean(hv_alpha_block3_Qmultiplicative)])
alpha_sem = np.array([np.nanstd(sham_alpha_block3)/np.sqrt(sham_num_days), np.nanstd(stim_alpha_block3)/np.sqrt(stim_num_days), np.nanstd(stim_alpha_block3_Qmultiplicative)/np.sqrt(stim_num_days), np.nanstd(hv_alpha_block3)/np.sqrt(hv_num_days), np.nanstd(hv_alpha_block3_Qmultiplicative)/np.sqrt(hv_num_days)])
beta_means = np.array([np.nanmean(sham_beta_block3[sham_beta_ind]), np.nanmean(stim_beta_block3[stim_beta_ind]), np.nanmean(stim_beta_block3_Qmultiplicative[stim_beta_ind]), np.nanmean(hv_beta_block3), np.nanmean(hv_beta_block3_Qmultiplicative)])
beta_sem = np.array([np.nanstd(sham_beta_block3[sham_beta_ind])/np.sqrt(sham_num_days-1), np.nanstd(stim_beta_block3[stim_beta_ind])/np.sqrt(stim_num_days-1), np.nanstd(stim_beta_block3_Qmultiplicative)/np.sqrt(stim_num_days), np.nanstd(hv_beta_block3)/np.sqrt(hv_num_days), np.nanstd(hv_beta_block3_Qmultiplicative)/np.sqrt(hv_num_days)])
gamma_means = np.array([np.nanmean(stim_gamma_block3_Qmultiplicative), np.nanmean(hv_gamma_block3_Qmultiplicative)])
gamma_sem = np.array([np.nanstd(stim_gamma_block3_Qmultiplicative)/np.sqrt(stim_num_days), np.nanstd(hv_gamma_block3_Qmultiplicative)/np.sqrt(hv_num_days)])


plt.figure()
plt.subplot(1,3,1)
plt.bar(np.arange(5), alpha_means,yerr=alpha_sem/2.,color='b')
plt.xticks(np.arange(5), ('Sham', 'Stim - Standard','Stim - Q Mult', 'Control - Standard', 'Control - Q Mult'))
plt.title('Alpha')
plt.subplot(1,3,2)
plt.bar(np.arange(5), beta_means,yerr=beta_sem/2.,color='r')
plt.xticks(np.arange(5), ('Sham', 'Stim - Standard','Stim - Q Mult', 'Control - Standard', 'Control - Q Mult'))
plt.title('Beta')
plt.subplot(1,3,3)
plt.bar(np.arange(2), gamma_means,yerr=gamma_sem/2.,color='g')
plt.xticks(np.arange(2), ('Stim - Q Mult', 'Control - Q Mult'))
plt.title('Gamma')
plt.show()

alpha_means = np.array([np.nanmean(sham_alpha_block3), np.nanmean(stim_alpha_block3_Qmultiplicative), np.nanmean(hv_alpha_block3_Qmultiplicative)])
alpha_sem = np.array([np.nanstd(sham_alpha_block3)/np.sqrt(sham_num_days), np.nanstd(stim_alpha_block3_Qmultiplicative)/np.sqrt(stim_num_days), np.nanstd(hv_alpha_block3_Qmultiplicative)/np.sqrt(hv_num_days)])
beta_means = np.array([np.nanmean(sham_beta_block3[sham_beta_ind]), np.nanmean(stim_beta_block3_Qmultiplicative[stim_beta_ind]), np.nanmean(hv_beta_block3_Qmultiplicative)])
beta_sem = np.array([np.nanstd(sham_beta_block3[sham_beta_ind])/np.sqrt(sham_num_days-1), np.nanstd(stim_beta_block3_Qmultiplicative[stim_beta_ind])/np.sqrt(stim_num_days), np.nanstd(hv_beta_block3_Qmultiplicative)/np.sqrt(hv_num_days)])
gamma_means = np.array([np.nanmean(stim_gamma_block3_Qmultiplicative[stim_gamma_ind]), np.nanmean(hv_gamma_block3_Qmultiplicative)])
gamma_sem = np.array([np.nanstd(stim_gamma_block3_Qmultiplicative[stim_gamma_ind])/np.sqrt(stim_num_days), np.nanstd(hv_gamma_block3_Qmultiplicative)/np.sqrt(hv_num_days)])


plt.figure()
plt.subplot(1,3,1)
plt.errorbar(np.arange(3), alpha_means,yerr=alpha_sem/2.,color='b',ecolor='b')
plt.xticks(np.arange(3), ('Sham', 'Stim - Q Mult', 'Control - Q Mult'))
plt.xlim((-0.1,2.1))
plt.ylim((0,0.25))
plt.title('Alpha')
plt.subplot(1,3,2)
plt.errorbar(np.arange(3), beta_means,yerr=beta_sem/2.,color='r',ecolor='r')
plt.xticks(np.arange(3), ('Sham', 'Stim - Q Mult', 'Control - Q Mult'))
plt.xlim((-0.1,2.1))
plt.ylim((0,10))
plt.title('Beta')
plt.subplot(1,3,3)
plt.errorbar(np.arange(2), gamma_means,yerr=gamma_sem/2.,color='g',ecolor='g')
plt.xticks(np.arange(2), ('Stim - Q Mult', 'Control - Q Mult'))
plt.xlim((-0.1,1.1))
plt.ylim((0,6.5))
plt.title('Gamma')
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.errorbar(np.arange(2), alpha_means[0:2],yerr=alpha_sem[0:2]/2.,color='b',ecolor='b')
plt.xticks(np.arange(3), ('Sham', 'Stim - Q Mult', 'Control - Q Mult'))
plt.xlim((-0.1,2.1))
#plt.ylim((0,0.25))
plt.title('Alpha')
plt.subplot(1,3,2)
plt.errorbar(np.arange(2), beta_means[0:2],yerr=beta_sem[0:2]/2.,color='r',ecolor='r')
plt.xticks(np.arange(3), ('Sham', 'Stim - Q Mult', 'Control - Q Mult'))
plt.xlim((-0.1,2.1))
plt.ylim((0,10))
plt.title('Beta')
plt.subplot(1,3,3)
plt.errorbar(np.arange(1), gamma_means[0],yerr=gamma_sem[0]/2.,color='g',ecolor='g')
plt.xticks(np.arange(2), ('Stim - Q Mult', 'Control - Q Mult'))
plt.xlim((-0.1,1.1))
plt.ylim((0,6.5))
plt.title('Gamma')
plt.show()


prob_choose_low_mean = (np.mean(stim_prob_choose_low), np.mean(sham_prob_choose_low), np.mean(hv_prob_choose_low))
prob_choose_low_sem = (float(np.std(stim_prob_choose_low))/stim_num_days, float(np.std(sham_prob_choose_low))/sham_num_days, float(np.std(hv_prob_choose_low))/hv_num_days)
ind = np.arange(3)
plt.figure()
plt.bar(ind, prob_choose_low_mean, width, color = 'c', yerr = prob_choose_low_sem)
plt.ylabel('P(Choose LV Target)')
plt.title('Target Selection Following a Trial with Stimulation')
plt.xticks(ind + width/2, ('Stim', 'Sham', 'HV Stim'))
plt.show()

prob_choose_left_mean_block1 = (np.mean(stim_prob_choose_left_block1), np.mean(sham_prob_choose_left_block1), np.mean(hv_prob_choose_left_block1))
prob_choose_left_sem_block1 = (float(np.std(stim_prob_choose_left_block1))/stim_num_days, float(np.std(sham_prob_choose_left_block1))/sham_num_days, float(np.std(hv_prob_choose_left_block1))/hv_num_days)

prob_choose_left_mean_block3 = (np.mean(stim_prob_choose_left_block3), np.mean(sham_prob_choose_left_block3), np.mean(hv_prob_choose_left_block3))
prob_choose_left_sem_block3 = (float(np.std(stim_prob_choose_left_block3))/stim_num_days, float(np.std(sham_prob_choose_left_block3))/sham_num_days, float(np.std(hv_prob_choose_left_block3))/hv_num_days)

prob_choose_left_and_low_mean_block1 = (np.mean(stim_prob_choose_left_and_low_block1), np.mean(sham_prob_choose_left_and_low_block1), np.mean(hv_prob_choose_left_and_low_block1))
prob_choose_left_and_low_sem_block1 = (float(np.std(stim_prob_choose_left_and_low_block1))/stim_num_days, float(np.std(sham_prob_choose_left_and_low_block1))/sham_num_days, float(np.std(hv_prob_choose_left_and_low_block1))/hv_num_days)

prob_choose_left_and_low_mean_block3 = (np.mean(stim_prob_choose_left_and_low_block3), np.mean(sham_prob_choose_left_and_low_block3), np.mean(hv_prob_choose_left_and_low_block3))
prob_choose_left_and_low_sem_block3 = (float(np.std(stim_prob_choose_left_and_low_block3))/stim_num_days, float(np.std(sham_prob_choose_left_and_low_block3))/sham_num_days, float(np.std(hv_prob_choose_left_and_low_block3))/hv_num_days)

prob_choose_left_and_high_mean_block1 = (np.mean(stim_prob_choose_left_and_high_block1), np.mean(sham_prob_choose_left_and_high_block1), np.mean(hv_prob_choose_left_and_high_block1))
prob_choose_left_and_high_sem_block1 = (float(np.std(stim_prob_choose_left_and_high_block1))/stim_num_days, float(np.std(sham_prob_choose_left_and_high_block1))/sham_num_days, float(np.std(hv_prob_choose_left_and_high_block1))/hv_num_days)

prob_choose_left_and_high_mean_block3 = (np.mean(stim_prob_choose_left_and_high_block3), np.mean(sham_prob_choose_left_and_high_block3), np.mean(hv_prob_choose_left_and_high_block3))
prob_choose_left_and_high_sem_block3 = (float(np.std(stim_prob_choose_left_and_high_block3))/stim_num_days, float(np.std(sham_prob_choose_left_and_high_block3))/sham_num_days, float(np.std(hv_prob_choose_left_and_high_block3))/hv_num_days)


ind = np.arange(3)
plt.figure()
plt.plot(ind, 0.5*np.ones(len(ind)),'k--')
plt.bar(ind, prob_choose_left_and_low_mean_block1, width/2, color = 'c', hatch = '//', yerr = prob_choose_left_and_low_sem_block1)
plt.bar(ind, prob_choose_left_and_high_mean_block1, width/2, color = 'c', bottom=prob_choose_left_and_low_mean_block1,yerr = prob_choose_left_and_high_sem_block1)
plt.bar(ind+width, prob_choose_left_and_low_mean_block3, width/2, color = 'm',hatch = '//', yerr = prob_choose_left_and_low_sem_block3)
plt.bar(ind+width, prob_choose_left_and_high_mean_block3, width/2, color = 'm',bottom=prob_choose_left_and_low_mean_block3,yerr = prob_choose_left_and_high_sem_block3)
plt.plot(ind, 0.5*np.ones(len(ind)),'k--')
plt.ylabel('P(Choose Left Target)')
plt.title('Target Selection')
plt.xticks(ind + width/2, ('Stim', 'Sham', 'HV'))
plt.ylim([0.0,1.05])
plt.show()
'''
fit_glm_block1: const, prev_reward1, prev_reward2, prev_reward3, prev_reward4, prev_reward5, prev_noreward1, prev_noreward2, prev_noreward3, prev_noreward4, prev_noreward5
fit_glm_block3: const, prev_reward1, prev_reward2, prev_reward3, prev_reward4, prev_reward5, prev_noreward1, prev_noreward2, prev_noreward3, prev_noreward4, prev_noreward5, prev_stim1, prev_stim2, prev_stim3, prev_stim4_prev_stim5
'''
"""
params_block1[counter_hdf,:] = fit_glm_block1.params
pvalues_block1[counter_hdf,:] = fit_glm_block1.pvalues
numtrials_block1[counter_hdf] = int(fit_glm_block1.nobs)

params_block3[counter_hdf,0:len(fit_glm_block3.params)] = fit_glm_block3.params
pvalues_block3[counter_hdf,0:len(fit_glm_block3.params)] = fit_glm_block3.pvalues
numtrials_block3[counter_hdf] = int(fit_glm_block3.nobs)



counter_hdf += 1


avg_params_block1 = np.mean(params_block1,axis=0)
avg_params_block3 = np.mean(params_block3,axis=0)
sem_params_block1 = np.std(params_block1,axis=0)/np.sqrt(len(hdf_list))
sem_params_block3 = np.std(params_block3,axis=0)/np.sqrt(len(hdf_list))

avg_prev_reward_block1 = avg_params_block1[1:6]
sem_prev_reward_block1 = sem_params_block1[1:6]
avg_prev_noreward_block1 = avg_params_block1[6:11]
sem_prev_noreward_block1 = sem_params_block1[6:11]

avg_prev_reward_block3 = avg_params_block3[1:6]
sem_prev_reward_block3 = sem_params_block3[1:6]
avg_prev_noreward_block3 = avg_params_block3[6:11]
sem_prev_noreward_block3 = sem_params_block3[6:11]
avg_prev_stim = avg_params_block3[11:16]
sem_prev_stim = sem_params_block3[11:16]

'''
plot avg betas with error bars. next: computer relative action values
'''
plt.figure()
plt.errorbar(np.arange(-1,-6,-1),avg_prev_reward_block1,color='b',yerr=sem_prev_reward_block1,label='Reward-Block 1')
plt.errorbar(np.arange(-1,-6,-1),avg_prev_noreward_block1,color='r',yerr=sem_prev_noreward_block1,label='No Reward-Block 1')
plt.errorbar(np.arange(-1,-6,-1),avg_prev_reward_block3,color='b',linestyle='--',yerr=sem_prev_reward_block3,label='Reward-Block 3')
plt.errorbar(np.arange(-1,-6,-1),avg_prev_noreward_block3,color='r',linestyle='--',yerr=sem_prev_noreward_block3,label='No Reward-Block 3')
plt.errorbar(np.arange(-1,-6,-1),avg_prev_stim,color='m',linestyle='--',yerr=sem_prev_stim,label='Stim')
plt.legend()


prev_reward_block1 = params_block1[1:6]
prev_noreward_block1 = params_block1[6:11]
prev_reward_block3 = params_block3[1:6]
prev_noreward_block3 = params_block3[6:11]
prev_stim = params_block3[11:16]

plt.figure()
plt.plot(np.arange(-1,-6,-1),prev_reward_block1,color='b',label='Reward-Block 1')
plt.plot(np.arange(-1,-6,-1),prev_noreward_block1,color='r',label='No Reward-Block 1')
plt.plot(np.arange(-1,-6,-1),prev_reward_block3,color='b',linestyle='--',label='Reward-Block 3')
plt.plot(np.arange(-1,-6,-1),prev_noreward_block3,color='r',linestyle='--',label='No Reward-Block 3')
plt.plot(np.arange(-1,-6,-1),prev_stim,color='m',linestyle='--',label='Stim')
plt.legend()

"""
'''
1. look at decsion rule and classification for trials  - not equal to null model
2. do regression for not separate regressors - better model
3. repeat analysis (w/o stim history) for first trial following stimulation - see _firstTrial.py. only difference is line 148 stating that we only include free-choice trials in block 3
where the previous trial was instructed (i.e. has stim)
4. what would the roc auc be for the null hypothesis? - unity line
5. can we predict if he was rewarded on the previous trial given his choice?  are there false alarms for positive predicts when he actually got stim instead of reward?

'''
