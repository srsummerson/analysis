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
                                        logLikelihoodRLPerformance_multiplicative_Pstimparameter, RLPerformance_additive_Pstimparameter, logLikelihoodRLPerformance_additive_Pstimparameter
from probabilisticRewardTaskPerformance import FreeChoicePilotTask_Behavior, FreeChoicePilotTask_Behavior_ProbChooseLow
from basicAnalysis import ComputeRSquared, ComputeEfronRSquared

'''
Do fit_regularized for regression
Plot difference in AIC/BIC
'''


# constant voltage and constant current stimulation: constant current starts at 20150508
# 12 sessions contant voltage, 19 sessions constant current, 15 sessions sham
"""
hdf_list_stim = ['\papa20150203_10.hdf','\papa20150210_13.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf','\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_03.hdf','\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf','\papa20150527_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_01.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']

# 14 good sham sessions
hdf_list_sham = ['\papa20150213_10.hdf','\papa20150217_05.hdf','\papa20150225_02.hdf',
    '\papa20150307_02.hdf','\papa20150308_06.hdf','\papa20150310_02.hdf','\papa20150506_09.hdf','\papa20150506_10.hdf',
    '\papa20150519_02.hdf','\papa20150519_03.hdf','\papa20150519_04.hdf','\papa20150527_01.hdf','\papa20150528_02.hdf']
"""
hdf_list_sham = ['\papa20150217_05.hdf','\papa20150305_02.hdf',
    '\papa20150310_02.hdf',
    '\papa20150519_02.hdf','\papa20150519_04.hdf','\papa20150528_02.hdf']

# sessions with good RL fit only
# 9 sessions contant voltage, 10 sessions constant current, 12 sessions sham
# 

hdf_list_stim = ['\papa20150203_10.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_01.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf']

hdf_list_sham_papa = ['\papa20150213_10.hdf','\papa20150217_05.hdf','\papa20150225_02.hdf',
    '\papa20150307_02.hdf','\papa20150308_06.hdf','\papa20150310_02.hdf','\papa20150506_09.hdf','\papa20150506_10.hdf',
    '\papa20150519_03.hdf','\papa20150519_04.hdf','\papa20150527_01.hdf','\papa20150528_02.hdf']

# constant voltage: 11 good stim sessions
# hdf_list_stim = [\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
#    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
#    '\papa20150306_07.hdf','\papa20150309_04.hdf']
"""
hdf_list_stim = ['\papa20150211_11.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf']
"""
# constant current: 18 good stim sessions
hdf_list_stim2 = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_03.hdf','\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_01.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']
hdf_list_stim2 = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']

#hdf_list = np.sum([hdf_list_stim,hdf_list_stim2])

'''

#hdf_list = np.sum([hdf_list_stim,hdf_list_sham])
# Exceptions: 5/25 - 2, 5/30 - 1, 2/18 - 4, 6/2 - 3, 2/19 - 9 (if doing first 100 trials), 3/3 - 3 (if doing first 100 trials)
'''
"""
hdf_list = ['\papa20150211_11.hdf',
    '\papa20150223_02.hdf','\papa20150224_02.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf','\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf',
    '\papa20150524_04.hdf',
    '\papa20150602_04.hdf']

"""
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





hdf_prefix = 'C:\Users\Carmena Lab\Dropbox\Carmena Lab\Papa\hdf'
stim_hdf_list = hdf_list_stim2
sham_hdf_list = hdf_list_sham_papa

global_max_trial_dist = 0
Q_initial = [0.5, 0.5]
alpha_true = 0.2
beta_true = 3
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
                                            params_block3[15]*prev_stim5_block3 + \
                                            params_block3[16]*fc_target_high_side_block3)

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
            prev_stim1_block3 = (stim_trials[i - 1]*(2*target3[i-1]- 1))  # = 1 if stim was delivered with the HV target and = -1 if stim was delivered with the LV target
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
            prev_stim1_block3.append(stim_trials[i - 1]*(2*target3[i-1]- 1))  # = 1 if stim was delivered with the HV and = -1 if stim was delivered with the LV target
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
        prev_stim1_block3, prev_stim2_block3, prev_stim3_block3, prev_stim4_block3, prev_stim5_block3, fc_target_high_side_block3))
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

color_stim =iter(cm.rainbow(np.linspace(0,1,stim_num_days)))
color_sham = iter(cm.rainbow(np.linspace(0,1,sham_num_days)))

stim_learning_ratio = np.zeros(stim_num_days)
stim_learning_ratio_Qadditive = np.zeros(stim_num_days)
stim_learning_ratio_Padditive = np.zeros(stim_num_days)
stim_learning_ratio_Qmultiplicative = np.zeros(stim_num_days)
stim_learning_ratio_Pmultiplicative = np.zeros(stim_num_days)
sham_learning_ratio = np.zeros(sham_num_days)
sham_learning_ratio_Qadditive = np.zeros(sham_num_days)
sham_learning_ratio_Padditive = np.zeros(sham_num_days)
sham_learning_ratio_Qmultiplicative = np.zeros(sham_num_days)
sham_learning_ratio_Pmultiplicative = np.zeros(sham_num_days)
stim_alpha_block1 = np.zeros(stim_num_days)
stim_beta_block1 = np.zeros(stim_num_days)
stim_alpha_block3 = np.zeros(stim_num_days)
stim_beta_block3 = np.zeros(stim_num_days)
stim_alpha_block3_Qadditive = np.zeros(stim_num_days)
stim_alpha_block3_Padditive = np.zeros(stim_num_days)
stim_beta_block3_Qadditive = np.zeros(stim_num_days)
stim_beta_block3_Padditive = np.zeros(stim_num_days)
stim_gamma_block3_Qadditive = np.zeros(stim_num_days)
stim_gamma_block3_Padditive = np.zeros(stim_num_days)
stim_alpha_block3_Qmultiplicative = np.zeros(stim_num_days)
stim_alpha_block3_Pmultiplicative = np.zeros(stim_num_days)
stim_beta_block3_Qmultiplicative = np.zeros(stim_num_days)
stim_beta_block3_Pmultiplicative = np.zeros(stim_num_days)
stim_gamma_block3_Qmultiplicative = np.zeros(stim_num_days)
stim_gamma_block3_Pmultiplicative = np.zeros(stim_num_days)
stim_BIC_block3 = np.zeros(stim_num_days)
stim_BIC_block3_Qadditive = np.zeros(stim_num_days)
stim_BIC_block3_Padditive = np.zeros(stim_num_days)
stim_BIC_block3_Qmultiplicative = np.zeros(stim_num_days)
stim_BIC_block3_Pmultiplicative = np.zeros(stim_num_days)
stim_AIC_block3 = np.zeros(stim_num_days)
stim_AIC_block3_Qadditive = np.zeros(stim_num_days)
stim_AIC_block3_Padditive = np.zeros(stim_num_days)
stim_AIC_block3_Qmultiplicative = np.zeros(stim_num_days)
stim_AIC_block3_Pmultiplicative = np.zeros(stim_num_days)
sham_alpha_block1 = np.zeros(sham_num_days)
sham_beta_block1 = np.zeros(sham_num_days)
sham_alpha_block3 = np.zeros(sham_num_days)
sham_beta_block3 = np.zeros(sham_num_days)
sham_alpha_block3_Qadditive = np.zeros(sham_num_days)
sham_alpha_block3_Padditive = np.zeros(sham_num_days)
sham_beta_block3_Qadditive = np.zeros(sham_num_days)
sham_beta_block3_Padditive = np.zeros(sham_num_days)
sham_gamma_block3_Qadditive = np.zeros(sham_num_days)
sham_gamma_block3_Padditive = np.zeros(sham_num_days)
sham_alpha_block3_Qmultiplicative = np.zeros(sham_num_days)
sham_alpha_block3_Pmultiplicative = np.zeros(sham_num_days)
sham_beta_block3_Qmultiplicative = np.zeros(sham_num_days)
sham_beta_block3_Pmultiplicative = np.zeros(sham_num_days)
sham_gamma_block3_Qmultiplicative = np.zeros(sham_num_days)
sham_gamma_block3_Pmultiplicative = np.zeros(sham_num_days)
sham_BIC_block3 = np.zeros(sham_num_days)
sham_AIC_block3 = np.zeros(sham_num_days)
sham_BIC_block3_Qadditive = np.zeros(sham_num_days)
sham_AIC_block3_Qadditive = np.zeros(sham_num_days)
sham_BIC_block3_Padditive = np.zeros(sham_num_days)
sham_AIC_block3_Padditive = np.zeros(sham_num_days)
sham_BIC_block3_Qmultiplicative = np.zeros(sham_num_days)
sham_AIC_block3_Qmultiplicative = np.zeros(sham_num_days)
sham_BIC_block3_Pmultiplicative = np.zeros(sham_num_days)
sham_AIC_block3_Pmultiplicative = np.zeros(sham_num_days)
stim_RLaccuracy_block3 = np.zeros(stim_num_days)
stim_RLaccuracy_block3_Qadditive = np.zeros(stim_num_days)
stim_RLaccuracy_block3_Padditive = np.zeros(stim_num_days)
stim_RLaccuracy_block3_Qmultiplicative = np.zeros(stim_num_days)
stim_RLaccuracy_block3_Pmultiplicative = np.zeros(stim_num_days)
sham_RLaccuracy_block3 = np.zeros(sham_num_days)
sham_RLaccuracy_block3_Qadditive = np.zeros(sham_num_days)
sham_RLaccuracy_block3_Padditive = np.zeros(sham_num_days)
sham_RLaccuracy_block3_Qmultiplicative = np.zeros(sham_num_days)
sham_RLaccuracy_block3_Pmultiplicative = np.zeros(sham_num_days)

sham_RLaccuracy_block3 = np.zeros(sham_num_days)
sham_RLaccuracy_block1 = np.zeros(sham_num_days)
sham_RL_max_loglikelihood3 = np.zeros(sham_num_days)
sham_RL_rsquared1 = np.zeros(sham_num_days)
sham_RL_rsquared3 = np.zeros(sham_num_days)
sham_RLpearsonr1 = np.zeros(sham_num_days)
sham_RLpearsonr3 = np.zeros(sham_num_days)
sham_RLprecision1 = np.zeros(sham_num_days)
sham_RLprecision3 = np.zeros(sham_num_days)

stim_prob_choose_low = np.zeros(stim_num_days)
sham_prob_choose_low = np.zeros(sham_num_days)
stim_prob_choose_left_block1 = np.zeros(stim_num_days)
sham_prob_choose_left_block1 = np.zeros(sham_num_days)
stim_prob_choose_left_block3 = np.zeros(stim_num_days)
sham_prob_choose_left_block3 = np.zeros(sham_num_days)

stim_prob_choose_left_and_low_block1 = np.zeros(stim_num_days)
stim_prob_choose_left_and_high_block1 = np.zeros(stim_num_days)
stim_prob_choose_left_and_low_block3 = np.zeros(stim_num_days)
stim_prob_choose_left_and_high_block3 = np.zeros(stim_num_days)

sham_prob_choose_left_and_low_block1 = np.zeros(sham_num_days)
sham_prob_choose_left_and_high_block1 = np.zeros(sham_num_days)
sham_prob_choose_left_and_low_block3 = np.zeros(sham_num_days)
sham_prob_choose_left_and_high_block3 = np.zeros(sham_num_days)

stim_counter = 0
sham_counter = 0

for name in stim_hdf_list:
    
    print name
    full_name = hdf_prefix + name

    '''
    Compute task performance.
    '''
    reward_block1, target_block1, trial_block1, target_side1, reward_block3, target_block3, trial_block3, target_side3, stim_trials_block = FreeChoicePilotTask_Behavior(full_name)
    b3_prob_choose_low, b3_prob_reward_low = FreeChoicePilotTask_Behavior_ProbChooseLow(full_name)
    prob_choose_low = FirstChoiceAfterStim(target_block3,trial_block3, stim_trials_block)

    #prob_choose_low = 1 - np.sum(target_block3[fc_trial_ind] - 1)/len(target_block3[fc_trial_ind])  # for prob of choosing low value target over all trials

    fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
    fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))
    target_freechoice_block3 = target_block3[fc_trial_ind_block3]
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
    Qlow_block3, Qhigh_block3, prob_low_block3_regular, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3)
    BIC3 = -2*max_loglikelihood3 + len(result3["x"])*np.log(target_block3.size)
    AIC3 = -2*max_loglikelihood3 + 2*len(result3["x"])

    # Accuracy of fit
    model3 = 0.33*(prob_low_block3_regular > 0.5) + 0.66*(np.less_equal(prob_low_block3_regular, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3[1:],(0.33*target_freechoice_block3))
    accuracy3 = float(np.sum(fit3))/fit3.size

    stim_alpha_block1[stim_counter] = alpha_ml_block1
    stim_beta_block1[stim_counter] = beta_ml_block1
    stim_alpha_block3[stim_counter] = alpha_ml_block3
    stim_beta_block3[stim_counter] = beta_ml_block3
    stim_BIC_block3[stim_counter] = BIC3
    stim_AIC_block3[stim_counter] = AIC3
    stim_RLaccuracy_block3[stim_counter] = accuracy3
    stim_learning_ratio[stim_counter] = float(alpha_ml_block3)/alpha_ml_block1
    

    '''
    Get fit with additive stimulation parameter in Q-value update equation
    '''
    nll_Qadditive = lambda *args: -logLikelihoodRLPerformance_additive_Qstimparameter(*args)
    result3_Qadditive = op.minimize(nll_Qadditive, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(0,None)])
    alpha_ml_block3_Qadditive, beta_ml_block3_Qadditive, gamma_ml_block3_Qadditive = result3_Qadditive["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3_Qadditive, max_loglikelihood3 = RLPerformance_additive_Qstimparameter([alpha_ml_block3_Qadditive,beta_ml_block3_Qadditive,gamma_ml_block3_Qadditive],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Qadditive = -2*max_loglikelihood3 + len(result3_Qadditive["x"])*np.log(target_block3.size)
    AIC3_Qadditive = -2*max_loglikelihood3 + 2*len(result3_Qadditive["x"])

     # Accuracy of fit
    model3 = 0.33*(prob_low_block3_Qadditive > 0.5) + 0.66*(np.less_equal(prob_low_block3_Qadditive, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3[1:],(0.33*target_freechoice_block3))
    accuracy3_Qadditive = float(np.sum(fit3))/fit3.size
    
    stim_alpha_block3_Qadditive[stim_counter] = alpha_ml_block3_Qadditive
    stim_beta_block3_Qadditive[stim_counter] = beta_ml_block3_Qadditive
    stim_gamma_block3_Qadditive[stim_counter] = gamma_ml_block3_Qadditive
    stim_BIC_block3_Qadditive[stim_counter] = BIC3_Qadditive
    stim_AIC_block3_Qadditive[stim_counter] = AIC3_Qadditive
    stim_RLaccuracy_block3_Qadditive[stim_counter] = accuracy3_Qadditive
    stim_learning_ratio_Qadditive[stim_counter] = float(alpha_ml_block3_Qadditive)/alpha_ml_block1

    '''
    Get fit with Multiplicative stimulation parameter in Q-value update equation
    '''
    nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter(*args)
    result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(0,None)])
    alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3_Qmultiplicative, max_loglikelihood3 = RLPerformance_multiplicative_Qstimparameter([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    print max_loglikelihood3
    BIC3_Qmultiplicative = -2*max_loglikelihood3 + len(result3_Qmultiplicative["x"])*np.log(target_block3.size)
    AIC3_Qmultiplicative = -2*max_loglikelihood3 + 2*len(result3_Qmultiplicative["x"])

     # Accuracy of fit
    model3 = 0.33*(prob_low_block3_Qmultiplicative > 0.5) + 0.66*(np.less_equal(prob_low_block3_Qmultiplicative, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3[1:],(0.33*target_freechoice_block3))
    accuracy3_Qmultiplicative = float(np.sum(fit3))/fit3.size

    stim_alpha_block3_Qmultiplicative[stim_counter] = alpha_ml_block3_Qmultiplicative
    stim_beta_block3_Qmultiplicative[stim_counter] = beta_ml_block3_Qmultiplicative
    stim_gamma_block3_Qmultiplicative[stim_counter] = gamma_ml_block3_Qmultiplicative
    stim_BIC_block3_Qmultiplicative[stim_counter] = BIC3_Qmultiplicative
    stim_AIC_block3_Qmultiplicative[stim_counter] = AIC3_Qmultiplicative
    stim_RLaccuracy_block3_Qmultiplicative[stim_counter] = accuracy3_Qmultiplicative
    stim_learning_ratio_Qmultiplicative[stim_counter] = float(alpha_ml_block3_Qmultiplicative)/alpha_ml_block1
    
    '''
    Get fit with additive stimulation parameter in P-value update equation
    '''
    nll_Padditive = lambda *args: -logLikelihoodRLPerformance_additive_Pstimparameter(*args)
    result3_Padditive = op.minimize(nll_Padditive, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(-1,None)])
    alpha_ml_block3_Padditive, beta_ml_block3_Padditive, gamma_ml_block3_Padditive = result3_Padditive["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3_Padditive, max_loglikelihood3 = RLPerformance_additive_Pstimparameter([alpha_ml_block3_Padditive,beta_ml_block3_Padditive,gamma_ml_block3_Padditive],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Padditive = -2*max_loglikelihood3 + len(result3_Padditive["x"])*np.log(target_block3.size)
    AIC3_Padditive = -2*max_loglikelihood3 + 2*len(result3_Padditive["x"])

     # Accuracy of fit
    model3 = 0.33*(prob_low_block3_Padditive > 0.5) + 0.66*(np.less_equal(prob_low_block3_Padditive, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3[1:],(0.33*target_freechoice_block3))
    accuracy3_Padditive = float(np.sum(fit3))/fit3.size
    
    stim_alpha_block3_Padditive[stim_counter] = alpha_ml_block3_Padditive
    stim_beta_block3_Padditive[stim_counter] = beta_ml_block3_Padditive
    stim_gamma_block3_Padditive[stim_counter] = gamma_ml_block3_Padditive
    stim_BIC_block3_Padditive[stim_counter] = BIC3_Padditive
    stim_AIC_block3_Padditive[stim_counter] = AIC3_Padditive
    stim_RLaccuracy_block3_Padditive[stim_counter] = accuracy3_Padditive
    stim_learning_ratio_Padditive[stim_counter] = float(alpha_ml_block3_Padditive)/alpha_ml_block1

    '''
    Get fit with Multiplicative stimulation parameter in P-value update equation
    '''
    nll_Pmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Pstimparameter(*args)
    result3_Pmultiplicative = op.minimize(nll_Pmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(-1,None)])
    alpha_ml_block3_Pmultiplicative, beta_ml_block3_Pmultiplicative, gamma_ml_block3_Pmultiplicative = result3_Pmultiplicative["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3_Pmultiplicative, max_loglikelihood3 = RLPerformance_multiplicative_Pstimparameter([alpha_ml_block3_Pmultiplicative,beta_ml_block3_Pmultiplicative,gamma_ml_block3_Pmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Pmultiplicative = -2*max_loglikelihood3 + len(result3_Pmultiplicative["x"])*np.log(target_block3.size)
    AIC3_Pmultiplicative = -2*max_loglikelihood3 + 2*len(result3_Pmultiplicative["x"])

     # Accuracy of fit
    model3 = 0.33*(prob_low_block3_Pmultiplicative > 0.5) + 0.66*(np.less_equal(prob_low_block3_Pmultiplicative, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3[:-1],(0.33*target_freechoice_block3))
    accuracy3_Pmultiplicative = float(np.sum(fit3))/model3.size

    stim_alpha_block3_Pmultiplicative[stim_counter] = alpha_ml_block3_Pmultiplicative
    stim_beta_block3_Pmultiplicative[stim_counter] = beta_ml_block3_Pmultiplicative
    stim_gamma_block3_Pmultiplicative[stim_counter] = gamma_ml_block3_Pmultiplicative
    stim_BIC_block3_Pmultiplicative[stim_counter] = BIC3_Pmultiplicative
    stim_AIC_block3_Pmultiplicative[stim_counter] = AIC3_Pmultiplicative
    stim_RLaccuracy_block3_Pmultiplicative[stim_counter] = accuracy3_Pmultiplicative
    stim_learning_ratio_Pmultiplicative[stim_counter] = float(alpha_ml_block3_Pmultiplicative)/alpha_ml_block1
    '''
    print b3_prob_choose_low[0]
    plt.figure()
    plt.plot(b3_prob_choose_low,'r',label='Behavior')
    plt.plot(prob_low_block3_regular,'b',label='Regular')
    plt.plot(prob_low_block3_Qadditive,'c',label='Q Additive')
    plt.plot(prob_low_block3_Qmultiplicative,'g',label='Q Multiplicative')
    plt.plot(prob_low_block3_Padditive,'m',label='P Additive')
    plt.plot(prob_low_block3_Pmultiplicative,'k',label='P Multiplicative')
    plt.legend()
    plt.show()
    '''
    stim_counter += 1
    
for name in sham_hdf_list:
    
    print name
    full_name = hdf_prefix + name

    '''
    Compute task performance.
    '''
    reward_block1, target_block1, trial_block1, target_side1, reward_block3, target_block3, trial_block3, target_side3, stim_trials_block = FreeChoicePilotTask_Behavior(full_name)
    b3_prob_choose_low, b3_prob_reward_low = FreeChoicePilotTask_Behavior_ProbChooseLow(full_name)
    prob_choose_low = FirstChoiceAfterStim(target_block3,trial_block3, stim_trials_block)

    #prob_choose_low = 1 - np.sum(target_block3[fc_trial_ind] - 1)/len(target_block3[fc_trial_ind])  # for prob of choosing low value target over all trials

    fc_trial_ind_block3 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block3),2)))
    fc_trial_ind_block1 = np.ravel(np.nonzero(np.equal(np.ravel(trial_block1),2)))
    target_freechoice_block3 = target_block3[fc_trial_ind_block3]
    target_freechoice_block1 = target_block1[fc_trial_ind_block1]
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
    
    prediction = 1 + (prob_low_block1 < 0.5)  # should be 1 = lv target, 2 = hv target
    sham_rsquared = ComputeEfronRSquared(target_freechoice_block1, prob_low_block1[1:])
    model1 = 0.33*(prob_low_block1 > 0.5) + 0.66*(np.less_equal(prob_low_block1, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    fit1 = np.equal(model1[1:],(0.33*target_freechoice_block1))

    pearsonr1 = np.corrcoef(prediction[1:], target_freechoice_block1)[1][0]
    precision = np.nanmean((prediction[1:] - target_freechoice_block1)**2)

    accuracy1 = float(np.sum(fit1))/fit1.size
    
    sham_RL_rsquared1[sham_counter] = sham_rsquared
    sham_RLprecision1[sham_counter] = precision

    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_block3, target_block3, trial_block3), bounds=[(0,1),(0,None)])

    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3)
    BIC3 = -2*max_loglikelihood3 + len(result3["x"])*np.log(target_block3.size)

     # Accuracy of fit
    prediction = 1 + (prob_low_block3 < 0.5)  # should be 1 = lv target, 2 = hv target
    model3 = 0.33*(prob_low_block3 > 0.5) + 0.66*(np.less_equal(prob_low_block3, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    fit3 = np.equal(model3[:-1],(0.33*target_freechoice_block3))

    #pearsonr3 = np.corrcoef(prediction[1:], target_freechoice_block3)[1][0]
    pearsonr3 = np.corrcoef(b3_prob_choose_low, prob_low_block3[1:])[1][0]
    
    precision = np.nanmean((prediction[1:] - target_freechoice_block3)**2)

    accuracy3 = float(np.sum(fit3))/fit3.size
    
    sham_rsquared = ComputeEfronRSquared(target_freechoice_block3, prob_low_block3[1:])
    
    sham_RLpearsonr1[sham_counter] = pearsonr1
    sham_RLpearsonr3[sham_counter] = pearsonr3
    sham_RLprecision3[sham_counter] = precision
    sham_RL_rsquared3[sham_counter] = sham_rsquared
    sham_RLaccuracy_block1[sham_counter] = accuracy1
    sham_RLaccuracy_block3[sham_counter] = accuracy3
    sham_RL_max_loglikelihood3[sham_counter] = max_loglikelihood3

    sham_alpha_block1[sham_counter] = alpha_ml_block1
    sham_beta_block1[sham_counter] = beta_ml_block1
    sham_alpha_block3[sham_counter] = alpha_ml_block3
    sham_beta_block3[sham_counter] = beta_ml_block3
    sham_BIC_block3[sham_counter] = BIC3
    sham_learning_ratio[sham_counter] = float(alpha_ml_block3)/alpha_ml_block1

    '''
    Get fit with additive stimulation parameter in Q-value update equation
    '''
    nll_Qadditive = lambda *args: -logLikelihoodRLPerformance_additive_Qstimparameter(*args)
    result3_Qadditive = op.minimize(nll_Qadditive, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(0,1)])
    alpha_ml_block3_Qadditive, beta_ml_block3_Qadditive, gamma_ml_block3_Qadditive = result3_Qadditive["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance_additive_Qstimparameter([alpha_ml_block3_Qadditive,beta_ml_block3_Qadditive,gamma_ml_block3_Qadditive],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Qadditive = -2*max_loglikelihood3 + len(result3_Qadditive["x"])*np.log(target_block3.size)
    
    sham_alpha_block3_Qadditive[sham_counter] = alpha_ml_block3_Qadditive
    sham_beta_block3_Qadditive[sham_counter] = beta_ml_block3_Qadditive
    sham_gamma_block3_Qadditive[sham_counter] = gamma_ml_block3_Qadditive
    sham_BIC_block3_Qadditive[sham_counter] = BIC3_Qadditive
    sham_learning_ratio_Qadditive[sham_counter] = float(alpha_ml_block3_Qadditive)/alpha_ml_block1

    '''
    Get fit with Multiplicative stimulation parameter in Q-value update equation
    '''
    nll_Qmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Qstimparameter(*args)
    result3_Qmultiplicative = op.minimize(nll_Qmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(0,None)])
    alpha_ml_block3_Qmultiplicative, beta_ml_block3_Qmultiplicative, gamma_ml_block3_Qmultiplicative = result3_Qmultiplicative["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance_additive_Qstimparameter([alpha_ml_block3_Qmultiplicative,beta_ml_block3_Qmultiplicative,gamma_ml_block3_Qmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Qmultiplicative = -2*max_loglikelihood3 + len(result3_Qmultiplicative["x"])*np.log(target_block3.size)

    sham_alpha_block3_Qmultiplicative[sham_counter] = alpha_ml_block3_Qmultiplicative
    sham_beta_block3_Qmultiplicative[sham_counter] = beta_ml_block3_Qmultiplicative
    sham_gamma_block3_Qmultiplicative[sham_counter] = gamma_ml_block3_Qmultiplicative
    sham_BIC_block3_Qmultiplicative[sham_counter] = BIC3_Qmultiplicative
    sham_learning_ratio_Qmultiplicative[sham_counter] = float(alpha_ml_block3_Qmultiplicative)/alpha_ml_block1

    '''
    Get fit with additive stimulation parameter in P-value update equation
    '''
    nll_Padditive = lambda *args: -logLikelihoodRLPerformance_additive_Pstimparameter(*args)
    result3_Padditive = op.minimize(nll_Padditive, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(-1,None)])
    alpha_ml_block3_Padditive, beta_ml_block3_Padditive, gamma_ml_block3_Padditive = result3_Padditive["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance_additive_Pstimparameter([alpha_ml_block3_Padditive,beta_ml_block3_Padditive,gamma_ml_block3_Padditive],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Padditive = -2*max_loglikelihood3 + len(result3_Padditive["x"])*np.log(target_block3.size)
    AIC3_Padditive = -2*max_loglikelihood3 + 2*len(result3_Padditive["x"])

     # Accuracy of fit
    model3 = 0.33*(prob_low_block3 > 0.5) + 0.66*(np.less_equal(prob_low_block3, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    fit3 = np.equal(model3[:-1],(0.33*target_freechoice_block3))
    accuracy3_Padditive = float(np.sum(fit3))/model3.size
    
    sham_alpha_block3_Padditive[sham_counter] = alpha_ml_block3_Padditive
    sham_beta_block3_Padditive[sham_counter] = beta_ml_block3_Padditive
    sham_gamma_block3_Padditive[sham_counter] = gamma_ml_block3_Padditive
    sham_BIC_block3_Padditive[sham_counter] = BIC3_Padditive
    sham_AIC_block3_Padditive[sham_counter] = AIC3_Padditive
    sham_RLaccuracy_block3_Padditive[sham_counter] = accuracy3_Padditive
    sham_learning_ratio_Padditive[sham_counter] = float(alpha_ml_block3_Padditive)/alpha_ml_block1

    '''
    Get fit with additive stimulation parameter in P-value update equation
    '''
    nll_Pmultiplicative = lambda *args: -logLikelihoodRLPerformance_multiplicative_Pstimparameter(*args)
    result3_Pmultiplicative = op.minimize(nll_Pmultiplicative, [alpha_true, beta_true, gamma_true], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_block3, target_block3, trial_block3, stim_trials_block), bounds=[(0,1),(0,None),(-1,None)])
    alpha_ml_block3_Pmultiplicative, beta_ml_block3_Pmultiplicative, gamma_ml_block3_Pmultiplicative = result3_Pmultiplicative["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance_additive_Pstimparameter([alpha_ml_block3_Pmultiplicative,beta_ml_block3_Pmultiplicative,gamma_ml_block3_Pmultiplicative],[Qlow_block1[-1],Qhigh_block1[-1]],reward_block3,target_block3, trial_block3, stim_trials_block)
    BIC3_Pmultiplicative = -2*max_loglikelihood3 + len(result3_Pmultiplicative["x"])*np.log(target_block3.size)
    AIC3_Pmultiplicative = -2*max_loglikelihood3 + 2*len(result3_Pmultiplicative["x"])

     # Accuracy of fit
    model3 = 0.33*(prob_low_block3 > 0.5) + 0.66*(np.less_equal(prob_low_block3, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    fit3 = np.equal(model3[:-1],(0.33*target_freechoice_block3))
    accuracy3_Pmultiplicative = float(np.sum(fit3))/model3.size

    sham_alpha_block3_Pmultiplicative[sham_counter] = alpha_ml_block3_Pmultiplicative
    sham_beta_block3_Pmultiplicative[sham_counter] = beta_ml_block3_Pmultiplicative
    sham_gamma_block3_Pmultiplicative[sham_counter] = gamma_ml_block3_Pmultiplicative
    sham_BIC_block3_Pmultiplicative[sham_counter] = BIC3_Pmultiplicative
    sham_AIC_block3_Pmultiplicative[sham_counter] = AIC3_Pmultiplicative
    sham_RLaccuracy_block3_Pmultiplicative[sham_counter] = accuracy3_Pmultiplicative
    sham_learning_ratio_Pmultiplicative[sham_counter] = float(alpha_ml_block3_Pmultiplicative)/alpha_ml_block1

    sham_counter += 1
    
    
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


#len_regress = np.min([len(reward3),100])
stim_len_regress = len(stim_reward3)
sham_len_regress = len(sham_reward3)

stim_fit_glm_block1, stim_fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(stim_reward1, stim_target1, stim_trial1, stim_target_side1,stim_reward3, stim_target3, stim_trial3, stim_target_side3,stim_stim_trials)
sham_fit_glm_block1, sham_fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(sham_reward1, sham_target1, sham_trial1, sham_target_side1,sham_reward3, sham_target3, sham_trial3, sham_target_side3,sham_stim_trials)

#stim_fit_glm_block1, stim_fit_glm_block3_hist1 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 1)
#stim_fit_glm_block1, stim_fit_glm_block3_hist2 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 2)
#stim_fit_glm_block1, stim_fit_glm_block3_hist3 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 3)
#stim_fit_glm_block1, stim_fit_glm_block3_hist4 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 4)
#stim_fit_glm_block1, stim_fit_glm_block3_hist5 = probabilisticFreeChoicePilotTask_logisticRegression_StimHist(stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 5)




#fit_glm_block1, fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression_sepRegressors(reward1, target1, trial1, reward3, target3, trial3, stim_trials)

print stim_fit_glm_block1.summary()
print sham_fit_glm_block1.summary()
print stim_fit_glm_block3.summary()
print sham_fit_glm_block3.summary()

stim_params_block1 = stim_fit_glm_block1.params
stim_params_block3 = stim_fit_glm_block3.params
sham_params_block1 = sham_fit_glm_block1.params
sham_params_block3 = sham_fit_glm_block3.params

#stim_params_block3_hist1 = stim_fit_glm_block3_hist1.params
#stim_params_block3_hist2 = stim_fit_glm_block3_hist2.params
#stim_params_block3_hist3 = stim_fit_glm_block3_hist3.params
#stim_params_block3_hist4 = stim_fit_glm_block3_hist4.params
#stim_params_block3_hist5 = stim_fit_glm_block3_hist5.params

stim_relative_action_value_block1, stim_relative_action_value_block3, stim_prob_choice_block1, stim_prob_choice_block3 = computeProbabilityChoiceWithRegressors(stim_params_block1, stim_params_block3,stim_reward1, stim_target1, stim_trial1, stim_target_side1,stim_reward3, stim_target3, stim_trial3, stim_target_side3,stim_stim_trials)
sham_relative_action_value_block1, sham_relative_action_value_block3, sham_prob_choice_block1, sham_prob_choice_block3 = computeProbabilityChoiceWithRegressors(sham_params_block1, sham_params_block3,sham_reward1, sham_target1, sham_trial1, sham_target_side1,sham_reward3, sham_target3, sham_trial3, sham_target_side3,sham_stim_trials)

#stim_relative_action_value_block1, stim_relative_action_value_block3_hist1, stim_prob_choice_block1, stim_prob_choice_block3_hist1 = computeProbabilityChoiceWithRegressors_StimHist(stim_params_block1, stim_params_block3_hist1,stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 1)
#stim_relative_action_value_block1, stim_relative_action_value_block3_hist2, stim_prob_choice_block1, stim_prob_choice_block3_hist2 = computeProbabilityChoiceWithRegressors_StimHist(stim_params_block1, stim_params_block3,stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 2)
#stim_relative_action_value_block1, stim_relative_action_value_block3_hist3, stim_prob_choice_block1, stim_prob_choice_block3_hist3 = computeProbabilityChoiceWithRegressors_StimHist(stim_params_block1, stim_params_block3,stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 3)
#stim_relative_action_value_block1, stim_relative_action_value_block3_hist4, stim_prob_choice_block1, stim_prob_choice_block3_hist4 = computeProbabilityChoiceWithRegressors_StimHist(stim_params_block1, stim_params_block3,stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 4)
#stim_relative_action_value_block1, stim_relative_action_value_block3_hist5, stim_prob_choice_block1, stim_prob_choice_block3_hist5 = computeProbabilityChoiceWithRegressors_StimHist(stim_params_block1, stim_params_block3,stim_reward1, stim_target1, stim_trial1, stim_reward3, stim_target3, stim_trial3, stim_stim_trials, 5)


#plt.figure(0)
#plt.show()

stim_sorted_ind = np.argsort(stim_relative_action_value_block3)
sham_sorted_ind = np.argsort(sham_relative_action_value_block3)

#stim_sorted_ind_hist1 = np.argsort(stim_relative_action_value_block3_hist1)
#stim_sorted_ind_hist2 = np.argsort(stim_relative_action_value_block3_hist2)
#stim_sorted_ind_hist3 = np.argsort(stim_relative_action_value_block3_hist3)
#stim_sorted_ind_hist4 = np.argsort(stim_relative_action_value_block3_hist4)
#stim_sorted_ind_hist5 = np.argsort(stim_relative_action_value_block3_hist5)

stim_relative_action_value_block3 = np.array(stim_relative_action_value_block3)
sham_relative_action_value_block3 = np.array(sham_relative_action_value_block3)

#stim_relative_action_value_block3_hist1 = np.array(stim_relative_action_value_block3_hist1)
#stim_relative_action_value_block3_hist2 = np.array(stim_relative_action_value_block3_hist2)
#stim_relative_action_value_block3_hist3 = np.array(stim_relative_action_value_block3_hist3)
#stim_relative_action_value_block3_hist4 = np.array(stim_relative_action_value_block3_hist4)
#stim_relative_action_value_block3_hist5 = np.array(stim_relative_action_value_block3_hist5)


xval = np.arange(-5,5,0.1)
stim_fit_block3 = (float(1)/(np.exp(-stim_params_block3[0] - xval) + 1))
sham_fit_block3 = (float(1)/(np.exp(-sham_params_block3[0] - xval) + 1))

#stim_fit_block3_hist1 = (float(1)/(np.exp(-stim_params_block3_hist1[0] - xval) + 1))
#stim_fit_block3_hist2 = (float(1)/(np.exp(-stim_params_block3_hist2[0] - xval) + 1))
#stim_fit_block3_hist3 = (float(1)/(np.exp(-stim_params_block3_hist3[0] - xval) + 1))
#stim_fit_block3_hist4 = (float(1)/(np.exp(-stim_params_block3_hist4[0] - xval) + 1))
#stim_fit_block3_hist5 = (float(1)/(np.exp(-stim_params_block3_hist5[0] - xval) + 1))


plt.figure(1)
plt.plot(stim_relative_action_value_block3[stim_sorted_ind],stim_prob_choice_block3[stim_sorted_ind],'r-*',label='LV Stim')
plt.plot(xval,stim_fit_block3,'r')
plt.plot(sham_relative_action_value_block3[sham_sorted_ind],sham_prob_choice_block3[sham_sorted_ind],'b-*',label='Sham')
plt.plot(xval,sham_fit_block3,'b')
#plt.plot(stim_relative_action_value_block3_hist1[stim_sorted_ind_hist1],stim_prob_choice_block3_hist1[stim_sorted_ind_hist1],'k-*',label='LV Stim - 1st trial')
#plt.plot(xval,stim_fit_block3_hist1,'k')
#plt.plot(stim_relative_action_value_block3_hist2[stim_sorted_ind_hist2],stim_prob_choice_block3_hist2[stim_sorted_ind_hist2],'c-*',label='LV Stim')
#plt.plot(xval,stim_fit_block3_hist2,'c')
#plt.plot(stim_relative_action_value_block3_hist3[stim_sorted_ind_hist3],stim_prob_choice_block3_hist3[stim_sorted_ind_hist3],'m-*',label='LV Stim')
#plt.plot(xval,stim_fit_block3_hist3,'m')
#plt.plot(stim_relative_action_value_block3_hist4[stim_sorted_ind_hist4],stim_prob_choice_block3_hist4[stim_sorted_ind_hist4],'g-*',label='LV Stim')
#plt.plot(xval,stim_fit_block3_hist4,'g')
#plt.plot(stim_relative_action_value_block3_hist5[stim_sorted_ind_hist5],stim_prob_choice_block3_hist5[stim_sorted_ind_hist5],'y-*',label='LV Stim')
#plt.plot(xval,stim_fit_block3_hist5,'y')
plt.legend(loc=4)
plt.xlabel('Relative Action Value')
plt.ylabel('P(Choose High-Value Target)')
plt.xlim([-5,5])
plt.ylim([0.0,1.05])
plt.show()


stim_prev_reward_block1 = stim_params_block1[1:6]
stim_prev_noreward_block1 = stim_params_block1[6:11]
stim_prev_reward_block3 = stim_params_block3[1:6]
stim_prev_noreward_block3 = stim_params_block3[6:11]
stim_prev_stim = stim_params_block3[11:16]

sham_prev_reward_block1 = sham_params_block1[1:6]
sham_prev_noreward_block1 = sham_params_block1[6:11]
sham_prev_reward_block3 = sham_params_block3[1:6]
sham_prev_noreward_block3 = sham_params_block3[6:11]
sham_prev_stim = sham_params_block3[11:16]

"""
plt.figure()
plt.plot(np.arange(-1,-6,-1),stim_prev_reward_block1,color='b',label='Stim: Reward-Block 1')
plt.plot(np.arange(-1,-6,-1),stim_prev_noreward_block1,color='r',label='Stim: No Reward-Block 1')
plt.plot(np.arange(-1,-6,-1),stim_prev_reward_block3,color='c',label='Stim: Reward-Block 3')
plt.plot(np.arange(-1,-6,-1),stim_prev_noreward_block3,color='m',label='Stim: No Reward-Block 3')
plt.plot(np.arange(-1,-6,-1),stim_prev_stim,color='k',label='Stim')
plt.plot(np.arange(-1,-6,-1),sham_prev_reward_block1,color='b',linestyle='--',label='Sham: Reward-Block 1')
plt.plot(np.arange(-1,-6,-1),sham_prev_noreward_block1,color='r',linestyle='--',label='Sham: No Reward-Block 1')
plt.plot(np.arange(-1,-6,-1),sham_prev_reward_block3,color='c',linestyle='--',label='Sham: Reward-Block 3')
plt.plot(np.arange(-1,-6,-1),sham_prev_noreward_block3,color='m',linestyle='--',label='Sham: No Reward-Block 3')
plt.plot(np.arange(-1,-6,-1),sham_prev_stim,color='k',linestyle='--',label='Sham')
plt.legend()
plt.show()
"""


alpha_means = (np.mean(stim_alpha_block1 - stim_alpha_block3), np.mean(sham_alpha_block1 - sham_alpha_block3), np.mean(stim_alpha_block3), np.mean(sham_alpha_block3))
alpha_sem = (float(np.std(stim_alpha_block1 - stim_alpha_block3))/stim_num_days, float(np.std(sham_alpha_block1 - sham_alpha_block3))/sham_num_days, float(np.std(stim_alpha_block3))/stim_num_days, float(np.std(sham_alpha_block3))/sham_num_days)
width = float(0.35)
ind = np.arange(4)
plt.figure()
plt.bar(ind, alpha_means, width, color='g', yerr=alpha_sem)
plt.ylabel('Alpha')
plt.title('Average Learning Rate')
plt.xticks(ind + width/2., ('Stim: B1 - B3', 'Sham: B1 - B3', 'Stim: B3', 'Sham: B3'))
plt.show()

print stim_BIC_block3_Qmultiplicative

plt.figure()
plt.subplot(1,2,1)
plt.scatter(1*np.ones(len(stim_BIC_block3)), -stim_BIC_block3 + stim_BIC_block3_Qadditive,c='b')
plt.scatter(2*np.ones(len(stim_BIC_block3)), -stim_BIC_block3 + stim_BIC_block3_Qmultiplicative,c='r')
plt.scatter(3*np.ones(len(stim_BIC_block3)), -stim_BIC_block3 + stim_BIC_block3_Padditive,c='m')
plt.scatter(4*np.ones(len(stim_BIC_block3)), -stim_BIC_block3 + stim_BIC_block3_Pmultiplicative,c='g')
plt.plot([-0.5,4.5],[0,0],'k--')
plt.ylabel('BIC Difference (Adjusted - Standard)')
labels = ["Q Additive","Q Multiplicative'","P Additive", "P Multiplicative"]
plt.xticks([1,2,3,4], labels, fontsize=8)
#plt.ylim([-10,50])
plt.title('Bayesian Information Criterion - Stim Days')
#plt.legend()
plt.subplot(1,2,2)
plt.scatter(1*np.ones(len(stim_BIC_block3)), stim_BIC_block3 - stim_BIC_block3_Qmultiplicative,c='r')
plt.scatter(2*np.ones(len(stim_BIC_block3)), -stim_BIC_block3_Qmultiplicative + stim_BIC_block3_Qadditive,c='b')
plt.scatter(3*np.ones(len(stim_BIC_block3)), -stim_BIC_block3_Qmultiplicative + stim_BIC_block3_Padditive,c='m')
plt.scatter(4*np.ones(len(stim_BIC_block3)), -stim_BIC_block3_Qmultiplicative + stim_BIC_block3_Pmultiplicative,c='g')
plt.plot([-0.5,4.5],[0,0],'k--')
plt.ylabel('BIC Difference (Adjusted - Q Multiplicative Model)')
labels = ["Standard","Q Additive'","P Additive", "P Multiplicative"]
plt.xticks([1,2,3,4], labels, fontsize=8)
plt.title('Bayesian Information Criterion - Stim Days')
#plt.ylim([-10,40])
#plt.legend()

plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.scatter(1*np.ones(len(stim_AIC_block3)), -stim_AIC_block3 + stim_AIC_block3_Qadditive,c='b')
plt.scatter(2*np.ones(len(stim_AIC_block3)), -stim_AIC_block3 + stim_AIC_block3_Qmultiplicative,c='r')
plt.scatter(3*np.ones(len(stim_AIC_block3)), -stim_AIC_block3 + stim_AIC_block3_Padditive,c='m')
plt.scatter(4*np.ones(len(stim_AIC_block3)), -stim_AIC_block3 + stim_AIC_block3_Pmultiplicative,c='g')
plt.plot([-0.5,4.5],[0,0],'k--')
plt.ylabel('AIC Difference (Adjusted - Standard)')
labels = ["Q Additive","Q Multiplicative'","P Additive", "P Multiplicative"]
plt.xticks([1,2,3,4], labels, fontsize=8)
#plt.ylim([-10,50])
plt.title('Akaike Information Criterion - Stim Days')
#plt.legend()
plt.subplot(1,2,2)
plt.scatter(1*np.ones(len(stim_AIC_block3)), stim_BIC_block3 - stim_AIC_block3_Qmultiplicative,c='r')
plt.scatter(2*np.ones(len(stim_AIC_block3)), -stim_AIC_block3_Qmultiplicative + stim_AIC_block3_Qadditive,c='b')
plt.scatter(3*np.ones(len(stim_AIC_block3)), -stim_AIC_block3_Qmultiplicative + stim_AIC_block3_Padditive,c='m')
plt.scatter(4*np.ones(len(stim_AIC_block3)), -stim_AIC_block3_Qmultiplicative + stim_AIC_block3_Pmultiplicative,c='g')
plt.plot([-0.5,4.5],[0,0],'k--')
plt.ylabel('AIC Difference (Adjusted - Q Multiplicative Model)')
labels = ["Standard","Q Additive'","P Additive", "P Multiplicative"]
plt.xticks([1,2,3,4], labels, fontsize=8)
plt.title('Akaike Information Criterion - Stim Days')
#plt.ylim([-10,40])
#plt.legend()

plt.show()

"""
# Luigi indices
BIC_indices = range(len(stim_BIC_block3))
BIC_indices.pop(5)
"""
BIC_indices = range(len(stim_BIC_block3))


avg_model_BIC = [np.nanmean(stim_BIC_block3[BIC_indices]), np.nanmean(stim_BIC_block3_Qadditive[BIC_indices]), np.nanmean(stim_BIC_block3_Qmultiplicative[BIC_indices]), np.nanmean(stim_BIC_block3_Padditive[BIC_indices]), np.nanmean(stim_BIC_block3_Pmultiplicative[BIC_indices])]
sem_model_BIC = [np.nanstd(stim_BIC_block3[BIC_indices])/len(stim_BIC_block3), np.nanstd(stim_BIC_block3_Qadditive[BIC_indices])/len(stim_BIC_block3), np.nanstd(stim_BIC_block3_Qmultiplicative[BIC_indices])/len(stim_BIC_block3), np.nanstd(stim_BIC_block3_Padditive[BIC_indices])/len(stim_BIC_block3), np.nanstd(stim_BIC_block3_Pmultiplicative[BIC_indices])/len(stim_BIC_block3)]
sem_model_BIC = np.array(sem_model_BIC)/2
plt.figure()
plt.errorbar(range(len(avg_model_BIC)), avg_model_BIC, yerr=sem_model_BIC,marker='o',color='c')
labels = ["Regular","Q Additive","Q Multiplicative","P Additive", "P Multiplicative"]
plt.xticks([0,1,2,3,4], labels, fontsize=8)
plt.xlim((-0.1,4.1))
plt.title('Bayesian Information Criterion')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(stim_RLaccuracy_block3,'r',label='Unadjusted')
plt.plot(stim_RLaccuracy_block3_Qadditive,'b',label='Additive Q Param')
plt.plot(stim_RLaccuracy_block3_Qmultiplicative,'c',label='Multiplicative Q Param')
plt.plot(stim_RLaccuracy_block3_Padditive,'g',label='Additive P Param')
plt.plot(stim_RLaccuracy_block3_Pmultiplicative,'k--',label='Multiplicative P Param')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Accuracy')
plt.title('Stim')
plt.subplot(1,2,2)
plt.plot(sham_RLaccuracy_block3,'r',label='Unadjusted')
plt.plot(sham_RLaccuracy_block3_Qadditive,'b',label='Additive Q Param')
plt.plot(sham_RLaccuracy_block3_Qmultiplicative,'c',label='Multiplicative Q Param')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Accuracy')
plt.title('Sham')
plt.show()

plt.figure()
plt.title('Stim')
plt.subplot(1,3,1)
plt.plot(stim_alpha_block3,'r',label='Unadjusted')
plt.plot(stim_alpha_block3_Qadditive,'b',label='Additive Q Param')
plt.plot(stim_alpha_block3_Qmultiplicative,'c',label='Multiplicative Q Param')
plt.plot(stim_alpha_block3_Padditive,'g',label='Additive P Param')
plt.plot(stim_alpha_block3_Pmultiplicative,'k--',label='Multiplicative P Param')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Alpha')
plt.subplot(1,3,2)
plt.plot(stim_beta_block3,'r',label='Unadjusted')
plt.plot(stim_beta_block3_Qadditive,'b',label='Additive Q Param')
plt.plot(stim_beta_block3_Qmultiplicative,'c',label='Multiplicative Q Param')
plt.plot(stim_beta_block3_Padditive,'g',label='Additive P Param')
plt.plot(stim_beta_block3_Pmultiplicative,'k--',label='Multiplicative P Param')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Beta')
plt.subplot(1,3,3)
plt.plot(stim_gamma_block3_Qadditive,'b',label='Additive Q Param')
plt.plot(stim_gamma_block3_Qmultiplicative,'c',label='Multiplicative Q Param')
plt.plot(stim_gamma_block3_Padditive,'g',label='Additive P Param')
plt.plot(stim_gamma_block3_Pmultiplicative,'k--',label='Multiplicative P Param')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Gamma')
plt.show()

adjusted_alpha_means = [np.mean(stim_alpha_block3),np.mean(stim_alpha_block3_Qadditive),np.mean(stim_alpha_block3_Qmultiplicative),np.mean(stim_alpha_block3_Padditive),np.mean(stim_alpha_block3_Pmultiplicative)]
adjusted_beta_means = [np.mean(stim_beta_block3),np.mean(stim_beta_block3_Qadditive),np.mean(stim_beta_block3_Qmultiplicative),np.mean(stim_beta_block3_Padditive),np.mean(stim_beta_block3_Pmultiplicative)]
adjusted_gamma_means = [np.mean(stim_gamma_block3_Qadditive),np.mean(stim_gamma_block3_Qmultiplicative),np.mean(stim_gamma_block3_Padditive),np.mean(stim_gamma_block3_Pmultiplicative)]
index = np.arange(5)

plt.figure()
plt.subplot(1,3,1)
plt.bar(index,adjusted_alpha_means,width/2, color='c')
plt.ylabel('avg alpha')
plt.xticks(index,('Regular','Q Additive','Q Multiplicative','P Additive','P Multiplicative'))
plt.subplot(1,3,2)
plt.bar(index,adjusted_beta_means,width/2,color='m')
plt.ylabel('avg beta')
plt.xticks(index,('Regular','Q Additive','Q Multiplicative','P Additive','P Multiplicative'))
plt.subplot(1,3,3)
plt.bar(index[1:],adjusted_gamma_means,width/2,color='y')
plt.ylabel('avg gamma')
plt.xticks(index,('Regular','Q Additive','Q Multiplicative','P Additive','P Multiplicative'))
plt.show()

prob_choose_low_mean = (np.mean(stim_prob_choose_low), np.mean(sham_prob_choose_low))
prob_choose_low_sem = (float(np.std(stim_prob_choose_low))/stim_num_days, float(np.std(sham_prob_choose_low))/sham_num_days)
ind = np.arange(2)
plt.figure()
plt.bar(ind, prob_choose_low_mean, width, color = 'c', yerr = prob_choose_low_sem)
plt.ylabel('P(Choose LV Target)')
plt.title('Target Selection Following a Trial with Stimulation')
plt.xticks(ind + width/2, ('Stim', 'Sham'))
plt.show()

prob_choose_left_mean_block1 = (np.mean(stim_prob_choose_left_block1), np.mean(sham_prob_choose_left_block1))
prob_choose_left_sem_block1 = (float(np.std(stim_prob_choose_left_block1))/stim_num_days, float(np.std(sham_prob_choose_left_block1))/sham_num_days)

prob_choose_left_mean_block3 = (np.mean(stim_prob_choose_left_block3), np.mean(sham_prob_choose_left_block3))
prob_choose_left_sem_block3 = (float(np.std(stim_prob_choose_left_block3))/stim_num_days, float(np.std(sham_prob_choose_left_block3))/sham_num_days)

prob_choose_left_and_low_mean_block1 = (np.mean(stim_prob_choose_left_and_low_block1), np.mean(sham_prob_choose_left_and_low_block1))
prob_choose_left_and_low_sem_block1 = (float(np.std(stim_prob_choose_left_and_low_block1))/stim_num_days, float(np.std(sham_prob_choose_left_and_low_block1))/sham_num_days)

prob_choose_left_and_low_mean_block3 = (np.mean(stim_prob_choose_left_and_low_block3), np.mean(sham_prob_choose_left_and_low_block3))
prob_choose_left_and_low_sem_block3 = (float(np.std(stim_prob_choose_left_and_low_block3))/stim_num_days, float(np.std(sham_prob_choose_left_and_low_block3))/sham_num_days)

prob_choose_left_and_high_mean_block1 = (np.mean(stim_prob_choose_left_and_high_block1), np.mean(sham_prob_choose_left_and_high_block1))
prob_choose_left_and_high_sem_block1 = (float(np.std(stim_prob_choose_left_and_high_block1))/stim_num_days, float(np.std(sham_prob_choose_left_and_high_block1))/sham_num_days)

prob_choose_left_and_high_mean_block3 = (np.mean(stim_prob_choose_left_and_high_block3), np.mean(sham_prob_choose_left_and_high_block3))
prob_choose_left_and_high_sem_block3 = (float(np.std(stim_prob_choose_left_and_high_block3))/stim_num_days, float(np.std(sham_prob_choose_left_and_high_block3))/sham_num_days)


ind = np.arange(2)
plt.figure()
plt.plot(ind, 0.5*np.ones(len(ind)),'k--')
plt.bar(ind, prob_choose_left_and_low_mean_block1, width/2, color = 'c', hatch = '//', yerr = prob_choose_left_and_low_sem_block1)
plt.bar(ind, prob_choose_left_and_high_mean_block1, width/2, color = 'c', bottom=prob_choose_left_and_low_mean_block1,yerr = prob_choose_left_and_high_sem_block1)
plt.bar(ind+width, prob_choose_left_and_low_mean_block3, width/2, color = 'm', hatch = '//',yerr = prob_choose_left_and_low_sem_block3)
plt.bar(ind+width, prob_choose_left_and_high_mean_block3, width/2, color = 'm', bottom=prob_choose_left_and_low_mean_block3,yerr = prob_choose_left_and_high_sem_block3)
plt.plot(ind, 0.5*np.ones(len(ind)),'k--')
plt.ylabel('P(Choose Left Target)')
plt.title('Target Selection')
plt.xticks(ind + width/2, ('Stim', 'Sham'))
plt.ylim([0.0,1.05])
plt.show()

"""
plt.figure()
plt.plot(range(0,stim_num_days),stim_learning_ratio,'r',label='Stim')
plt.plot(range(0,sham_num_days),sham_learning_ratio,'b',label='Sham')
plt.plot(range(0,stim_num_days),stim_alpha_block1,'r*',label='Stim - Block1')
plt.plot(range(0,sham_num_days),sham_alpha_block1,'b*',label='Sham - Block1')
plt.plot(range(0,stim_num_days),stim_alpha_block3,'ro',label='Stim - Block3')
plt.plot(range(0,sham_num_days),sham_alpha_block3,'bo',label='Sham - Block3')
plt.legend()
plt.show()
"""


'''
1. look at decsion rule and classification for trials  - not equal to null model
2. do regression for not separate regressors - better model
3. repeat analysis (w/o stim history) for first trial following stimulation - see _firstTrial.py. only difference is line 148 stating that we only include free-choice trials in block 3
where the previous trial was instructed (i.e. has stim)
4. what would the roc auc be for the null hypothesis? - unity line
5. can we predict if he was rewarded on the previous trial given his choice?  are there false alarms for positive predicts when he actually got stim instead of reward?

'''
