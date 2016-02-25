from scipy import stats
import statsmodels.api as sm
import scipy
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import scipy.optimize as op
import tables
import numpy as np
import matplotlib.pyplot as plt
from logLikelihoodRLPerformance import RLPerformance, logLikelihoodRLPerformance
from probabilisticRewardTaskPerformance import FreeChoicePilotTask_Behavior



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
"""
# 14 good sham sessions
hdf_list_sham = ['\papa20150213_10.hdf','\papa20150217_05.hdf','\papa20150225_02.hdf','\papa20150305_02.hdf',
    '\papa20150307_02.hdf','\papa20150308_06.hdf','\papa20150310_02.hdf','\papa20150506_09.hdf','\papa20150506_10.hdf',
    '\papa20150519_02.hdf','\papa20150519_03.hdf','\papa20150519_04.hdf','\papa20150527_01.hdf','\papa20150528_02.hdf']

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
# hdf_list_stim = ['\papa20150210_13.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf',
#    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
#    '\papa20150306_07.hdf','\papa20150309_04.hdf']
hdf_list_stim = ['\papa20150211_11.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf']
# constant current: 18 good stim sessions
# hdf_list_stim2 = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
#    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
#    '\papa20150524_03.hdf','\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
#    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_01.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
#    '\papa20150602_04.hdf']
hdf_list_stim2 = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']

#hdf_list_stim = np.sum([hdf_list_stim,hdf_list_stim2])

#hdf_list = np.sum([hdf_list_stim,hdf_list_sham])
# Exceptions: 5/25 - 2, 5/30 - 1, 2/18 - 4, 6/2 - 3, 2/19 - 9 (if doing first 100 trials), 3/3 - 3 (if doing first 100 trials)
hdf_list = ['\papa20150211_11.hdf',
    '\papa20150223_02.hdf','\papa20150224_02.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf','\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_04.hdf','\papa20150525_01.hdf',
    '\papa20150530_02.hdf','\papa20150601_02.hdf',
    '\papa20150602_04.hdf']
hdf_prefix = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Papa\hdf'


global_max_trial_dist = 0
Q_initial = [0.5, 0.5]
alpha_true = 0.2
beta_true = 0.2

def probabilisticFreeChoicePilotTask_logisticRegression(reward1, target1, trial1, reward3, target3, trial3, stim_trials):

    
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
        if (trial3[i] == 2):
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
            prev_stim1_block3.append(stim_trials[i - 1])
            prev_stim2_block3.append(stim_trials[i - 2])
            prev_stim3_block3.append(stim_trials[i - 3])
            prev_stim4_block3.append(stim_trials[i - 4])
            prev_stim5_block3.append(stim_trials[i - 5])


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

    model_glm_block1 = sm.GLM(fc_target_low_block1,x,family = sm.families.Binomial())
    model_glm_block3 = sm.GLM(fc_target_low_block3,y,family = sm.families.Binomial())
    fit_glm_block1 = model_glm_block1.fit()
    fit_glm_block3 = model_glm_block3.fit()
    print fit_glm_block1.predict()
    
    '''
    Oraganize data and regress with LogisticRegression
    '''
    """
    d_block1 = {'target_selection': fc_target_low_block1, 
            'prev_reward1': prev_reward1_block1, 
            'prev_reward2': prev_reward2_block1, 
            'prev_reward3': prev_reward3_block1, 
            'prev_reward4': prev_reward4_block1, 
            'prev_reward5': prev_reward5_block1, 
            'prev_noreward1': prev_noreward1_block1, 
            'prev_noreward2': prev_noreward2_block1,
            'prev_noreward3': prev_noreward3_block1, 
            'prev_noreward4': prev_noreward4_block1, 
            'prev_noreward5': prev_noreward5_block1, 
            'prev_stim': prev_stim_block1}
    df_block1 = pd.DataFrame(d_block1)

    y_block1, X_block1 = dmatrices('target_selection ~ prev_reward1 + prev_reward2 + prev_reward3 + \
                                    prev_reward4 + prev_reward5 + prev_noreward1 + prev_noreward2 + \
                                    prev_noreward3 + prev_noreward4 + prev_noreward5 + prev_stim', df_block1,
                                    return_type = "dataframe")
    print X_block1.columns
    # flatten y_block1 into 1-D array
    y_block1 = np.ravel(y_block1)
    
    d_block3 = {'target_selection': fc_target_low_block3, 
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
            'prev_stim': prev_stim_block3}
    df_block3 = pd.DataFrame(d_block3)

    y_block3, X_block3 = dmatrices('target_selection ~ prev_reward1 + prev_reward2 + prev_reward3 + \
                                    prev_reward4 + prev_reward5 + prev_noreward1 + prev_noreward2 + \
                                    prev_noreward3 + prev_noreward4 + prev_noreward5 + prev_stim', df_block3,
                                    return_type = "dataframe")
    # flatten y_block3 into 1-D array
    y_block3 = np.ravel(y_block3)

    # instantiate a logistic regression model, and fit with X and y
    model_block1 = LogisticRegression()
    model_block3 = LogisticRegression()
    model_block1 = model_block1.fit(X_block1, y_block1)
    model_block3 = model_block3.fit(X_block3, y_block3)

    # check the accuracy on the training set
    print 'Model accuracy for Block1:',model_block1.score(X_block1, y_block1)
    print 'Null accuracy rate:',np.max([y_block1.mean(),1 - y_block1.mean()])

    # examine the coefficients
    print pd.DataFrame(zip(X_block1.columns, np.transpose(model_block1.coef_)))
    """
    return fit_glm_block1, fit_glm_block3


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
            prev_stim1_block3.append(stim_trials[i - 1])
            prev_stim2_block3.append(stim_trials[i - 2])
            prev_stim3_block3.append(stim_trials[i - 3])
            prev_stim4_block3.append(stim_trials[i - 4])
            prev_stim5_block3.append(stim_trials[i - 5])


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

    model_glm_block1 = sm.GLM(fc_target_low_block1,x,family = sm.families.Binomial())
    model_glm_block3 = sm.GLM(fc_target_low_block3,y,family = sm.families.Binomial())
    fit_glm_block1 = model_glm_block1.fit()
    fit_glm_block3 = model_glm_block3.fit()

    
    
   
    return fit_glm_block1, fit_glm_block3


num_days = len(hdf_list)

params_block1 = np.zeros([num_days,11])
params_block3 = np.zeros([num_days,16])
numtrials_block1 = np.zeros(num_days)
numtrials_block3 = np.zeros(num_days)
pvalues_block1 = np.zeros([num_days,11])
pvalues_block3 = np.zeros([num_days,16])

counter_hdf = 0
reward1 = []
target1 = []
trial1 = []
reward3 = []
target3 = []
trial3 = []
stim_trials = []


for name in hdf_list:
    
    print name
    full_name = hdf_prefix + name

    '''
    Compute task performance.
    '''
    reward_block1, target_block1, trial_block1, reward_block3, target_block3, trial_block3, stim_trials_block = FreeChoicePilotTask_Behavior(full_name)

    reward1.extend(reward_block1.tolist())
    target1.extend(target_block1.tolist())
    trial1.extend(trial_block1.tolist())
    reward3.extend(reward_block3.tolist())
    target3.extend(target_block3.tolist())
    trial3.extend(trial_block3.tolist())
    stim_trials.extend(stim_trials_block.tolist()) 

    """
    '''
    Get soft-max decision fit
    '''
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward1, target1, trial1), bounds=[(0,1),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, trial1)
    
    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward3, target3, trial3), bounds=[(0,1),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)
    """
    
'''
Perform logistic regression
'''

reward1 = np.ravel(reward1)
target1 = np.ravel(target1)
trial1 = np.ravel(trial1)
reward3 = np.ravel(reward3)
target3 = np.ravel(target3)
trial3 = np.ravel(trial3)
stim_trials = np.ravel(stim_trials)


#len_regress = np.min([len(reward3),100])
len_regress = len(reward3)

fit_glm_block1, fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression(reward1, target1, trial1, reward3, target3, trial3, stim_trials)
#fit_glm_block1, fit_glm_block3 = probabilisticFreeChoicePilotTask_logisticRegression_sepRegressors(reward1, target1, trial1, reward3, target3, trial3, stim_trials)

print fit_glm_block1.summary()
print fit_glm_block3.summary()

'''
fit_glm_block1: const, prev_reward1, prev_reward2, prev_reward3, prev_reward4, prev_reward5, prev_noreward1, prev_noreward2, prev_noreward3, prev_noreward4, prev_noreward5
fit_glm_block3: const, prev_reward1, prev_reward2, prev_reward3, prev_reward4, prev_reward5, prev_noreward1, prev_noreward2, prev_noreward3, prev_noreward4, prev_noreward5, prev_stim1, prev_stim2, prev_stim3, prev_stim4_prev_stim5
'''
params_block1[counter_hdf,:] = fit_glm_block1.params
pvalues_block1[counter_hdf,:] = fit_glm_block1.pvalues
numtrials_block1[counter_hdf] = int(fit_glm_block1.nobs)

params_block3[counter_hdf,0:len(fit_glm_block3.params)] = fit_glm_block3.params
pvalues_block3[counter_hdf,0:len(fit_glm_block3.params)] = fit_glm_block3.pvalues
numtrials_block3[counter_hdf] = int(fit_glm_block3.nobs)


"""
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

"""
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


'''
5. regressor for side of screen
6. regress with data concatenated across days. add accuracy and null accuracy rates. add cross-validation. 
'''
