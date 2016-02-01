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
hdf_list = ['papa20150530_01.hdf']
hdf_prefix = '/storage/rawdata/hdf/'


global_max_trial_dist = 0

for name in hdf_list:
    counter_block1 = 0.0
    counter_block3 = 0.0
    print name
    full_name = hdf_prefix + name
    hdf = tables.openFile(full_name)

    state = hdf.root.task_msgs[:]['msg']
    state_time = hdf.root.task_msgs[:]['time']
    trial_type = hdf.root.task[:]['target_index']
    # reward schedules
    reward_scheduleH = hdf.root.task[:]['reward_scheduleH']
    reward_scheduleL = hdf.root.task[:]['reward_scheduleL']
      
    ind_wait_states = np.ravel(np.nonzero(state == 'wait'))
    ind_target_states = np.ravel(np.nonzero(state == 'target'))
    ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
    num_successful_trials = ind_check_reward_states.size
    instructed_or_freechoice = trial_type[state_time[ind_check_reward_states]]
    rewarded_reward_scheduleH = reward_scheduleH[state_time[ind_target_states]]
    rewarded_reward_scheduleL = reward_scheduleL[state_time[ind_target_states]]

    target1 = np.zeros(100)
    reward1 = np.zeros(target1.size)
    target3 = np.zeros(ind_check_reward_states.size-200)
    #target3 = np.zeros(np.min([num_successful_trials-200,100]))
    reward3 = np.zeros(target3.size)
    target_freechoice_block1 = np.zeros(70)
    reward_freechoice_block1 = np.zeros(70)
    target_freechoice_block3 = []
    reward_freechoice_block3 = np.zeros(num_successful_trials-200)
    #reward_freechoice_block3 = np.zeros(np.min([num_successful_trials-200,100]))
    trial1 = np.zeros(target1.size)
    trial3 = np.zeros(target3.size)
    stim_trials = np.zeros(target3.size)

    '''
    Target choices for all (free-choice only) and associated reward assignments
    '''
    for i in range(0,100):
        target_state1 = state[ind_check_reward_states[i] - 2]
        trial1[i] = instructed_or_freechoice[i]
        if target_state1 == 'hold_targetL':
            target1[i] = 1
            reward1[i] = rewarded_reward_scheduleL[i]
        else:
            target1[i] = 2
            reward1[i] = rewarded_reward_scheduleH[i]
        if trial1[i] == 2:
            target_freechoice_block1[counter_block1] = target1[i]
            reward_freechoice_block1[counter_block1] = reward1[i]
            counter_block1 += 1
    for i in range(200,num_successful_trials):
    #for i in range(200,np.min([num_successful_trials,300])):
        target_state3 = state[ind_check_reward_states[i] - 2]
        trial3[i-200] = instructed_or_freechoice[i]
        if target_state3 == 'hold_targetL':
            target3[i-200] = 1
            reward3[i-200] = rewarded_reward_scheduleL[i]
            if trial3[i-200]==1:   # instructed trial to low-value targer paired with stim
                stim_trials[i-200] = 1
            else:
                stim_trials[i-200] = 0
        else:
            target3[i-200] = 2
            reward3[i-200] = rewarded_reward_scheduleH[i]
            stim_trials[i-200] = 0
        if trial3[i-200] == 2:
            target_freechoice_block3.append(target3[i-200])
            reward_freechoice_block3[counter_block3] = reward3[i-200]
            counter_block3 += 1

    '''
    Previous rewards and no rewards
    '''
    fc_target_high = []
    prev_reward1 = []
    prev_reward2 = []
    prev_reward3 = []
    prev_reward4 = []
    prev_reward5 = []
    prev_noreward1 = []
    prev_noreward2 = []
    prev_noreward3 = []
    prev_noreward4 = []
    prev_noreward5 = []
    prev_stim1 = []
    for i in range(5,100):
        if trial1[i] == 2:
            fc_target_high.append(target1[i] - 1)   # = 0 if selected low-value, = 1 if selected high-value
            prev_reward1.append((2*target1[i-1] - 3)*reward1[i-1])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2.append((2*target1[i-2] - 3)*reward1[i-2])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3.append((2*target1[i-3] - 3)*reward1[i-3])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4.append((2*target1[i-4] - 3)*reward1[i-4])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5.append((2*target1[i-5] - 3)*reward1[i-5])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1.append((2*target1[i-1] - 3)*(1 - reward1[i-1]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2.append((2*target1[i-2] - 3)*(1 - reward1[i-2]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3.append((2*target1[i-3] - 3)*(1 - reward1[i-3]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4.append((2*target1[i-4] - 3)*(1 - reward1[i-4]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5.append((2*target1[i-5] - 3)*(1 - reward1[i-5]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim1.append(0)
    num_regress_block1 = len(fc_target_high)
    for i in range(205,num_successful_trials):
        if trial3[i - 200] == 2:
            fc_target_high.append(target3[i - 200] - 1)   # = 0 if selected low-value, = 1 if selected high-value
            prev_reward1.append((2*target3[i-201] - 3)*reward3[i-201])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward2.append((2*target3[i-202] - 3)*reward3[i-202])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward3.append((2*target3[i-203] - 3)*reward3[i-203])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward4.append((2*target3[i-204] - 3)*reward3[i-204])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_reward5.append((2*target3[i-205] - 3)*reward3[i-205])  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward1.append((2*target3[i-201] - 3)*(1 - reward3[i-201]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward2.append((2*target3[i-202] - 3)*(1 - reward3[i-202]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward3.append((2*target3[i-203] - 3)*(1 - reward3[i-203]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward4.append((2*target3[i-204] - 3)*(1 - reward3[i-204]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_noreward5.append((2*target3[i-205] - 3)*(1 - reward3[i-205]))  # = -1 if selected low-value and rewarded, = 1 if selected high-value and rewarded
            prev_stim1.append(stim_trials[i - 201])

    '''
    Turn everything into an array
    '''
    fc_target_high = np.array(fc_target_high)
    prev_reward1 = np.array(prev_reward1)
    prev_reward2 = np.array(prev_reward2)
    prev_reward3 = np.array(prev_reward3)
    prev_reward4 = np.array(prev_reward4)
    prev_reward5 = np.array(prev_reward5)
    prev_noreward1 = np.array(prev_noreward1)
    prev_noreward2 = np.array(prev_noreward2)
    prev_noreward3 = np.array(prev_noreward3)
    prev_noreward4 = np.array(prev_noreward4)
    prev_noreward5 = np.array(prev_noreward5)
    prev_stim1 = np.array(prev_stim1)

    '''
    Oraganize data and regress 
    '''
    const_logit_block1 = np.ones(num_regress_block1)
    const_logit_all = np.ones(fc_target_high.size)
    x = np.vstack((prev_reward1,prev_reward2,prev_reward3,prev_reward4,prev_reward5,prev_noreward1,prev_noreward2,prev_noreward3,prev_noreward4,prev_noreward5,prev_stim1))
    x = np.transpose(x)
    x = sm.add_constant(x,prepend='False')
    d = {'FC Target Selection': fc_target_high, 'Prev Reward 1': prev_reward1, 'Prev Reward 2': prev_reward2, 'Prev Reward 3': prev_reward3, 
            'Prev Reward 4': prev_reward4, 'Prev Reward 5': prev_reward5, 'Prev No Reward 1': prev_noreward1, 'Prev No Reward 2': prev_noreward2,
            'Prev No Reward 3': prev_noreward3, 'Prev No Reward 4': prev_noreward4, 'Prev No Reward 5': prev_noreward5, 'Prev Stim': prev_stim1,
            'Const': const_logit_all}
    df = pd.DataFrame(d)

    model_glm = sm.GLM(fc_target_high,x,family = sm.families.Binomial())
    fit_glm = model_glm.fit()
    fit_glm.summary()
