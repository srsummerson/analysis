from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance
import scipy.optimize as op
import tables
import numpy as np
import matplotlib.pyplot as plt

"""
#hdf_list = ['papa20150213_10.hdf','papa20150217_05.hdf','papa20150225_02.hdf','papa20150301_02.hdf','papa20150305_02.hdf',
#            'papa20150307_02.hdf','papa20150308_06.hdf','papa20150310_02.hdf']
#hdf_list with days where learning in block A was not sufficient are excluded
hdf_list = ['papa20150213_10.hdf','papa20150217_05.hdf','papa20150225_02.hdf','papa20150305_02.hdf',
            'papa20150307_02.hdf','papa20150308_06.hdf','papa20150310_02.hdf']
"""
#hdf_list = ['\papa20150210_13.hdf','\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf','\papa20150218_04.hdf',
#            '\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf','\papa20150306_07.hdf',
#            '\papa20150309_04.hdf']
hdf_list = ['\papa20150211_11.hdf','\papa20150214_18.hdf','\papa20150216_05.hdf','\papa20150218_04.hdf',
            '\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf','\papa20150306_07.hdf']
hdf_data = dict()
for x in hdf_list:
    hdf_data[x] = []

hdf_prefix = 'C:\Users\Samantha\Dropbox\Carmena Lab\Papa\hdf'
Q_initial = [0.5, 0.5]
alpha_true = 0.25
beta_true = 0.25

for name in hdf_list:
    counter_block1 = 0.0
    counter_block3 = 0.0
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

    target1 = np.zeros(ind_check_reward_states.size)
    reward1 = np.zeros(target1.size)
    target3 = np.zeros(ind_check_reward_states.size)
    reward3 = np.zeros(target3.size)
    target_freechoice_block1 = np.zeros(70)
    reward_freechoice_block1 = np.zeros(70)
    target_freechoice_block3 = np.zeros(num_successful_trials-200)
    reward_freechoice_block3 = np.zeros(num_successful_trials-200)

    '''
    Target choices for all (free-choice only) and associated reward assignments
    '''
    for i in range(0,100):
        target_state1 = state[ind_check_reward_states[i] - 2]
        trial1 = instructed_or_freechoice[i]
        if target_state1 == 'hold_targetL':
            target1[i] = 1
            reward1[i] = rewarded_reward_scheduleL[i]
        else:
            target1[i] = 2
            reward1[i] = rewarded_reward_scheduleH[i]
        if trial1 == 2:
            target_freechoice_block1[counter_block1] = target1[i]
            reward_freechoice_block1[counter_block1] = reward1[i]
            counter_block1 += 1
    for i in range(200,num_successful_trials):
        target_state3 = state[ind_check_reward_states[i] - 2]
        trial3 = instructed_or_freechoice[i]
        if target_state3 == 'hold_targetL':
            target3[i] = 1
            reward3[i] = rewarded_reward_scheduleL[i]
        else:
            target3[i] = 2
            reward3[i] = rewarded_reward_scheduleH[i]
        if trial3 == 2:
            target_freechoice_block3[counter_block3] = target3[i]
            reward_freechoice_block3[counter_block3] = reward3[i]
            counter_block3 += 1

    max_block3 = 70*(counter_block3 > 70) + counter_block3*(counter_block3 < 70)
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward_freechoice_block1, target_freechoice_block1), bounds=[(None,None),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    Qlow_block1, Qhigh_block1, prob_low_block1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward_freechoice_block1,target_freechoice_block1)
    # what is RL model performance with parameters fit from Block A in Block A'
    Qlow_block3, Qhigh_block3, prob_low_block3 = RLPerformance([alpha_ml_block1,beta_ml_block1],[Qlow_block1[-1],Qhigh_block1[-1]],reward_freechoice_block3,target_freechoice_block3)
    # what are parameters fit from Block A' with initial Q values from end of Block A
    result3 = op.minimize(nll, [alpha_ml_block1, beta_ml_block1], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_freechoice_block3[:max_block3], target_freechoice_block3[:max_block3]), bounds=[(None,None),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]

    hdf_data[name] = [alpha_ml_block1,alpha_ml_block3,beta_ml_block1,beta_ml_block3]


