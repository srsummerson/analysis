from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance, logLikelihoodRLPerformance_random
from scipy import stats
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

hdf_data_stim = dict()
BIC_stim = dict()
pseudo_rsquared_stim = dict()
modelerror_afterstim_stim = dict()
prob_lowvalue_afterstim_stim = dict()

hdf_data_sham = dict()
BIC_sham = dict()
pseudo_rsquared_sham = dict()
modelerror_afterstim_sham = dict()
prob_lowvalue_afterstim_sham = dict()
prob_lowvalue_afterstim_noreward_stim = dict()
prob_lowvalue_afterstim_withreward_stim = dict()
prob_lowvalue_afterstim_noreward_sham = dict()
prob_lowvalue_afterstim_withreward_sham = dict()
modelerror_afterstim_stim_refit = dict()
modelerror_afterstim_sham_refit = dict()
prob_chooselow_stim = dict()
prob_chooselow_sham = dict()
modelerror_stim = dict()
modelerror_sham = dict()

prob_switch_given_rewarded_block1 = dict()
prob_switch_given_notrewarded_block1 = dict()
prob_switch_given_rewarded_block3 = dict()
prob_switch_given_notrewarded_block3 = dict()


for x in hdf_list_stim:
    hdf_data_stim[x] = []
    BIC_stim[x] = []
    pseudo_rsquared_stim[x] = []
    modelerror_afterstim_stim[x] = []
    modelerror_afterstim_stim_refit[x] = []
    prob_lowvalue_afterstim_stim[x] = []
    prob_lowvalue_afterstim_withreward_stim[x] = []
    prob_lowvalue_afterstim_noreward_stim[x] = []
    prob_chooselow_stim[x] = []

    prob_switch_given_rewarded_block1[x] = []
    prob_switch_given_notrewarded_block1[x] = []
    prob_switch_given_rewarded_block3[x] = []
    prob_switch_given_notrewarded_block3[x] = []

for x in hdf_list_sham:
    hdf_data_sham[x] = []
    BIC_sham[x] = []
    pseudo_rsquared_sham[x] = []
    modelerror_afterstim_sham[x] = []
    modelerror_afterstim_sham_refit[x] = []
    prob_lowvalue_afterstim_sham[x] = []
    prob_lowvalue_afterstim_withreward_sham[x] = []
    prob_lowvalue_afterstim_noreward_sham[x] = []
    prob_chooselow_sham[x] = [x]

    prob_switch_given_rewarded_block1[x] = []
    prob_switch_given_notrewarded_block1[x] = []
    prob_switch_given_rewarded_block3[x] = []
    prob_switch_given_notrewarded_block3[x] = []

hdf_list = np.sum([hdf_list_stim,hdf_list_sham])

hdf_prefix = 'C:\Users\Samantha\Dropbox\Carmena Lab\Papa\hdf'
Q_initial = [0.5, 0.5]
alpha_true = 0.2
beta_true = 0.2

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
    #target3 = np.zeros(ind_check_reward_states.size-200)
    target3 = np.zeros(np.min([num_successful_trials-200,100]))
    reward3 = np.zeros(target3.size)
    target_freechoice_block1 = np.zeros(70)
    switch_freechoice_block1 = np.zeros(target_freechoice_block1.size)  # switch vector: 1 if current freechoice is a switch from previous choice
    priortrial_rewarded_block1 = np.zeros(target_freechoice_block1.size)
    reward_freechoice_block1 = np.zeros(70)
    target_freechoice_block3 = []
    switch_freechoice_block3 = []                                       # switch vector: 1 if current freechoice is a switch from previous choice
    priortrial_rewarded_block3 = []
    #reward_freechoice_block3 = np.zeros(num_successful_trials-200)
    reward_freechoice_block3 = np.zeros(np.min([num_successful_trials-200,100]))
    trial1 = np.zeros(target1.size)
    trial3 = np.zeros(target3.size)
    stim_trials = []

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
            if i > 0:
                switch_freechoice_block1[counter_block1] = (target1[i]!=target1[i-1])
                priortrial_rewarded_block1[counter_block1] = reward1[i-1]
            reward_freechoice_block1[counter_block1] = reward1[i]
            counter_block1 += 1
    #for i in range(200,num_successful_trials):
    for i in range(200,np.min([num_successful_trials,300])):
        target_state3 = state[ind_check_reward_states[i] - 2]
        prev_target_state = state[ind_check_reward_states[i-1] - 2]
        switched = (target_state3 != prev_target_state)    # do it this way to account for first free choice trial switch from last instructed trial in Block B
        previously_rewarded = (prev_target_state =='hold_targetL')*rewarded_reward_scheduleL[i-1] + (prev_target_state=='hold_targetH')*rewarded_reward_scheduleH[i-1]
        trial3[i-200] = instructed_or_freechoice[i]
        if target_state3 == 'hold_targetL':
            target3[i-200] = 1
            reward3[i-200] = rewarded_reward_scheduleL[i]
            if trial3[i-200]==1:   # instructed trial to low-value targer paired with stim
                stim_trials.append(i)
        else:
            target3[i-200] = 2
            reward3[i-200] = rewarded_reward_scheduleH[i]
        if trial3[i-200] == 2:
            target_freechoice_block3.append(target3[i-200])
            switch_freechoice_block3.append(switched)
            priortrial_rewarded_block3.append(previously_rewarded)
            reward_freechoice_block3[counter_block3] = reward3[i-200]
            counter_block3 += 1

    max_block3 = 70*(counter_block3 > 70) + counter_block3*(counter_block3 < 70)
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward1, target1, trial1), bounds=[(0,1),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    # RL model fit for Block A
    Qlow_block1, Qhigh_block1, prob_low_block1, max_loglikelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, trial1)
    loglikelihood_random1 = logLikelihoodRLPerformance_random([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, trial1)
    BIC1 = -2*max_loglikelihood1 + 2*np.log(target1.size)
    pseudo_rsquared1 = float(loglikelihood_random1 - max_loglikelihood1)/loglikelihood_random1
    # what is RL model performance with parameters fit from Block A in Block A'

    Qlow_block3, Qhigh_block3, prob_low_block3, max_loglikelihood3 = RLPerformance([alpha_ml_block1,beta_ml_block1],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)
    loglikelihood_random3 = logLikelihoodRLPerformance_random([alpha_ml_block1,beta_ml_block1],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)
    BIC3 = -2*max_loglikelihood3 + 2*np.log(target1.size)
    pseudo_rsquared3 = float(loglikelihood_random3 - max_loglikelihood3)/loglikelihood_random3
    # what are parameters fit from Block A' with initial Q values from end of Block A
    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward3, target3, trial3), bounds=[(0,1),(0,None)])
    #result3 = op.minimize(nll, [alpha_ml_block1, beta_ml_block1], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_freechoice_block3[:max_block3], target_freechoice_block3[:max_block3]), bounds=[(0,1),(0,None)])
    #result3 = op.minimize(nll, [alpha_ml_block1, beta_ml_block1], args=(Q_initial, reward_freechoice_block3[:max_block3], target_freechoice_block3[:max_block3]), bounds=[(0,1),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3_refit, Qhigh_block3_refit, prob_low_block3_refit, max_loglikelihood3_refit = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)
    loglikelihood_random3_refit = logLikelihoodRLPerformance_random([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)
    BIC3_refit = -2*max_loglikelihood3_refit + 2*np.log(target3.size)
    pseudo_rsquared3_refit = float(loglikelihood_random3_refit - max_loglikelihood3_refit)/loglikelihood_random3_refit

    ######################
    ### SWITCHING analysis
    ######################

    # Block A
    rate_switch1 = float(switch_freechoice_block1.sum())/switch_freechoice_block1.size
    prob_switch_given_rewarded1 = float(np.sum(np.logical_and(switch_freechoice_block1[1:], priortrial_rewarded_block1[1:])))/np.sum(priortrial_rewarded_block1[1:]) # prob(switch|rewarded)
    prob_switch_given_notrewarded1 = float(np.sum(np.logical_and(switch_freechoice_block1[1:], np.logical_not(priortrial_rewarded_block1[1:]))))/np.sum(np.logical_not(priortrial_rewarded_block1[1:])) # prob(switch|not rewarded)
    
    # Block A'
    switch_freechoice_block3 = np.array(switch_freechoice_block3)
    priortrial_rewarded_block3 = np.array(priortrial_rewarded_block3).flatten()
    rate_switch3 = float(switch_freechoice_block3.sum())/switch_freechoice_block3.size
    prob_switch_given_rewarded3 = float(np.sum(np.logical_and(switch_freechoice_block3, priortrial_rewarded_block3)))/np.sum(priortrial_rewarded_block3) # prob(switch|rewarded)
    prob_switch_given_notrewarded3 = float(np.sum(np.logical_and(switch_freechoice_block3, np.logical_not(priortrial_rewarded_block3))))/np.sum(np.logical_not(priortrial_rewarded_block3)) # prob(switch|not rewarded)

    '''
    # plot fig of Qlow_block 1, Qlow_block3
    plt.figure()
    plt.plot(range(1,Qlow_block1.size+1),Qlow_block1,'b',label="Block A")
    plt.plot(range(1,Qlow_block3.size+1),Qlow_block3,'r',label="Block A'")
    plt.axis([1,Qlow_block1.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Q-value')
    plt.yscale('log')
    plt.title('%s' % name)
    plt.show()
    '''
    totalprob_chooselow = np.sum(target_freechoice_block3==np.ones(len(target_freechoice_block3)))/float(len(target_freechoice_block3))

    # calculate accuracy of fit for Block A
    model1 = 0.33*(prob_low_block1 > 0.5) + 0.66*(np.less_equal(prob_low_block1, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    target_freechoice_block1 = np.array(target_freechoice_block1)
    fit1 = np.equal(model1[:-1],(0.33*target_freechoice_block1))
    shuffled_indices = range(0,target_freechoice_block1.size)
    np.random.shuffle(shuffled_indices)
    shuffle_fit1 = np.equal(model1[:-1],(0.33*target_freechoice_block1[shuffled_indices]))
    accuracy1 = float(np.sum(fit1))/model1.size
    shuffle_accuracy1 = float(np.sum(shuffle_fit1))/model1.size
    # plot fig of prob_low_block1 versus choice
    
    plt.figure()
    plt.subplot(121)
    plt.plot(range(1,target_freechoice_block1.size+1),target_freechoice_block1*0.33,'b',label="Data")
    plt.plot(range(1,prob_low_block1.size+1),model1,'r--',label="Model")
    plt.axis([1,target_freechoice_block1.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Target Choice')
    plt.title('Block A - %s - Accuracy %f - Chance %f' % (name[1:-4],accuracy1,shuffle_accuracy1),fontsize=8)
    plt.legend()
    
    
    # calculate accuracy of fit applied to block A'
    model3 = 0.33*(prob_low_block3[:-1] > 0.5) + 0.66*(np.less_equal(prob_low_block3[:-1], 0.5))  #discard last element since it predicts selection for trial after final trial
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3,(0.33*target_freechoice_block3))
    shuffled_indices = range(0,target_freechoice_block3.size)
    np.random.shuffle(shuffled_indices)
    shuffle_fit3 = np.equal(model3,(0.33*target_freechoice_block3[shuffled_indices]))
    accuracy3 = float(np.sum(fit3))/model3.size
    shuffle_accuracy3 = float(np.sum(shuffle_fit3))/model3.size
    # plot fig of prob_low_block3 versus choice
    plt.subplot(122)
    plt.plot(range(1,target_freechoice_block3.size+1),target_freechoice_block3*0.33,'b',label="Data")
    plt.plot(range(1,model3.size+1),model3,'r--',label="Model")
    plt.axis([1,model3.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Target Choice')
    plt.title("Block A' - %s - Accuracy %f - Chance %f" % (name[1:-4],accuracy3,shuffle_accuracy3),fontsize=8)
    plt.legend()
    plt.savefig('Papa_RL_figs/RLfit_%s.svg' % name[1:-4])    # save this filetype for AI editing
    plt.savefig('Papa_RL_figs/RLfit_%s.png' % name[1:-4])    # save this filetype for easy viewing
    plt.close()

    # calculate acurracy of re-fit parameters for block A'
    model3_refit = 0.33*(prob_low_block3_refit[:-1] > 0.5) + 0.66*(np.less_equal(prob_low_block3_refit[:-1], 0.5))  #discard last element since it predicts selection for trial after final trial
    refit3 = np.equal(model3_refit,(0.33*target_freechoice_block3))
    np.random.shuffle(shuffled_indices)
    shuffle_refit3 = np.equal(model3_refit,(0.33*target_freechoice_block3[shuffled_indices]))
    accuracy3_refit = float(np.sum(refit3))/model3_refit.size
    shuffle_accuracy3_refit = float(np.sum(shuffle_refit3))/model3_refit.size
    plt.figure()
    plt.plot(range(1,target_freechoice_block3.size+1),target_freechoice_block3*0.33,'b',label="Data")
    plt.plot(range(1,model3_refit.size+1),model3_refit,'r--',label="Refit Model")
    plt.axis([1,model3_refit.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Target Choice')
    plt.title("Block A' - %s - Accuracy %f - Chance %f" % (name[1:-4],accuracy3_refit,shuffle_accuracy3_refit),fontsize=8)
    plt.legend()
    plt.savefig('Papa_RL_figs/RLfit_%s_refit.svg' % name[1:-4])    # save this filetype for AI editing
    plt.savefig('Papa_RL_figs/RLfit_%s_refit.png' % name[1:-4])    # save this filetype for easy viewing
    plt.close()

    

    # model for trials immediately following stim trial
    stim_trials = np.array(stim_trials) - 200   # array of stim trial indices in Block A', aligned to zero
    max_trial_dist = np.amax(stim_trials[1:] - stim_trials[:-1])
    max_trial_dsit = np.amax([max_trial_dist,300 - stim_trials[-1]])  # use when doing analysis for first 100 trials only in Block A'
    #max_trial_dist = np.amax([max_trial_dist,num_successful_trials - stim_trials[-1] - 200])
    global_max_trial_dist = np.amax([global_max_trial_dist,max_trial_dist])
    error_vec = np.zeros(max_trial_dist,dtype=float) # placeholder vector counting number of errors
    error_vec_refit = np.zeros(max_trial_dist,dtype=float)
    after_stim_trials = np.zeros(max_trial_dist) # placeholder vector counting number of trials following stim trial
    choose_lowvalue = np.zeros(max_trial_dist,dtype=float)
    choose_lowvalue_reward = np.zeros(max_trial_dist,dtype=float)   # choose low value target in free choice trials following a stim trial with reward
    choose_lowvalue_noreward = np.zeros(max_trial_dist,dtype=float) # choose low value target in free choice trials following a stim trial without reward
    counter_error_mat = 0
    after_stim_trials_reward = np.zeros(max_trial_dist,dtype=float)
    after_stim_trials_noreward = np.zeros(max_trial_dist,dtype=float)
    
    for stim_index in stim_trials:
        if stim_index == stim_trials[-1]:
            error = np.not_equal(model3[stim_index:],0.33*target_freechoice_block3[stim_index:]) # returns true if the model does not match the true behavior
            error_refit = np.not_equal(model3_refit[stim_index:],0.33*target_freechoice_block3[stim_index:])
            error_vec[:error.size] += error
            error_vec_refit[:error_refit.size] += error_refit
            after_stim_trials[:error.size] += np.ones(error.size) # count number of trials are being considered between stim trials
            lowvalue_choice = np.equal(target_freechoice_block3[stim_index:],np.ones(target_freechoice_block3[stim_index:].size)) # 1 if chose low -value, 2 if chose high-value target
            choose_lowvalue[:lowvalue_choice.size] += lowvalue_choice # count number of times low value target is choosen N number of trials after stim trial
        else:
            error = np.not_equal(model3[stim_index:stim_trials[counter_error_mat+1]],0.33*target_freechoice_block3[stim_index:stim_trials[counter_error_mat+1]]) # returns true if the model does not match the true behavior
            error_refit = np.not_equal(model3_refit[stim_index:stim_trials[counter_error_mat+1]],0.33*target_freechoice_block3[stim_index:stim_trials[counter_error_mat+1]]) # returns true if the model does not match the true behavior
            error_vec[:error.size] += error
            error_vec_refit[:error_refit.size] += error_refit
            after_stim_trials[:error.size] += np.ones(error.size) # count number of trials are being considered for errors
            lowvalue_choice = np.equal(target_freechoice_block3[stim_index:stim_trials[counter_error_mat+1]],np.ones(target_freechoice_block3[stim_index:stim_trials[counter_error_mat+1]].size))
            choose_lowvalue[:lowvalue_choice.size] += lowvalue_choice
        if reward3[stim_index] == 1:
            choose_lowvalue_reward[:lowvalue_choice.size] += lowvalue_choice
            after_stim_trials_reward[:error.size] += np.ones(error.size) # count number of trials are being considered between stim trials
        else:
            choose_lowvalue_noreward[:lowvalue_choice.size] += lowvalue_choice
            after_stim_trials_noreward[:error.size] += np.ones(error.size) # count number of trials are being considered between stim trials
        counter_error_mat += 1

    # Save model fit parameters and probabilities
    errors_after_stim = error_vec/after_stim_trials
    errors_after_stim_refit = error_vec_refit/after_stim_trials
    prob_choose_lowvalue = choose_lowvalue/after_stim_trials
    prob_choose_lowvalue_reward = choose_lowvalue_reward/after_stim_trials_reward
    prob_choose_lowvalue_noreward = choose_lowvalue_noreward/after_stim_trials_noreward

    if name in hdf_list_stim:
        hdf_data_stim[name] = [alpha_ml_block1,beta_ml_block1]
        BIC_stim[name] = [BIC1,BIC3,BIC3_refit]
        pseudo_rsquared_stim[name] = [pseudo_rsquared1,pseudo_rsquared3,pseudo_rsquared3_refit]
        modelerror_afterstim_stim[name] = errors_after_stim
        modelerror_afterstim_stim_refit[name] = errors_after_stim_refit
        prob_lowvalue_afterstim_stim[name] = prob_choose_lowvalue
        prob_lowvalue_afterstim_withreward_stim[name] = prob_choose_lowvalue_reward
        prob_lowvalue_afterstim_noreward_stim[name] = prob_choose_lowvalue_noreward
        prob_switch_given_rewarded_block1[name] = prob_switch_given_rewarded1
        prob_switch_given_notrewarded_block1[name] = prob_switch_given_notrewarded1
        prob_switch_given_rewarded_block3[name] = prob_switch_given_rewarded3
        prob_switch_given_notrewarded_block3[name] = prob_switch_given_notrewarded3
        prob_chooselow_stim[name] = totalprob_chooselow
        modelerror_stim[name] = accuracy3
    else:
        hdf_data_sham[name] = [alpha_ml_block1,beta_ml_block1]
        BIC_sham[name] = [BIC1,BIC3,BIC3_refit]
        pseudo_rsquared_sham[name] = [pseudo_rsquared1,pseudo_rsquared3,pseudo_rsquared3_refit]
        modelerror_afterstim_sham[name] = errors_after_stim
        modelerror_afterstim_sham_refit[name] = errors_after_stim_refit
        prob_lowvalue_afterstim_sham[name] = prob_choose_lowvalue
        prob_lowvalue_afterstim_withreward_sham[name] = prob_choose_lowvalue_reward
        prob_lowvalue_afterstim_noreward_sham[name] = prob_choose_lowvalue_noreward
        prob_switch_given_rewarded_block1[name] = prob_switch_given_rewarded1
        prob_switch_given_notrewarded_block1[name] = prob_switch_given_notrewarded1
        prob_switch_given_rewarded_block3[name] = prob_switch_given_rewarded3
        prob_switch_given_notrewarded_block3[name] = prob_switch_given_notrewarded3
        prob_chooselow_sham[name] = totalprob_chooselow
        modelerror_sham[name] = accuracy3
    hdf.close()     # close hdf table file

# compute average likelihood of model error and choosing low value target for trials after stim
global_max_trial_dist
mean_modelerror_afterstim_stim = []
std_modelerror_afterstim_stim = []
mean_modelerror_afterstim_stim_refit = []
std_modelerror_afterstim_stim_refit = []
mean_prob_lowvalue_afterstim_stim = []
std_prob_lowvalue_afterstim_stim = []
mean_prob_lowvalue_afterstim_reward_stim = []
std_prob_lowvalue_afterstim_reward_stim = []
mean_prob_lowvalue_afterstim_noreward_stim = []
std_prob_lowvalue_afterstim_noreward_stim = []

mean_modelerror_afterstim_sham = []
std_modelerror_afterstim_sham = []
mean_modelerror_afterstim_sham_refit = []
std_modelerror_afterstim_sham_refit = []
mean_prob_lowvalue_afterstim_sham = []
std_prob_lowvalue_afterstim_sham = []
mean_prob_lowvalue_afterstim_reward_sham = []
std_prob_lowvalue_afterstim_reward_sham = []
mean_prob_lowvalue_afterstim_noreward_sham = []
std_prob_lowvalue_afterstim_noreward_sham = []


for i in range(0,global_max_trial_dist):
    # for the ith trial after stim
    prob_error = []
    prob_error_refit = []
    prob_lowvalue = []
    prob_lowvalue_reward = []
    prob_lowvalue_noreward = []
    for name in hdf_list_stim:
        modelerror = modelerror_afterstim_stim[name]
        modelerror_refit = modelerror_afterstim_stim_refit[name]
        lowvalue = prob_lowvalue_afterstim_stim[name]
        lowvalue_reward = prob_lowvalue_afterstim_withreward_stim[name]
        lowvalue_noreward = prob_lowvalue_afterstim_noreward_stim[name]
        if i < modelerror.size:
            prob_error.append(modelerror[i])
            prob_lowvalue.append(lowvalue[i])
            prob_lowvalue_reward.append(lowvalue_reward[i])
            prob_lowvalue_noreward.append(lowvalue_noreward[i])
        if i < modelerror_refit.size:
            prob_error_refit.append(modelerror_refit[i])
    mean_prob_error = np.mean(prob_error)
    std_prob_error = np.std(prob_error)/np.sqrt(len(prob_error))
    mean_prob_error_refit = np.mean(prob_error_refit)
    std_prob_error_refit = np.std(prob_error_refit)/np.sqrt(len(prob_error_refit))
    mean_prob_lowvalue = np.mean(prob_lowvalue)
    std_prob_lowvalue = np.std(prob_lowvalue)/np.sqrt(len(prob_lowvalue))
    mean_prob_lowvalue_reward = np.mean(prob_lowvalue_reward)
    std_prob_lowvalue_reward = np.std(prob_lowvalue_reward)/np.sqrt(len(prob_lowvalue_reward))
    mean_prob_lowvalue_noreward = np.mean(prob_lowvalue_noreward)
    std_prob_lowvalue_noreward = np.std(prob_lowvalue_noreward)/np.sqrt(len(prob_lowvalue_noreward))
    mean_modelerror_afterstim_stim.append(mean_prob_error)
    std_modelerror_afterstim_stim.append(std_prob_error)
    mean_modelerror_afterstim_stim_refit.append(mean_prob_error_refit)
    std_modelerror_afterstim_stim_refit.append(std_prob_error_refit)
    mean_prob_lowvalue_afterstim_stim.append(mean_prob_lowvalue)
    std_prob_lowvalue_afterstim_stim.append(std_prob_lowvalue)
    mean_prob_lowvalue_afterstim_reward_stim.append(mean_prob_lowvalue_reward)
    std_prob_lowvalue_afterstim_reward_stim.append(std_prob_lowvalue_reward)
    mean_prob_lowvalue_afterstim_noreward_stim.append(mean_prob_lowvalue_noreward)
    std_prob_lowvalue_afterstim_noreward_stim.append(std_prob_lowvalue_noreward)

    prob_error = []
    prob_error_refit = []
    prob_lowvalue = []
    prob_lowvalue_reward = []
    prob_lowvalue_noreward = []
    for name in hdf_list_sham:
        modelerror = modelerror_afterstim_sham[name]
        lowvalue = prob_lowvalue_afterstim_sham[name]
        lowvalue_reward = prob_lowvalue_afterstim_withreward_sham[name]
        lowvalue_noreward = prob_lowvalue_afterstim_noreward_sham[name]
        if i < modelerror.size:
            prob_error.append(modelerror[i])
            prob_lowvalue.append(lowvalue[i])
            prob_lowvalue_reward.append(lowvalue_reward[i])
            prob_lowvalue_noreward.append(lowvalue_noreward[i])
        if i < modelerror_refit.size:
            prob_error_refit.append(modelerror_refit[i])
    mean_prob_error = np.mean(prob_error)
    std_prob_error = np.std(prob_error)/float(np.sqrt(len(prob_error)))
    mean_prob_error_refit = np.mean(prob_error_refit)
    std_prob_error_refit = np.std(prob_error_refit)/np.sqrt(len(prob_error_refit))
    mean_prob_lowvalue = np.mean(prob_lowvalue)
    std_prob_lowvalue = np.std(prob_lowvalue)/float(np.sqrt(len(prob_lowvalue)))
    mean_prob_lowvalue_reward = np.mean(prob_lowvalue_reward)
    std_prob_lowvalue_reward = np.std(prob_lowvalue_reward)/np.sqrt(len(prob_lowvalue_reward))
    mean_prob_lowvalue_noreward = np.mean(prob_lowvalue_noreward)
    std_prob_lowvalue_noreward = np.std(prob_lowvalue_noreward)/np.sqrt(len(prob_lowvalue_noreward))
    mean_modelerror_afterstim_sham.append(mean_prob_error)
    std_modelerror_afterstim_sham.append(std_prob_error)
    mean_modelerror_afterstim_sham_refit.append(mean_prob_error_refit)
    std_modelerror_afterstim_sham_refit.append(std_prob_error_refit)
    mean_prob_lowvalue_afterstim_sham.append(mean_prob_lowvalue)
    std_prob_lowvalue_afterstim_sham.append(std_prob_lowvalue)
    mean_prob_lowvalue_afterstim_reward_sham.append(mean_prob_lowvalue_reward)
    std_prob_lowvalue_afterstim_reward_sham.append(std_prob_lowvalue_reward)
    mean_prob_lowvalue_afterstim_noreward_sham.append(mean_prob_lowvalue_noreward)
    std_prob_lowvalue_afterstim_noreward_sham.append(std_prob_lowvalue_noreward)

# overall probability of choosing low value target in Block A'
prob_lowtarg_stim = []
prob_lowtarg_sham = []
prob_error_stim = []
prob_error_sham = []
prob_error_stim_refit = []
prob_error_sham_refit = []
for name in hdf_list_stim:
    prob_lowtarg_stim.append(prob_chooselow_stim[name])
    prob_error_stim.append(modelerror_stim[name])
for name in hdf_list_sham:
    prob_lowtarg_sham.append(prob_chooselow_sham[name])
    prob_error_sham.append(modelerror_sham[name])

avg_rate_lowtarg_stim = np.mean(prob_lowtarg_stim)
avg_rate_lowtarg_sham = np.mean(prob_lowtarg_sham)
avg_rate_modelerror_stim = np.mean(prob_error_stim)
avg_rate_modelerror_sham = np.mean(prob_error_sham)

######################################################
### OVERALL PROBABILITY OF SWITCHING IN BLOCK A VS A'
######################################################

prob_switch_reward_block1_stim = []
prob_switch_noreward_block1_stim = []
prob_switch_reward_block3_stim = []
prob_switch_noreward_block3_stim = []

prob_switch_reward_block1_sham = []
prob_switch_noreward_block1_sham = []
prob_switch_reward_block3_sham = []
prob_switch_noreward_block3_sham = []

for name in hdf_list_stim:
    prob_switch_reward_block1_stim.append(prob_switch_given_rewarded_block1[name])
    prob_switch_noreward_block1_stim.append(prob_switch_given_notrewarded_block1[name])
    prob_switch_reward_block3_stim.append(prob_switch_given_rewarded_block3[name])
    prob_switch_noreward_block3_stim.append(prob_switch_given_notrewarded_block3[name])
for name in hdf_list_sham:
    prob_switch_reward_block1_sham.append(prob_switch_given_rewarded_block1[name])
    prob_switch_noreward_block1_sham.append(prob_switch_given_notrewarded_block1[name])
    prob_switch_reward_block3_sham.append(prob_switch_given_rewarded_block3[name])
    prob_switch_noreward_block3_sham.append(prob_switch_given_notrewarded_block3[name])

t_switch_reward_block1, p_switch_reward_block1 = stats.ttest_ind(prob_switch_reward_block1_stim,prob_switch_reward_block1_sham,equal_var=False)
t_switch_noreward_block1, p_switch_noreward_block1 = stats.ttest_ind(prob_switch_noreward_block1_stim,prob_switch_noreward_block1_sham,equal_var=False)
t_switch_reward_block3, p_switch_reward_block3 = stats.ttest_ind(prob_switch_reward_block3_stim,prob_switch_reward_block3_sham,equal_var=False)
t_switch_noreward_block3, p_switch_noreward_block3 = stats.ttest_ind(prob_switch_noreward_block3_stim,prob_switch_noreward_block3_sham,equal_var=False)

avg_prob_switch_given_rewarded_block1_stim = np.mean(prob_switch_reward_block1_stim)
avg_prob_switch_given_notrewarded_block1_stim = np.mean(prob_switch_noreward_block1_stim)
avg_prob_switch_given_rewarded_block3_stim = np.mean(prob_switch_reward_block3_stim)
avg_prob_switch_given_notrewarded_block3_stim = np.mean(prob_switch_noreward_block3_stim)
avg_prob_switch_given_rewarded_block1_sham = np.mean(prob_switch_reward_block1_sham)
avg_prob_switch_given_notrewarded_block1_sham = np.mean(prob_switch_noreward_block1_sham)
avg_prob_switch_given_rewarded_block3_sham = np.mean(prob_switch_reward_block3_sham)
avg_prob_switch_given_notrewarded_block3_sham = np.mean(prob_switch_noreward_block3_sham)

plt.figure()
plt.plot([1,2,3,4],[avg_prob_switch_given_rewarded_block1_stim,avg_prob_switch_given_notrewarded_block1_stim,avg_prob_switch_given_rewarded_block3_stim,avg_prob_switch_given_notrewarded_block3_stim],'b',label='Stim')
plt.plot([1,2,3,4],[avg_prob_switch_given_rewarded_block1_sham,avg_prob_switch_given_notrewarded_block1_sham,avg_prob_switch_given_rewarded_block3_sham,avg_prob_switch_given_notrewarded_block3_sham],'g',label='Sham')
plt.text(1,0.7,p_switch_reward_block1)
plt.text(2,0.75,p_switch_noreward_block1)
plt.text(3,0.7,p_switch_reward_block3)
plt.text(4,0.75,p_switch_noreward_block3)
plt.axis([0, 5, 0, 1])
labels = ["Block A:\n Switch|Rewarded","Block A:\n Switch|Not Rewarded","Block A':\n Switch|Rewarded","Block A':\n Switch|Not Rewarded"]
plt.xticks([1,2,3,4], labels, fontsize=8)
plt.title('Switching Probabilities')
plt.ylabel('Probability')
plt.legend()
plt.savefig('Papa_RL_figs/switching_probability-voltage_only.svg')
plt.savefig('Papa_RL_figs/switching_probability-voltage_only.png')

###############################
### Measures of goodness-of-fit
###############################

BIC_block1_stim = []
BIC_block1_sham = []
BIC_block3_stim = []
BIC_block3_sham = []
BIC_block3_refit_stim = []
pseudo_rsquared_block1_stim = []
pseudo_rsquared_block1_sham = []
pseudo_rsquared_block3_stim = []
pseudo_rsquared_block3_sham = []
pseudo_rsquared_block3_refit_stim = []


for name in hdf_list_stim:
    BIC_block1_stim.append(BIC_stim[name][0])
    BIC_block3_stim.append(BIC_stim[name][1])
    BIC_block3_refit_stim.append(BIC_stim[name][2])
    pseudo_rsquared_block1_stim.append(pseudo_rsquared_stim[name][0])
    pseudo_rsquared_block3_stim.append(pseudo_rsquared_stim[name][1])
    pseudo_rsquared_block3_refit_stim.append(pseudo_rsquared_stim[name][2])
for name in hdf_list_sham:
    BIC_block1_sham.append(BIC_sham[name][0])
    BIC_block3_sham.append(BIC_sham[name][1])
    pseudo_rsquared_block1_sham.append(pseudo_rsquared_sham[name][0])
    pseudo_rsquared_block3_sham.append(pseudo_rsquared_sham[name][1])

# plots of goodness of fit
plt.figure()
plt.scatter(0.95*np.ones(len(BIC_block1_stim)), BIC_block1_stim,c='b',label='Stim')
plt.scatter(1.05*np.ones(len(BIC_block1_sham)), BIC_block1_sham,c='g',label='Sham')
plt.scatter(1.95*np.ones(len(BIC_block3_stim)), BIC_block3_stim,c='b')
plt.scatter(2.05*np.ones(len(BIC_block3_sham)), BIC_block3_sham,c='g')
plt.scatter(3*np.ones(len(BIC_block3_refit_stim)), BIC_block3_refit_stim,c='b')
plt.ylabel('BIC')
labels = ["Block A","Block A'","Block A' - Refit params"]
plt.xticks([1,2,3], labels, fontsize=8)
plt.title('Bayesian Information Criterion')
plt.legend()
plt.savefig('Papa_RL_figs/BIC-voltage_only.svg')
plt.savefig('Papa_RL_figs/BIC-voltage_only.png')
plt.close()

plt.figure()
plt.scatter(0.95*np.ones(len(pseudo_rsquared_block1_stim)), pseudo_rsquared_block1_stim,c='b',label='Stim')
plt.scatter(1.05*np.ones(len(pseudo_rsquared_block1_sham)), pseudo_rsquared_block1_sham,c='g',label='Sham')
plt.scatter(1.95**np.ones(len(pseudo_rsquared_block3_stim)), pseudo_rsquared_block3_stim,c='b')
plt.scatter(2.05*np.ones(len(pseudo_rsquared_block3_sham)), pseudo_rsquared_block3_sham,c='g')
plt.scatter(3*np.ones(len(pseudo_rsquared_block3_refit_stim)), pseudo_rsquared_block3_refit_stim,c='b')
labels = ["Block A","Block A'","Block A' - Refit params"]
plt.xticks([1,2,3], labels, fontsize=8)
plt.title('Pseudo R-Squared')
plt.legend()
plt.savefig('Papa_RL_figs/Pseudo_R2-voltage_only.svg')
plt.savefig('Papa_RL_figs/Pseudo_R2-voltage_only.png')
plt.close()

# plots of model errors and choosing low value target as a function of trials post-stim trial

plt.figure()
plt.errorbar(range(1,len(mean_modelerror_afterstim_stim)+1),mean_modelerror_afterstim_stim,yerr=std_modelerror_afterstim_stim,label='Stim')
plt.errorbar(range(1,len(mean_modelerror_afterstim_sham)+1),mean_modelerror_afterstim_sham,yerr=std_modelerror_afterstim_sham,label='Sham')
plt.plot(range(1,len(mean_modelerror_afterstim_stim)+1),avg_rate_modelerror_stim*np.ones(len(mean_modelerror_afterstim_stim)),'b',label='Stim overall avg')
plt.plot(range(1,len(mean_modelerror_afterstim_sham)+1),avg_rate_modelerror_sham*np.ones(len(mean_modelerror_afterstim_sham)),'g',label='Sham overal avg')
plt.title('Model Error likelihood')
plt.xlabel('Trials post-IC trial with stimulation')
plt.ylabel('Probability of Model Error')
plt.legend()
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim-voltage_only.svg')
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim-voltage_only.png')
plt.close()

plt.figure()
plt.errorbar(range(1,len(mean_modelerror_afterstim_stim_refit)+1),mean_modelerror_afterstim_stim_refit,yerr=std_modelerror_afterstim_stim_refit,label='Stim')
plt.errorbar(range(1,len(mean_modelerror_afterstim_sham_refit)+1),mean_modelerror_afterstim_sham_refit,yerr=std_modelerror_afterstim_sham_refit,label='Sham')
#plt.plot(range(1,len(mean_modelerror_afterstim_stim)+1),avg_rate_modelerror_stim*np.ones(len(mean_modelerror_afterstim_stim_refit)),'b',label='Stim overall avg')
#plt.plot(range(1,len(mean_modelerror_afterstim_sham)+1),avg_rate_modelerror_sham*np.ones(len(mean_modelerror_afterstim_sham)),'g',label='Sham overal avg')
plt.title('Model Error likelihood with Re-fit Parameters')
plt.xlabel('Trials post-IC trial with stimulation')
plt.ylabel('Probability of Model Error')
plt.legend()
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim_refit-voltage_only.svg')
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim_refit-voltage_only.png')
plt.close()

plt.figure()
plt.errorbar(range(1,len(mean_modelerror_afterstim_stim_refit)+1),mean_modelerror_afterstim_stim_refit,yerr=std_modelerror_afterstim_stim_refit,label='Stim - refit')
plt.errorbar(range(1,len(mean_modelerror_afterstim_stim)+1),mean_modelerror_afterstim_stim,yerr=std_modelerror_afterstim_stim,label='Stim - original')
#plt.plot(range(1,len(mean_modelerror_afterstim_stim)+1),avg_rate_modelerror_stim*np.ones(len(mean_modelerror_afterstim_stim_refit)),'b',label='Stim overall avg')
#plt.plot(range(1,len(mean_modelerror_afterstim_sham)+1),avg_rate_modelerror_sham*np.ones(len(mean_modelerror_afterstim_sham)),'g',label='Sham overal avg')
plt.title('Model Error Comparison with Re-fit Parameters')
plt.xlabel('Trials post-IC trial with stimulation')
plt.ylabel('Probability of Model Error')
plt.legend()
plt.savefig('Papa_RL_figs/modelerror_comparison_afterstim_refit-voltage_only.svg')
plt.savefig('Papa_RL_figs/modelerror_comparison_afterstim_refit-voltage_only.png')
plt.close()

plt.figure()
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),mean_prob_lowvalue_afterstim_stim,label='Stim',yerr=std_prob_lowvalue_afterstim_stim)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_sham)+1),mean_prob_lowvalue_afterstim_sham,label='Sham',yerr=std_prob_lowvalue_afterstim_sham)
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_stim*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'b',label='Stim overall avg')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_sham)+1),avg_rate_lowtarg_sham*np.ones(len(mean_prob_lowvalue_afterstim_sham)),'g',label='Sham overal avg')
plt.title('Low-Value Target Selection')
plt.xlabel('Trials post-IC trial with stimulation')
plt.ylabel('Probability of Selecting Low-Value Target')
plt.legend()
plt.savefig('Papa_RL_figs/prob_lowvalue_afterstim-voltage_only.svg')
plt.savefig('Papa_RL_figs/prob_lowvalue_afterstim-voltage_only.png')
plt.close()

plt.figure()
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),mean_prob_lowvalue_afterstim_stim,label='Stim',yerr=std_prob_lowvalue_afterstim_stim)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_sham)+1),mean_prob_lowvalue_afterstim_sham,label='Sham',yerr=std_prob_lowvalue_afterstim_sham)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_reward_stim)+1),mean_prob_lowvalue_afterstim_reward_stim,label='Stim with reward',yerr=std_prob_lowvalue_afterstim_reward_stim)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_noreward_stim)+1),mean_prob_lowvalue_afterstim_noreward_stim,label='Stim w/o reward',yerr=std_prob_lowvalue_afterstim_noreward_stim)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_reward_sham)+1),mean_prob_lowvalue_afterstim_reward_sham,label='Sham with reward',yerr=std_prob_lowvalue_afterstim_reward_sham)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_noreward_sham)+1),mean_prob_lowvalue_afterstim_noreward_sham,label='Sham w/o reward',yerr=std_prob_lowvalue_afterstim_noreward_sham)
plt.title('Low-Value Target Selection')
plt.xlabel('Trials post-IC trial with stimulation')
plt.ylabel('Probability of Selecting Low-Value Target')
plt.legend()
plt.savefig('Papa_RL_figs/prob_lowvalue_afterstim_rewardcomparison-voltage_only.svg')
plt.savefig('Papa_RL_figs/prob_lowvalue_afterstim_rewardcomparison-voltage_only.png')
plt.close()




