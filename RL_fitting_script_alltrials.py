from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance
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
modelerror_afterstim_stim = dict()
prob_lowvalue_afterstim_stim = dict()

hdf_data_sham = dict()
modelerror_afterstim_sham = dict()
prob_lowvalue_afterstim_sham = dict()
modelerror_afterstim_stim_refit = dict()
modelerror_afterstim_sham_refit = dict()
prob_chooselow_stim = dict()
prob_chooselow_blocks100_stim = dict()
prob_chooselow_sham = dict()
modelerror_stim = dict()
modelerror_sham = dict()


for x in hdf_list_stim:
    hdf_data_stim[x] = []
    modelerror_afterstim_stim[x] = []
    modelerror_afterstim_stim_refit[x] = []
    prob_lowvalue_afterstim_stim[x] = []
    prob_chooselow_stim[x] = []
    prob_chooselow_blocks100_stim[x] = []

for x in hdf_list_sham:
    hdf_data_sham[x] = []
    modelerror_afterstim_sham[x] = []
    modelerror_afterstim_sham_refit[x] = []
    prob_lowvalue_afterstim_sham[x] = []
    prob_chooselow_sham[x] = [x]

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
                stim_trials.append(i)
        else:
            target3[i-200] = 2
            reward3[i-200] = rewarded_reward_scheduleH[i]
        if trial3[i-200] == 2:
            target_freechoice_block3.append(target3[i-200])
            reward_freechoice_block3[counter_block3] = reward3[i-200]
            counter_block3 += 1

    max_block3 = 70*(counter_block3 > 70) + counter_block3*(counter_block3 < 70)
    nll = lambda *args: -logLikelihoodRLPerformance(*args)
    result1 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward1, target1, trial1), bounds=[(0,1),(0,None)])
    alpha_ml_block1, beta_ml_block1 = result1["x"]
    # RL model fit for Block A
    Qlow_block1, Qhigh_block1, prob_low_block1,log_likelihood1 = RLPerformance([alpha_ml_block1,beta_ml_block1],Q_initial,reward1,target1, trial1)
    # what is RL model performance with parameters fit from Block A in Block A'

    Qlow_block3, Qhigh_block3, prob_low_block3, log_likelihood3 = RLPerformance([alpha_ml_block1,beta_ml_block1],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)
    
    # what are parameters fit from Block A' with initial Q values from end of Block A
    result3 = op.minimize(nll, [alpha_true, beta_true], args=(Q_initial, reward3, target3, trial3), bounds=[(0,1),(0,None)])
    #result3 = op.minimize(nll, [alpha_ml_block1, beta_ml_block1], args=([Qlow_block1[-1],Qhigh_block1[-1]], reward_freechoice_block3[:max_block3], target_freechoice_block3[:max_block3]), bounds=[(0,1),(0,None)])
    #result3 = op.minimize(nll, [alpha_ml_block1, beta_ml_block1], args=(Q_initial, reward_freechoice_block3[:max_block3], target_freechoice_block3[:max_block3]), bounds=[(0,1),(0,None)])
    alpha_ml_block3, beta_ml_block3 = result3["x"]
    Qlow_block3_refit, Qhigh_block3_refit, prob_low_block3_refit,log_likelihood3_refit = RLPerformance([alpha_ml_block3,beta_ml_block3],[Qlow_block1[-1],Qhigh_block1[-1]],reward3,target3, trial3)

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
    num_blocks100 = len(target_freechoice_block3)/70   # there are 70 free choice trials in every 100 trials
    totalprob_chooselow_blocks100 = []
    for i in range(0,num_blocks100):
        totalprob_chooselow_blocks100.append(np.sum(target_freechoice_block3[i*70:(i+1)*70]==np.ones(70))/float(70))

    # calculate accuracy of fit for Block A
    model1 = 0.33*(prob_low_block1 > 0.5) + 0.66*(np.less_equal(prob_low_block1, 0.5))  # scaling by 0.33 and 0.66 just for plotting purposes
    fit1 = np.equal(model1[:-1],(0.33*target_freechoice_block1))
    accuracy1 = float(np.sum(fit1))/model1.size
    # plot fig of prob_low_block1 versus choice
    """
    plt.figure()
    plt.subplot(121)
    plt.plot(range(1,target_freechoice_block1.size+1),target_freechoice_block1*0.33,'b',label="Data")
    plt.plot(range(1,prob_low_block1.size+1),model1,'r--',label="Model")
    plt.axis([1,target_freechoice_block1.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Target Choice')
    plt.title('Block A - %s - Accuracy %f' % (name[1:-4],accuracy1),fontsize=8)
    plt.legend()
    """
    
    # calculate accuracy of fit applied to block A'
    model3 = 0.33*(prob_low_block3[:-1] > 0.5) + 0.66*(np.less_equal(prob_low_block3[:-1], 0.5))  #discard last element since it predicts selection for trial after final trial
    target_freechoice_block3 = np.array(target_freechoice_block3)
    fit3 = np.equal(model3,(0.33*target_freechoice_block3))
    accuracy3 = float(np.sum(fit3))/model3.size
    # plot fig of prob_low_block3 versus choice
    plt.subplot(122)
    plt.plot(range(1,target_freechoice_block3.size+1),target_freechoice_block3*0.33,'b',label="Data")
    plt.plot(range(1,model3.size+1),model3,'r--',label="Model")
    plt.axis([1,model3.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Target Choice')
    plt.title("Block A' - %s - Accuracy %f" % (name[1:-4],accuracy3),fontsize=8)
    plt.legend()
    plt.savefig('Papa_RL_figs/RLfit_%s.svg' % name[1:-4])    # save this filetype for AI editing
    plt.savefig('Papa_RL_figs/RLfit_%s.png' % name[1:-4])    # save this filetype for easy viewing
    plt.close()

    # calculate acurracy of re-fit parameters for block A'
    model3_refit = 0.33*(prob_low_block3_refit[:-1] > 0.5) + 0.66*(np.less_equal(prob_low_block3_refit[:-1], 0.5))  #discard last element since it predicts selection for trial after final trial
    refit3 = np.equal(model3_refit,(0.33*target_freechoice_block3))
    accuracy3_refit = float(np.sum(refit3))/model3_refit.size
    plt.figure()
    plt.plot(range(1,target_freechoice_block3.size+1),target_freechoice_block3*0.33,'b',label="Data")
    plt.plot(range(1,model3_refit.size+1),model3_refit,'r--',label="Refit Model")
    plt.axis([1,model3_refit.size,0, 1])
    plt.xlabel('Trials')
    plt.ylabel('Target Choice')
    plt.title("Block A' - %s - Accuracy %f" % (name[1:-4],accuracy3_refit),fontsize=8)
    plt.legend()
    plt.savefig('Papa_RL_figs/RLfit_%s_refit.svg' % name[1:-4])    # save this filetype for AI editing
    plt.savefig('Papa_RL_figs/RLfit_%s_refit.png' % name[1:-4])    # save this filetype for easy viewing
    plt.close()

    

    # model for trials immediately following stim trial
    stim_trials = np.array(stim_trials) - 200
    max_trial_dist = np.amax(stim_trials[1:] - stim_trials[:-1])
    #max_trial_dsit = np.amax([max_trial_dist,300 - stim_trials[-1]])  # use when doing analysis for first 100 trials only in Block A'
    max_trial_dist = np.amax([max_trial_dist,num_successful_trials - stim_trials[-1] - 200])
    global_max_trial_dist = np.amax([global_max_trial_dist,max_trial_dist])
    error_vec = np.zeros(max_trial_dist,dtype=float) # placeholder vector counting number of errors
    error_vec_refit = np.zeros(max_trial_dist,dtype=float)
    after_stim_trials = np.zeros(max_trial_dist) # placeholder vector counting number of trials following stim trial
    choose_lowvalue = np.zeros(max_trial_dist,dtype=float)
    counter_error_mat = 0
    
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
        counter_error_mat += 1

    # Save model fit parameters and probabilities
    errors_after_stim = error_vec/after_stim_trials
    errors_after_stim_refit = error_vec_refit/after_stim_trials
    prob_choose_lowvalue = choose_lowvalue/after_stim_trials

    if name in hdf_list_stim:
        hdf_data_stim[name] = [alpha_ml_block1,beta_ml_block1]
        modelerror_afterstim_stim[name] = errors_after_stim
        modelerror_afterstim_stim_refit[name] = errors_after_stim_refit
        prob_lowvalue_afterstim_stim[name] = prob_choose_lowvalue
        prob_chooselow_stim[name] = totalprob_chooselow
        prob_chooselow_blocks100_stim[name] = totalprob_chooselow_blocks100
        modelerror_stim[name] = accuracy3
    else:
        hdf_data_sham[name] = [alpha_ml_block1,beta_ml_block1]
        modelerror_afterstim_sham[name] = errors_after_stim
        modelerror_afterstim_sham_refit[name] = errors_after_stim_refit
        prob_lowvalue_afterstim_sham[name] = prob_choose_lowvalue
        prob_chooselow_sham[name] = totalprob_chooselow
        modelerror_sham[name] = accuracy3

# compute average likelihood of model error and choosing low value target for trials after stim
global_max_trial_dist
mean_modelerror_afterstim_stim = []
std_modelerror_afterstim_stim = []
mean_modelerror_afterstim_stim_refit = []
std_modelerror_afterstim_stim_refit = []
mean_prob_lowvalue_afterstim_stim = []
std_prob_lowvalue_afterstim_stim = []

mean_modelerror_afterstim_sham = []
std_modelerror_afterstim_sham = []
mean_modelerror_afterstim_sham_refit = []
std_modelerror_afterstim_sham_refit = []
mean_prob_lowvalue_afterstim_sham = []
std_prob_lowvalue_afterstim_sham = []


for i in range(0,global_max_trial_dist):
    # for the ith trial after stim
    prob_error = []
    prob_error_refit = []
    prob_lowvalue = []
    for name in hdf_list_stim:
        modelerror = modelerror_afterstim_stim[name]
        modelerror_refit = modelerror_afterstim_stim_refit[name]
        lowvalue = prob_lowvalue_afterstim_stim[name]
        if i < modelerror.size:
            prob_error.append(modelerror[i])

            prob_lowvalue.append(lowvalue[i])
        if i < modelerror_refit.size:
            prob_error_refit.append(modelerror_refit[i])
    mean_prob_error = np.mean(prob_error)
    std_prob_error = np.std(prob_error)/np.sqrt(len(prob_error))
    mean_prob_error_refit = np.mean(prob_error_refit)
    std_prob_error_refit = np.std(prob_error_refit)/np.sqrt(len(prob_error_refit))
    mean_prob_lowvalue = np.mean(prob_lowvalue)
    std_prob_lowvalue = np.std(prob_lowvalue)/np.sqrt(len(prob_lowvalue))
    mean_modelerror_afterstim_stim.append(mean_prob_error)
    std_modelerror_afterstim_stim.append(std_prob_error)
    mean_modelerror_afterstim_stim_refit.append(mean_prob_error_refit)
    std_modelerror_afterstim_stim_refit.append(std_prob_error_refit)
    mean_prob_lowvalue_afterstim_stim.append(mean_prob_lowvalue)
    std_prob_lowvalue_afterstim_stim.append(std_prob_lowvalue)

    prob_error = []
    prob_error_refit = []
    prob_lowvalue = []
    for name in hdf_list_sham:
        modelerror = modelerror_afterstim_sham[name]
        lowvalue = prob_lowvalue_afterstim_sham[name]
        if i < modelerror.size:
            prob_error.append(modelerror[i])
            
            prob_lowvalue.append(lowvalue[i])
        if i < modelerror_refit.size:
            prob_error_refit.append(modelerror_refit[i])
    mean_prob_error = np.mean(prob_error)
    std_prob_error = np.std(prob_error)/float(np.sqrt(len(prob_error)))
    mean_prob_error_refit = np.mean(prob_error_refit)
    std_prob_error_refit = np.std(prob_error_refit)/np.sqrt(len(prob_error_refit))
    mean_prob_lowvalue = np.mean(prob_lowvalue)
    std_prob_lowvalue = np.std(prob_lowvalue)/float(np.sqrt(len(prob_lowvalue)))
    mean_modelerror_afterstim_sham.append(mean_prob_error)
    std_modelerror_afterstim_sham.append(std_prob_error)
    mean_modelerror_afterstim_sham_refit.append(mean_prob_error_refit)
    std_modelerror_afterstim_sham_refit.append(std_prob_error_refit)
    mean_prob_lowvalue_afterstim_sham.append(mean_prob_lowvalue)
    std_prob_lowvalue_afterstim_sham.append(std_prob_lowvalue)

# overall probability of choosing low value target in Block A'
prob_lowtarg_stim = []
prob_lowtarg_blocks100_stim = np.zeros(5)   # only care about first 500 trials in Block A'
prob_lowtarg_sham = []
prob_error_stim = []
prob_error_sham = []
prob_error_stim_refit = []
prob_error_sham_refit = []
count_blocks100 = np.zeros(5)   # only care about first 500 trials in block A'
for name in hdf_list_stim:
    prob_lowtarg_stim.append(prob_chooselow_stim[name])
    prob_error_stim.append(modelerror_stim[name])
    lowtarg_blocks100_stim = prob_chooselow_blocks100_stim[name]   # vector of variable length
    prob_lowtarg_blocks100_stim[:np.min([5,len(lowtarg_blocks100_stim)])] += lowtarg_blocks100_stim[:np.min([5,len(lowtarg_blocks100_stim)])]
    count_blocks100[:np.min([5,len(lowtarg_blocks100_stim)])] += np.ones(np.min([5,len(lowtarg_blocks100_stim)]))

for name in hdf_list_sham:
    prob_lowtarg_sham.append(prob_chooselow_sham[name])
    prob_error_sham.append(modelerror_sham[name])

avg_rate_lowtarg_stim = np.mean(prob_lowtarg_stim)
avg_rate_lowtarg_blocks100_stim = prob_lowtarg_blocks100_stim/count_blocks100   # should be a vector of length 5
avg_rate_lowtarg_sham = np.mean(prob_lowtarg_sham)
avg_rate_modelerror_stim = np.mean(prob_error_stim)
avg_rate_modelerror_sham = np.mean(prob_error_sham)


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
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim_alltrials.svg')
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim_alltrials.png')
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
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim_alltrials_refit.svg')
plt.savefig('Papa_RL_figs/modelerror_likelihood_afterstim_alltrials_refit.png')
plt.close()

plt.figure()
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),mean_prob_lowvalue_afterstim_stim,label='Stim',yerr=std_prob_lowvalue_afterstim_stim)
plt.errorbar(range(1,len(mean_prob_lowvalue_afterstim_sham)+1),mean_prob_lowvalue_afterstim_sham,label='Sham',yerr=std_prob_lowvalue_afterstim_sham)
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_stim*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'b',label='Stim overall avg')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_sham)+1),avg_rate_lowtarg_sham*np.ones(len(mean_prob_lowvalue_afterstim_sham)),'g',label='Sham overal avg')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_blocks100_stim[0]*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'m',label='Trials 0-100')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_blocks100_stim[1]*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'c',label='Trials 100-200')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_blocks100_stim[2]*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'k',label='Trials 200-300')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_blocks100_stim[3]*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'y',label='Trials 300-400')
plt.plot(range(1,len(mean_prob_lowvalue_afterstim_stim)+1),avg_rate_lowtarg_blocks100_stim[4]*np.ones(len(mean_prob_lowvalue_afterstim_stim)),'r',label='Trials 400-500')
plt.title('Low-Value Target Selection')
plt.xlabel('Trials post-IC trial with stimulation')
plt.ylabel('Probability of Selecting Low-Value Target')
plt.legend()
plt.savefig('Papa_RL_figs/prob_lowvalue_afterstim_alltrials.svg')
plt.savefig('Papa_RL_figs/prob_lowvalue_afterstim_alltrials.png')
plt.close()

"""
add code for: adding stim parameter
"""



