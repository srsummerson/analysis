import numpy as np
import matplotlib.pyplot as plt


def PeriStimulusFreeChoiceBehavior(stim_trial_ind, choice, ntrials):
    

    num_stim_trials = len(stim_trial_ind)
    aligned_lv_choices = np.zeros((num_stim_trials,ntrials))  # look at ntrials free-choice trials out of from stim trial
    aligned_lv_choices_rewarded = np.zeros((num_stim_trials,ntrials))   # look at ntrials free-choice trials out from stim trial when stim trial was rewarded
    aligned_lv_choices_unrewarded = np.zeros((num_stim_trials,ntrials))
    number_aligned_choices = np.zeros(ntrials)
    counter_stimtrials_used = 0

    for i in range(0,num_stim_trials-1):
        ind_stim = stim_trial_ind[i]
        max_ind_out = np.min([ntrials,stim_trial_ind[i+1]-stim_trial_ind[i]-1])
        if max_ind_out > 0:
            aligned_lv_choices[i,0:max_ind_out] = (2 - choice[ind_stim+1:ind_stim+max_ind_out+1])
            number_aligned_choices[0:max_ind_out] += np.ones(max_ind_out)
            counter_stimtrials_used += 1
        else:
            aligned_lv_choices[i,:] = np.zeros(ntrials)

    prob_choose_low_aligned = np.sum(aligned_lv_choices,axis=0)
    prob_choose_low_aligned = prob_choose_low_aligned/number_aligned_choices

    return prob_choose_low_aligned


def simulateRLPerformance_regular(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials):
    '''   
     This method simulates a Q-learning model with a multiplicative parameter used to model stimulation. 
     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.

     Inputs:
        - parameters: 2-tuple of parameter values 
        - Q_initial: 2-tuple of initial estimates for low and high value action
        - reward_probabilities: 2-tuple of the reward probabilities for each choice
        - stim_trial_probability: percentage of trials on which stimulation occurs
    '''
    Q_low = np.zeros(num_trials+1)
    Q_high = np.zeros(num_trials+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    rewardL = reward_probabilities[0]
    rewardH = reward_probabilities[1]
    num_rewarded_low = int(num_trials*rewardL)
    num_rewarded_high = int(num_trials*rewardH)

    reward_schedule_low_ind = np.random.choice(np.arange(num_trials), num_rewarded_low, replace=False)
    reward_schedule_low = np.zeros(num_trials)
    reward_schedule_low[reward_schedule_low_ind] = 1
    reward_schedule_high_ind = np.random.choice(np.arange(num_trials), num_rewarded_high, replace=False)
    reward_schedule_high = np.zeros(num_trials)
    reward_schedule_high[reward_schedule_high_ind] = 1

    alpha = parameters[0]
    beta = parameters[1]
    num_stim = int(num_trials*stim_trial_probability)
    num_freechoice = num_trials - num_stim
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    choice = np.zeros(num_trials)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 0  # start counter at one so that q-values and prob value are index-aligned for trials
    
    stim_trial_inds = np.random.choice(np.arange(num_trials), num_stim, replace=False)  # indices of stim trials
    stim_trial = np.zeros(num_trials)
    stim_trial[stim_trial_inds] = 1

    for i in range(0,num_trials):

        Q_low_dec = Q_low[i]
        if stim_trial[i]==0:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            #log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            choice[i] = 1*(prob_choice_low[counter] >= prob_choice_high[counter]) + 2*(prob_choice_low[counter] < prob_choice_high[counter])
            counter += 1
        else:
            choice[i] = 1

        delta_low = float(reward_schedule_low[i]) - Q_low_dec
        delta_high = float(reward_schedule_high[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        
    return stim_trial_inds, choice, Q_low, Q_high, reward_schedule_low, reward_schedule_high

def simulateRLPerformance_multiplicative_Qstimparameter(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials):
    '''   
     This method simulates a Q-learning model with a multiplicative parameter used to model stimulation. 
     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.

     Inputs:
        - parameters: 3-tuple of parameter values 
        - Q_initial: 2-tuple of initial estimates for low and high value action
        - reward_probabilities: 2-tuple of the reward probabilities for each choice
        - stim_trial_probability: percentage of trials on which stimulation occurs
    '''
    Q_low = np.zeros(num_trials+1)
    Q_high = np.zeros(num_trials+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    rewardL = reward_probabilities[0]
    rewardH = reward_probabilities[1]
    num_rewarded_low = int(num_trials*rewardL)
    num_rewarded_high = int(num_trials*rewardH)

    reward_schedule_low_ind = np.random.choice(np.arange(num_trials), num_rewarded_low, replace=False)
    reward_schedule_low = np.zeros(num_trials)
    reward_schedule_low[reward_schedule_low_ind] = 1
    reward_schedule_high_ind = np.random.choice(np.arange(num_trials), num_rewarded_high, replace=False)
    reward_schedule_high = np.zeros(num_trials)
    reward_schedule_high[reward_schedule_high_ind] = 1

    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    num_stim = int(num_trials*stim_trial_probability)
    num_freechoice = num_trials - num_stim
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    choice = np.zeros(num_trials)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 0  # start counter at one so that q-values and prob value are index-aligned for trials
    
    stim_trial_inds = np.random.choice(np.arange(num_trials), num_stim, replace=False)  # indices of stim trials
    stim_trial = np.zeros(num_trials)
    stim_trial[stim_trial_inds] = 1

    for i in range(0,num_trials):

        Q_low_dec = Q_low[i] + gamma*Q_low[i]*(stim_trial[i]==1)
        if stim_trial[i]==0:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            #log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            choice[i] = 1*(prob_choice_low[counter] >= prob_choice_high[counter]) + 2*(prob_choice_low[counter] < prob_choice_high[counter])
            counter += 1
        else:
            choice[i] = 1

        delta_low = float(reward_schedule_low[i]) - Q_low_dec
        delta_high = float(reward_schedule_high[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        
    return stim_trial_inds, choice, Q_low, Q_high, reward_schedule_low, reward_schedule_high

def simulateRLPerformance_multiplicative_Qstimparameter_HV(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials):
    '''   
     This method simulates a Q-learning model with a multiplicative parameter used to model stimulation. 
     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.

     Inputs:
        - parameters: 3-tuple of parameter values 
        - Q_initial: 2-tuple of initial estimates for low and high value action
        - reward_probabilities: 2-tuple of the reward probabilities for each choice
        - stim_trial_probability: percentage of trials on which stimulation occurs
    '''
    Q_low = np.zeros(num_trials+1)
    Q_high = np.zeros(num_trials+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    rewardL = reward_probabilities[0]
    rewardH = reward_probabilities[1]
    num_rewarded_low = int(num_trials*rewardL)
    num_rewarded_high = int(num_trials*rewardH)

    reward_schedule_low_ind = np.random.choice(np.arange(num_trials), num_rewarded_low, replace=False)
    reward_schedule_low = np.zeros(num_trials)
    reward_schedule_low[reward_schedule_low_ind] = 1
    reward_schedule_high_ind = np.random.choice(np.arange(num_trials), num_rewarded_high, replace=False)
    reward_schedule_high = np.zeros(num_trials)
    reward_schedule_high[reward_schedule_high_ind] = 1

    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    num_stim = int(num_trials*stim_trial_probability)
    num_freechoice = num_trials - num_stim
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    choice = np.zeros(num_trials)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 0  # start counter at one so that q-values and prob value are index-aligned for trials
    
    stim_trial_inds = np.random.choice(np.arange(num_trials), num_stim, replace=False)  # indices of stim trials
    stim_trial = np.zeros(num_trials)
    stim_trial[stim_trial_inds] = 1

    for i in range(0,num_trials):

        Q_high_dec = Q_high[i] + gamma*Q_high[i]*(stim_trial[i]==1)
        if stim_trial[i]==0:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            #log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            choice[i] = 1*(prob_choice_low[counter] >= prob_choice_high[counter]) + 2*(prob_choice_low[counter] < prob_choice_high[counter])
            counter += 1
        else:
            choice[i] = 1

        delta_low = float(reward_schedule_low[i]) - Q_low[i]
        delta_high = float(reward_schedule_high[i]) - Q_high_dec
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        
    return stim_trial_inds, choice, Q_low, Q_high, reward_schedule_low, reward_schedule_high


def simulate_RLPerformance_shamVstim(sham_parameters, stim_parameters):
    num_sims = 1000
    num_trials_per_sim = 100
    num_trials = 7
    
    sham_alpha = sham_parameters[0]
    sham_beta = sham_parameters[1]

    stim_alpha = stim_parameters[0]
    stim_beta = stim_parameters[1]
    stim_gamma = stim_parameters[2]

    stim_prob_low = np.zeros(len(stim_gamma))
    stim_prob_high = np.zeros(len(stim_gamma))
    #prob_next_low = np.zeros(len(stim_gamma))
    stim_prob_low_std = np.zeros(len(stim_gamma))
    stim_prob_high_std = np.zeros(len(stim_gamma))
    #prob_next_low_std = np.zeros(len(stim_gamma))
    stim_prob_choose_low_aligned = np.zeros([len(stim_gamma),num_trials])

    sham_prob_low = np.zeros(len(sham_alpha))
    sham_prob_high = np.zeros(len(sham_alpha))
    sham_prob_low_std = np.zeros(len(sham_alpha))
    sham_prob_high_std = np.zeros(len(sham_alpha))
    sham_prob_choose_low_aligned = np.zeros([len(sham_alpha), num_trials])

    Q_initial = [0.5,0.5]
    reward_probabilities = [0.4,0.8]
    stim_trial_probability = 0.3

    for ind in range(len(stim_gamma)):
        parameters = [stim_alpha[ind], stim_beta[ind], stim_gamma[ind]]
        prob_choose_low = np.zeros(num_sims)
        prob_choose_high = np.zeros(num_sims)
        prob_choose_next_low = np.zeros(num_sims)
        prob_choose_next_high = np.zeros(num_sims)
        prob_choose_low_aligned = np.zeros([num_sims, num_trials])

        for sim in range(num_sims):
            stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_multiplicative_Qstimparameter_HV(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
            
            regular_trial_inds = np.array([i for i in range(0,num_trials_per_sim) if i not in stim_trial_inds])
            stim_next_inds = [index+1 for index in stim_trial_inds if (index+1) in regular_trial_inds]

            prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
            prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

            prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
            prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))

            prob_choose_low_aligned[sim,:] = PeriStimulusFreeChoiceBehavior(stim_trial_inds, choice, num_trials)

        stim_prob_low[ind] = np.nanmean(prob_choose_low)
        stim_prob_high[ind] = np.nanmean(prob_choose_high)
        stim_prob_low_std[ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
        stim_prob_high_std[ind] = np.nanstd(prob_choose_high)/np.sqrt(num_sims - 1)
        stim_prob_choose_low_aligned[ind,:] = np.nanmean(prob_choose_low_aligned,axis = 0)

    for ind in range(len(sham_alpha)):
        parameters = [sham_alpha[ind], sham_beta[ind]]
        prob_choose_low = np.zeros(num_sims)
        prob_choose_high = np.zeros(num_sims)
        prob_choose_next_low = np.zeros(num_sims)
        prob_choose_next_high = np.zeros(num_sims)
        prob_choose_low_aligned = np.zeros([num_sims, num_trials])

        for sim in range(num_sims):
            stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_regular(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
            
            regular_trial_inds = np.array([i for i in range(0,num_trials_per_sim) if i not in stim_trial_inds])
            stim_next_inds = [index+1 for index in stim_trial_inds if (index+1) in regular_trial_inds]

            prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
            prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

            prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
            prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))

            prob_choose_low_aligned[sim,:] = PeriStimulusFreeChoiceBehavior(stim_trial_inds, choice, num_trials)

        sham_prob_low[ind] = np.nanmean(prob_choose_low)
        sham_prob_high[ind] = np.nanmean(prob_choose_high)
        sham_prob_low_std[ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
        sham_prob_high_std[ind] = np.nanstd(prob_choose_high)/np.sqrt(num_sims - 1)
        sham_prob_choose_low_aligned[ind,:] = np.nanmean(prob_choose_low_aligned, axis = 0)

    stim_prob_low_avg = np.nanmean(stim_prob_low)
    stim_prob_low_sem = np.nanstd(stim_prob_low)/np.sqrt(len(stim_prob_low) - 1)
    sham_prob_low_avg = np.nanmean(sham_prob_low)
    sham_prob_low_sem = np.nanstd(sham_prob_low)/np.sqrt(len(sham_prob_low) - 1)

    stim_prob_low_aligned_avg = np.nanmean(stim_prob_choose_low_aligned, axis = 0)
    stim_prob_low_aligned_sem = np.nanstd(stim_prob_choose_low_aligned, axis = 0)/np.sqrt(len(stim_gamma)-1)

    sham_prob_low_aligned_avg = np.nanmean(sham_prob_choose_low_aligned, axis = 0)
    sham_prob_low_aligned_sem = np.nanstd(sham_prob_choose_low_aligned, axis = 0)/np.sqrt(len(sham_alpha)-1)

    width = float(0.35)
    days = np.arange(2)
    plt.figure()
    plt.bar(days[0], stim_prob_low_avg, width, color='g', yerr=stim_prob_low_sem/2., ecolor='k', label='stim')
    plt.bar(days[1], sham_prob_low_avg, width, color='c', yerr=sham_prob_low_sem/2., ecolor='k', label='sham')
    plt.ylim((0,0.2))
    plt.ylabel('Prob LV Choice')
    plt.show()

    trial_ind = np.arange(0,num_trials)
    # Get linear fit to total prob
    m_stim,b_stim = np.polyfit(trial_ind, stim_prob_low_aligned_avg, 1)
    m_sham,b_sham = np.polyfit(trial_ind, sham_prob_low_aligned_avg, 1)

    plt.figure()
    plt.bar(trial_ind, stim_prob_low_aligned_avg, width/2, color = 'c', yerr = stim_prob_low_aligned_sem/2)
    plt.plot(trial_ind,(m_stim*trial_ind + b_stim),'c--')
    plt.bar(trial_ind + width/2, sham_prob_low_aligned_avg, width/2, color = 'm', yerr = sham_prob_low_aligned_sem/2)
    plt.plot(trial_ind+width/2, (m_sham*trial_ind + b_sham),'m--')
    plt.ylabel('P(Choose LV Target)')
    plt.title('Target Selection')
    xticklabels = [str(num + 1) for num in trial_ind]
    plt.xticks(trial_ind + width/2, xticklabels)
    plt.xlabel('Trials post-stimulation')
    #plt.ylim([0.0,0.32])
    plt.xlim([-0.1,num_trials + 0.4])
    plt.legend()
    
    plt.show()


    return stim_prob_low, sham_prob_low, stim_prob_low_aligned_avg, sham_prob_low_aligned_avg

def simulate_RLPerformance_shamVstimHV(sham_parameters, stim_parameters):
    num_sims = 1000
    num_trials_per_sim = 100
    num_trials = 7
    
    sham_alpha = sham_parameters[0]
    sham_beta = sham_parameters[1]

    stim_alpha = stim_parameters[0]
    stim_beta = stim_parameters[1]
    stim_gamma = stim_parameters[2]

    stim_prob_low = np.zeros(len(stim_gamma))
    stim_prob_high = np.zeros(len(stim_gamma))
    #prob_next_low = np.zeros(len(stim_gamma))
    stim_prob_low_std = np.zeros(len(stim_gamma))
    stim_prob_high_std = np.zeros(len(stim_gamma))
    #prob_next_low_std = np.zeros(len(stim_gamma))
    stim_prob_choose_low_aligned = np.zeros([len(stim_gamma),num_trials])

    sham_prob_low = np.zeros(len(sham_alpha))
    sham_prob_high = np.zeros(len(sham_alpha))
    sham_prob_low_std = np.zeros(len(sham_alpha))
    sham_prob_high_std = np.zeros(len(sham_alpha))
    sham_prob_choose_low_aligned = np.zeros([len(sham_alpha),num_trials])


    Q_initial = [0.5,0.5]
    reward_probabilities = [0.4,0.8]
    stim_trial_probability = 0.3

    for ind in range(len(stim_gamma)):
        parameters = [stim_alpha[ind], stim_beta[ind], stim_gamma[ind]]
        prob_choose_low = np.zeros(num_sims)
        prob_choose_high = np.zeros(num_sims)
        prob_choose_next_low = np.zeros(num_sims)
        prob_choose_next_high = np.zeros(num_sims)
        prob_choose_low_aligned = np.zeros([num_sims,num_trials])

        for sim in range(num_sims):
            stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_multiplicative_Qstimparameter_HV(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
            
            regular_trial_inds = np.array([i for i in range(0,num_trials_per_sim) if i not in stim_trial_inds])
            stim_next_inds = [index+1 for index in stim_trial_inds if (index+1) in regular_trial_inds]

            prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
            prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

            prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
            prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))

            prob_choose_low_aligned[sim,:] = PeriStimulusFreeChoiceBehavior(stim_trial_inds, choice, num_trials)

        stim_prob_low[ind] = np.nanmean(prob_choose_low)
        stim_prob_high[ind] = np.nanmean(prob_choose_high)
        stim_prob_low_std[ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
        stim_prob_high_std[ind] = np.nanstd(prob_choose_high)/np.sqrt(num_sims - 1)
        stim_prob_choose_low_aligned[ind,:] = np.nanmean(prob_choose_low_aligned,axis = 0)


    for ind in range(len(sham_alpha)):
        parameters = [sham_alpha[ind], sham_beta[ind]]
        prob_choose_low = np.zeros(num_sims)
        prob_choose_high = np.zeros(num_sims)
        prob_choose_next_low = np.zeros(num_sims)
        prob_choose_next_high = np.zeros(num_sims)
        prob_choose_low_aligned = np.zeros([num_sims,num_trials])

        for sim in range(num_sims):
            stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_regular(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
            
            regular_trial_inds = np.array([i for i in range(0,num_trials_per_sim) if i not in stim_trial_inds])
            stim_next_inds = [index+1 for index in stim_trial_inds if (index+1) in regular_trial_inds]

            prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
            prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

            prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
            prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))

            prob_choose_low_aligned[sim,:] = PeriStimulusFreeChoiceBehavior(stim_trial_inds, choice, num_trials)


        sham_prob_low[ind] = np.nanmean(prob_choose_low)
        sham_prob_high[ind] = np.nanmean(prob_choose_high)
        sham_prob_low_std[ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
        sham_prob_high_std[ind] = np.nanstd(prob_choose_high)/np.sqrt(num_sims - 1)
        sham_prob_choose_low_aligned[ind,:] = np.nanmean(prob_choose_low_aligned,axis = 0)

    stim_prob_low_avg = np.nanmean(stim_prob_low)
    stim_prob_low_sem = np.nanstd(stim_prob_low)/np.sqrt(len(stim_prob_low) - 1)
    sham_prob_low_avg = np.nanmean(sham_prob_low)
    sham_prob_low_sem = np.nanstd(sham_prob_low)/np.sqrt(len(sham_prob_low) - 1)

    stim_prob_low_aligned_avg = np.nanmean(stim_prob_choose_low_aligned, axis = 0)
    stim_prob_low_aligned_sem = np.nanstd(stim_prob_choose_low_aligned, axis = 0)/np.sqrt(len(stim_gamma)-1)

    sham_prob_low_aligned_avg = np.nanmean(sham_prob_choose_low_aligned, axis = 0)
    sham_prob_low_aligned_sem = np.nanstd(sham_prob_choose_low_aligned, axis = 0)/np.sqrt(len(sham_alpha)-1)

    width = float(0.35)
    days = np.arange(2)
    plt.figure()
    plt.bar(days[0], stim_prob_low_avg, width, color='y', yerr=stim_prob_low_sem/2., ecolor='k', label='HV stim')
    plt.bar(days[1], sham_prob_low_avg, width, color='c', yerr=sham_prob_low_sem/2., ecolor='k', label='sham')
    plt.ylim((0,0.2))
    plt.ylabel('Prob LV Choice')
    plt.show()

    trial_ind = np.arange(0,num_trials)
    # Get linear fit to total prob
    m_stim,b_stim = np.polyfit(trial_ind, stim_prob_low_aligned_avg, 1)
    m_sham,b_sham = np.polyfit(trial_ind, sham_prob_low_aligned_avg, 1)

    plt.figure()
    plt.bar(trial_ind, stim_prob_low_aligned_avg, width/2, color = 'y', yerr = stim_prob_low_aligned_sem/2)
    plt.plot(trial_ind,(m_stim*trial_ind + b_stim),'y--')
    plt.bar(trial_ind + width/2, sham_prob_low_aligned_avg, width/2, color = 'm', yerr = sham_prob_low_aligned_sem/2)
    plt.plot(trial_ind+width/2, (m_sham*trial_ind + b_sham),'m--')
    plt.ylabel('P(Choose LV Target)')
    plt.title('Target Selection')
    xticklabels = [str(num + 1) for num in trial_ind]
    plt.xticks(trial_ind + width/2, xticklabels)
    plt.xlabel('Trials post-stimulation')
    #plt.ylim([0.0,0.32])
    plt.xlim([-0.1,num_trials + 0.4])
    plt.legend()
    
    plt.show()

    return stim_prob_low, sham_prob_low

def simulate_overparameters_Qmultiplicative():
    num_sims = 10
    num_trials_per_sim = 100

    stim_alpha = np.arange(0.01,1,0.01)
    stim_beta = 10
    stim_gamma = np.arange(0.1,10,0.2)

    stim_prob_low = np.zeros([len(stim_alpha),len(stim_gamma)])
    stim_prob_low_std = np.zeros([len(stim_alpha),len(stim_gamma)])

    Q_initial = [0.5,0.5]
    reward_probabilities = [0.4,0.8]
    stim_trial_probability = 0.3

    for ind in range(len(stim_gamma)):
        print ind
        for kind in range(len(stim_alpha)):
            parameters = [stim_alpha[kind], stim_beta, stim_gamma[ind]]
            prob_choose_low = np.zeros(num_sims)
            prob_choose_high = np.zeros(num_sims)
            prob_choose_next_low = np.zeros(num_sims)
            prob_choose_next_high = np.zeros(num_sims)

            for sim in range(num_sims):
                stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_multiplicative_Qstimparameter(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
                
                regular_trial_inds = np.array([i for i in range(0,num_trials_per_sim) if i not in stim_trial_inds])
                stim_next_inds = [index+1 for index in stim_trial_inds if (index+1) in regular_trial_inds]

                prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
                prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

                prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
                prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))

            stim_prob_low[kind,ind] = np.nanmean(prob_choose_low)
            stim_prob_low_std[kind,ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
            
    fig = plt.figure()
    ax = plt.imshow(stim_prob_low,aspect='auto', origin='lower')
    yticks = np.arange(0, len(stim_alpha), 5)
    yticklabels = ['{0:.2f}'.format(stim_alpha[j]) for j in yticks]
    xticks = np.arange(0, len(stim_gamma), 5)
    xticklabels = ['{0:.2f}'.format(stim_gamma[j]) for j in xticks]
    plt.yticks(yticks, yticklabels)
    plt.xticks(xticks, xticklabels)
    plt.ylabel('alpha')
    plt.xlabel('gamma')
    plt.title('Prob LV Choice')
    fig.colorbar(ax)
        
    plt.show()

    return stim_prob_low

def simulate_overparameters_Qmultiplicative_HV():
    num_sims = 10
    num_trials_per_sim = 100

    stim_alpha = np.arange(0.01,1,0.01)
    stim_beta = 3
    stim_gamma = np.arange(0.1,10,0.2)

    stim_prob_low = np.zeros([len(stim_alpha),len(stim_gamma)])
    stim_prob_low_std = np.zeros([len(stim_alpha),len(stim_gamma)])

    Q_initial = [0.5,0.5]
    reward_probabilities = [0.4,0.8]
    stim_trial_probability = 0.3

    for ind in range(len(stim_gamma)):
        print ind
        for kind in range(len(stim_alpha)):
            parameters = [stim_alpha[kind], stim_gamma[ind], stim_beta]
            prob_choose_low = np.zeros(num_sims)
            prob_choose_high = np.zeros(num_sims)
            prob_choose_next_low = np.zeros(num_sims)
            prob_choose_next_high = np.zeros(num_sims)

            for sim in range(num_sims):
                stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_multiplicative_Qstimparameter_HV(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
                
                regular_trial_inds = np.array([i for i in range(0,num_trials_per_sim) if i not in stim_trial_inds])
                stim_next_inds = [index+1 for index in stim_trial_inds if (index+1) in regular_trial_inds]

                prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
                prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

                prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
                prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))

            stim_prob_low[kind,ind] = np.nanmean(prob_choose_low)
            stim_prob_low_std[kind,ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
            
    fig = plt.figure()
    ax = plt.imshow(stim_prob_low,aspect='auto', origin='lower')
    yticks = np.arange(0, len(stim_alpha), 5)
    yticklabels = ['{0:.2f}'.format(stim_alpha[j]) for j in yticks]
    xticks = np.arange(0, len(stim_gamma), 5)
    xticklabels = ['{0:.2f}'.format(stim_gamma[j]) for j in xticks]
    plt.yticks(yticks, yticklabels)
    plt.xticks(xticks, xticklabels)
    plt.ylabel('alpha')
    plt.xlabel('gamma')
    plt.title('Prob LV Choice w/ HV Stim')
    fig.colorbar(ax)
        
    plt.show()

    return stim_prob_low

