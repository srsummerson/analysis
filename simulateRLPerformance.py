import numpy as np
import matplotlib.pyplot as plt

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


num_sims = 1000
num_trials_per_sim = 100
alpha = 0.5
beta = 3
gamma = [0.2, 0.4,0.6, 0.8]
prob_low = np.zeros(len(gamma))
prob_high = np.zeros(len(gamma))
prob_next_low = np.zeros(len(gamma))
prob_low_std = np.zeros(len(gamma))
prob_high_std = np.zeros(len(gamma))
prob_next_low_std = np.zeros(len(gamma))
Q_initial = [0.5,0.5]
reward_probabilities = [0.4,0.8]
stim_trial_probability = 0.3

for gamma_ind in range(len(gamma)):
    parameters = [alpha, beta, gamma[gamma_ind]]
    prob_choose_low = np.zeros(num_sims)
    prob_choose_high = np.zeros(num_sims)
    prob_choose_next_low = np.zeros(num_sims)
    prob_choose_next_high = np.zeros(num_sims)

    for sim in range(num_sims):
        stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_multiplicative_Qstimparameter(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
        
        regular_trial_inds = np.array([ind for ind in range(0,num_trials_per_sim) if ind not in stim_trial_inds])
        stim_next_inds = [ind+1 for ind in stim_trial_inds if (ind+1) in regular_trial_inds]

        prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
        prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

        prob_choose_next_low[sim] = np.sum(choice[stim_next_inds]==1)/float(len(stim_next_inds))
        prob_choose_next_high[sim] = np.sum(choice[stim_next_inds]==2)/float(len(stim_next_inds))


    prob_low[gamma_ind] = np.nanmean(prob_choose_low)
    prob_high[gamma_ind] = np.nanmean(prob_choose_high)
    prob_low_std[gamma_ind] = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
    prob_high_std[gamma_ind] = np.nanstd(prob_choose_high)/np.sqrt(num_sims - 1)

    prob_next_low[gamma_ind] = np.nanmean(prob_choose_next_low)
    prob_next_low_std[gamma_ind] = np.std(prob_choose_next_low)/np.sqrt(num_sims - 1)

plt.figure()
plt.errorbar(gamma, prob_low,yerr=prob_low_std/2.,color='r',ecolor='r')
plt.errorbar(gamma, prob_next_low,yerr=prob_next_low_std/2.,color='m',ecolor='m')
plt.xlabel('gamma')
plt.ylabel('Prob Choose LV')
plt.show()


alpha = 0.1
beta = 6
Q_initial = [0.5,0.5]
reward_probabilities = [0.4,0.8]
stim_trial_probability = 0.3

parameters = [alpha, beta]
prob_choose_low = np.zeros(num_sims)
prob_choose_high = np.zeros(num_sims)

for sim in range(num_sims):
    stim_trial_inds, choice, Q_low, Q_high, reward_low, reward_high = simulateRLPerformance_regular(parameters, Q_initial, reward_probabilities, stim_trial_probability, num_trials_per_sim)
    
    regular_trial_inds = np.array([ind for ind in range(0,num_trials_per_sim) if ind not in stim_trial_inds])

    prob_choose_low[sim] = np.sum(choice[regular_trial_inds]==1)/float(len(regular_trial_inds))
    prob_choose_high[sim] = np.sum(choice[regular_trial_inds]==2)/float(len(regular_trial_inds))

prob_low_sham = np.nanmean(prob_choose_low)
prob_high_sham = np.nanmean(prob_choose_high)
prob_low_std_sham = np.nanstd(prob_choose_low)/np.sqrt(num_sims - 1)
prob_high_std_sham = np.nanstd(prob_choose_high)/np.sqrt(num_sims - 1)