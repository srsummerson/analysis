import numpy as np
import random
import math
import scipy.optimize as op

def logLikelihoodRLPerformance(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha and beta. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), and the trial type (instructed_or_freechoice). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Q[t] is the value at the beginning of trial t, reward_schedule[t] is the reward at the end of trial t, 
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        
    return log_prob_total

def logLikelihoodRLPerformance_random(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice):
    '''   
     This function computes the log likelihood of purely random choices (i.e. probability of choosing each target is equal and independent of value function) 
     for a given set of RL Q-learning parameters: alpha and beta. This assumes a softmax decision policy. The inputs to the function include the parameters 
     (parameters), the initial Q values (Q_initial), whether a trial was rewarded or not (reward_schedule), the selected target on a trial (choice), and the 
     trial type (instructed_or_freechoice). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        #if i==0:
        #    Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
        #    Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        #else:
        #    Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1])
        #    Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = 0.5
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
    return log_prob_total


def RLPerformance(parameters, Q_initial, reward_schedule,choice, instructed_or_freechoice):
    '''   
     This function computes the RK model fit of the data for a given set of RL Q-learning parameters: alpha and beta. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), and the trial type (instructed_or_freechoice). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 
     '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
    return Q_low, Q_high, prob_choice_low, log_prob_total

def logLikelihoodRLPerformance_additive_Qstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):

        Q_low[i] = Q_low[i] + gamma*(stim_trial[i]==1)
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)

        '''
        Q_low[i+1] = Q_low[i] + gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]) 
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        if instructed_or_freechoice[i]==2:
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]  - gamma*(stim_trial[i]==1))))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        '''
        
    return log_prob_total

def RLPerformance_additive_Qstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        
        Q_low[i] = Q_low[i] + gamma*(stim_trial[i]==1)
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)

        '''
        Q_low[i+1] = Q_low[i] + gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]) 
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        if instructed_or_freechoice[i]==2:
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]  - gamma*(stim_trial[i]==1))))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        '''
    return Q_low, Q_high, prob_choice_low, log_prob_total

def logLikelihoodRLPerformance_multiplicative_Qstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        """
        Q_low_dec = Q_low[i] + gamma*Q_low[i]*(stim_trial[i]==1)
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low_dec
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i] - Q_low[i]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i] + Q_low[i]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i] - Q_low[i]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i]*gamma*(stim_trial[i]==1) + Q_low[i]*(stim_trial[i]==0) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            #Q_low[i] = Q_low[i-1] + Q_low[i-1]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1))
            Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i-1]*gamma*(stim_trial[i]==1) + Q_low[i-1]*(stim_trial[i]==0) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1])
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i] + Q_low[i]*gamma*(stim_trial[i]==1))))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            if i==0:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            else:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1) - Q_low[i-1]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        
    return log_prob_total

def RLPerformance_multiplicative_Qstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        """
        Q_low_dec = Q_low[i] + gamma*Q_low[i]*(stim_trial[i]==1)
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low_dec
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)

        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i] - Q_low[i]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i] + Q_low[i]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i] - Q_low[i]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i]*gamma*(stim_trial[i]==1) + Q_low[i]*(stim_trial[i]==0) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            #Q_low[i] = Q_low[i-1] + Q_low[i-1]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1))
            Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i-1]*gamma*(stim_trial[i]==1) + Q_low[i-1]*(stim_trial[i]==0) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1])
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i] + Q_low[i]*gamma*(stim_trial[i]==1))))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            if i==0:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            else:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1) - Q_low[i-1]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        
    return Q_low, Q_high, prob_choice_low, log_prob_total

def RLPerformance_multiplicative_Qstimparameter_withQstimOutput(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_low_adjusted = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        if i==0:
            Q_low_adjusted[i] = Q_low[i]*gamma*(stim_trial[i]==1) + Q_low[i]
            Q_low[i] = Q_low[i]*gamma*(stim_trial[i]==1) + Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i] - Q_low[i]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]*gamma)*(stim_trial[i]==1) + (Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]))*(stim_trial[i]==0)
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low_adjusted[i-1] = Q_low[i-1]*gamma*(stim_trial[i]==1) + Q_low[i-1]
            Q_low[i] = Q_low[i-1]*gamma*(stim_trial[i]==1) + Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1))
            #Q_low[i] = Q_low[i-1]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]*gamma)*(stim_trial[i]==1) + (Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]))*(stim_trial[i]==0)
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            if i==0:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            else:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1) - Q_low[i-1]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
    return Q_low, Q_high, prob_choice_low, log_prob_total, Q_low_adjusted


def logLikelihoodRLPerformance_additive_Pstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the P-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + gamma*(stim_trial[i]==1)
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]) 
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]) 
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + gamma*(stim_trial[i]==1)
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        """
    return log_prob_total

def RLPerformance_additive_Pstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the P-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):

        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + gamma*(stim_trial[i]==1)
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)

        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]) 
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]) 
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + gamma*(stim_trial[i]==1)
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        """
    return Q_low, Q_high, prob_choice_low, log_prob_total

def logLikelihoodRLPerformance_multiplicative_Pstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     multiplicative parameter to the P-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):

        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = (stim_trial[i]==0)*1./(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + gamma*(stim_trial[i]==1)*1./(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)

        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]) 
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]) 
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = gamma*(stim_trial[i]==1)*float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + (stim_trial[i]==0)*float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) 
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        """
    return log_prob_total

def RLPerformance_multiplicative_Pstimparameter(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     multiplicative parameter to the P-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):

        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = (stim_trial[i]==0)*1./(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + gamma*(stim_trial[i]==1)*1./(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)
        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]) 
            Q_high[i] = Q_high[i] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low[i] = Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]) 
            Q_high[i] = Q_high[i-1] + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = gamma*(stim_trial[i]==1)*float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) + (stim_trial[i]==0)*float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]))) 
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        """
    return Q_low, Q_high, prob_choice_low, log_prob_total

def logLikelihoodRLPerformance_multiplicative_Qstimparameter_HVTarget(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        """
        Q_high_dec = Q_high[i] + gamma*Q_high[i]*(stim_trial[i]==1)
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high_dec
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)


        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
            #Q_low[i] = Q_low[i]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]*gamma)*(stim_trial[i]==1) + (Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]))*(stim_trial[i]==0)
            Q_high[i] = Q_high[i]*gamma*(stim_trial[i]==1) + Q_high[i]*(stim_trial[i]==0) + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low[i] = Q_low[i-1] +  alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1])
            #Q_low[i] = Q_low[i-1]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]*gamma)*(stim_trial[i]==1) + (Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]))*(stim_trial[i]==0)
            Q_high[i] = Q_high[i-1]*gamma*(stim_trial[i]==1) + Q_high[i-1]*(stim_trial[i]==0) + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i]*gamma*(stim_trial[i]==1) + Q_high[i]*(stim_trial[i]==0) - Q_low[i])))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            if i==0:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            else:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1) - Q_low[i-1]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        
    return log_prob_total

def RLPerformance_multiplicative_Qstimparameter_HVTarget(parameters, Q_initial, reward_schedule,choice,instructed_or_freechoice,stim_trial):
    '''   
     This function computes the log likelihood of the data for a given set of RL Q-learning parameters: alpha, beta, and gamma. This assumes a softmax 
     decision policy. The inputs to the function include the parameters (parameters), the initial Q values (Q_initial), whether a trial was
     rewarded or not (reward_schedule), the selected target on a trial (choice), the trial type (instructed_or_freechoice) and whether stimulation
     was administered (stim_trial). 

     The Q-values here update for all trials, whereas the probability of selecting the low-value and high-value targets only updates for the free-choice
     trials. 

     Gamma is the parameter capturing the effect of stimulation on the decision-policy. In this model, it is considered that stimulation is an 
     additive parameter to the Q-value function.
    '''
    num_freechoice = np.sum(instructed_or_freechoice) - instructed_or_freechoice.size
    Q_low = np.zeros(reward_schedule.size+1)
    Q_high = np.zeros(reward_schedule.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    gamma = parameters[2]   # parameter for 
    prob_choice_low = np.zeros(num_freechoice+1)    # calculate up to what would be the following trial after the last trial
    prob_choice_high = np.zeros(num_freechoice+1)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5
    log_prob_total = 0.0
    counter = 1  # start counter at one so that q-values and prob value are index-aligned for trials
    
    for i in range(0,choice.size):
        """
        Q_high_dec = Q_high[i] + gamma*Q_high[i]*(stim_trial[i]==1)
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1

        delta_low = float(reward_schedule[i]) - Q_low[i]
        delta_high = float(reward_schedule[i]) - Q_high_dec
        Q_low[i+1] = Q_low[i] + alpha*(choice[i]==1)*(delta_low)
        Q_high[i+1] = Q_high[i] + alpha*(choice[i]==2)*(delta_high)

        """
        if i==0:
            Q_low[i] = Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
            #Q_low[i] = Q_low[i]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]*gamma)*(stim_trial[i]==1) + (Q_low[i] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i]))*(stim_trial[i]==0)
            Q_high[i] = Q_high[i]*gamma*(stim_trial[i]==1) + Q_high[i]*(stim_trial[i]==0) + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        else:
            Q_low[i] = Q_low[i-1] +  alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1])
            #Q_low[i] = Q_low[i-1]*gamma*(stim_trial[i]==1) + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]*gamma)*(stim_trial[i]==1) + (Q_low[i-1] + alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i-1]))*(stim_trial[i]==0)
            Q_high[i] = Q_high[i-1]*gamma*(stim_trial[i]==1) + Q_high[i-1]*(stim_trial[i]==0) + alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i-1])
        if instructed_or_freechoice[i]==2:
            prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i]*gamma*(stim_trial[i]==1) + Q_high[i]*(stim_trial[i]==0) - Q_low[i])))
            #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
            prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            if i==0:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i]*gamma*(stim_trial[i]==1) - Q_low[i]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            else:
                prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i-1] - Q_low[i-1]*gamma*(stim_trial[i]==1) - Q_low[i-1]*(stim_trial[i]==0))))
                #prob_choice_low[counter] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
                prob_choice_high[counter] = float(1) - prob_choice_low[counter]
            '''
            log_prob_total = float(log_prob_total) + np.log(prob_choice_low[counter]*(choice[i]==1) + prob_choice_high[counter]*(choice[i]==2))
            counter += 1
        
    return Q_low, Q_high, prob_choice_low, log_prob_total