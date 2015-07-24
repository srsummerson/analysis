import numpy as np
import random
import math
import scipy.optimize as op

def logLikelihoodRLPerformance(parameters, Q_initial, reward_schedule,choice):
    Q_low = np.zeros(choice.size+1)
    Q_high = np.zeros(choice.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    prob_choice_low = np.zeros(choice.size)
    prob_choice_high = np.zeros(choice.size)
    log_prob_total = 0.0
    for i in range(0,choice.size):
        Q_low[i+1] += alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
        Q_high[i+1] += alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        #prob_choice_low[i] = np.exp(beta*Q_low[i+1])/(np.exp(beta*Q_low[i+1]) + np.exp(beta*Q_high[i+1]))
        prob_choice_low[i] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
        prob_choice_high[i] = float(1) - prob_choice_low[i]
        log_prob_total = float(log_prob_total) + np.log(prob_choice_low[i]*(choice[i]==1) + prob_choice_high[i]*(choice[i]==2))
    return log_prob_total

def RLPerformance(parameters, Q_initial, reward_schedule,choice):
    Q_low = np.zeros(choice.size+1)
    Q_high = np.zeros(choice.size+1)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]
    alpha = parameters[0]
    beta = parameters[1]
    prob_choice_low = np.zeros(choice.size)
    prob_choice_high = np.zeros(choice.size)
    log_prob_total = 0.0
    for i in range(0,choice.size):
        Q_low[i+1] += alpha*(choice[i]==1)*(float(reward_schedule[i]) - Q_low[i])
        Q_high[i+1] += alpha*(choice[i]==2)*(float(reward_schedule[i]) - Q_high[i])
        #prob_choice_low[i] = np.exp(beta*Q_low[i+1])/(np.exp(beta*Q_low[i+1]) + np.exp(beta*Q_high[i+1]))
        prob_choice_low[i] = float(1)/(1 + np.exp(beta*(Q_high[i] - Q_low[i])))
        prob_choice_high[i] = float(1) - prob_choice_low[i]
        log_prob_total = float(log_prob_total) + np.log(prob_choice_low[i]*(choice[i]==1) + prob_choice_high[i]*(choice[i]==2))
    return Q_low, Q_high, prob_choice_low