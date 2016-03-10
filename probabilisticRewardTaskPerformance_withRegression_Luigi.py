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
from logLikelihoodRLPerformance import RLPerformance, logLikelihoodRLPerformance
from probabilisticRewardTaskPerformance import FreeChoicePilotTask_Behavior


hdf_list_stim = ['\luig20160204_15_te1382.hdf','\luig20160208_07_te1401.hdf','\luig20160212_08_te1429.hdf','\luig20160217_06_te1451.hdf',
                '\luig20160229_11_te1565.hdf','\luig20160301_07_te1572.hdf','\luig20160301_09_te1574.hdf']
hdf_list_sham = ['\luig20160213_05_te1434.hdf','\luig20160219_04_te1473.hdf','\luig20160221_05_te1478.hdf']
#hdf_list_hv = ['\luig20160218_10_te1469.hdf','\luig20160223_09_te1506.hdf','\luig20160223_11_te1508.hdf','\luig20160224_11_te1519.hdf',
#                '\luig20160224_15_te1523.hdf']
hdf_list_hv = ['\luig20160218_10_te1469.hdf','\luig20160223_09_te1506.hdf','\luig20160223_11_te1508.hdf','\luig20160224_11_te1519.hdf', '\luig20160224_15_te1523.hdf']


hdf_list = hdf_list_sham

hdf_prefix = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Luigi\hdf'


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
            prev_stim1_block3.append(2*stim_trials[i - 1] - 1)  # = 1 if stim was delivered and = -1 if stim was not delivered
            prev_stim2_block3.append(2*stim_trials[i - 2] - 1)
            prev_stim3_block3.append(2*stim_trials[i - 3] - 1)
            prev_stim4_block3.append(2*stim_trials[i - 4] - 1)
            prev_stim5_block3.append(2*stim_trials[i - 5] - 1)


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

    '''
    # check the accuracy on the training set
    print 'Model accuracy for Block1:',model_block1.score(X_block1, y_block1)
    print 'Null accuracy rate for Block1:',np.max([y_block1.mean(),1 - y_block1.mean()])

    print 'Model accuracy for Block3:',model_block3.score(X_block3, y_block3)
    print 'Null accuracy rate for Block3:',np.max([y_block3.mean(),1 - y_block3.mean()])
    '''

    # examine the coefficients
    print pd.DataFrame(zip(X_block1.columns, np.transpose(model_block1.coef_)))
    print pd.DataFrame(zip(X_block3.columns, np.transpose(model_block3.coef_)))
    
    #return fit_glm_block1, fit_glm_block3
    return model_block1, model_block3, predicted_block1, predicted_block3

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

    """
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

    '''
    # check the accuracy on the training set
    print 'Model accuracy for Block1:',model_block1.score(X_block1, y_block1)
    print 'Null accuracy rate for Block1:',np.max([y_block1.mean(),1 - y_block1.mean()])

    print 'Model accuracy for Block3:',model_block3.score(X_block3, y_block3)
    print 'Null accuracy rate for Block3:',np.max([y_block3.mean(),1 - y_block3.mean()])
    '''

    # examine the coefficients
    print pd.DataFrame(zip(X_block1.columns, np.transpose(model_block1.coef_)))
    print pd.DataFrame(zip(X_block3.columns, np.transpose(model_block3.coef_)))

    
    #return fit_glm_block1, fit_glm_block3
    return model_block1, model_block3, predicted_block1, predicted_block3

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
    '''
    len_block3 = len(reward_block3)
    first_100 = np.min([100,len_block3])
    reward_block3 = reward_block3[0:first_100]
    target_block3 = target_block3[0:first_100]
    trial_block3 = trial_block3[0:first_100]
    stim_trials_block = stim_trials_block[0:first_100]
    '''

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

fit_glm_block1, fit_glm_block3, predicted_block1, predicted_block3 = probabilisticFreeChoicePilotTask_logisticRegression(reward1, target1, trial1, reward3, target3, trial3, stim_trials)
#fit_glm_block1, fit_glm_block3, predicted_block1, predicted_block3 = probabilisticFreeChoicePilotTask_logisticRegression_sepRegressors(reward1, target1, trial1, reward3, target3, trial3, stim_trials)

#print fit_glm_block1.summary()
#print fit_glm_block3.summary()

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
1. look at decsion rule and classification for trials  - not equal to null model
2. do regression for not separate regressors - better model
3. repeat analysis (w/o stim history) for first trial following stimulation - see _firstTrial.py. only difference is line 148 stating that we only include free-choice trials in block 3
where the previous trial was instructed (i.e. has stim)
4. what would the roc auc be for the null hypothesis? - unity line
5. can we predict if he was rewarded on the previous trial given his choice?  are there false alarms for positive predicts when he actually got stim instead of reward?

'''
