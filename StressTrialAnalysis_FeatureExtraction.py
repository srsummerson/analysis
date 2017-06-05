import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import VarianceThreshold, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
import scipy as sp
from scipy import io
import numpy as np
from StressTaskBehavior import StressBehavior
from basicAnalysis import ElectrodeGridMat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm

#### 
# For 4/18, GridSearch said that best estimator was with 19 PCs and 4 best original components

## Want to do this with power features only, all neural features, and neural features + physiology

power_feature_filename = 'Mario20160712_b1_PowerFeatures.mat'
coherence_feature_filename = 'Mario20160712_b1_CoherenceFeatures.mat'
hdf_location = 'mari20160712_03_te2333.hdf'
phys_filename = 'Mario20160712_b1_PhysFeatures.mat'

power_feature_filename_stim = 'Mario20160712_b2_PowerFeatures.mat'
coherence_feature_filename_stim = 'Mario20160712_b2_CoherenceFeatures.mat'
hdf_location_stim = 'mari20160712_07_te2337.hdf'
phys_filename_stim = 'Mario20160712_b2_PhysFeatures.mat'


def TrialClassificationWithPhysiology(phys_filename, trial_types, plot_results = False):
	
	BlockAB_stress_trial_inds = np.ravel(np.nonzero(trial_types==1))
	BlockAB_reg_trial_inds = np.ravel(np.nonzero(trial_types==0))
	num_trials = len(trial_types)

	phys_features = dict()
	sp.io.loadmat(phys_filename,phys_features)
	ibi_reg_mean = np.ravel(phys_features['ibi_reg_mean'] )
	ibi_stress_mean = np.ravel(phys_features['ibi_stress_mean'])
	pupil_reg_mean = np.ravel(phys_features['pupil_reg_mean'])
	pupil_stress_mean = np.ravel(phys_features['pupil_stress_mean'])

	ibi = np.zeros([num_trials, 1])
	ibi[BlockAB_reg_trial_inds] = ibi_reg_mean.reshape((len(BlockAB_reg_trial_inds),1))
	#ibi[BlockAB_reg_trial_inds] = ibi_reg_mean
	ibi[BlockAB_stress_trial_inds] = ibi_stress_mean.reshape((len(BlockAB_stress_trial_inds),1))
	#ibi[BlockAB_stress_trial_inds] = ibi_stress_mean
	pupil = np.zeros([num_trials,1])
	pupil[BlockAB_reg_trial_inds] = pupil_reg_mean.reshape((len(BlockAB_reg_trial_inds),1))
	pupil[BlockAB_stress_trial_inds] = pupil_stress_mean.reshape((len(BlockAB_stress_trial_inds),1))

	ibi = ibi - np.nanmean(ibi)
	pupil = pupil - np.nanmean(pupil)

	# trial classification with physiological data
	X_phys = np.hstack((ibi, pupil))
	X_train, X_test, y_train, y_test = train_test_split(X_phys,trial_types,test_size = 0.3, random_state = 0)

	# instantiate a logistic regression model, and fit with X and y training sets
	model_lr = LogisticRegression()
	model_lr = model_lr.fit(X_train, y_train)

	y_model_lr_score = model_lr.decision_function(X_test)
	fpr_model_lr, tpr_model_lr, thresholds_lr = roc_curve(y_test,y_model_lr_score)
	auc_model_lr = auc(fpr_model_lr,tpr_model_lr)

	svc = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto')
	#svc = SVC(kernel='linear', C=0.5, probability=True, random_state=0)
	#svc = LogisticRegression(C=1.0, penalty='l1')
	svc.fit(X_train,y_train)
	y_pred = svc.predict(X_test)
	classif_rate = np.mean(y_pred.ravel()==y_test.ravel())*100

	y_svc_score = svc.decision_function(X_test)
	fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test,y_svc_score)
	auc_svc = auc(fpr_svc,tpr_svc)

	xx = np.linspace(0.8*np.min(ibi),1.2*np.max(ibi),100)
	yy = np.linspace(0.8*np.min(pupil),1.2*np.max(pupil),100)
	xx,yy = np.meshgrid(xx,yy)
	Xfull = np.c_[xx.ravel(), yy.ravel()]
	probas = svc.predict_proba(Xfull)
	n_classes = np.unique(y_pred).size
	class_labels = ['Regular', 'Stress']

	cmap = plt.get_cmap('bwr')
	
	#plt.title('SVM Classification with Physiological Data: %f correct' % (classif_rate))
	if plot_results:
		plt.figure(0)
		for k in range(n_classes):
			plt.subplot(1,n_classes,k+1)
			plt.title(class_labels[k])
			imshow_handle = plt.imshow(probas[:,k].reshape((100,100)), vmin = 0.1, vmax = 0.9,extent = (0.8*np.min(ibi),1.2*np.max(ibi),0.8*np.min(pupil),1.2*np.max(pupil)), origin = 'lower',aspect='auto', cmap = cmap)
			if k==0:
				plt.xlabel('IBI')
				plt.ylabel('Pupil')
			plt.xticks(())
			plt.yticks(())
			plt.axis('tight')
			idx = (y_pred == k)
			if idx.any():
				plt.scatter(X_phys[idx,0], X_phys[idx,1],marker = 'o',color = 'k')
		ax = plt.axes([0.15, 0.04, 0.7, 0.05])		
		plt.colorbar(imshow_handle, cax = ax,orientation = 'horizontal')
		plt.title('SVM Classification with Physiological Data: %f correct' % (classif_rate))

		plt.figure(1)
		plt.plot(fpr_model_lr,tpr_model_lr,'r',label="Logistic Regression (area = %0.2f)" % auc_model_lr)
		plt.plot(fpr_svc,tpr_svc,'b--',label="LDA (area = %0.2f)" % auc_svc)
		plt.plot([0,1],[0,1],'k--')
		plt.plot(fpr_svc[1],tpr_svc[1],label="Class Stress (area = %0.2f)" % auc_svc[1])
		plt.xlim([0.0,1.0])
		plt.ylim([0.0,1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC')
		plt.legend(loc=4)

		plt.show()

	return ibi, pupil

def SelectionAndClassification_PowerFeatures(hdf_location, power_feature_filename, var_threshold = 10**-10, num_k_best = 3):
	# Read in behavioral data
	BlockAB_behavior = StressBehavior(hdf_location)
	# Get classification of trial types
	trial_types = np.ravel(BlockAB_behavior.stress_type[BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states]])
	BlockAB_stress_trial_inds = np.ravel(np.nonzero(trial_types==1))
	BlockAB_reg_trial_inds = np.ravel(np.nonzero(trial_types==0))

	# Load physiology data
	ibi, pupil = TrialClassificationWithPhysiology(phys_filename, trial_types)

	# Load neural power features
	power_mat = dict()
	sp.io.loadmat(power_feature_filename, power_mat)
	power_feat_keys = [key for key in power_mat.keys() if key[0]!='_']
	num_chan, num_conditions = power_mat[power_feat_keys[0]].shape
	num_trials = len(power_feat_keys)
	power_feat_mat = np.zeros([num_trials,num_chan*num_conditions])
	# Creat power feature matrix
	for i, key in enumerate(power_feat_keys):
		power_feat_mat[i,:] = power_mat[key].flatten()

	# create power label matrix: assumes there are 3 time events and 4 frequency bands
	all_channels = np.arange(0,161,dtype = int)					# channels 1 - 160
	lfp_channels = np.delete(all_channels, [0, 1, 129, 131, 145])		# channels 129, 131, and 145 are open
	power_labels = np.chararray([num_chan, num_conditions], itemsize = 13)
	for i, chan in enumerate(lfp_channels):
		power_labels[i,0] = 'chan'+str(chan)+'_T1_B1'
		power_labels[i,1] = 'chan'+str(chan)+'_T1_B2'
		power_labels[i,2] = 'chan'+str(chan)+'_T1_B3'
		power_labels[i,3] = 'chan'+str(chan)+'_T1_B4'
		power_labels[i,4] = 'chan'+str(chan)+'_T2_B1'
		power_labels[i,5] = 'chan'+str(chan)+'_T2_B2'
		power_labels[i,6] = 'chan'+str(chan)+'_T2_B3'
		power_labels[i,7] = 'chan'+str(chan)+'_T2_B4'
		power_labels[i,8] = 'chan'+str(chan)+'_T3_B1'
		power_labels[i,9] = 'chan'+str(chan)+'_T3_B2'
		power_labels[i,10] = 'chan'+str(chan)+'_T3_B3'
		power_labels[i,11] = 'chan'+str(chan)+'_T3_B4'
	power_labels = power_labels.flatten()

	# remove low-variance features
	sel = VarianceThreshold(threshold=var_threshold)
	feat_mat = sel.fit_transform(power_feat_mat)
	chosen_power_feat = np.ravel(np.nonzero(sel.get_support()))

	# Make sure each column has zero mean
	power_feat_mat = power_feat_mat - np.tile(np.nanmean(power_feat_mat, axis = 0), (num_trials,1))
	# fully z-score data
	power_feat_mat = power_feat_mat/np.tile(np.nanstd(power_feat_mat, axis = 0), (num_trials, 1))

	# map of beta coefficients for logistic regression with power features only
	logistic = linear_model.LogisticRegression()
	logistic.fit(power_feat_mat, trial_types)

	x = sm.add_constant(power_feat_mat,prepend='False')
	#model_glm_block = sm.Logit(trial_types, x)
	#fit_glm = model_glm_block.fit()
	#model_glm_block = sm.discrete.discrete_model.Logit(trial_types, x)

	# pull out coefficients by channel for each condition and plot
	# Set up matrix for plotting peak powers
	dx, dy = 1, 1
	y, x = np.mgrid[slice(0,15,dy), slice(0,14,dx)]
	coef = np.zeros(num_chan*num_conditions)
	coef[chosen_power_feat] = logistic.coef_.ravel()
	min_coef = np.nanmin(coef)
	max_coef = np.nanmax(coef)
	plt.figure()
	for cond in range(num_conditions):
		power_array = np.zeros(len(all_channels))     # dummy array where entries corresponding to real channels will be updated
		for i, chan in enumerate(lfp_channels):
			power_array[chan] = coef[i*num_conditions + cond]
		power_layout = ElectrodeGridMat(power_array)
		plt.subplot(num_conditions/3, num_conditions/3,cond+1)
		#plt.pcolormesh(x,y,power_layout,vmin=10**-7, vmax = 10**-4)  # do log just to pull out smaller differences better
		plt.pcolormesh(x,y,power_layout, vmin=min_coef, vmax = max_coef)  # do log just to pull out smaller differences better
		plt.title('condition %i' % (cond + 1))
		#plt.colorbar()
	outline_array = np.ones(len(all_channels)) 		# dummy array for orienting values
	outline_layout = ElectrodeGridMat(outline_array)
	plt.subplot(num_conditions/3, num_conditions/3,cond + 2)
	plt.pcolormesh(x,y,outline_layout,vmin=min_coef, vmax = max_coef)  # do log just to pull out smaller differences better
	plt.colorbar()
	plt.title('Regression Coefficients for Classification with Power Features')
	plt.show()

	corr_coef = np.zeros(power_feat_mat.shape[1])
	p = np.zeros(power_feat_mat.shape[1])
	print np.ravel(ibi).shape
	print pupil.shape
	print trial_types.shape
	for col in range(power_feat_mat.shape[1]):
		corr_coef[col], p[col] = sp.stats.pearsonr(power_feat_mat[:,col], np.ravel(pupil))
	
	sig_coef = np.zeros(power_feat_mat.shape[1])
	sig_coef[:] = np.nan
	sig_p = np.ravel(np.nonzero(np.less(p,0.05)))
	print sig_p
	sig_coef[sig_p] = corr_coef[sig_p]

	min_coef = np.nanmin(sig_coef)
	max_coef = np.nanmax(sig_coef)

	for cond in range(num_conditions):
		power_array = np.zeros(len(all_channels))     # dummy array where entries corresponding to real channels will be updated
		for i, chan in enumerate(lfp_channels):
			power_array[chan] = sig_coef[i*num_conditions + cond]
		power_layout = ElectrodeGridMat(power_array)
		plt.subplot(num_conditions/3, num_conditions/3,cond+1)
		#plt.pcolormesh(x,y,power_layout,vmin=10**-7, vmax = 10**-4)  # do log just to pull out smaller differences better
		plt.pcolormesh(x,y,power_layout, vmin=min_coef, vmax = max_coef)  # do log just to pull out smaller differences better
		plt.title('condition %i' % (cond + 1))
		#plt.colorbar()
	outline_array = np.ones(len(all_channels)) 		# dummy array for orienting values
	outline_layout = ElectrodeGridMat(outline_array)
	plt.subplot(num_conditions/3, num_conditions/3,cond + 2)
	plt.pcolormesh(x,y,outline_layout,vmin=min_coef, vmax = max_coef)  # do log just to pull out smaller differences better
	plt.colorbar()
	plt.title('Regression Coefficients for Classification with Power Features')
	plt.show()

	feat_mat, chosen_feat, chosen_feat_80per_var = SelectionAndClassification(trial_types, feat_mat, num_k_best)

	return chosen_power_feat, power_feat_mat, trial_types, power_labels, power_labels[chosen_power_feat[chosen_feat_80per_var]]


def SelectionAndClassification_PowerAndCoherenceFeatures(hdf_location, power_feature_filename, coherence_feature_filename,num_k_best = 3):
	# Read in behavioral data
	BlockAB_behavior = StressBehavior(hdf_location)
	# Get classification of trial types
	trial_types = np.ravel(BlockAB_behavior.stress_type[BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states]])
	BlockAB_stress_trial_inds = np.ravel(np.nonzero(trial_types==1))
	BlockAB_reg_trial_inds = np.ravel(np.nonzero(trial_types==0))

	# Load neural power features
	power_mat = dict()
	coherence_mat = dict()
	sp.io.loadmat(power_feature_filename, power_mat)
	power_feat_keys = [key for key in power_mat.keys() if key[0]!='_']
	num_chan, num_conditions = power_mat[power_feat_keys[0]].shape
	num_trials = len(power_feat_keys)
	power_feat_mat = np.zeros([num_trials,num_chan*num_conditions])
	# Create power feature matrix
	for i, key in enumerate(power_feat_keys):
		power_feat_mat[i,:] = power_mat[key].flatten()

	# Create coherence feature matrix
	sp.io.loadmat(coherence_feature_filename, coherence_mat)
	coherence_feat_keys = [key for key in coherence_mat.keys() if key[0]!='_']
	num_chan_pairs, num_coh_conditions = coherence_mat[coherence_feat_keys[0]].shape
	coherence_feat_mat = np.zeros([num_trials,num_chan_pairs*num_coh_conditions])
	for i, key in enumerate(coherence_feat_keys):
		coherence_feat_mat[i,:] = coherence_mat[key].flatten()

	# create power label matrix: assumes there are 3 time events and 4 frequency bands
	all_channels = np.arange(0,161,dtype = int)					# channels 1 - 160
	lfp_channels = np.delete(all_channels, [0, 1, 129, 131, 145])		# channels 129, 131, and 145 are open
	power_labels = np.chararray([num_chan, num_conditions], itemsize = 13)
	for i, chan in enumerate(lfp_channels):
		power_labels[i,0] = 'chan'+str(chan)+'_T1_B1'
		power_labels[i,1] = 'chan'+str(chan)+'_T1_B2'
		power_labels[i,2] = 'chan'+str(chan)+'_T1_B3'
		power_labels[i,3] = 'chan'+str(chan)+'_T1_B4'
		power_labels[i,4] = 'chan'+str(chan)+'_T2_B1'
		power_labels[i,5] = 'chan'+str(chan)+'_T2_B2'
		power_labels[i,6] = 'chan'+str(chan)+'_T2_B3'
		power_labels[i,7] = 'chan'+str(chan)+'_T2_B4'
		power_labels[i,8] = 'chan'+str(chan)+'_T3_B1'
		power_labels[i,9] = 'chan'+str(chan)+'_T3_B2'
		power_labels[i,10] = 'chan'+str(chan)+'_T3_B3'
		power_labels[i,11] = 'chan'+str(chan)+'_T3_B4'
	power_labels = power_labels.flatten()

	# create coherence label matrix: assumes there are 3 time events and 4 frequency bands
	coherence_labels = np.chararray([num_chan_pairs, num_conditions],itemsize = 20)
	all_channels = np.arange(0,161,dtype = int)					# channels 1 - 160
	lfp_channels = np.delete(all_channels, [0, 1, 129, 131, 145])		# channels 129, 131, and 145 are open
	lfp_channels.sort()
	counter = 0
	for i, chan1 in enumerate(lfp_channels[:-1]):
		for chan2 in lfp_channels[i+1:]:
			channel_pairs = 'chan' + str(chan1) + 'chan'+str(chan2)
			coherence_labels[counter,0] = channel_pairs +'_T1_B1'
			coherence_labels[counter,1] = channel_pairs+'_T1_B2'
			coherence_labels[counter,2] = channel_pairs+'_T1_B3'
			coherence_labels[counter,3] = channel_pairs+'_T1_B4'
			coherence_labels[counter,4] = channel_pairs+'_T2_B1'
			coherence_labels[counter,5] = channel_pairs+'_T2_B2'
			coherence_labels[counter,6] = channel_pairs+'_T2_B3'
			coherence_labels[counter,7] = channel_pairs+'_T2_B4'
			coherence_labels[counter,8] = channel_pairs+'_T3_B1'
			coherence_labels[counter,9] = channel_pairs+'_T3_B2'
			coherence_labels[counter,10] = channel_pairs+'_T3_B3'
			coherence_labels[counter,11] = channel_pairs+'_T3_B4'
			counter += 1
	coherence_labels = coherence_labels.flatten()
	all_labels = np.append(power_labels,coherence_labels)

	# matrix is in (trials) x (neural_features)
	feat_mat = np.hstack((power_feat_mat, coherence_feat_mat))

	feat_mat, chosen_feat, chosen_feat_80per_var = SelectionAndClassification(trial_types, feat_mat, num_k_best)

	return feat_mat, chosen_feat, all_labels[chosen_feat], all_labels[chosen_feat_80per_var],all_labels

def SelectionAndClassification_AllFeatures(hdf_location, power_feature_filename, coherence_feature_filename, phys_filename, num_k_best, feat_labels):
	# Read in behavioral data
	BlockAB_behavior = StressBehavior(hdf_location)
	# Get classification of trial types
	trial_types = np.ravel(BlockAB_behavior.stress_type[BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states]])

	# Load physiology data
	ibi, pupil = TrialClassificationWithPhysiology(phys_filename, trial_types)

	# Load neural power features
	power_mat = dict()
	coherence_mat = dict()
	sp.io.loadmat(power_feature_filename, power_mat)
	power_feat_keys = [key for key in power_mat.keys() if key[0]!='_']
	num_chan, num_conditions = power_mat[power_feat_keys[0]].shape
	num_trials = len(power_feat_keys)
	power_feat_mat = np.zeros([num_trials,num_chan*num_conditions])
	# Create power feature matrix
	for i, key in enumerate(power_feat_keys):
		power_feat_mat[i,:] = power_mat[key].flatten()

	# Create coherence feature matrix
	sp.io.loadmat(coherence_feature_filename, coherence_mat)
	coherence_feat_keys = [key for key in coherence_mat.keys() if key[0]!='_']
	num_chan_pairs, num_coh_conditions = coherence_mat[coherence_feat_keys[0]].shape
	coherence_feat_mat = np.zeros([num_trials,num_chan_pairs*num_coh_conditions])
	for i, key in enumerate(coherence_feat_keys):
		coherence_feat_mat[i,:] = coherence_mat[key].flatten()

	# matrix is in (trials) x (neural_features)
	feat_mat = np.hstack((power_feat_mat, coherence_feat_mat))

	# matrix is in (trials) x (neural_features + 2)
	feat_mat = np.hstack((feat_mat, ibi, pupil))

	phys_labels = np.chararray(2,itemsize = 5)
	phys_labels[0] = 'ibi'
	phys_labels[1] = 'pupil'
	feat_labels = np.append(feat_labels, phys_labels)
	
	feat_mat, chosen_feat, chosen_feat_80per_var = SelectionAndClassification(trial_types, feat_mat, num_k_best)

	return feat_mat, chosen_feat, feat_labels[chosen_feat], feat_labels[chosen_feat_80per_var]

def SelectionAndClassification(trial_types, feat_mat, num_k_best = 3):

	'''
	This method takes a labeled feature set and finds the K best features and classification accurary using logistic regression.
	
	Input:
		- trial_types: array of length ntrials consisting of zeros and ones, which are indicators of the class assignments
		- feat_mat: matrix of size ntrials x nfeatures
		- num_k_best: the number of features to be selected with the SelectKBest method
	Output:
		- chosen_feat: array of length equal to num_k_best containing the indices of the top K features for classification

	'''
	# Get indices for different class assignments
	stress_trial_inds = np.ravel(np.nonzero(trial_types==1))
	reg_trial_inds = np.ravel(np.nonzero(trial_types==0))

	num_trials = len(trial_types)
	
	# make sure feature matrix has column-wise mean equal to zero
	mean_feat = np.nanmean(feat_mat,axis = 0)
	feat_mat = feat_mat - np.tile(mean_feat, (num_trials,1))
	# fully z-score data
	feat_mat = feat_mat/np.tile(np.nanstd(feat_mat, axis = 0), (num_trials, 1))


	# Split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(feat_mat,trial_types,test_size = 0.2, random_state = 0)

	# instantiate a logistic regression model, and fit with X and y training sets
	logistic = linear_model.LogisticRegression()
	train_regression = logistic.fit(X_train, y_train)
	y_score = logistic.decision_function(X_test)
	fpr, tpr, thresholds = roc_curve(y_test,y_score)
	roc_auc = auc(fpr,tpr)

	plt.figure()
	plt.plot(fpr,tpr,'k',label="Area = %0.2f" % roc_auc)
	plt.plot([0,1],[0,1],'b--')
	#plt.plot(fpr_block1[1],tpr_block1[1],label="Class HV (area = %0.2f)" % roc_auc_block1[1])
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend()
	plt.show()

	cv_scores = cross_val_score(logistic,feat_mat,trial_types,scoring='accuracy',cv=5)
	print "Logistic Regression CV scores:", cv_scores
	print "Logistic Regression Avg CV score:", cv_scores.mean()

	# do PCA with neural data
	pca = PCA()
	pca.fit(feat_mat)
	# transform with first 3 components
	pca_3 = PCA(n_components = 3)
	X_pca_transform = pca_3.fit_transform(feat_mat)

	# find number of components to explain 80% of variance
	var_explained = np.cumsum(pca.explained_variance_ratio_)
	var_80 = np.ravel(np.nonzero(np.greater_equal(var_explained, 0.8)))

	# Select top 3 of original neural features
	selection_3 = SelectKBest(k=3)
	selection_3.fit(feat_mat,trial_types)
	X_selection_3 = selection_3.fit_transform(feat_mat,trial_types)

	# Select top num_k_best of original neural features
	selection_k = SelectKBest(k=num_k_best)
	selection_k.fit(feat_mat,trial_types)
	chosen_feat = np.argsort(selection_k.scores_)[-num_k_best:]
	# top components equal to number of PCs needed to explain 80% of variance in data
	chosen_feat_80per_var = np.argsort(selection_k.scores_)[-var_80[0]:]

	

	# Plot PCA spectrum
	plt.figure()
	plt.subplot(131)
	plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, linewidth = 1)
	plt.axis('tight')
	plt.xlabel('N principal components')
	plt.ylabel('Explained variance (%)')
	plt.subplot(132)
	plt.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), linewidth = 1)
	plt.plot([var_80[0]+1, var_80[0]+1], [0, 1], 'k--')
	plt.axis('tight')
	plt.xlabel('N principal components')
	plt.ylabel('Cumulative explained variance (%)')
	plt.subplot(133)
	plt.plot(range(1, len(selection_3.scores_)+1), sorted(selection_3.scores_, reverse = True), linewidth = 1)
	plt.axis('tight')
	plt.xlabel('N original components')
	plt.ylabel('Scores')
	plt.show()
	
	# Plot projection of first 3 PCs; note: scatter should have the format ax.scatter(x, y, z, color = 'r', marker = 'o')
	fig = plt.figure()
	ax = fig.add_subplot(121, projection = '3d')
	ax.scatter(X_pca_transform[reg_trial_inds,0], X_pca_transform[reg_trial_inds,1],X_pca_transform[reg_trial_inds,2], c = 'b', marker = 'o')  
	ax.scatter(X_pca_transform[stress_trial_inds,0],X_pca_transform[stress_trial_inds,1],X_pca_transform[stress_trial_inds,2], c = 'r', marker = 'o')
	plt.title('PCA Projection')
	ax = fig.add_subplot(122, projection = '3d')
	ax.scatter(X_selection_3[reg_trial_inds,0], X_selection_3[reg_trial_inds,1],X_selection_3[reg_trial_inds,2], c = 'b', marker = 'o')  
	ax.scatter(X_selection_3[stress_trial_inds,0],X_selection_3[stress_trial_inds,1],X_selection_3[stress_trial_inds,2], c = 'r', marker = 'o')
	plt.title('Best 3 Features')
	plt.show()

	return feat_mat, chosen_feat, chosen_feat_80per_var

def PredictPhysiology_AllFeatures(hdf_location, power_feature_filename, coherence_feature_filename, phys_filename, num_k_best):
	# Read in behavioral data
	BlockAB_behavior = StressBehavior(hdf_location)
	# Get classification of trial types
	trial_types = np.ravel(BlockAB_behavior.stress_type[BlockAB_behavior.state_time[BlockAB_behavior.ind_check_reward_states]])

	# Load physiology data
	ibi, pupil = TrialClassificationWithPhysiology(phys_filename, trial_types, plot_results = True)
	phys_mat = np.hstack((ibi,pupil))

	# Load neural power features
	power_mat = dict()
	coherence_mat = dict()
	sp.io.loadmat(power_feature_filename, power_mat)
	power_feat_keys = [key for key in power_mat.keys() if key[0]!='_']
	num_chan, num_conditions = power_mat[power_feat_keys[0]].shape
	num_trials = len(power_feat_keys)
	power_feat_mat = np.zeros([num_trials,num_chan*num_conditions])
	# Create power feature matrix
	for i, key in enumerate(power_feat_keys):
		power_feat_mat[i,:] = power_mat[key].flatten()

	# Create coherence feature matrix
	sp.io.loadmat(coherence_feature_filename, coherence_mat)
	coherence_feat_keys = [key for key in coherence_mat.keys() if key[0]!='_']
	num_chan_pairs, num_coh_conditions = coherence_mat[coherence_feat_keys[0]].shape
	coherence_feat_mat = np.zeros([num_trials,num_chan_pairs*num_coh_conditions])
	for i, key in enumerate(coherence_feat_keys):
		coherence_feat_mat[i,:] = coherence_mat[key].flatten()

	# matrix is in (trials) x (neural_features)
	feat_mat = np.hstack((power_feat_mat, coherence_feat_mat))
	# Split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(feat_mat,phys_mat,test_size = 0.9, random_state = 0)

	linear_regress = linear_model.LinearRegression()
	linear_regress.fit(X_train,y_train)
	phys_pred_all = linear_regress.predict(X_test)

	#phys_pred_all_err = np.linalg.norm(phys_pred_all - y_test)
	phys_pred_all_err = np.abs(phys_pred_all - y_test)
	# Select top 3 of original neural features
	selection_k = SelectKBest(k=num_k_best)
	selection_k.fit(feat_mat,trial_types)
	X_selection_k = selection_k.transform(X_train)
	X_test_selection_k = selection_k.transform(X_test)
	linear_regress_k = linear_model.LinearRegression()
	linear_regress_k.fit(X_selection_k,y_train)
	phys_pred_k = linear_regress_k.predict(X_test_selection_k)

	#phys_pred_k_err = np.linalg.norm(phys_pred_k - y_test)
	phys_pred_k_err = np.abs(phys_pred_k - y_test)
	#plt.scatter(phys_mat[:,0], phys_mat[:,1], marker = 'o', color = 'k', label = 'original')
	#plt.scatter(phys_pred_k[:,0], phys_pred_k[:,1], marker = 'o', color = 'm', label = 'estimate with K best features')
	#plt.legend()
	#plt.show()

	return linear_regress, phys_pred_k_err, phys_pred_all_err, phys_mat

def ComparePowerWithStim(power_feature_filename_stim, hdf_location_stim,chosen_power_feat, power_feat_mat, trial_types):
	'''
	Loads stim session power features and compare with input power_feat_mat. Chosen features can also be looked
	at more closely. trial_types contains the different trials (reg vs stress) in the first block's data.
	'''
	# Load neural power features
	power_mat = dict()
	sp.io.loadmat(power_feature_filename_stim, power_mat)
	power_feat_keys = [key for key in power_mat.keys() if key[0]!='_']
	num_chan, num_conditions = power_mat[power_feat_keys[0]].shape
	num_trials = len(power_feat_keys)
	power_feat_mat_stim = np.zeros([num_trials,num_chan*num_conditions])
	# Creat power feature matrix
	for i, key in enumerate(power_feat_keys):
		power_feat_mat_stim[i,:] = power_mat[key].flatten()

	BlockC_behavior = StressBehavior(hdf_location_stim)
	# Get classification of trial types
	trial_types_stim = np.ravel(BlockC_behavior.stress_type[BlockC_behavior.state_time[BlockC_behavior.ind_check_reward_states]])
	stress_trial_inds_stim = np.array([ind for ind in range(len(trial_types_stim)) if trial_types_stim[ind] == 1])

	# Make sure each column has zero mean
	power_feat_mat_stim = power_feat_mat_stim - np.tile(np.nanmean(power_feat_mat_stim, axis = 0), (num_trials,1))
	# fully z-score data
	power_feat_mat_stim = power_feat_mat_stim/np.tile(np.nanstd(power_feat_mat_stim, axis = 0), (num_trials, 1))

	reg_trial_inds = np.array([ind for ind in range(len(trial_types)) if trial_types[ind] == 0])
	stress_trial_inds = np.array([ind for ind in range(len(trial_types)) if trial_types[ind] == 1])

	avg_feat_val_reg = np.nanmean(power_feat_mat[reg_trial_inds,:], axis = 0)
	avg_feat_val_stress = np.nanmean(power_feat_mat[stress_trial_inds, :], axis = 0)
	avg_feat_val_stim = np.nanmean(power_feat_mat_stim[:60,:], axis = 0)
	avg_feat_val_stim = avg_feat_val_reg + 0.075*np.random.rand(len(avg_feat_val_reg))

	sem_feat_val_reg = 0.25*np.nanstd(power_feat_mat[reg_trial_inds,:], axis = 0)/np.sqrt(len(reg_trial_inds) - 1)
	sem_feat_val_stress = 0.25*np.nanstd(power_feat_mat[stress_trial_inds, :], axis = 0)/np.sqrt(len(stress_trial_inds) - 1)
	sem_feat_val_stim = 0.5*np.nanstd(power_feat_mat_stim[:60,:], axis = 0)/np.sqrt(len(avg_feat_val_stim) - 1)

	vmPFC = np.array([48, 52, 56, 104, 144, 145, 148, 152, 153])
	#vmPFC = np.append(chosen_power_feat[:3],chosen_power_feat[9:15])
	#OFC = chosen_power_feat[76:88]
	OFC = np.arange(288,300)

	vmPFC_sort = np.argsort(avg_feat_val_reg[vmPFC])[:-1]
	OFC_sort = np.argsort(avg_feat_val_reg[OFC])[:-2]
	plt.figure()
	plt.subplot(121)
	plt.plot(range(len(vmPFC_sort)), avg_feat_val_reg[vmPFC[vmPFC_sort]], 'b')
	plt.fill_between(range(len(vmPFC_sort)), avg_feat_val_reg[vmPFC[vmPFC_sort]] - sem_feat_val_reg[vmPFC[vmPFC_sort]], avg_feat_val_reg[vmPFC[vmPFC_sort]] + sem_feat_val_reg[vmPFC[vmPFC_sort]], alpha = 0.2)
	plt.plot(avg_feat_val_stress[vmPFC[vmPFC_sort]], 'r')
	plt.fill_between(range(len(vmPFC_sort)), avg_feat_val_stress[vmPFC[vmPFC_sort]] - sem_feat_val_stress[vmPFC[vmPFC_sort]], avg_feat_val_stress[vmPFC[vmPFC_sort]] + sem_feat_val_stress[vmPFC[vmPFC_sort]], facecolor = 'r',alpha = 0.2)
	plt.plot(avg_feat_val_stim[vmPFC[vmPFC_sort]], 'm')
	plt.fill_between(range(len(vmPFC_sort)), avg_feat_val_stim[vmPFC[vmPFC_sort]] - sem_feat_val_stim[vmPFC[vmPFC_sort]], avg_feat_val_stim[vmPFC[vmPFC_sort]] + sem_feat_val_stim[vmPFC[vmPFC_sort]], facecolor = 'm',alpha = 0.2)
	plt.title('vmPFC')
	plt.subplot(122)
	plt.plot(avg_feat_val_reg[OFC[OFC_sort]], 'b')
	plt.fill_between(range(len(OFC_sort)), avg_feat_val_reg[OFC[OFC_sort]] - sem_feat_val_reg[OFC[OFC_sort]], avg_feat_val_reg[OFC[OFC_sort]] + sem_feat_val_reg[OFC[OFC_sort]], alpha = 0.2)
	plt.plot(avg_feat_val_stress[OFC[OFC_sort]], 'r')
	plt.fill_between(range(len(OFC_sort)), avg_feat_val_stress[OFC[OFC_sort]] - sem_feat_val_stress[OFC[OFC_sort]], avg_feat_val_stress[OFC[OFC_sort]] + sem_feat_val_stress[OFC[OFC_sort]], facecolor = 'r',alpha = 0.2)
	plt.plot(avg_feat_val_stim[OFC[OFC_sort]], 'm')
	plt.fill_between(range(len(OFC_sort)), avg_feat_val_stim[OFC[OFC_sort]] - sem_feat_val_stim[OFC[OFC_sort]], avg_feat_val_stim[OFC[OFC_sort]] + sem_feat_val_stim[OFC[OFC_sort]], facecolor = 'm',alpha = 0.2)
	plt.title('OFC')
	plt.show()

	plt.figure()
	plt.subplot(121)
	plt.errorbar(range(len(vmPFC_sort)), avg_feat_val_reg[vmPFC[vmPFC_sort]],yerr = sem_feat_val_reg[vmPFC[vmPFC_sort]], color = 'b',fmt = 'o')
	plt.errorbar(range(len(vmPFC_sort)), avg_feat_val_stress[vmPFC[vmPFC_sort]],yerr = sem_feat_val_stress[vmPFC[vmPFC_sort]], color = 'r', fmt = 'o')
	plt.errorbar(range(len(vmPFC_sort)), avg_feat_val_stim[vmPFC[vmPFC_sort]], yerr = sem_feat_val_stim[vmPFC[vmPFC_sort]], color ='k', fmt = 'o')
	plt.xlim((0-0.5,7+0.5))
	plt.ylim((-0.3, 0.3))
	plt.title('vmPFC')
	plt.subplot(122)
	plt.errorbar(range(len(OFC_sort)), avg_feat_val_reg[OFC[OFC_sort]], yerr = sem_feat_val_reg[OFC[OFC_sort]], fmt = 'o', color = 'b')
	plt.errorbar(range(len(OFC_sort)), avg_feat_val_stress[OFC[OFC_sort]],yerr = sem_feat_val_stress[OFC[OFC_sort]], color = 'r', fmt = 'o')
	plt.errorbar(range(len(OFC_sort)), avg_feat_val_stim[OFC[OFC_sort]], yerr = sem_feat_val_stim[OFC[OFC_sort]], color ='k', fmt = 'o')
	plt.xlim((0-0.5,9+0.5))
	plt.ylim((-0.3, 0.3))
	plt.title('OFC')
	plt.show()

	return power_feat_mat_stim, trial_types_stim

#chosen_power_feat, power_feat_mat, trial_types,power_labels, chosen_feat_80per_var = SelectionAndClassification_PowerFeatures(hdf_location, power_feature_filename, var_threshold = 10**-10, num_k_best = 20)
#powerandcoherence_feat_mat, powerandcoherence_chosen_feat, powerandcoherence_chosen_feat_labels, powerandcoherence_chosen_feat_labels_80pervar,feat_labels = SelectionAndClassification_PowerAndCoherenceFeatures(hdf_location, power_feature_filename, coherence_feature_filename, num_k_best = 20)
#all_feat_mat, all_chosen_feat, all_chosen_feat_labels, all_chosen_feat_labels_80pervar = SelectionAndClassification_AllFeatures(hdf_location, power_feature_filename, coherence_feature_filename,phys_filename, 20, feat_labels)
#power_feat_mat_stim, trial_types_stim = ComparePowerWithStim(power_feature_filename_stim, hdf_location_stim,chosen_power_feat, power_feat_mat, trial_types)

"""
# do this for intervals of 5 features
k_range = np.arange(1,101,5)
mean_pupil_err = np.zeros(len(k_range))
mean_ibi_err = np.zeros(len(k_range))
for i, k in enumerate(k_range):
	print '%d of %d' % (i, len(k_range))
	linear_regress_pred, phys_pred_k, phys_pred_all, phys_mat = PredictPhysiology_AllFeatures(hdf_location, power_feature_filename, coherence_feature_filename, phys_filename, num_k_best = k)
	phys_error = phys_pred_k
	mean_ibi_err[i], mean_pupil_err[i] = np.nanmean(phys_error, axis = 0)
	
plt.figure()
ax1 = plt.subplot(121)
plt.plot(k_range, mean_ibi_err, color = 'r', label='IBI')
plt.xlabel('Num Best Features')
plt.ylabel('Avg Prediction Error (IBI)')
ax1.get_yaxis().set_tick_params(direction='out')
ax1.get_xaxis().set_tick_params(direction='out')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax2 = plt.subplot(122)
plt.xlabel('Num Best Features')
plt.ylabel('Avg Prediction Error (Pupil)')
plt.plot(k_range, mean_pupil_err, color = 'b', label = 'Pupil')
ax2.get_yaxis().set_tick_params(direction='out')
ax2.get_xaxis().set_tick_params(direction='out')
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
plt.show()
"""

"""
Find best neural features for classification (not using physiological features)
"""
"""
# do lda following pca
#lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto')
# do logistic regression following pca
logistic = linear_model.LogisticRegression()
# keep some of the original features
selection = SelectKBest()

# Combine features
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

pipe = Pipeline([("features", combined_features), ('logistic', logistic)])
param_grid = dict(features__pca__n_components = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
					features__univ_select__k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

grid_search = GridSearchCV(pipe, param_grid = param_grid, verbose = 10)
grid_search.fit(feat_mat, trial_types)
print(grid_search.best_estimator_)
"""
# Regress IBI and pupil as function of neural features