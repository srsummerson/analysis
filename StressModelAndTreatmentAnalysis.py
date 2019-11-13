import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
import sys
import statsmodels.api as sm
from neo import io
import scipy.io as spio
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from basicAnalysis import LDAforFeatureSelection, plot_step_lda
from scipy import signal
from scipy import stats
from scipy.ndimage import filters
from matplotlib import mlab
from syncHDF import create_syncHDF_TDTloaded
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse, highpassFilterData
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD, computePowerFeatures, computeAllCoherenceFeatures, computePowerFeatures_overTime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from basicAnalysis import highpassFilterData, lowpassFilterData, bandpassFilterData
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import os.path
import time

from StressTaskBehavior import StressBehaviorWithDrugs_CenterOut

"""
Main file for analyzing Luigi and Mario's data from the experiments comparing closed-loop stimulation
with benzodiazepines in terms of the anxiolytic response, where beta-carbolines were used to generate
an anxiogenic response. 

To do:


"""
class StressTask_PhysData():
	'''
	Class for operating on pre-computed peripheral physiology data. 
	This is for the stress task which has 3 blocks: baseline, stress, and treatment.
	Peripheral physiology data is pre-computed (1) per trial, looking at the first 100 ms
	after a hold_center state and (2) over time, per 10 s time windows that 
	'''

	def __init__(self, phys_mat_filenames):
		# Read  data from .mat file
		self.filename = phys_mat_filenames[0][59:-20]
		self.phys_features_base = spio.loadmat(phys_mat_filenames[0])
		self.phys_features_stress = spio.loadmat(phys_mat_filenames[1])
		self.phys_features_treat = spio.loadmat(phys_mat_filenames[2])
		
		# Pull out per-trial data
		self.ibi_trials_base = self.phys_features_base['ibi_trials_mean'].flatten()
		self.pupil_trials_base = self.phys_features_base['pupil_trials_mean'].flatten()
		self.ibi_trials_stress = self.phys_features_stress['ibi_trials_mean'].flatten()
		self.pupil_trials_stress = self.phys_features_stress['pupil_trials_mean'].flatten()
		self.ibi_trials_treat = self.phys_features_treat['ibi_trials_mean'].flatten()
		self.pupil_trials_treat = self.phys_features_treat['pupil_trials_mean'].flatten()


		
		# Pull out per-time block data
		self.ibi_time_base = self.phys_features_base['ibi_time_mean'].flatten()
		self.pupil_time_base = self.phys_features_base['pupil_time_mean'].flatten()
		self.ind_time_base = self.phys_features_base['ind_time'].flatten()
		self.ibi_time_stress = self.phys_features_stress['ibi_time_mean'].flatten()
		self.pupil_time_stress = self.phys_features_stress['pupil_time_mean'].flatten()
		self.ind_time_stress = self.phys_features_stress['ind_time'].flatten()
		self.ibi_time_treat = self.phys_features_treat['ibi_time_mean'].flatten()
		self.pupil_time_treat = self.phys_features_treat['pupil_time_mean'].flatten()
		self.ind_time_treat = self.phys_features_treat['ind_time'].flatten()

	def logistic_regression_baseline_vs_stress(self,trial_or_time,**kwargs):
		'''
		Method that performs logistic regression of the task state using IBI and PD
		as regressors. Either physiology data taken over trials or time may be used.
		'''
		if trial_or_time == 'trial':
			ibi_base = self.ibi_trials_base
			ibi_stress = self.ibi_trials_stress
			pupil_base = self.pupil_trials_base
			pupil_stress = self.pupil_trials_stress
		else:
			ibi_base = self.ibi_time_base
			ibi_stress = self.ibi_time_stress
			pupil_base = self.pupil_time_base
			pupil_stress = self.pupil_time_stress
			
		# organize data into trial x data matrix, where data = (ibi, pd)
		trials1 = len(ibi_base)
		trials2 = len(ibi_stress)
		X_mat = np.zeros((trials1 + trials2,2))
		ibi_total = np.append(ibi_base, ibi_stress)
		pupil_total = np.append(pupil_base, pupil_stress)
		X_mat[:trials1,0] = (ibi_base - np.nanmean(ibi_total))/np.nanstd(ibi_total)
		X_mat[trials1:,0] = (ibi_stress - np.nanmean(ibi_total))/np.nanstd(ibi_total)
		X_mat[:trials1,1] = (pupil_base - np.nanmean(pupil_total))/np.nanstd(pupil_total)
		X_mat[trials1:,1] = (pupil_stress - np.nanmean(pupil_total))/np.nanstd(pupil_total)
		#X_mat[:,2] = np.ones(trials1 + trials2)

		# find any rows with nan entries in phys data matrix
		inds = np.ravel(np.nonzero(1 - np.sum(np.isnan(X_mat),axis = 1)))

		y = np.zeros(trials1 + trials2)
		y[trials1:] = 1

		# only keep data from rows that don't have nans
		X_mat = X_mat[inds,:]
		y = y[inds]

		# Logistic regression as was done with closed-loop stim
		

		#x_successful = np.vstack((np.append(ibi_reg_mean_adj, ibi_stress_mean_adj), np.append(pupil_reg_mean_adj, pupil_stress_mean_adj)))
		#x_successful = np.transpose(x_successful)
		x_successful = sm.add_constant(X_mat,prepend='False')
		y_successful = y

		#print("Regression with successful trials")
		#print("x1: IBI")
		#print("x2: Pupil Dilation")
		model_glm = sm.Logit(y_successful,x_successful)
		fit_glm = model_glm.fit()
		#print(fit_glm.summary())

		'''
		# Model evaluation using 10-fold cross-validation
		scores_logreg = cross_val_score(LogisticRegression(),X_mat,y,scoring='accuracy',cv=5)
		np.random.shuffle(X_mat)
		mc_sim = 2
		scores_shuffle = np.zeros((mc_sim,5))
		for j in range(mc_sim):
			scores_shuffle[j,:] = cross_val_score(LogisticRegression(),X_mat,y,scoring='accuracy',cv=5)
		scores_logreg_shuffle = np.nanmean(scores_shuffle, axis = 0)
		#print("CV scores for decoding over %s :" %(trial_or_time), scores_logreg)
		#print("Avg CV score:", scores_logreg.mean())
		#print("Avg shuffle score:", scores_logreg_shuffle.mean())
		
		return scores_logreg, scores_logreg_shuffle, X_mat
		'''
		return fit_glm

	def ModulationDepth(self, trial_or_time, **kwargs):
		'''
		A measure of how much the Stress and Treatment block data has changed from the baseline block. 
		The current measure of modulation depth is: (data_block - avg(data_base))/avg(data_base).

		Inputs:
		- trial_or_time: string; can be either 'time' or 'trial', depending on which data you want to plot
		- show_fig: Boolean; indicate whether you want the plot to be shown
		- save_fig: Boolean; indicate whether you want the plot to be saved

		Outputs:
		- ibi_md_stress: array; modulation depth values for ibi during stress block
		- pupil_md_stress: array; modulation depth values for pupil during stress block
		- ibi_md_treat: array; modulation depth values for ibi during treatment block
		- pupil_md_treat: array; modulation depth values for pupil during treatment block
		'''
		# Defining the optional input parameters to the method
		show_fig = kwargs.get('show_fig', False)				# variable that determines whether plot is shown, default is True
		save_fig = kwargs.get('save_fig', False)				# variable that determines whether plot is saved, default is True

		# Pull out the relevant data
		if trial_or_time == 'trial':
			ibi_base = self.ibi_trials_base
			ibi_stress = self.ibi_trials_stress
			ibi_treat = self.ibi_trials_treat
			pupil_base = self.pupil_trials_base
			pupil_stress = self.pupil_trials_stress
			pupil_treat = self.pupil_trials_treat
		else:
			ibi_base = self.ibi_time_base
			ibi_stress = self.ibi_time_stress
			ibi_treat = self.ibi_time_treat
			pupil_base = self.pupil_time_base
			pupil_stress = self.pupil_time_stress
			pupil_treat = self.pupil_time_treat

		# Compute the average values from the baseline block
		ibi_base_mean = np.nanmean(ibi_base)
		pupil_base_mean = np.nanmean(pupil_base)

		# Compute the modulation depth
		ibi_md_stress = (ibi_stress - ibi_base_mean)/ibi_base_mean
		pupil_md_stress = (pupil_stress - pupil_base_mean)/pupil_base_mean
		ibi_md_treat = (ibi_treat - ibi_base_mean)/ibi_base_mean
		pupil_md_treat = (pupil_treat - pupil_base_mean)/pupil_base_mean

		# Plot data
		plt.figure()
		plt.subplot(2,2,1)
		plt.plot(ibi_md_stress,'r')
		plt.title('ibi')
		plt.subplot(2,2,2)
		plt.plot(pupil_md_stress,'r')
		plt.title('pupil')
		plt.subplot(2,2,3)
		plt.plot(ibi_md_treat,'b')
		plt.subplot(2,2,4)
		plt.plot(pupil_md_treat,'b')

		if show_fig:
			plt.show()
		if save_fig:
			plt.savefig('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Plots/'+self.filename+'_ModulationDepth.svg')

		return ibi_md_stress, pupil_md_stress, ibi_md_treat, pupil_md_treat

	def IBIvsPup_ScatterPlot(self, trial_or_time, **kwargs):
		'''
		Method to plot the scatter plot of IBI vs PD values, along with their covariance ellipse.
		All blocks are plotted simulataneously.
		Each point in the scatter plot corresponds to data from either (1) beginning of a trial or (2) a time chunk.

		Inputs:
		- trial_or_time: string; can be either 'time' or 'trial', depending on which data you want to plot
		- show_fig: Boolean; indicate whether you want the plot to be shown
		- save_fig: Boolean; indicate whether you want the plot to be saved
		'''
		# Defining the optional input parameters to the method
		show_fig = kwargs.get('show_fig', True)				# variable that determines whether plot is shown, default is True
		save_fig = kwargs.get('save_fig', True)				# variable that determines whether plot is saved, default is True
		

		# Pull out the relevant data
		if trial_or_time == 'trial':
			ibi_base = self.ibi_trials_base
			ibi_stress = self.ibi_trials_stress
			ibi_treat = self.ibi_trials_treat
			pupil_base = self.pupil_trials_base
			pupil_stress = self.pupil_trials_stress
			pupil_treat = self.pupil_trials_treat
		else:
			ibi_base = self.ibi_time_base
			ibi_stress = self.ibi_time_stress
			ibi_treat = self.ibi_time_treat
			pupil_base = self.pupil_time_base
			pupil_stress = self.pupil_time_stress
			pupil_treat = self.pupil_time_treat

		# Compute covariance of IBI and PD data for plot

		inds_ibi_base = np.ravel(np.nonzero(~np.isnan(ibi_base)))
		inds_pupil_base = np.ravel(np.nonzero(~np.isnan(pupil_base)))
		inds_base = np.array([int(ind) for ind in inds_ibi_base if (ind in inds_pupil_base)])
		if len(inds_base)>0:
			points_base = np.array([ibi_base[inds_base],pupil_base[inds_base]])
		else:
			points_base = np.array([ibi_base, pupil_base])

		inds_ibi_stress = np.ravel(np.nonzero(~np.isnan(ibi_stress)))
		inds_pupil_stress = np.ravel(np.nonzero(~np.isnan(pupil_stress)))
		inds_stress = np.array([int(ind) for ind in inds_ibi_stress if (ind in inds_pupil_stress)])
		if len(inds_stress)>0:
			points_stress = np.array([ibi_stress[inds_stress],pupil_stress[inds_stress]])
		else:
			points_stress = np.array([ibi_stress,pupil_stress])
		
		inds_ibi_treat = np.ravel(np.nonzero(~np.isnan(ibi_treat)))
		inds_pupil_treat = np.ravel(np.nonzero(~np.isnan(pupil_treat)))
		inds_treat = np.array([int(ind) for ind in inds_ibi_treat if (ind in inds_pupil_treat)])
		if len(inds_treat):
			points_treat = np.array([ibi_treat[inds_treat],pupil_treat[inds_treat]])
		else:
			points_treat = np.array([ibi_treat,pupil_treat])
		
		cov_base = np.cov(points_base)
		cov_stress = np.cov(points_stress)
		cov_treat = np.cov(points_treat)
		mean_vec_base = [np.nanmean(ibi_base),np.nanmean(pupil_base)]
		mean_vec_stress = [np.nanmean(ibi_stress),np.nanmean(pupil_stress)]
		mean_vec_treat = [np.nanmean(ibi_treat),np.nanmean(pupil_treat)]
		

		# Plot figure
		cmap_stress = mpl.cm.autumn
		cmap_treat = mpl.cm.winter
		cmap_base = mpl.cm.gray

		plt.figure()
		for i in range(0,len(ibi_base)):
			plt.plot(ibi_base[i],pupil_base[i],color=cmap_base(i/float(len(ibi_base))),marker='o',markersize=10)
		plot_cov_ellipse(cov_base,mean_vec_base,fc='k',ec='None',a=0.2)
		for i in range(0,len(ibi_stress)):
		    plt.plot(ibi_stress[i],pupil_stress[i],color=cmap_stress(i/float(len(ibi_stress))),marker='o',markersize=10)
		plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
		for i in range(0,len(ibi_treat)):
			plt.plot(ibi_treat[i],pupil_treat[i],color=cmap_treat(i/float(len(ibi_treat))),marker='o', markersize=10)
		plot_cov_ellipse(cov_treat,mean_vec_treat,fc='b',ec='None',a=0.2)
		
		plt.xlabel('Mean Trial IBI (s)')
		plt.ylabel('Mean Trial PD (AU)')
		plt.title('Data points taken over %s' % (trial_or_time))
		#plt.xlim((0.32,0.52))
		#plt.ylim((1,3))
		sm_base = plt.cm.ScalarMappable(cmap=cmap_base, norm=plt.Normalize(vmin=0, vmax=1))
		# fake up the array of the scalar mappable. Urgh...
		sm_base._A = []
		cbar = plt.colorbar(sm_base,ticks=[0,1], orientation='vertical')
		cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
		sm_stress = plt.cm.ScalarMappable(cmap=cmap_stress, norm=plt.Normalize(vmin=0, vmax=1))
		# fake up the array of the scalar mappable. Urgh...
		sm_stress._A = []
		cbar = plt.colorbar(sm_stress,ticks=[0,1], orientation='vertical')
		cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
		sm_treat = plt.cm.ScalarMappable(cmap=cmap_treat, norm=plt.Normalize(vmin=0, vmax=1))
		# fake up the array of the scalar mappable. Urgh...
		sm_treat._A = []
		cbar = plt.colorbar(sm_treat,ticks=[0,1], orientation='vertical')
		cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
		#plt.ylim((1.75,2.27))
		#plt.xlim((0.30,0.44))

		if show_fig:
			plt.show()
		if save_fig:
			plt.savefig('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Plots/'+self.filename+'_IBIPupilCovariance.svg')

		return

	def TimeSeries_Plots(self, trial_or_time, **kwargs):
		'''
		Method to plot values over time of IBI and PD.
		All blocks are plotted individually.

		Inputs:
		- trial_or_time: string; can be either 'time' or 'trial', depending on which data you want to plot
		- show_fig: Boolean; indicate whether you want the plot to be shown
		- save_fig: Boolean; indicate whether you want the plot to be saved
		'''
		# Defining the optional input parameters to the method
		show_fig = kwargs.get('show_fig', True)				# variable that determines whether plot is shown, default is True
		save_fig = kwargs.get('save_fig', True)				# variable that determines whether plot is saved, default is True
		

		# Pull out the relevant data
		if trial_or_time == 'trial':
			ibi_base = self.ibi_trials_base
			ibi_stress = self.ibi_trials_stress
			ibi_treat = self.ibi_trials_treat
			pupil_base = self.pupil_trials_base
			pupil_stress = self.pupil_trials_stress
			pupil_treat = self.pupil_trials_treat
			x_base = range(len(ibi_base))
			x_stress = range(len(ibi_stress))
			x_treat = range(len(ibi_treat))

		else:
			ibi_base = self.ibi_time_base
			ibi_stress = self.ibi_time_stress
			ibi_treat = self.ibi_time_treat
			pupil_base = self.pupil_time_base
			pupil_stress = self.pupil_time_stress
			pupil_treat = self.pupil_time_treat
			x_base = self.ind_time_base
			print(len(x_base))
			print(len(ibi_base))
			x_stress = self.ind_time_stress
			print(len(x_stress))
			print(len(ibi_stress))
			x_treat = self.ind_time_treat
			print(len(x_treat))
			print(len(ibi_treat))

		# Plot data over time
		plt.figure()
		plt.subplot(2,3,1)
		plt.plot(x_base, ibi_base,'r')
		plt.ylabel('IBI (s)')
		plt.title('Baseline')
		plt.subplot(2,3,4)
		plt.plot(x_base, pupil_base, 'g')
		plt.ylabel('PD (a.u.)')
		plt.subplot(2,3,2)
		plt.plot(x_stress, ibi_stress, 'y')
		plt.title('Stress')
		plt.subplot(2,3,5)
		plt.plot(x_stress, pupil_stress, 'b')
		plt.subplot(2,3,3)
		plt.plot(x_treat, ibi_treat, 'c')
		plt.title('Treatment')
		plt.xlabel('%s' % (trial_or_time))
		
		plt.subplot(2,3,6)
		plt.plot(x_treat, pupil_treat, 'm')
		plt.xlabel('%s' % (trial_or_time))
		

		if show_fig:
			plt.show()
		if save_fig:
			plt.savefig('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Plots/'+self.filename+'_TimeSeries.svg')

		return


class StressTask_PowerFeatureData():
	'''
	Class for operating on pre-computed peripheral physiology data. 
	This is for the stress task which has 3 blocks: baseline, stress, and treatment.
	Peripheral physiology data is pre-computed over time, per 5 s time windows,
	and stored in a dictionary. Each dictionary key is a timepoint, and each element is a 
	channel x power matrix.
	'''
	def __init__(self, power_mat_filenames, num_channels):
		# Read data from .mat file
		self.filename = power_mat_filenames[0][59:-21]
		self.lfp_features_base = spio.loadmat(power_mat_filenames[0])
		self.lfp_features_stress = spio.loadmat(power_mat_filenames[1])
		self.lfp_features_treat = spio.loadmat(power_mat_filenames[2])

		self.num_channels = num_channels

		# Pull out per-trial data
		self.num_time_points_base = len(self.lfp_features_base.keys()) - 3		# account for the fact that there are 3 useful keys: __header__, __version__, and __globals__
		self.num_time_points_stress = len(self.lfp_features_stress.keys()) - 3
		self.num_time_points_treat = len(self.lfp_features_treat.keys()) - 3

		self.power_features_base = np.empty([self.num_time_points_base, num_channels, 4])		# there are 157 channels and 4 power bands over which power was computed
		self.power_features_stress = np.empty([self.num_time_points_stress, num_channels, 4])
		self.power_features_treat = np.empty([self.num_time_points_treat, num_channels, 4])

		for i in range(self.num_time_points_base):
			self.power_features_base[i,:,:] = self.lfp_features_base[str(i)]

		for j in range(self.num_time_points_stress):
			self.power_features_stress[j,:,:] = self.lfp_features_stress[str(j)]

		for k in range(self.num_time_points_treat):
			self.power_features_treat[k,:,:] = self.lfp_features_treat[str(k)]

		return

	def logistic_decoding_with_power(self, **kwargs):
		'''
		Train logistic decoder with power values from baseline and stress blocks. Use data from the end of the stress block such that the 
		number of training points from each block is equal. Then, test decoder on treatment blocks.

		Inputs:
		- bands: array; array indicating which power bands to make figure for according to the following convention: 0 = [8,13] Hz,1 = [13,30] Hz,2 = [40,70] Hz,3 = [70,200] Hz

		Outputs:
		- 
		'''

		# Defining the optional input parameters to the method
		
		power1 = self.power_features_base			# result should be timepoints x channels matrix x band that is transposed, so channels x timepoints
		power2 = self.power_features_stress
		power3 = self.power_features_treat

		power1 = power1.reshape(power1.shape[0], power1.shape[1]*power1.shape[2])
		power2 = power2.reshape(power2.shape[0], power2.shape[1]*power2.shape[2])
		power3 = power3.reshape(power3.shape[0], power3.shape[1]*power3.shape[2])

		num_timepoints_per_block = power1.shape[0]
		power_train = np.vstack((power1,power2[-num_timepoints_per_block:,:]))		# should be (2*timepoints from baseline) x (channels x bands) 

		power_all = np.vstack((power_train, power3))
		power_all = power_all.T

		power_train = power_train.T
		# z score by power from all blocks
		power_train_mean = np.dot(np.nanmean(power_train,axis = 1).reshape((power_train.shape[0],1)), np.ones((1,power_train.shape[1])))
		power_train_std = np.dot(np.nanstd(power_train,axis = 1).reshape((power_train.shape[0],1)), np.ones((1,power_train.shape[1])))
		power_train_zscore = (power_train - power_train_mean)/power_train_std

		power_all_mean = np.dot(np.nanmean(power_train,axis = 1).reshape((power_all.shape[0],1)), np.ones((1,power_all.shape[1])))
		power_all_std = np.dot(np.nanmean(power_train,axis = 1).reshape((power_all.shape[0],1)), np.ones((1,power_all.shape[1])))
		power_all_zscore = (power_all - power_all_mean)/power_all_std


		power_train_zscore = power_train_zscore.T
		power_all_zscore = power_all_zscore.T

		#power_train_zscore = power_all_zscore[:2*num_timepoints_per_block,:]
		
		# find any rows with nan entries in phys data matrix
		inds = np.ravel(np.nonzero(1 - np.sum(np.isnan(power_train_zscore),axis = 1)))
		inds_all = np.ravel(np.nonzero(1 - np.sum(np.isnan(power_all_zscore), axis = 1)))

		y = np.zeros(2*num_timepoints_per_block)
		y[num_timepoints_per_block:] = 1

		# only keep data from rows that don't have nans
		X_mat = power_train_zscore[inds,:]
		y1 = y[inds]

		W = LDAforFeatureSelection(X_mat,y1,'x',1)
		
		# Transform the samples onto the new subspace
		X_lda = power_all_zscore[inds_all,:].dot(W)
		y = np.append(y, 2*np.ones(power3.shape[0]))
		y2 = y[inds_all]

		plot_step_lda(X_lda,y2,['Base','Stress','Treat'])
		#plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename+'_b'+str(block_num)+'_LDA.svg')
		#plt.close()
		plt.show()

		print("LDA using Power Features:")
		clf_all = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto')
		clf_all.fit(X_mat, y1)
		scores = cross_val_score(LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto'),X_mat,y1,scoring='accuracy',cv=10)
		print("CV (10-fold) scores:", scores)
		print("Avg CV score:", scores.mean())
		pred_treat = clf_all.predict(power_all_zscore[inds_all[2*num_timepoints_per_block:],:])
		print("Fraction treatment classified as stress:", pred_treat.mean())
		
		# Logistic regression as was done with closed-loop stim
		'''

		#x_successful = np.vstack((np.append(ibi_reg_mean_adj, ibi_stress_mean_adj), np.append(pupil_reg_mean_adj, pupil_stress_mean_adj)))
		#x_successful = np.transpose(x_successful)
		x_successful = sm.add_constant(X_mat,prepend='False')
		y_successful = y

		#print("Regression with successful trials")
		#print("x1: IBI")
		#print("x2: Pupil Dilation")
		model_glm = sm.Logit(y_successful,x_successful)
		fit_glm = model_glm.fit()
		#print(fit_glm.summary())
		'''
		return 


	def power_heatmap_overtime(self, **kwargs):
		'''
		Plot power over time for all three epochs, with channels organized according to channel groups if desired.
		Plot heatmap of power per band or only for indicated band.

		Inputs:
		- t_bin_size: float; size of time bin used for computing power
		- t_overlap: float; size of overlap in time bins used for computing power
		- channel_groups: liexitst; list of groups of channel numbers for purpose of organizing in plot
		- channel_group_labels: list; list of strings describing each channel grouping
		- bands: array; array indicating which power bands to make figure for according to the following convention: 0 = [8,13] Hz,1 = [13,30] Hz,2 = [40,70] Hz,3 = [70,200] Hz

		Outputs:
		- 
		'''

		# Defining the optional input parameters to the method
		t_bin_size = kwargs.get('t_bin_size', 5.)							# time bin size for power feature computation, defaults to 5 seconds
		t_overlap = kwargs.get('t_overlap', t_bin_size/2.)					# overlap of time bins for power feature computation, defaults to t_bin_size/2
		channel_groups = kwargs.get('channel_groups', [[np.arange(self.num_channels)]]) 	# default to list of all channels
		bands = kwargs.get('bands', np.arange(4))

		group_size = [len(group) for group in channel_groups]
		group_size = np.cumsum(group_size)

		power_bands = [[8,13],[13,30],[40,70],[70,200]]

		event_indices_base = np.arange(0, self.num_time_points_base*(t_bin_size - t_overlap), t_bin_size - t_overlap)	# times in seconds
		event_indices_base = event_indices_base/60.																		# times in minutes

		plt.figure(figsize = (30,20))
		# Compute for baseline
		for b,band in enumerate(bands):
			for g,group in enumerate(channel_groups):
				power1 = self.power_features_base[:,group,band].T			# result should be timepoints x channels matrix that is transposed, so channels x timepoints
				power2 = self.power_features_stress[:,group,band].T
				power3 = self.power_features_treat[:,group,band].T
				if g > 0:
					power_base = np.vstack((power_base, power1))				# result is channel groups stacked on top of each other, so channel groups x timepoints
					power_stress = np.vstack((power_stress, power2))
					power_treat = np.vstack((power_treat, power3))
				else:
					power_base = power1
					power_stress = power2
					power_treat = power3

			power_base = np.squeeze(power_base)
			power_stress = np.squeeze(power_stress)
			power_treat = np.squeeze(power_treat)

			power_all = np.hstack((power_base, power_stress))
			power_all = np.hstack((power_all, power_treat))
			power_all_mean = np.dot(np.nanmean(power_all,axis = 1).reshape((power_all.shape[0],1)), np.ones((1,power_all.shape[1])))
			power_all_std = np.dot(np.nanstd(power_all,axis = 1).reshape((power_all.shape[0],1)), np.ones((1,power_all.shape[1])))
			power_all_zscore = (power_all - power_all_mean)/power_all_std
			# Z-score
			
			power_base_mean = np.dot(np.nanmean(power_base,axis = 1).reshape((power_base.shape[0],1)), np.ones((1,power_base.shape[1])))
			power_base_std = np.dot(np.nanstd(power_base,axis = 1).reshape((power_base.shape[0],1)), np.ones((1,power_base.shape[1])))
			power_base_zscore = (power_base - power_base_mean)/power_base_std
			
			power_stress_mean = np.dot(np.nanmean(power_stress,axis = 1).reshape((power_stress.shape[0],1)), np.ones((1,power_stress.shape[1])))
			power_stress_std = np.dot(np.nanstd(power_stress,axis = 1).reshape((power_stress.shape[0],1)), np.ones((1,power_stress.shape[1])))
			power_stress_zscore = (power_stress - power_stress_mean)/power_stress_std
			
			power_treat_mean = np.dot(np.nanmean(power_treat,axis = 1).reshape((power_treat.shape[0],1)), np.ones((1,power_treat.shape[1])))
			power_treat_std = np.dot(np.nanstd(power_treat,axis = 1).reshape((power_treat.shape[0],1)), np.ones((1,power_treat.shape[1])))
			power_treat_zscore = (power_treat - power_treat_mean)/power_treat_std
			
			'''
			plt.figure(1)
			plt.subplot(len(bands),3,b*3 + 1)
			plt.imshow(power_base_zscore,interpolation="sinc", aspect='auto', cmap="rainbow", vmin = -2.5, vmax = 5)
			#plt.xticks(range(0,49,12),[-1,"0\nGo cue",1,2,3],rotation=0)
			for i in range(len(channel_groups)):
				plt.axhline(group_size[i], color='k')
			plt.yticks([])
			plt.ylabel("Channels")
			plt.xlabel("Time (min)")
			plt.title("Baseline power: %i - %i Hz" % (power_bands[band][0], power_bands[band][1]))
			plt.grid(None)
			plt.colorbar()

			plt.subplot(len(bands),3,b*3 + 2)
			plt.imshow(power_stress_zscore,interpolation="lanczos", aspect='auto', cmap="rainbow", vmin = -2.5, vmax = 5)
			#plt.xticks(range(0,49,12),[-1,"0\nGo cue",1,2,3],rotation=0)
			for i in range(len(channel_groups))[:-1]:
				plt.axhline(group_size[i], color='k')
			plt.yticks([])
			plt.ylabel("Channels")
			plt.xlabel("Time (min)")
			plt.title("Stress power: %i - %i Hz" % (power_bands[band][0], power_bands[band][1]))
			plt.grid(None)
			plt.colorbar()

			plt.subplot(len(bands),3,b*3 + 3)
			plt.imshow(power_treat_zscore,interpolation="gaussian", aspect='auto', cmap="rainbow", vmin = -2.5, vmax = 5)
			#plt.xticks(range(0,49,12),[-1,"0\nGo cue",1,2,3],rotation=0)
			for i in range(len(channel_groups))[:-1]:
				plt.axhline(group_size[i], color='k')
			plt.yticks([])
			plt.ylabel("Channels")
			plt.xlabel("Time (min)")
			plt.title("Treatment power: %i - %i Hz" % (power_bands[band][0], power_bands[band][1]))
			plt.grid(None)
			plt.colorbar()
			'''
			plt.figure(1)
			plt.subplot(len(bands),1,b+1)
			plt.imshow(power_all_zscore,interpolation="gaussian", aspect='auto', cmap="rainbow", vmin = -2.5, vmax = 5)
			#plt.xticks(range(0,49,12),[-1,"0\nGo cue",1,2,3],rotation=0)
			for i in range(len(channel_groups))[:-1]:
				plt.axhline(group_size[i], color='k')
			plt.axvline(power_base_zscore.shape[1], color = 'k')
			plt.axvline(power_base_zscore.shape[1] + power_stress_zscore.shape[1], color = 'k')
			xticks = np.arange(0, self.num_time_points_base + self.num_time_points_stress + self.num_time_points_treat, 10*60/(t_bin_size - t_overlap))
			xticklabels = np.arange(0,(self.num_time_points_base + self.num_time_points_stress + self.num_time_points_treat)*60/(t_bin_size - t_overlap),10)
			plt.xticks(xticks,xticklabels)
			plt.yticks([])
			plt.ylabel("Channels")
			plt.xlabel("Time (min)")
			plt.title("Treatment power: %i - %i Hz" % (power_bands[band][0], power_bands[band][1]))
			plt.grid(None)
			plt.colorbar()
		
		plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + self.filename + "_power_heatmap_overtime.svg")
		plt.close()		

		return group_size, power_all
	
	def power_average_overtime(self, **kwargs):
		'''
		Plot power over time for all three epochs, with channels organized according to channel groups if desired.
		Plot average of power per band where averaging is across all channels within the indicated group.

		Inputs:
		- t_bin_size: float; size of time bin used for computing power
		- t_overlap: float; size of overlap in time bins used for computing power
		- channel_groups: liexitst; list of groups of channel numbers for purpose of organizing in plot
		- channel_group_labels: list; list of strings describing each channel grouping
		- bands: array; array indicating which power bands to make figure for according to the following convention: 0 = [8,13] Hz,1 = [13,30] Hz,2 = [40,70] Hz,3 = [70,200] Hz

		Outputs:
		- 
		'''

		# Defining the optional input parameters to the method
		t_bin_size = kwargs.get('t_bin_size', 5.)							# time bin size for power feature computation, defaults to 5 seconds
		t_overlap = kwargs.get('t_overlap', t_bin_size/2.)					# overlap of time bins for power feature computation, defaults to t_bin_size/2
		channel_groups = kwargs.get('channel_groups', [[np.arange(self.num_channels)]]) 	# default to list of all channels
		channel_group_labels = kwargs.get('channel_group_labels', ['all'])
		bands = kwargs.get('bands', np.arange(4))

		group_size = [len(group) for group in channel_groups]
		group_size = np.cumsum(group_size)

		power_bands = [[8,13],[13,30],[40,70],[70,200]]

		event_indices = np.arange(0, (self.num_time_points_base + self.num_time_points_stress + self.num_time_points_treat)*(t_bin_size - t_overlap), t_bin_size - t_overlap)	# times in seconds
		event_indices = event_indices/60.

		c = signal.gaussian(99, 1)																		# times in minutes
		colors = ['b','r','g','m']

		plt.figure(figsize = (30,20))
		# Compute for baseline
		for b,band in enumerate(bands):
			for g,group in enumerate(channel_groups):
				power_base = self.power_features_base[:,group,band].T			# result should be timepoints x channels matrix that is transposed, so channels x timepoints
				power_stress = self.power_features_stress[:,group,band].T
				power_treat = self.power_features_treat[:,group,band].T
			
				power_base = np.squeeze(power_base)
				power_stress = np.squeeze(power_stress)
				power_treat = np.squeeze(power_treat)

				power_all = np.hstack((power_base, power_stress))
				power_all = np.hstack((power_all, power_treat))
				power_all_mean = np.dot(np.nanmean(power_all,axis = 1).reshape((power_all.shape[0],1)), np.ones((1,power_all.shape[1])))
				power_all_std = np.dot(np.nanstd(power_all,axis = 1).reshape((power_all.shape[0],1)), np.ones((1,power_all.shape[1])))
				power_all_zscore = (power_all - power_all_mean)/power_all_std

				power_all_avg = np.nanmean(power_all_zscore,axis = 0)		# result should be length timepoints
				power_all_sem = np.nanstd(power_all_zscore,axis = 0)/np.sqrt(self.num_channels)

				# Lowpass filter the average signal to smooth it out
				power_all_avg = filters.convolve1d(power_all_avg, c/c.sum())

				plt.figure(1)
				plt.subplot(len(bands),len(channel_groups),b*len(channel_groups) + g +  1)
				plt.plot(event_indices,power_all_avg,color = colors[g], label = '%s' % (channel_group_labels[g]))
				plt.fill_between(event_indices,power_all_avg - power_all_sem,power_all_avg + power_all_sem, facecolor = colors[g], alpha = 0.5)
				plt.xlabel("Time (min)")
				plt.ylabel("Average z-scored power")
				plt.title("Band: %i - %i Hz, Channel group: %s" % (power_bands[band][0], power_bands[band][1], channel_group_labels[g]))
				plt.legend()
		
		plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + self.filename + "_power_average_overtime.svg")
		plt.close()	

		return

	def power_average_per_condition(self, **kwargs):
		'''
		Plot power over time for all three epochs, with channels organized according to channel groups if desired.
		Plot average of power per band where averaging is across all channels within the indicated group.

		Inputs:
		- t_bin_size: float; size of time bin used for computing power
		- t_overlap: float; size of overlap in time bins used for computing power
		- channel_groups: liexitst; list of groups of channel numbers for purpose of organizing in plot
		- channel_group_labels: list; list of strings describing each channel grouping
		- bands: array; array indicating which power bands to make figure for according to the following convention: 0 = [8,13] Hz,1 = [13,30] Hz,2 = [40,70] Hz,3 = [70,200] Hz
		- normalization: Boolean; indicate whether normalization with all power should be performed

		Outputs:
		- 
		'''

		# Defining the optional input parameters to the method
		channel_groups = kwargs.get('channel_groups', [[np.arange(self.num_channels)]]) 	# default to list of all channels
		channel_group_labels = kwargs.get('channel_group_labels', ['all'])
		bands = kwargs.get('bands', np.arange(4))
		normalization = kwargs.get('normalization', True)

		group_size = [len(group) for group in channel_groups]
		group_size = np.cumsum(group_size)

		power_bands = [[8,13],[13,30],[40,70],[70,200]]

		colors = ['b','r','g','m']

		plt.figure(figsize = (30,20))

		average_power = np.empty((len(channel_groups),len(bands), 3))				# size is channel groups x num frequency bands x behavioral conditions
		sem_power = np.empty((len(bands), 3))				# size if frequency bands x behavioral condition
		
		for g,group in enumerate(channel_groups):
			plt.figure(1)
			ind = np.arange(3)
			plt.subplot(len(channel_groups),1,g+1)
			for b,band in enumerate(bands):
				power_base = self.power_features_base[:,group,band].T			# result should be timepoints x channels matrix that is transposed, so channels x timepoints
				power_stress = self.power_features_stress[:,group,band].T
				power_treat = self.power_features_treat[:,group,band].T
			
				power_base = np.squeeze(power_base)
				power_stress = np.squeeze(power_stress)
				power_treat = np.squeeze(power_treat)

				#Z-score
				power_all = np.hstack((power_base, power_stress))
				power_all = np.hstack((power_all, power_treat))

				power_all_base_mean = np.dot(np.nanmean(power_all,axis = 1).reshape((power_base.shape[0],1)), np.ones((1,power_base.shape[1])))
				power_all_base_std = np.dot(np.nanstd(power_all,axis = 1).reshape((power_base.shape[0],1)), np.ones((1,power_base.shape[1])))
				power_base_zscore = (power_base - power_all_base_mean)/power_all_base_std
				

				power_all_stress_mean = np.dot(np.nanmean(power_all,axis = 1).reshape((power_stress.shape[0],1)), np.ones((1,power_stress.shape[1])))
				power_all_stress_std = np.dot(np.nanstd(power_all,axis = 1).reshape((power_stress.shape[0],1)), np.ones((1,power_stress.shape[1])))
				power_stress_zscore = (power_stress - power_all_stress_mean)/power_all_stress_std
				

				power_all_treat_mean = np.dot(np.nanmean(power_all,axis = 1).reshape((power_treat.shape[0],1)), np.ones((1,power_treat.shape[1])))
				power_all_treat_std = np.dot(np.nanstd(power_all,axis = 1).reshape((power_treat.shape[0],1)), np.ones((1,power_treat.shape[1])))
				power_treat_zscore = (power_treat - power_all_treat_mean)/power_all_treat_std

				if normalization==False:
					power_base_zscore = power_base 
					power_stress_zscore = power_stress
					power_treat_zscore = power_treat 
				

				average_power[g,b,0] = np.nanmean(power_base_zscore)
				average_power[g,b,1] = np.nanmean(power_stress_zscore)
				average_power[g,b,2] = np.nanmean(power_treat_zscore)

				sem_power[b,0] = np.nanstd(power_base_zscore)/np.sqrt(power_base.size)
				sem_power[b,1] = np.nanstd(power_stress_zscore)/np.sqrt(power_stress.size)
				sem_power[b,2] = np.nanstd(power_treat_zscore)/np.sqrt(power_treat.size)
				
			
				plt.errorbar(ind, average_power[g,b,:],yerr = sem_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[band][0], power_bands[band][1]))
			plt.title("Channel group: %s" % (channel_group_labels[g]))
			plt.legend()
		
		#plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + self.filename + "_power_average_per_condition.svg")
		plt.close()	
		#plt.show()
		return average_power


def StressTaskAnalysis_ComputePowerFeatures(hdf_filenames, filenames, block_nums, TDT_tank, power_bands, **kwargs):

	"""
	Inputs: 
	- hdf_filenames: list; list of three hdf filenames corresponding to the blocks of behavior in the following order: baseline, stress, treatment
	- names: list; list of the parent folders where the neural data is stored for each of the blocks in the following order: baseline, stress, treatment
	- block_nums: list; list of integers with the corresponding block numbers of the recording within each parent folder
	- power_bands: list; list of power bands, e.g. [[4,8], [13,30]] is a list defining two frequency bands: 4 - 8 Hz, and 13 - 30 Hz 
	- t_bin_size: float; length of time bins to chunk time into (in seconds) to compute features over, default is 5 seconds
	- t_overlap: float; length of time (in seconds) that time bins should overlap, default is t_bin_size/2 seconds
	- len_window: float; length of time (in seconds) to chunk time into to compute peripheral physiology values, default is 5 seconds
	- need_syncHDF: Boolean; indicates if syncHDF file should be generated with this script

	Example inputs: 
	- hdf_filenames = ['mari20160409_07_te1961.hdf', 'mario20160409_08_te1962.hdf', 'mari20160409_10_te1964.hdf']
	- filenames = ['Mario20160409', 'Mario20160409', 'Mario20160409-2'
	- block_nums = [1,2,1]
	"""
	
	'''
	Define method parameters.
	'''

	# Defining the optional input parameters to the method
	t_bin_size = kwargs.get('t_bin_size', 5.)				# time bin size for power feature computation, defaults to 5 seconds
	t_overlap = kwargs.get('t_overlap', t_bin_size/2.)		# overlap of time bins for power feature computation, defaults to t_bin_size/2
	len_window = kwargs.get('len_window', 5.)				# time bin size (non-overlapping) for peripheral physiology
	need_syncHDF = kwargs.get('need_syncHDF', False)

	# Defining naming conventions and other variables
	print(filenames[0])
	#TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Luigi - Neural Data/'+filename
	TDT_tanks = [(TDT_tank + name) for name in filenames]
	hdf_locations = [('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/'+hdf) for hdf in hdf_filenames]
	pf_location = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/PowerFeatures/'
	pf_filename_base = pf_location + filenames[0]+'_b'+str(block_nums[0])+'_PowerFeatures.mat'
	pf_filename_stress = pf_location + filenames[1] + '_b'+str(block_nums[1])+'_PowerFeatures.mat'
	pf_filename_treat = pf_location + filenames[2] + '_b'+str(block_nums[2]) + '_PowerFeatures.mat'
	phys_filename_base = pf_location + filenames[0]+'_b'+str(block_nums[0])+'_PhysFeatures.mat'
	phys_filename_stress = pf_location + filenames[1] + '_b'+str(block_nums[1])+'_PhysFeatures.mat'
	phys_filename_treat = pf_location + filenames[2] + '_b'+str(block_nums[2]) + '_PhysFeatures.mat'

	mat_filename_base = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/syncHDF/' + filenames[0]+'_b'+str(block_nums[0])+'_syncHDF.mat'
	mat_filename_stress = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/syncHDF/' + filenames[1]+'_b'+str(block_nums[1])+'_syncHDF.mat'
	mat_filename_treat = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/syncHDF/' + filenames[2]+'_b'+str(block_nums[2])+'_syncHDF.mat'
	
	lfp_channels = np.arange(1,161,dtype = int)					# channels 1 - 160
	lfp_channels = np.delete(lfp_channels, [129, 131, 145])		# channels 129, 131, and 145 are open

	bands = [[8,13],[13,30],[40,70],[70,200]]
	# bands = [[8,18], [14-21],[101,298]]

	'''
	Compute power and physiological features. If pre-computed and previuosly saved, do not re-compute. 
	'''

	if os.path.exists(pf_filename_base)&os.path.exists(pf_filename_stress)&os.path.exists(pf_filename_treat)&os.path.exists(phys_filename_base)&os.path.exists(phys_filename_stress)&os.path.exists(phys_filename_treat):
		print("Power features previously computed.")
		
		
	else:
	
		'''
		Load behavior data.
		'''
		base_behavior = StressBehaviorWithDrugs_CenterOut(hdf_locations[0])
		stress_behavior = StressBehaviorWithDrugs_CenterOut(hdf_locations[1])
		treat_behavior = StressBehaviorWithDrugs_CenterOut(hdf_locations[2])

		lfp_ind_hold_center_states_base_trials = base_behavior.get_state_TDT_LFPvalues(base_behavior.ind_hold_center_states,mat_filename_base)[0]      	# aligns to end of hold
		lfp_ind_hold_center_states_stress_trials = stress_behavior.get_state_TDT_LFPvalues(stress_behavior.ind_hold_center_states,mat_filename_stress)[0]
		lfp_ind_hold_center_states_treat_trials = treat_behavior.get_state_TDT_LFPvalues(treat_behavior.ind_hold_center_states,mat_filename_treat)[0]

		print("Behavior data loaded.")

		'''
		Load all neural, pupil, and pulse data. FYI, Luigi's data is always 3 blocks in the same tank, whereas Mario's baseline and stress blocks are in one tank and treatment block is in another tank
		'''
		print("Loading TDT data.")
		print(need_syncHDF)
	
		r = io.TdtIO(TDT_tanks[0])
		bl = r.read_block(lazy=False,cascade=True)
		print("First tank read.")

		if need_syncHDF:
			print("Creating syncHDF file")
			create_syncHDF_TDTloaded(filenames[0], TDT_tank[0], bl)

		lfp_base = dict()
		lfp_stress = dict()
		lfp_treat = dict()
		
		# Get all TDT data from baseline block (Block 1)
		block_num = block_nums[0]
		for sig in bl.segments[block_num-1].analogsignals:
			if (sig.name == "b'PupD' 1"):
				pupil_data_base = np.array(sig)
				pupil_samprate = sig.sampling_rate.item()
			if (sig.name == "b'HrtR' 1"):
				pulse_data_base = np.array(sig)
				pulse_samprate = sig.sampling_rate.item()
			if (sig.name[0:7] == "b'LFP1'"):
				channel = sig.channel_index
				if (channel in lfp_channels)&(channel < 97):
					lfp_samprate = sig.sampling_rate.item()
					lfp_base[channel] = np.array(sig)
			if (sig.name[0:7] == "b'LFP2'"):
				channel = sig.channel_index
				if (channel + 96) in lfp_channels:
					channel_name = channel + 96
					lfp_base[channel_name] = np.array(sig)

		# Get all TDT data from stress block (Block 2)
		block_num = block_nums[1]
		for sig in bl.segments[block_num-1].analogsignals:
			if (sig.name == "b'PupD' 1"):
				pupil_data_stress = np.array(sig)
				pupil_samprate = sig.sampling_rate.item()
			if (sig.name == "b'HrtR' 1"):
				pulse_data_stress = np.array(sig)
				pulse_samprate = sig.sampling_rate.item()
			if (sig.name[0:7] == "b'LFP1'"):
				channel = sig.channel_index
				if (channel in lfp_channels)&(channel < 97):
					lfp_samprate = sig.sampling_rate.item()
					lfp_stress[channel] = np.array(sig)
			if (sig.name[0:7] == "b'LFP2'"):
				channel = sig.channel_index
				if (channel + 96) in lfp_channels:
					channel_name = channel + 96
					lfp_stress[channel_name] = np.array(sig)

		if TDT_tanks[0]==TDT_tanks[2]:				# check if this is Luigi and all data is in the same tank

			# Get all TDT data from treatment block (Block 3)
			block_num = block_nums[2]
			for sig in bl.segments[block_num-1].analogsignals:
				if (sig.name == "b'PupD' 1"):
					pupil_data_treat = np.array(sig)
					pupil_samprate = sig.sampling_rate.item()
				if (sig.name == "b'HrtR' 1"):
					pulse_data_treat = np.array(sig)
					pulse_samprate = sig.sampling_rate.item()
				if (sig.name[0:7] == "b'LFP1'"):
					channel = sig.channel_index
					if (channel in lfp_channels)&(channel < 97):
						lfp_samprate = sig.sampling_rate.item()
						lfp_treat[channel] = np.array(sig)
				if (sig.name[0:7] == "b'LFP2'"):
					channel = sig.channel_index
					if (channel + 96) in lfp_channels:
						channel_name = channel + 96
						lfp_treat[channel_name] = np.array(sig)
		else:
			r1 = io.TdtIO(TDT_tanks[2])
			bl1 = r1.read_block(lazy=False,cascade=True)
			print("Separate tank for treatment block read.")

			if need_syncHDF:
				print("Creating syncHDF file")
				create_syncHDF_TDTloaded(filenames[2], TDT_tank[2], bl)

			# Get all TDT data from treatment block (Block 3)
			block_num = block_nums[2]
			for sig in bl1.segments[block_num-1].analogsignals:
				if (sig.name == "b'PupD' 1"):
					pupil_data_treat = np.array(sig)
					pupil_samprate = sig.sampling_rate.item()
				if (sig.name == "b'HrtR' 1"):
					pulse_data_treat = np.array(sig)
					pulse_samprate = sig.sampling_rate.item()
				if (sig.name[0:7] == "b'LFP1'"):
					channel = sig.channel_index
					if (channel in lfp_channels)&(channel < 97):
						lfp_samprate = sig.sampling_rate.item()
						lfp_treat[channel] = np.array(sig)
				if (sig.name[0:7] == "b'LFP2'"):
					channel = sig.channel_index
					if (channel + 96) in lfp_channels:
						channel_name = channel + 96
						lfp_treat[channel_name] = np.array(sig)


		print("Finished loading TDT data.")


		'''
		Load syncing data for behavior and TDT recording
		'''
		print("Loading syncing data.")

		

		hdf_times_base = dict()
		hdf_times_stress = dict()
		hdf_times_treat = dict()
		sp.io.loadmat(mat_filename_base, hdf_times_base)
		sp.io.loadmat(mat_filename_stress, hdf_times_stress)
		sp.io.loadmat(mat_filename_treat, hdf_times_treat)
		dio_tdt_sample_base = np.ravel(hdf_times_base['tdt_samplenumber'])
		dio_tdt_sample_stress = np.ravel(hdf_times_stress['tdt_samplenumber'])
		dio_tdt_sample_treat = np.ravel(hdf_times_treat['tdt_samplenumber'])
		

		'''
		Process pupil and pulse data
		'''
		
		# Find IBIs and pupil data for all trials.
		samples_pupil_trials_base = int(0.1*pupil_samprate)*np.ones(len(lfp_ind_hold_center_states_base_trials))  # look at first 100 ms
		samples_pupil_trials_stress = int(0.1*pupil_samprate)*np.ones(len(lfp_ind_hold_center_states_stress_trials))  # look at first 100 ms 
		samples_pupil_trials_treat = int(0.1*pupil_samprate)*np.ones(len(lfp_ind_hold_center_states_treat_trials))  # look at first 100 ms

		# pupil is for 100 ms, but pulse is for 1 s
		samples_pulse_trials_base = int(1.*pupil_samprate)*np.ones(len(lfp_ind_hold_center_states_base_trials))  # look at first 100 ms
		samples_pulse_trials_stress = int(1.*pupil_samprate)*np.ones(len(lfp_ind_hold_center_states_stress_trials))  # look at first 100 ms 
		samples_pulse_trials_treat = int(1.*pupil_samprate)*np.ones(len(lfp_ind_hold_center_states_treat_trials))  # look at first 100 ms

		
		ibi_trials_base_mean, ibi_trials_base_std, pupil_trials_base_mean, pupil_trials_base_std, nbins_ibi_trials_base, ibi_trials_base_hist, nbins_pupil_trials_base, pupil_trials_base_hist = getIBIandPuilDilation(pulse_data_base, lfp_ind_hold_center_states_base_trials,samples_pulse_trials_base, pulse_samprate,pupil_data_base, lfp_ind_hold_center_states_base_trials,samples_pupil_trials_base,pupil_samprate)
		ibi_trials_stress_mean, ibi_trials_stress_std, pupil_trials_stress_mean, pupil_trials_stress_std, nbins_ibi_trials_stress, ibi_trials_stress_hist, nbins_pupil_trials_stress, pupil_trials_stress_hist = getIBIandPuilDilation(pulse_data_stress, lfp_ind_hold_center_states_stress_trials,samples_pulse_trials_stress, pulse_samprate,pupil_data_stress, lfp_ind_hold_center_states_stress_trials,samples_pupil_trials_stress,pupil_samprate)
		ibi_trials_treat_mean, ibi_trials_treat_std, pupil_trials_treat_mean, pupil_trials_treat_std, nbins_ibi_trials_treat, ibi_trials_treat_hist, nbins_pupil_trials_treat, pupil_trials_treat_hist = getIBIandPuilDilation(pulse_data_treat, lfp_ind_hold_center_states_treat_trials,samples_pulse_trials_treat, pulse_samprate,pupil_data_treat, lfp_ind_hold_center_states_treat_trials,samples_pupil_trials_treat,pupil_samprate)

		# Find IBIs and pupil data for time epochs
		len_window_samples = np.floor(len_window*pulse_samprate) 	# number of samples in designated time window
		start_sample_base = dio_tdt_sample_base[0]
		stop_sample_base = dio_tdt_sample_base[-1]
		start_sample_stress = dio_tdt_sample_stress[0]
		stop_sample_stress = dio_tdt_sample_stress[-1]
		start_sample_treat = dio_tdt_sample_treat[0]
		stop_sample_treat = dio_tdt_sample_treat[-1]
		
		pulse_ind_time_base = np.arange(start_sample_base, stop_sample_base, len_window_samples)
		pupil_ind_time_base = pulse_ind_time_base
		pulse_ind_time_stress = np.arange(start_sample_stress, stop_sample_stress, len_window_samples)
		pupil_ind_time_stress = pulse_ind_time_stress
		pulse_ind_time_treat = np.arange(start_sample_treat, stop_sample_treat, len_window_samples)
		pupil_ind_time_treat = pulse_ind_time_treat

		samples_pupil_time_base = len_window_samples*np.ones(len(pulse_ind_time_base))
		samples_pupil_time_stress = len_window_samples*np.ones(len(pulse_ind_time_stress))
		samples_pupil_time_treat = len_window_samples*np.ones(len(pulse_ind_time_treat))
		
		ibi_time_base_mean, ibi_time_base_std, pupil_time_base_mean, pupil_time_base_std, nbins_ibi_time_base, ibi_time_base_hist, nbins_pupil_time_base, pupil_time_base_hist = getIBIandPuilDilation(pulse_data_base, pulse_ind_time_base,samples_pupil_time_base, pulse_samprate,pupil_data_base, pupil_ind_time_base,samples_pupil_time_base,pupil_samprate)
		ibi_time_stress_mean, ibi_time_stress_std, pupil_time_stress_mean, pupil_time_stress_std, nbins_ibi_time_stress, ibi_time_stress_hist, nbins_pupil_time_stress, pupil_time_stress_hist = getIBIandPuilDilation(pulse_data_stress, pulse_ind_time_stress,samples_pupil_time_stress, pulse_samprate,pupil_data_stress, pupil_ind_time_stress,samples_pupil_time_stress,pupil_samprate)
		ibi_time_treat_mean, ibi_time_treat_std, pupil_time_treat_mean, pupil_time_treat_std, nbins_ibi_time_treat, ibi_time_treat_hist, nbins_pupil_time_treat, pupil_time_treat_hist = getIBIandPuilDilation(pulse_data_treat, pulse_ind_time_treat,samples_pupil_time_treat, pulse_samprate,pupil_data_treat, pupil_ind_time_treat,samples_pupil_time_treat,pupil_samprate)
		
		'''
		Save peripheral physiology to .mat files
		'''

		phys_features_base = dict()
		phys_features_base['ibi_trials_mean'] = ibi_trials_base_mean
		phys_features_base['ibi_time_mean'] = ibi_time_base_mean
		phys_features_base['pupil_trials_mean'] = pupil_trials_base_mean
		phys_features_base['pupil_time_mean'] = pupil_time_base_mean
		phys_features_base['ind_time'] = pupil_ind_time_base
		sp.io.savemat(phys_filename_base,phys_features_base)

		phys_features_stress = dict()
		phys_features_stress['ibi_trials_mean'] = ibi_trials_stress_mean
		phys_features_stress['ibi_time_mean'] = ibi_time_stress_mean
		phys_features_stress['pupil_trials_mean'] = pupil_trials_stress_mean
		phys_features_stress['pupil_time_mean'] = pupil_time_stress_mean
		phys_features_stress['ind_time'] = pupil_ind_time_stress
		sp.io.savemat(phys_filename_stress,phys_features_stress)

		phys_features_treat = dict()
		phys_features_treat['ibi_trials_mean'] = ibi_trials_treat_mean
		phys_features_treat['ibi_time_mean'] = ibi_time_treat_mean
		phys_features_treat['pupil_trials_mean'] = pupil_trials_treat_mean
		phys_features_treat['pupil_time_mean'] = pupil_time_treat_mean
		phys_features_treat['ind_time'] = pupil_ind_time_treat
		sp.io.savemat(phys_filename_treat,phys_features_treat)
		
		'''
		Process neural data: get power in designated frequency bands per trial.
		Ouputs are dictionarys where each entry (i.e. key) corresponds to one trial/time window, and then
		saved in each entry is a C x K matrix, where C is the number of channels and K is the number of features.
		'''
		
		
		lfp_power_trials_base = []
		lfp_power_time_base = []
		lfp_power_trials_stress = []
		lfp_power_time_stress = []
		lfp_power_trials_treat = []
		lfp_power_time_treat = []
		
		X_trials_base = []
		X_trials_stress = []
		X_trials_treat = []
		
		t_window = [0.4]		# in seconds
		
		'''
		print("Computing features over trials.")
		print("Computing LFP power features for baseline block.")

		lfp_ind_hold_center_states_base_trials = np.reshape(lfp_ind_hold_center_states_base_trials,(len(lfp_ind_hold_center_states_base_trials),1))
		lfp_ind_hold_center_states_stress_trials = np.reshape(lfp_ind_hold_center_states_stress_trials,(len(lfp_ind_hold_center_states_stress_trials),1))
		lfp_ind_hold_center_states_treat_trials = np.reshape(lfp_ind_hold_center_states_treat_trials,(len(lfp_ind_hold_center_states_treat_trials),1))
		

		t = time.time()
		lfp_features_trials_base = computePowerFeatures(lfp_base, lfp_samprate, bands, lfp_ind_hold_center_states_base_trials, t_window)
		print("Computing LFP power features for stress block.")
		lfp_features_trials_stress = computePowerFeatures(lfp_stress, lfp_samprate, bands, lfp_ind_hold_center_states_stress_trials, t_window)
		print("Computing LFP power features for treatment block.")
		lfp_features_trials_treat = computePowerFeatures(lfp_treat, lfp_samprate, bands, lfp_ind_hold_center_states_treat_trials, t_window)
		elapsed = (time.time() - t)/60.
		print("Finished LFP power features over trials: took %f minutes." % (elapsed))
		t_bin_size = 10.			# 10 s time bins
		'''
		print("Computing features over time.")
		print("Computing LFP power features for baseline block.")
		t = time.time()
		lfp_features_time_base, ind_time_base = computePowerFeatures_overTime(lfp_base, lfp_samprate, bands, t_bin_size = t_bin_size, t_overlap = t_bin_size/2., t_start = start_sample_base, t_stop = stop_sample_base)
		print("Computing LFP power features for stress block.")
		lfp_features_time_stress, ind_time_stress = computePowerFeatures_overTime(lfp_stress, lfp_samprate, bands, t_bin_size = t_bin_size, t_overlap = t_bin_size/2., t_start = start_sample_stress, t_stop = stop_sample_stress)
		print("Computing LFP power features for treatment block.")
		lfp_features_time_treat, ind_time_treat = computePowerFeatures_overTime(lfp_treat, lfp_samprate, bands, t_bin_size = t_bin_size, t_overlap = t_bin_size/2., t_start = start_sample_treat, t_stop = stop_sample_treat)
		elapsed = (time.time() - t)/60.
		print("Finished LFP power features over time: took %f minutes." % (elapsed))
		
		'''
		Save power features to .mat files
		'''
		'''
		lfp_features_base = dict()
		lfp_features_base['trials'] = lfp_features_trials_base
		lfp_features_base['time'] = lfp_features_time_base
		lfp_features_base['ind_time'] = ind_time_base
		lfp_features_stress = dict()
		lfp_features_stress['trials'] = lfp_features_trials_stress
		lfp_features_stress['time'] = lfp_features_time_stress
		lfp_features_stress['ind_time'] = ind_time_stress
		lfp_features_treat = dict()
		lfp_features_treat['trials'] = lfp_features_trials_treat
		lfp_features_treat['time'] = lfp_features_time_treat
		lfp_features_treat['ind_time'] = ind_time_treat
		'''
		#sp.io.savemat(pf_filename_base,lfp_features_base)
		sp.io.savemat(pf_filename_base,lfp_features_time_base)
		#sp.io.savemat(pf_filename_stress,lfp_features_stress)
		sp.io.savemat(pf_filename_stress, lfp_features_time_stress)
		#sp.io.savemat(pf_filename_treat,lfp_features_treat)
		sp.io.savemat(pf_filename_treat, lfp_features_time_treat)
		
	
	return 