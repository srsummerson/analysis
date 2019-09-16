import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
import sys
import statsmodels.api as sm
from neo import io
import scipy.io as spio
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse, highpassFilterData
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD, computePowerFeatures, computeAllCoherenceFeatures, computePowerFeatures_overTime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import os.path
import time

from StressTaskBehavior import StressBehaviorWithDrugs_CenterOut

"""
Main file for analyzing Luigi and Mario's data from the experiments comparing closed-loop stimulation
with benzodiazepines in terms of the anxiolytic response, where beta-carbolines were used to generate
an anxiogenic response. 

To do:
- finishing generating all syncHDF files
- fix files that couldn't be process with syncHDF
- LPF the pupil and heart rate data to get rid of large + fast fluctuations 
- test compute power features code
- test class for phys data
- test class for power data

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
		points_base = np.array([ibi_base,pupil_base])
		points_stress = np.array([ibi_stress,pupil_stress])
		points_treat = np.array([ibi_treat,pupil_treat])
		cov_base = np.cov(points_base)
		cov_stress = np.cov(points_stress)
		cov_treat = np.cov(points_treat)
		mean_vec_base = [np.nanmean(ibi_base),np.nanmean(pupil_base)]
		mean_vec_stress = [np.nanmean(ibi_stress),np.nanmean(pupil_stress)]
		mean_vec_treat = [np.nanmean(ibi_treat),np.nanmean(pupil_treat)]
		

		# Plot figure
		cmap_stress = mpl.cm.autumn
		cmap_base = mpl.cm.winter
		cmap_treat = mpl.cm.gray

		plt.figure()
		for i in range(0,len(ibi_base)):
			plt.plot(ibi_base[i],pupil_base[i],color=cmap_base(i/float(len(ibi_base))),marker='o')
		plot_cov_ellipse(cov_base,mean_vec_base,fc='b',ec='None',a=0.2)
		for i in range(0,len(ibi_stress)):
		    plt.plot(ibi_stress[i],pupil_stress[i],color=cmap_stress(i/float(len(ibi_stress))),marker='o')
		plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
		for i in range(0,len(ibi_treat)):
			plt.plot(ibi_treat[i],pupil_treat[i],color=cmap_treat(i/float(len(ibi_treat))),marker='o')
		plot_cov_ellipse(cov_treat,mean_vec_treat,fc='k',ec='None',a=0.2)
		
		plt.xlabel('Mean Trial IBI (s)')
		plt.ylabel('Mean Trial PD (AU)')
		plt.title('Data points taken over %s' % (trial_or_time))
		plt.xlim((0.35,0.46))
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
		#plt.ylim((-0.05,1.05))
		#plt.xlim((-0.05,1.05))

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
		plt.title('Baseline')
		plt.subplot(2,3,4)
		plt.plot(x_base, pupil_base, 'g')
		plt.subplot(2,3,2)
		plt.plot(x_stress, ibi_stress, 'y')
		plt.title('Stress')
		plt.subplot(2,3,5)
		plt.plot(x_stress, pupil_stress, 'b')
		plt.subplot(2,3,3)
		plt.plot(x_treat, ibi_treat, 'c')
		plt.title('Treatment')
		plt.xlabel('%s' % (trial_or_time))
		plt.ylabel('IBI (s)')
		plt.subplot(2,3,6)
		plt.plot(x_treat, pupil_treat, 'm')
		plt.xlabel('%s' % (trial_or_time))
		plt.ylabel('PD (a.u.)')

		if show_fig:
			plt.show()
		if save_fig:
			plt.savefig('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Plots/'+self.filename+'_TimeSeries.svg')

		return


class StressTask_PowerFeatureData():
	'''
	Class for operating on pre-computed peripheral physiology data. 
	This is for the stress task which has 3 blocks: baseline, stress, and treatment.
	Peripheral physiology data is pre-computed (1) per trial, looking at the first 100 ms
	after a hold_center state and (2) over time, per 10 s time windows that 
	'''
	def __init__(self, power_mat_filenames):
		# Read data from .mat file
		self.lfp_features_base = spio.loadmat(power_mat_filenames[0])
		self.lfp_features_stress = spio.loadmat(power_mat_filenames[1])
		self.lfp_features_treat = spio.loadmat(power_mat_filenames[2])
		# Pull out per-trial data
		self.lfp_features_trials_base = self.lfp_features_base['trials']
		self.lfp_features_trials_stress = self.lfp_features_stress['trials']
		self.lfp_features_trials_treat = self.lfp_features_treat['trials']
		# Pull out per-time block data
		self.lfp_features_time_base = self.lfp_features_base['time']
		self.lfp_features_time_stress = self.lfp_features_stress['time']
		self.lfp_features_time_treat = self.lfp_features_treat['time']

		return


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
		Load all neural, pupil, and pulse data. FYI, Luigi's data is always 3 blocks in the same tank, whereas Mario's baseline and stress blocks are in one tank and treatment block is in another tank
		'''
		print("Loading TDT data.")
	
		r = io.TdtIO(TDT_tanks[0])
		bl = r.read_block(lazy=False,cascade=True)
		print("First tank read.")
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
			r = io.TdtIO(TDT_tanks[2])
			bl = r.read_block(lazy=False,cascade=True)
			print("Reading separate tank for treatment block.")

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


		print("Finished loading TDT data.")

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