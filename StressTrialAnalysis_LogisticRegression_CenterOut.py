import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
import sys
import statsmodels.api as sm
from neo import io
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from StressTaskBehavior import StressBehavior_CenterOut
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse, LDAforFeatureSelection
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score


'''
This is for computing the regression coefficients used in closed-loop stimulation during the center-out version of the stress task. This code should be run to compute
these coefficients after Blocks A (regular trials) and B (stress trials) have been completed. First, syncHDF.py must be run in order to generate the syncHDF.mat file
and then this script may be run. 

The top three variables should be updated depending on the file name(s) for the data in Blocks A and B prior to executing the code.

'''

hdf_filenames = ['mari20180807_03_te1065.hdf']
filename = ['Mario20180807']
block_num = [1]

#TDT_tank = ['/backup/subnetsrig/storage/tdt/'+name for name in filename]
TDT_tank = ['/home/srsummerson/storage/tdt/'+name for name in filename]
hdf_location = ['/storage/rawdata/hdf/'+hdf_name for hdf_name in hdf_filenames]


'''
Loop through files to extract TDT sample numbers relevant to data times of interest for the different trial types.
'''
for i, hdf in enumerate(hdf_location):
	'''
	Load behavior data, and syncing data for behavior and TDT recording.
	''' 
	sb = StressBehavior_CenterOut(hdf)
	mat_filename = filename[i]+'_b'+str(block_num[i])+'_syncHDF.mat'

	if i ==0:
		num_trials = np.array([len(sb.ind_reward_states)])
		response_times = sb.trial_times
		stress_type = sb.stress_type

		tdt_ind_hold_center = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,syncHDF_file)
	else:
		num_trials = np.append(num_trials, len(sb.ind_reward_states))
		response_times = np.append(response_times, sb.trial_times)
		stress_type = np.append(stress_type, sb.stress_type)

		inds = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,syncHDF_file)
		tdt_ind_hold_center = np.append(tdt_ind_hold_center, inds)

	
'''
Load pupil dilation and heart rate data
'''
pulse_data = dict()
pupil_data = dict()

if (len(TDT_tank)==1) or (TDT_tank[0]==TDT_tank[1]):

	r = io.TdtIO(TDT_tank[0])
	bl = r.read_block(lazy=False,cascade=True)
	print "File read."

	for j in range(len(block_num)):

		# Get Pulse and Pupil Data
		for sig in bl.segments[block_num[j]-1].analogsignals:
			if (sig.name == 'PupD 1'):
				pupil_data[j] = np.ravel(sig)
				pupil_samprate = sig.sampling_rate.item()
			if (sig.name == 'HrtR 1'):
				pulse_data[j] = np.ravel(sig)
				pulse_samprate = sig.sampling_rate.item()
else:
	for j in range(len(block_num)):
		r = io.TdtIO(TDT_tank[j])
		bl = r.read_block(lazy=False,cascade=True)
		print "File read."

		# Get Pulse and Pupil Data
		for sig in bl.segments[block_num[j]-1].analogsignals:
			if (sig.name == 'PupD 1'):
				pupil_data[j] = np.ravel(sig)
				pupil_samprate = sig.sampling_rate.item()
			if (sig.name == 'HrtR 1'):
				pulse_data[j] = np.ravel(sig)
				pulse_samprate = sig.sampling_rate.item()


'''
Process pupil and pulse data: STOPPED HERE. NEED TO divide data into stress and reg trials within loop
'''

# Find IBIs and pupil data for all successful stress trials. 
samples_pulse = np.floor(response_times*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil = np.floor(response_times*pupil_samprate)

trial_start = 0
ibi_stress_mean = np.array([])
ibi_reg_mean = np.array([])
pupil_stress_mean = np.array([])
pupil_reg_mean = np.array([])

for k in range(len(hdf_filenames)):
	pulse_d = np.ravel(pulse_data[k])
	 
	trial_end = trial_start + num_trials[k]
	
	pulse_ind = tdt_ind_hold_center[trial_start:trial_end]
	nsamples_pulse = samples_pulse[trial_start:trial_end]
	pupil_d = np.ravel(pupil_data[k])
	pupil_ind = tdt_ind_hold_center[trial_start:trial_end]
	nsamples_pupil = samples_pupil[trial_start:trial_end]

	ibi_mean, ibi_std, pupil_mean, pupil_std, nbins_ibi, ibi_hist, nbins_pupil, pupil_hist = getIBIandPuilDilation(pulse_d, pulse_ind,nsamples_pulse, pulse_samprate,pupil_d, pupil_ind,nsamples_pupil,pupil_samprate)
	trial_start = trial_end

	ind_stress = np.ravel(np.nonzero(stress_type[trial_start:trial_end]))
	ind_reg = np.ravel(np.nonzero(~stress_type[trial_start:trial_end]))

	ibi_stress_mean = np.append(ibi_stress_mean, ibi_mean[ind_stress])
	ibi_reg_mean = np.append(ibi_reg_mean, ibi_mean[ind_reg])
	pupil_stress_mean = np.append(pupil_stress_mean, pupil_mean[ind_stress])
	pupil_reg_mean = np.append(pupil_reg_mean, pupil_mean[ind_reg])


num_successful_stress = len(ibi_stress_mean)
num_successful_reg = len(ibi_reg_mean)


y_successful_stress = np.ones(num_successful_stress)
y_successful_reg = np.zeros(num_successful_reg)
y_successful = np.append(y_successful_reg,y_successful_stress)

'''
Do regression as well: 0 = regular trial, 1 = stress trial
'''

x_successful = np.vstack((np.append(ibi_reg_mean, ibi_stress_mean), np.append(pupil_reg_mean, pupil_stress_mean)))
x_successful = np.transpose(x_successful)
x_successful = sm.add_constant(x_successful,prepend='False')

print "Regression with successful trials"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()

norm_ibi_stress_mean = ibi_stress_mean 
norm_pupil_stress_mean = pupil_stress_mean 
norm_ibi_reg_mean = ibi_reg_mean 
norm_pupil_reg_mean = pupil_reg_mean 

points_stress = np.array([norm_ibi_stress_mean,norm_pupil_stress_mean])
points_reg = np.array([norm_ibi_reg_mean,norm_pupil_reg_mean])
cov_stress = np.cov(points_stress)
cov_reg = np.cov(points_reg)
mean_vec_stress = [np.nanmean(norm_ibi_stress_mean),np.nanmean(norm_pupil_stress_mean)]
mean_vec_reg = [np.nanmean(norm_ibi_reg_mean),np.nanmean(norm_pupil_reg_mean)]

cmap_stress = mpl.cm.autumn
cmap_reg = mpl.cm.winter

plt.figure()
for i in range(0,len(ibi_stress_mean)):
    #plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o',markeredgecolor=None,markeredgewidth=0.0)
    plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o')
plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
for i in range(0,len(ibi_reg_mean)):
	plt.plot(norm_ibi_reg_mean[i],norm_pupil_reg_mean[i],color=cmap_reg(i/float(len(ibi_reg_mean))),marker='o')
plot_cov_ellipse(cov_reg,mean_vec_reg,fc='b',ec='None',a=0.2)
#plt.legend()
plt.xlabel('Mean Trial IBI (s)')
plt.ylabel('Mean Trial PD (AU)')
plt.title('Successful Trials')
sm_reg = plt.cm.ScalarMappable(cmap=cmap_reg, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_reg._A = []
cbar = plt.colorbar(sm_reg,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
sm_stress = plt.cm.ScalarMappable(cmap=cmap_stress, norm=plt.Normalize(vmin=0, vmax=1))
# fake up the array of the scalar mappable. Urgh...
sm_stress._A = []
cbar = plt.colorbar(sm_stress,ticks=[0,1], orientation='vertical')
cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
#plt.ylim((-0.05,1.05))
#plt.xlim((-0.05,1.05))
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename[0]+'_b'+str(block_num[0])+'_IBIPupilCovariance.svg')





