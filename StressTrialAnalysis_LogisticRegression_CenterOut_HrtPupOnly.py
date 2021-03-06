import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
import sys
import csv
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

#### still needs to be fixed - more samples in csv file than TDT neo reading

hdf_filenames = ['mari20180902_03_te1223.hdf']
filename = ['Mario20180902']
block_num = [1]

#TDT_tank = ['/backup/subnetsrig/storage/tdt/'+name for name in filename]
TDT_tank = ['/home/srsummerson/storage/tdt/'+name for name in filename]
hdf_location = ['/storage/rawdata/hdf/'+hdf_name for hdf_name in hdf_filenames]
mat_location = '/storage/syncHDF/'

PupD_filename = ['/home/srsummerson/storage/tdt/' + filename[ind] + '/' + filename[ind] + '_Block-' + str(block_num[ind]) + '_PupD.csv' for ind in range(len(filename))]
HrtR_filename = ['/home/srsummerson/storage/tdt/' + filename[ind] + '/' + filename[ind] + '_Block-' + str(block_num[ind]) + '_HrtR.csv' for ind in range(len(filename))]

#HrtR_filename = ['/home/srsummerson/storage/tdt/Mario20160320_plex/Mario20160320_Block-1_HrtR.csv']


'''
Loop through files to extract TDT sample numbers relevant to data times of interest for the different trial types.
'''
for i, hdf in enumerate(hdf_location):
	'''
	Load behavior data, and syncing data for behavior and TDT recording.
	''' 
	sb = StressBehavior_CenterOut(hdf)
	mat_filename = filename[i]+'_b'+str(block_num[i])+'_syncHDF.mat'
	mat_filename = mat_location + mat_filename

	if i ==0:
		num_trials = np.array([len(sb.ind_reward_states)])
		response_times = sb.trial_times
		stress_type = sb.stress_trial

		tdt_ind_hold_center, tdt_sr = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,mat_filename)
	else:
		num_trials = np.append(num_trials, len(sb.ind_reward_states))
		response_times = np.append(response_times, sb.trial_times)
		stress_type = np.append(stress_type, sb.stress_trial)

		inds, tdt_sr = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,mat_filename)
		tdt_ind_hold_center = np.append(tdt_ind_hold_center, inds)

	
'''
Load pupil dilation and heart rate data
'''
pulse_data = dict()
pupil_data = dict()

for j in range(len(block_num)):
	f = open(PupD_filename[j], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k for i in data for k in i]
	pupil_data[j] = np.array([float(val) for val in datal])
	#pupil_data[j] = get_csv_data_singlechannel(PupD_filename[j])
	pupil_samprate = 3051.757813

	f = open(HrtR_filename[j], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k for i in data for k in i]
	pulse_data[j] = np.array([float(val) for val in datal])
	#pulse_data[j] = get_csv_data_singlechannel(HrtR_filename[j])
	pulse_samprate = 3051.757813


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

	print "Block %i" % (k)
	print len(pulse_d)
	ibi_mean, ibi_std, pupil_mean, pupil_std, nbins_ibi, ibi_hist, nbins_pupil, pupil_hist = getIBIandPuilDilation(pulse_d, pulse_ind,nsamples_pulse, pulse_samprate,pupil_d, pupil_ind,nsamples_pupil,pupil_samprate)
	# trial_start = trial_end

	ind_stress = np.ravel(np.nonzero(stress_type[trial_start:trial_end]))
	ind_reg = np.ravel(np.nonzero(stress_type[trial_start:trial_end]-1))
	ibi_stress_mean = np.append(ibi_stress_mean, np.array(ibi_mean)[ind_stress])
	ibi_reg_mean = np.append(ibi_reg_mean, np.array(ibi_mean)[ind_reg])
	pupil_stress_mean = np.append(pupil_stress_mean, np.array(pupil_mean)[ind_stress])
	pupil_reg_mean = np.append(pupil_reg_mean, np.array(pupil_mean)[ind_reg])

	trial_start = trial_end # moved

# delete PD values less than 0
ibi_stress_mean_adj = np.array([ibi_stress_mean[i] for i in range(len(ibi_stress_mean)) if (pupil_stress_mean[i] > -3)and(~np.isnan(ibi_stress_mean[i]))])
ibi_reg_mean_adj = np.array([ibi_reg_mean[i] for i in range(len(ibi_reg_mean)) if (pupil_reg_mean[i] > -3)and(~np.isnan(ibi_reg_mean[i]))])
pupil_stress_mean_adj = np.array([pupil_stress_mean[i] for i in range(len(pupil_stress_mean)) if (pupil_stress_mean[i] > -3)and(~np.isnan(ibi_stress_mean[i]))])
pupil_reg_mean_adj = np.array([pupil_reg_mean[i] for i in range(len(pupil_reg_mean)) if (pupil_reg_mean[i] > -3)and(~np.isnan(ibi_reg_mean[i]))])


num_successful_stress = len(ibi_stress_mean_adj)
num_successful_reg = len(ibi_reg_mean_adj)


y_successful_stress = np.ones(num_successful_stress)
y_successful_reg = np.zeros(num_successful_reg)
y_successful = np.append(y_successful_reg,y_successful_stress)

'''
Do regression as well: 0 = regular trial, 1 = stress trial
'''

x_successful = np.vstack((np.append(ibi_reg_mean_adj, ibi_stress_mean_adj), np.append(pupil_reg_mean_adj, pupil_stress_mean_adj)))
x_successful = np.transpose(x_successful)
x_successful = sm.add_constant(x_successful,prepend='False')

print "Regression with successful trials"
print "x1: IBI"
print "x2: Pupil Dilation"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()

norm_ibi_stress_mean = ibi_stress_mean_adj 
norm_pupil_stress_mean = pupil_stress_mean_adj 
norm_ibi_reg_mean = ibi_reg_mean_adj
norm_pupil_reg_mean = pupil_reg_mean_adj

points_stress = np.array([norm_ibi_stress_mean,norm_pupil_stress_mean])
points_reg = np.array([norm_ibi_reg_mean,norm_pupil_reg_mean])
cov_stress = np.cov(points_stress)
cov_reg = np.cov(points_reg)
mean_vec_stress = [np.nanmean(norm_ibi_stress_mean),np.nanmean(norm_pupil_stress_mean)]
mean_vec_reg = [np.nanmean(norm_ibi_reg_mean),np.nanmean(norm_pupil_reg_mean)]

cmap_stress = mpl.cm.autumn
cmap_reg = mpl.cm.winter

plt.figure()
for i in range(0,len(ibi_stress_mean_adj)):
    #plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o',markeredgecolor=None,markeredgewidth=0.0)
    plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o')
plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
for i in range(0,len(ibi_reg_mean_adj)):
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
