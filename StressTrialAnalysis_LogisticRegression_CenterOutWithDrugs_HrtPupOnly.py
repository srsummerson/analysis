import numpy as np 
import scipy as sp
import matplotlib as mpl
import tables
import sys
import csv
import statsmodels.api as sm
from neo import io
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from StressTaskBehavior import StressBehaviorWithDrugs_CenterOut
from syncHDF_withCSV import syncHDF_withCSV
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


hdf_filenames = ['mari20181009_09_te1390.hdf', 'mari20181009_10_te1391.hdf', 'mari20181009_11_te1392.hdf'] 			# list of hdf files for block A and B
filename = ['Mario20181009', 'Mario20181009', 'Mario20181009-1'] 							# list of TDT tanks for blocks A and B
block_num = [1, 2, 1] 										# corresponding TDT block numbers of the tanks for blocks A and B of behavior
DIOx_csv = True 										# Boolean indicating if DIO is from CSV or syncHDF file


#TDT_tank = ['/backup/subnetsrig/storage/tdt/'+name for name in filename]
TDT_tank = ['/home/srsummerson/storage/tdt/'+name for name in filename]
hdf_location = ['/storage/rawdata/hdf/'+hdf_name for hdf_name in hdf_filenames]
mat_location = '/storage/syncHDF/'

PupD_filename = ['/home/srsummerson/storage/tdt/' + filename[ind] + '/' + filename[ind] + '_Block-' + str(block_num[ind]) + '_PupD.csv' for ind in range(len(filename))]
HrtR_filename = ['/home/srsummerson/storage/tdt/' + filename[ind] + '/' + filename[ind] + '_Block-' + str(block_num[ind]) + '_HrtR.csv' for ind in range(len(filename))]


DIOx_filename = [None] * 4

for l in range(len(block_num)):
	DIOx_filename[l] = ['/home/srsummerson/storage/tdt/' + filename[l] + '/' + filename[l] + '_Block-' + str(block_num[l]) + '_DIOx_CH1.csv', 
						'/home/srsummerson/storage/tdt/' + filename[l] + '/' + filename[l] + '_Block-' + str(block_num[l]) + '_DIOx_CH2.csv', 
						'/home/srsummerson/storage/tdt/' + filename[l] + '/' + filename[l] + '_Block-' + str(block_num[l]) + '_DIOx_CH3.csv',
						'/home/srsummerson/storage/tdt/' + filename[l] + '/' + filename[l] + '_Block-' + str(block_num[l]) + '_DIOx_CH4.csv']



if DIOx_csv == True:
	print 'Making syncHDF file from DIO CSV files'
	for m in range(len(hdf_filenames)):
		hdf_times = syncHDF_withCSV(filename[m], TDT_tank[m], block_num[m], DIOx_filename[m])

#### ADD IN INDICATING IF DIO IS FROM CSV FILE

len_window = 5.0 										# number of seconds used for computing pulse and pupil mean values over time

'''
Loop through files to extract TDT sample numbers relevant to data times of interest for the different trial types.
'''
for i, hdf in enumerate(hdf_location):
	'''
	Load behavior data, and syncing data for behavior and TDT recording.
	''' 
	sb = StressBehaviorWithDrugs_CenterOut(hdf)
	mat_filename = filename[i]+'_b'+str(block_num[i])+'_syncHDF.mat'
	mat_filename = mat_location + mat_filename

	hdf_times = dict()
	sp.io.loadmat(mat_filename, hdf_times)
	dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])

	if i ==0:
		num_trials = np.array([len(sb.ind_reward_states)])
		response_times = sb.trial_times
		#stress_type = sb.stress_trial

		tdt_ind_hold_center, tdt_sr = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,mat_filename)
		
		reg_start_sample = dio_tdt_sample[0]
		reg_stop_sample = dio_tdt_sample[-1]
	elif i == 1:
		num_trials = np.append(num_trials, len(sb.ind_reward_states))
		response_times = np.append(response_times, sb.trial_times)
		#stress_type = np.append(stress_type, sb.stress_trial)

		inds, tdt_sr = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,mat_filename)
		tdt_ind_hold_center = np.append(tdt_ind_hold_center, inds)
		stress_start_sample = dio_tdt_sample[0]
		stress_stop_sample = dio_tdt_sample[-1]

	elif i == 2:
		num_trials = np.append(num_trials, len(sb.ind_reward_states))
		response_times = np.append(response_times, sb.trial_times)
		#stress_type = np.append(stress_type, sb.stress_trial)

		inds, tdt_sr = sb.get_state_TDT_LFPvalues(sb.ind_hold_center_states,mat_filename)
		tdt_ind_hold_center = np.append(tdt_ind_hold_center, inds)
		blockC_start_sample = dio_tdt_sample[0]
		blockC_stop_sample = dio_tdt_sample[-1]
	
'''
Load pupil dilation and heart rate data

'''
pulse_data = dict()
pupil_data = dict()

for j in range(len(block_num)):
	print "Loading phys data from Block %i" % (j+1)
	f = open(PupD_filename[j], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k if k!= '' else np.nan for i in data for k in i]
	pupil_data[j] = np.array([float(val) for val in datal])
	#pupil_data[j] = get_csv_data_singlechannel(PupD_filename[j])
	pupil_samprate = 3051.757813

	f = open(HrtR_filename[j], 'r')
	reader = csv.reader(f)
	data = list(reader)
	datal = [k if k!= '' else np.nan for i in data for k in i]
	pulse_data[j] = np.array([float(val) for val in datal])
	pulse_data[j] = np.array([val for val in pulse_data[j] if val > 1.0]) 			# get rid of padded values that sometimes appear in CSV file
	#pulse_data[j] = get_csv_data_singlechannel(HrtR_filename[j])
	pulse_samprate = 3051.757813



# Find IBIs and pupil data for all successful stress trials. 
samples_pulse = np.floor(response_times*pulse_samprate) 	#number of samples in trial interval for pulse signal
samples_pupil = np.floor(response_times*pupil_samprate)

trial_start = 0
ibi_stress_mean = np.array([])
ibi_reg_mean = np.array([])
ibi_blockC_mean = np.array([])
pupil_stress_mean = np.array([])
pupil_reg_mean = np.array([])
pupil_blockC_mean = np.array([])


for k in range(len(hdf_filenames)):
	print "Block %i - samples by trials" % (k)
	if num_trials[k] > 0: 
		trial_end = trial_start + num_trials[k]

		pulse_d = np.ravel(pulse_data[k])
		pulse_ind = tdt_ind_hold_center[trial_start:trial_end]
		nsamples_pulse = samples_pulse[trial_start:trial_end]
		pupil_d = np.ravel(pupil_data[k])
		pupil_ind = tdt_ind_hold_center[trial_start:trial_end]
		nsamples_pupil = samples_pupil[trial_start:trial_end]

		ibi_mean, ibi_std, pupil_mean, pupil_std, nbins_ibi, ibi_hist, nbins_pupil, pupil_hist = getIBIandPuilDilation(pulse_d, pulse_ind,nsamples_pulse, pulse_samprate,pupil_d, pupil_ind,nsamples_pupil,pupil_samprate)
	elif num_trials[k] == 0:
		ibi_mean = np.array([])
		ibi_std = np.array([])
		pupil_mean = np.array([])
		pupil_std = np.array([])

	'''
	Compute values based on time windows
	'''
	
	len_window_samples = np.floor(len_window*pulse_samprate) 	# number of samples in designated time window
	#pulse_ind_time = np.arange(pulse_ind[0], pulse_ind[-1], len_window_samples)
	#pupil_ind_time = np.arange(pupil_ind[0], pupil_ind[-1], len_window_samples)


	if k==0:
		pulse_ind_time = np.arange(reg_start_sample, reg_stop_sample, len_window_samples)
		pupil_ind_time = pulse_ind_time

	elif k==1:
		pulse_ind_time = np.arange(stress_start_sample, stress_stop_sample, len_window_samples)
		pupil_ind_time = pulse_ind_time
	elif k==2:
		pulse_ind_time = np.arange(blockC_start_sample, blockC_stop_sample, len_window_samples)
		pupil_ind_time = pulse_ind_time
	nsamples_pulse_window = len_window_samples*np.ones(len(pulse_ind_time))
	nsamples_pupil_window = len_window_samples*np.ones(len(pupil_ind_time))

	print "Block %i - samples in time windows" % (k)
	
	ibi_mean_time, ibi_std_time, pupil_mean_time, pupil_std_time, nbins_ibi_time, ibi_hist_time, nbins_pupil_time, pupil_hist_time = getIBIandPuilDilation(pulse_d, pulse_ind_time,nsamples_pulse_window, pulse_samprate,pupil_d, pupil_ind_time,nsamples_pupil_window,pupil_samprate)
	#
	# trial_start = trial_end

	if k==0:
		ibi_reg_mean = np.array(ibi_mean)
		pupil_reg_mean = np.array(pupil_mean)
		ibi_reg_mean_time = np.array(ibi_mean_time)
		pupil_reg_mean_time = np.array(pupil_mean_time)
	elif k==1:
		ibi_stress_mean = np.array(ibi_mean)
		pupil_stress_mean = np.array(pupil_mean)
		ibi_stress_mean_time = np.array(ibi_mean_time)
		pupil_stress_mean_time = np.array(pupil_mean_time)
	elif k==2:
		ibi_blockC_mean = np.array(ibi_mean)
		pupil_blockC_mean = np.array(pupil_mean)
		ibi_blockC_mean_time = np.array(ibi_mean_time)
		pupil_blockC_mean_time = np.array(pupil_mean_time)

	trial_start = trial_end # moved


# delete PD values less than 0
ibi_stress_mean_adj = np.array([ibi_stress_mean[i] for i in range(len(ibi_stress_mean)) if (pupil_stress_mean[i] > -3)and(~np.isnan(ibi_stress_mean[i]))])
ibi_reg_mean_adj = np.array([ibi_reg_mean[i] for i in range(len(ibi_reg_mean)) if (pupil_reg_mean[i] > -3)and(~np.isnan(ibi_reg_mean[i]))])
pupil_stress_mean_adj = np.array([pupil_stress_mean[i] for i in range(len(pupil_stress_mean)) if (pupil_stress_mean[i] > -3)and(~np.isnan(ibi_stress_mean[i]))])
pupil_reg_mean_adj = np.array([pupil_reg_mean[i] for i in range(len(pupil_reg_mean)) if (pupil_reg_mean[i] > -3)and(~np.isnan(ibi_reg_mean[i]))])
if len(hdf_filenames)==3:
	ibi_blockC_mean_adj = np.array([ibi_blockC_mean[i] for i in range(len(ibi_blockC_mean)) if (pupil_blockC_mean[i] > -3)and(~np.isnan(ibi_blockC_mean[i]))])
	pupil_blockC_mean_adj = np.array([pupil_blockC_mean[i] for i in range(len(pupil_blockC_mean)) if (pupil_blockC_mean[i] > -3)and(~np.isnan(ibi_blockC_mean[i]))])


ibi_stress_mean_adj_time = np.array([ibi_stress_mean_time[i] for i in range(len(ibi_stress_mean_time)) if (pupil_stress_mean_time[i] > -3)and(~np.isnan(ibi_stress_mean_time[i]))])
ibi_reg_mean_adj_time = np.array([ibi_reg_mean_time[i] for i in range(len(ibi_reg_mean_time)) if (pupil_reg_mean_time[i] > -3)and(~np.isnan(ibi_reg_mean_time[i]))])
pupil_stress_mean_adj_time = np.array([pupil_stress_mean_time[i] for i in range(len(pupil_stress_mean_time)) if (pupil_stress_mean_time[i] > -3)and(~np.isnan(ibi_stress_mean_time[i]))])
pupil_reg_mean_adj_time = np.array([pupil_reg_mean_time[i] for i in range(len(pupil_reg_mean_time)) if (pupil_reg_mean_time[i] > -3)and(~np.isnan(ibi_reg_mean_time[i]))])
if len(hdf_filenames)==3:
	ibi_blockC_mean_adj_time = np.array([ibi_blockC_mean_time[i] for i in range(len(ibi_blockC_mean_time)) if (pupil_blockC_mean_time[i] > -3)and(~np.isnan(ibi_blockC_mean_time[i]))])
	pupil_blockC_mean_adj_time = np.array([pupil_blockC_mean_time[i] for i in range(len(pupil_blockC_mean_time)) if (pupil_blockC_mean_time[i] > -3)and(~np.isnan(ibi_blockC_mean_time[i]))])


num_successful_stress = len(ibi_stress_mean_adj)
num_successful_reg = len(ibi_reg_mean_adj)
if len(hdf_filenames)==3:
	num_successful_blockC = len(ibi_blockC_mean_adj)

num_stress_time = len(ibi_stress_mean_adj_time)
num_reg_time = len(ibi_reg_mean_adj_time)
if len(hdf_filenames)==3:
	num_blockC_time = len(ibi_blockC_mean_adj_time)

y_successful_stress = np.ones(num_successful_stress)
y_successful_reg = np.zeros(num_successful_reg)
y_successful = np.append(y_successful_reg,y_successful_stress)

y_stress_time = np.ones(num_stress_time)
y_reg_time = np.zeros(num_reg_time)
y_time = np.append(y_reg_time,y_stress_time)

'''
Do regression as well: 0 = regular trial, 1 = stress trial
'''

x_successful = np.vstack((np.append(ibi_reg_mean_adj, ibi_stress_mean_adj), np.append(pupil_reg_mean_adj, pupil_stress_mean_adj)))
x_successful = np.transpose(x_successful)
x_successful = sm.add_constant(x_successful,prepend='False')

x_time = np.vstack((np.append(ibi_reg_mean_adj_time, ibi_stress_mean_adj_time), np.append(pupil_reg_mean_adj_time, pupil_stress_mean_adj_time)))
x_time = np.transpose(x_time)
x_time = sm.add_constant(x_time,prepend='False')

print "Regression with successful trials"
print "x1: IBI"
print "x2: Pupil Dilation"
model_glm = sm.Logit(y_successful,x_successful)
fit_glm = model_glm.fit()
print fit_glm.summary()


print "Regression with time windows"
print "x1: IBI"
print "x2: Pupil Dilation"
model_glm_time = sm.Logit(y_time,x_time)
fit_glm_time = model_glm_time.fit()
print fit_glm_time.summary()


norm_ibi_stress_mean = ibi_stress_mean_adj 
norm_pupil_stress_mean = pupil_stress_mean_adj 
norm_ibi_reg_mean = ibi_reg_mean_adj
norm_pupil_reg_mean = pupil_reg_mean_adj
if len(hdf_filenames)==3:
	norm_ibi_blockC_mean = ibi_blockC_mean_adj
	norm_pupil_blockC_mean = pupil_blockC_mean_adj

points_stress = np.array([norm_ibi_stress_mean,norm_pupil_stress_mean])
points_reg = np.array([norm_ibi_reg_mean,norm_pupil_reg_mean])
cov_stress = np.cov(points_stress)
cov_reg = np.cov(points_reg)
mean_vec_stress = [np.nanmean(norm_ibi_stress_mean),np.nanmean(norm_pupil_stress_mean)]
mean_vec_reg = [np.nanmean(norm_ibi_reg_mean),np.nanmean(norm_pupil_reg_mean)]
if len(hdf_filenames)==3:
	points_blockC = np.array([norm_ibi_blockC_mean,norm_pupil_blockC_mean])
	cov_blockC = np.cov(points_blockC)
	mean_vec_blockC = [np.nanmean(norm_ibi_blockC_mean),np.nanmean(norm_pupil_blockC_mean)]

norm_ibi_stress_mean_time = ibi_stress_mean_adj_time 
norm_pupil_stress_mean_time = pupil_stress_mean_adj_time
norm_ibi_reg_mean_time = ibi_reg_mean_adj_time
norm_pupil_reg_mean_time = pupil_reg_mean_adj_time
if len(hdf_filenames)==3:
	norm_ibi_blockC_mean_time = ibi_blockC_mean_adj_time 
	norm_pupil_blockC_mean_time = pupil_blockC_mean_adj_time

points_stress_time = np.array([norm_ibi_stress_mean_time,norm_pupil_stress_mean_time])
points_reg_time = np.array([norm_ibi_reg_mean_time,norm_pupil_reg_mean_time])
cov_stress_time = np.cov(points_stress_time)
cov_reg_time = np.cov(points_reg_time)
mean_vec_stress_time = [np.nanmean(norm_ibi_stress_mean_time),np.nanmean(norm_pupil_stress_mean_time)]
mean_vec_reg_time = [np.nanmean(norm_ibi_reg_mean_time),np.nanmean(norm_pupil_reg_mean_time)]
if len(hdf_filenames)==3:
	points_blockC_time = np.array([norm_ibi_blockC_mean_time,norm_pupil_blockC_mean_time])
	cov_blockC_time = np.cov(points_blockC_time)
	mean_vec_blockC_time = [np.nanmean(norm_ibi_blockC_mean_time),np.nanmean(norm_pupil_blockC_mean_time)]


cmap_stress = mpl.cm.autumn
cmap_reg = mpl.cm.winter
cmap_blockC = mpl.cm.gray

plt.figure()
for i in range(0,len(ibi_stress_mean_adj)):
    #plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o',markeredgecolor=None,markeredgewidth=0.0)
    plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o')
plot_cov_ellipse(cov_stress,mean_vec_stress,fc='r',ec='None',a=0.2)
for i in range(0,len(ibi_reg_mean_adj)):
	plt.plot(norm_ibi_reg_mean[i],norm_pupil_reg_mean[i],color=cmap_reg(i/float(len(ibi_reg_mean))),marker='o')
plot_cov_ellipse(cov_reg,mean_vec_reg,fc='b',ec='None',a=0.2)
if len(hdf_filenames)==3:
	for i in range(0,len(ibi_blockC_mean_adj)):
		plt.plot(norm_ibi_blockC_mean[i],norm_pupil_blockC_mean[i],color=cmap_blockC(i/float(len(ibi_blockC_mean))),marker='o')
	plot_cov_ellipse(cov_blockC,mean_vec_blockC,fc='k',ec='None',a=0.2)
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
if len(hdf_filenames)==3:
	sm_blockC = plt.cm.ScalarMappable(cmap=cmap_blockC, norm=plt.Normalize(vmin=0, vmax=1))
	# fake up the array of the scalar mappable. Urgh...
	sm_blockC._A = []
	cbar = plt.colorbar(sm_blockC,ticks=[0,1], orientation='vertical')
	cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
#plt.ylim((-0.05,1.05))
#plt.xlim((-0.05,1.05))
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename[0]+'_b'+str(block_num[0])+'_IBIPupilCovariance.svg')


plt.figure()
for i in range(0,len(ibi_stress_mean_adj_time)):
    #plt.plot(norm_ibi_stress_mean[i],norm_pupil_stress_mean[i],color=cmap_stress(i/float(len(ibi_stress_mean))),marker='o',markeredgecolor=None,markeredgewidth=0.0)
    plt.plot(norm_ibi_stress_mean_time[i],norm_pupil_stress_mean_time[i],color=cmap_stress(i/float(len(ibi_stress_mean_time))),marker='o')
plot_cov_ellipse(cov_stress_time,mean_vec_stress_time,fc='r',ec='None',a=0.2)
for i in range(0,len(ibi_reg_mean_adj_time)):
	plt.plot(norm_ibi_reg_mean_time[i],norm_pupil_reg_mean_time[i],color=cmap_reg(i/float(len(ibi_reg_mean_time))),marker='o')
plot_cov_ellipse(cov_reg_time,mean_vec_reg_time,fc='b',ec='None',a=0.2)
if len(hdf_filenames)==3:
	for i in range(0,len(ibi_blockC_mean_adj_time)):
		plt.plot(norm_ibi_blockC_mean_time[i],norm_pupil_blockC_mean_time[i],color=cmap_blockC(i/float(len(ibi_blockC_mean_time))),marker='o')
	plot_cov_ellipse(cov_blockC_time,mean_vec_blockC_time,fc='k',ec='None',a=0.2)
#plt.legend()
plt.xlabel('Mean Trial IBI (s)')
plt.ylabel('Mean Trial PD (AU)')
plt.title('Time Windows')
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
if len(hdf_filenames)==3:
	sm_blockC = plt.cm.ScalarMappable(cmap=cmap_blockC, norm=plt.Normalize(vmin=0, vmax=1))
	# fake up the array of the scalar mappable. Urgh...
	sm_blockC._A = []
	cbar = plt.colorbar(sm_blockC,ticks=[0,1], orientation='vertical')
	cbar.ax.set_xticklabels(['Early', 'Late'])  # horizontal colorbar
#plt.ylim((-0.05,1.05))
#plt.xlim((-0.05,1.05))
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename[0]+'_b'+str(block_num[0])+'_IBIPupilCovariance_TimeWindows.svg')


# Recall, len_window is the length of the time windows used to compute the data points
# Make tick marks every minute

plt.figure()
plt.subplot(121)
plt.plot(range(len(ibi_reg_mean_time)), ibi_reg_mean_time,'b', label = 'Reg')
plt.plot(range(len(ibi_reg_mean_time), len(ibi_reg_mean_time) + len(ibi_stress_mean_time)), ibi_stress_mean_time, 'r', label = 'Stress')
if len(hdf_filenames)==3:
	plt.plot(range(len(ibi_reg_mean_time) + len(ibi_stress_mean_time), len(ibi_reg_mean_time) + len(ibi_stress_mean_time) + len(ibi_blockC_mean_time)), ibi_blockC_mean_time, 'k', label='Block C')
	xticklabels = np.arange(0,(len(ibi_reg_mean_time) + len(ibi_stress_mean_time) + len(ibi_blockC_mean_time))*len_window/60.,len_window/60.)  # labels in minutes
else:
	xticklabels = np.arange(0,(len(ibi_reg_mean_time) + len(ibi_stress_mean_time))*len_window/60.,len_window/60.)  # labels in minutes
plt.legend()
plt.title('IBI - Reg vs Stress')
xticks = np.arange(0, len(xticklabels), 5 * 60./len_window) # tick every 5 mins
xticks = [int(val) for val in xticks]
xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
plt.xticks(xticks, xticklabels)
plt.xlabel('Minutes')

plt.subplot(122)
plt.plot(range(len(pupil_reg_mean_time)), pupil_reg_mean_time,'b', label = 'Reg')
plt.plot(range(len(pupil_reg_mean_time), len(pupil_reg_mean_time) + len(pupil_stress_mean_time)), pupil_stress_mean_time, 'r', label = 'Stress')
plt.title('Pupil - Reg vs Stress')
if len(hdf_filenames)==3:
	plt.plot(range(len(pupil_reg_mean_time) + len(pupil_stress_mean_time), len(pupil_reg_mean_time) + len(pupil_stress_mean_time) + len(pupil_blockC_mean_time)), pupil_blockC_mean_time, 'k', label='Block C')
	xticklabels = np.arange(0,(len(ibi_reg_mean_time) + len(ibi_stress_mean_time) + len(ibi_blockC_mean_time))*len_window/60.,len_window/60.)  # labels in minutes
else:
	xticklabels = np.arange(0,(len(ibi_reg_mean_time) + len(ibi_stress_mean_time))*len_window/60.,len_window/60.)  # labels in minutes
plt.legend()
plt.ylim((1, 3))
xticks = np.arange(0, len(xticklabels), 5 * 60./len_window) # tick every 5 mins
xticks = [int(val) for val in xticks]
xticklabels = ['{0:.1f}'.format(xticklabels[k]) for k in xticks]
plt.xticks(xticks, xticklabels)
plt.xlabel('Minutes')
plt.savefig('/home/srsummerson/code/analysis/StressPlots/'+filename[0]+'_b'+str(block_num[0])+'_IBIPupil_TimeWindows.svg')
