import numpy as np 
import scipy as sp
import pandas
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from StressTaskBehavior import StressBehavior


# list of hdf files for data containing blocks of stress and regular trials
stress_excel_spreadsheet = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/Mood Bias Task/Stress Task AB Blocks Log.xlsx'
df = pandas.read_excel(stress_excel_spreadsheet)  			# read excel spreadsheet with task infor
files = df['Data file'].values 								# load column containing names of all files
stress_hdf_files = [name for name in files if name[-3:]=='hdf']	# extract only names of hdf files

pick_files = range(len(stress_hdf_files))
pick_files = np.delete(pick_files, [0, 1, 31, 35, 39])

stress_hdf_files_adjust = [stress_hdf_files[i] for i in range(len(stress_hdf_files)) if i in pick_files]

#stress_hdf_files = ['mari20161013_03_te2598.hdf', 'mari20161026_04_te2634.hdf']
#hdf_prefix = '/storage/rawdata/hdf/'
hdf_prefix = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/hdf/'

cmap = mpl.cm.hsv

print 'Loaded all file names.'

def CompareRT_RegularVStress(stress_hdf_files, hdf_prefix): 

	all_reg_rt = []
	all_stress_rt = []

	avg_reg_rt = np.zeros(len(stress_hdf_files))
	avg_stress_rt = np.zeros(len(stress_hdf_files))

	for i, name in enumerate(stress_hdf_files):
		print 'File %i of %i' % (i, len(stress_hdf_files))
		# load stress data into class object StressBehavior
		hdf_location = hdf_prefix +name
		stress_data = StressBehavior(hdf_location)

		trial_times_percentage_session, num_times_per_session = stress_data.PlotTrialsCompleted(20)

		# find which trials are stress trials
		trial_type = np.ravel(stress_data.stress_type[stress_data.state_time[stress_data.ind_check_reward_states]])
		reg_trial_inds = np.array([ind for ind in range(len(trial_type)) if trial_type[ind] == 0])
		stress_trial_inds = np.array([ind for ind in range(len(trial_type)) if trial_type[ind] == 1])

		rt_reg, total_vel = stress_data.compute_rt_per_trial_FreeChoiceTask(reg_trial_inds)
		rt_stress, total_vel = stress_data.compute_rt_per_trial_FreeChoiceTask(stress_trial_inds)

		all_reg_rt.append(rt_reg.tolist())
		all_stress_rt.append(rt_stress.tolist())

		avg_reg_rt[i] = np.nanmean(rt_reg)
		avg_stress_rt[i] = np.nanmean(rt_stress)

		regular_times = trial_times_percentage_session[:num_times_per_session[0]]
		stress_times = trial_times_percentage_session[num_times_per_session[0]:num_times_per_session[0] + num_times_per_session[1]]
		
		# normalize to pecentage of session
		regular_times = regular_times/float(stress_times[-1])
		stress_times = stress_times/float(stress_times[-1])
		plt.figure(1)
		ax1 = plt.subplot(111)
		plt.plot(stress_times,np.ones(len(stress_times)) + i,'|', color='r')
		plt.plot(regular_times,np.ones(len(regular_times)) + i,'|', color = 'b')

	plt.xlim((0,1))
	plt.xlabel('Session (%)')
	ax1.get_yaxis().set_tick_params(direction='out')
	ax1.get_xaxis().set_tick_params(direction='out')
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	plt.show()
	t, p = stats.ttest_ind(avg_reg_rt, avg_stress_rt)
	u_value, p_value = stats.mannwhitneyu(avg_reg_rt, avg_stress_rt)

	width = 0.5
	fig = plt.figure(2)
	ax = plt.subplot(111)
	plt.bar(0, [np.nanmean(avg_reg_rt)], width, color = 'b', ecolor = 'k', yerr = [np.nanstd(avg_reg_rt)/np.sqrt(len(avg_reg_rt) - 1.)])
	plt.bar(1, [np.nanmean(avg_stress_rt)], width, color = 'r', ecolor = 'k', yerr = [ np.nanstd(avg_stress_rt)/np.sqrt(len(avg_stress_rt) - 1.)])
	plt.ylabel('Reaction Time (s)')
	plt.xticks(np.arange(2)+width/2., ('Regular', 'Stress'))
	plt.title('Avg. RTs over %i Sessions: (t,p) = (%f,%f)' % (len(stress_hdf_files), t, p))
	plt.xlim((-0.1, 1 + width + 0.1))
	ax.text(0.6, 0.13, 'Regular: m = %f, Stress: m = %f' % (np.nanmean(avg_reg_rt), np.nanmean(avg_stress_rt)))
	plt.show()

	fig = plt.figure(3)
	ax = plt.subplot(111)
	plt.boxplot([avg_reg_rt, avg_stress_rt],0,'')
	plt.ylabel('Reaction Time (s)')
	plt.xticks(np.arange(2)+1., ('Regular', 'Stress'))
	plt.title('Avg. RTs over %i Sessions: (u,p) = (%f,%f)' % (len(stress_hdf_files), u_value, p_value))
	ax.text(0.6, 0.28, 'Regular: med = %f, Stress: med = %f' % (np.median(avg_reg_rt), np.median(avg_stress_rt)))
	
	plt.show()
	return avg_reg_rt, avg_stress_rt

CompareRT_RegularVStress(stress_hdf_files_adjust, hdf_prefix)