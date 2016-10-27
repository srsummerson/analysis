import numpy as np 
import scipy as sp
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from StressTaskBehavior import StressBehavior


# list of hdf files for data containing blocks of stress and regular trials
stress_hdf_files = ['mari20161013_03_te2598.hdf', 'mari20161026_04_te2634.hdf']
#hdf_prefix = '/storage/rawdata/hdf/'
hdf_prefix = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/hdf/'


def CompareRT_RegularVStress(stress_hdf_files, hdf_prefix): 

	all_reg_rt = []
	all_stress_rt = []

	avg_reg_rt = np.zeros(len(stress_hdf_files))
	avg_stress_rt = np.zeros(len(stress_hdf_files))

	for i, name in enumerate(stress_hdf_files):

		# load stress data into class object StressBehavior
		hdf_location = hdf_prefix +name
		stress_data = StressBehavior(hdf_location)

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


	t, p = stats.ttest_ind(avg_reg_rt, avg_stress_rt)

	width = 0.5
	plt.bar(0, [np.nanmean(avg_reg_rt)], width, color = 'b', ecolor = 'k', yerr = [np.nanstd(avg_reg_rt)/np.sqrt(len(avg_reg_rt) - 1.)])
	plt.bar(1, [np.nanmean(avg_stress_rt)], width, color = 'r', ecolor = 'k', yerr = [ np.nanstd(avg_stress_rt)/np.sqrt(len(avg_stress_rt) - 1.)])
	plt.ylabel('Reaction Time (s)')
	plt.xticks(np.arange(2)+width/2., ('Regular', 'Stress'))
	plt.title('Avg. RTs over %i Sessions: (t,p) = (%f,%f)' % (len(stress_hdf_files), t, p))
	plt.xlim((-0.1, 1 + width + 0.1))
	plt.show()

	return

CompareRT_RegularVStress(stress_hdf_files, hdf_prefix)