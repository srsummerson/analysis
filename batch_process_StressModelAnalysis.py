from StressModelAndTreatmentAnalysis import StressTaskAnalysis_ComputePowerFeatures, StressTask_PhysData
from StressTaskBehavior import StressBehaviorWithDrugs_CenterOut
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from statsmodels.stats.anova import anova_lm
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.formula.api as sm

TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/'
os.chdir(TDT_tank)



dir = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/"
TDT_tank_luigi = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Luigi - Neural Data/"
TDT_tank_mario = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/"
block_nums_luigi = [1,2,3]
block_nums_mario = [1,2,1]
power_bands = [[4,8],[13,30],[40,70]]


hdf_list_control_luigi = [['luig20181026_08_te1489.hdf', 'luig20181026_09_te1490.hdf', 'luig20181026_10_te1491.hdf'], \
			['luig20181028_07_te1506.hdf', 'luig20181028_08_te1507.hdf', 'luig20181028_09_te1508.hdf'], \
			['luig20181030_02_te1518.hdf', 'luig20181030_03_te1519.hdf', 'luig20181030_04_te1520.hdf'], \
			['luig20181112_03_te1561.hdf', 'luig20181112_04_te1562.hdf', 'luig20181112_05_te1563.hdf'], \
			['luig20181114_02_te1569.hdf', 'luig20181114_03_te1570.hdf', 'luig20181114_04_te1571.hdf'], \
			['luig20181115_02_te1573.hdf', 'luig20181115_03_te1574.hdf', 'luig20181115_04_te1575.hdf'], \
			['luig20181116_02_te1577.hdf', 'luig20181116_03_te1578.hdf', 'luig20181116_04_te1579.hdf'], \
			['luig20181119_03_te1591.hdf', 'luig20181119_04_te1592.hdf', 'luig20181119_05_te1593.hdf'], \
			]
hdf_list_stim_luigi = [['luig20181027_06_te1497.hdf', 'luig20181027_07_te1498.hdf', 'luig20181027_08_te1499.hdf'], \
			['luig20181029_06_te1514.hdf', 'luig20181029_07_te1515.hdf', 'luig20181029_08_te1516.hdf'], \
			['luig20181031_02_te1522.hdf', 'luig20181031_03_te1523.hdf', 'luig20181031_04_te1524.hdf'], \
			['luig20181111_03_te1556.hdf', 'luig20181111_04_te1557.hdf', 'luig20181111_05_te1558.hdf'], \
			['luig20181113_02_te1565.hdf', 'luig20181113_03_te1566.hdf', 'luig20181113_04_te1567.hdf'], \
			['luig20181117_02_te1581.hdf', 'luig20181117_03_te1582.hdf', 'luig20181117_04_te1583.hdf'], \
			['luig20181118_03_te1586.hdf', 'luig20181118_04_te1587.hdf', 'luig20181118_05_te1588.hdf'], \
			['luig20181120_02_te1595.hdf', 'luig20181120_03_te1596.hdf', 'luig20181120_04_te1597.hdf'], \
			]

filenames_control_luigi = [['Luigi20181026','Luigi20181026','Luigi20181026'], \
			['Luigi20181028','Luigi20181028','Luigi20181028'], \
			['Luigi20181030','Luigi20181030','Luigi20181030'], \
			['Luigi20181112','Luigi20181112','Luigi20181112'], \
			['Luigi20181114','Luigi20181114','Luigi20181114'], \
			['Luigi20181115','Luigi20181115','Luigi20181115'], \
			['Luigi20181116','Luigi20181116','Luigi20181116'], \
			['Luigi20181119','Luigi20181119','Luigi20181119'], \
			]
filenames_stim_luigi = [['Luigi20181027','Luigi20181027','Luigi20181027'], \
			['Luigi20181029','Luigi20181029','Luigi20181029'], \
			['Luigi20181031','Luigi20181031','Luigi20181031'], \
			['Luigi20181111','Luigi20181111','Luigi20181111'], \
			['Luigi20181113','Luigi20181113','Luigi20181113'], \
			['Luigi20181117','Luigi20181117','Luigi20181117'], \
			['Luigi20181118','Luigi20181118','Luigi20181118'], \
			['Luigi20181120','Luigi20181120','Luigi20181120'], \
			]

hdf_list_control_mario = [['mari20181004_02_te1350.hdf', 'mari20181004_03_te1351.hdf', 'mari20181004_04_te1352.hdf'], \
			['mari20181007_03_te1367.hdf','mari20181007_04_te1368.hdf', 'mari20181007_05_te1369.hdf'], \
			['mari20181008_02_te1375.hdf', 'mari20181008_04_te1377.hdf', 'mari20181008_05_te1378.hdf'], \
			['mari20181011_04_te1409.hdf', 'mari20181011_05_te1410.hdf', 'mari20181011_07_te1412.hdf'], \
			['mari20181013_02_te1422.hdf', 'mari20181013_03_te1423.hdf', 'mari20181013_05_te1425.hdf'], \
			['mari20181017_02_te1441.hdf', 'mari20181017_04_te1443.hdf', 'mari20181017_05_te1444.hdf'], \
			['mari20181025_03_te1478.hdf', 'mari20181025_05_te1480.hdf', 'mari20181025_06_te1481.hdf'], \
			['mari20181027_02_te1493.hdf', 'mari20181027_03_te1494.hdf', 'mari20181027_04_te1495.hdf'], \
			['mari20181101_04_te1528.hdf', 'mari20181101_05_te1529.hdf', 'mari20181101_06_te1530.hdf'], \
			['mari20181105_02_te1532.hdf', 'mari20181105_03_te1533.hdf', 'mari20181105_04_te1534.hdf'], \
			]
hdf_list_stim_mario = [['mari20181006_02_te1361.hdf', 'mari20181006_03_te1362.hdf', 'mari20181006_05_te1364.hdf'], \
			['mari20181009_09_te1390.hdf', 'mari20181009_10_te1391.hdf', 'mari20181009_11_te1392.hdf'], \
			['mari20181010_02_te1394.hdf', 'mari20181010_03_te1395.hdf', 'mari20181010_06_te1398.hdf'], \
			['mari20181018_02_te1446.hdf', 'mari20181018_03_te1447.hdf', 'mari20181018_04_te1448.hdf'], \
			['mari20181022_03_te1465.hdf', 'mari20181022_04_te1466.hdf', 'mari20181022_05_te1467.hdf'], \
			['mari20181024_02_te1469.hdf', 'mari20181024_03_te1470.hdf', 'mari20181024_04_te1471.hdf'], \
			['mari20181026_02_te1483.hdf', 'mari20181026_03_te1484.hdf', 'mari20181026_04_te1485.hdf'], \
			['mari20181028_02_te1501.hdf', 'mari20181028_03_te1502.hdf', 'mari20181028_05_te1504.hdf'], \
			['mari20181029_02_te1510.hdf', 'mari20181029_03_te1511.hdf', 'mari20181029_04_te1512.hdf'], \
			['mari20181107_04_te1538.hdf', 'mari20181107_05_te1539.hdf', 'mari20181107_06_te1540.hdf'], \
			['mari20181108_04_te1544.hdf', 'mari20181108_05_te1545.hdf', 'mari20181108_08_te1548.hdf'], \
			['mari20181109_03_te1551.hdf', 'mari20181109_04_te1552.hdf', 'mari20181109_05_te1553.hdf'], \
			]

filenames_control_mario = [['Mario20181004', 'Mario20181004','Mario20181004-1'], \
			['Mario20181007', 'Mario20181007','Mario20181007-1'], \
			['Mario20181008', 'Mario20181008','Mario20181008-1'], \
			['Mario20181011', 'Mario20181011','Mario20181011-1'], \
			['Mario20181013', 'Mario20181013','Mario20181013-1'], \
			['Mario20181017', 'Mario20181017','Mario20181017-1'], \
			['Mario20181025', 'Mario20181025','Mario20181025-1'], \
			['Mario20181027', 'Mario20181027','Mario20181027-1'], \
			['Mario20181101', 'Mario20181101','Mario20181101-1'], \
			['Mario20181105', 'Mario20181105','Mario20181105-1'], \
			]
filenames_stim_mario = [['Mario20181006', 'Mario20181006','Mario20181006-1'], \
			['Mario20181009', 'Mario20181009','Mario20181009-1'], \
			['Mario20181010', 'Mario20181010','Mario20181010-1'], \
			['Mario20181018', 'Mario20181018','Mario20181018-1'], \
			['Mario20181022', 'Mario20181022','Mario20181022-1'], \
			['Mario20181024', 'Mario20181024','Mario20181024-1'], \
			['Mario20181026', 'Mario20181026','Mario20181026-1'], \
			['Mario20181028', 'Mario20181028','Mario20181028-1'], \
			['Mario20181029', 'Mario20181029','Mario20181029-1'], \
			['Mario20181107', 'Mario20181107','Mario20181107-1'], \
			['Mario20181108', 'Mario20181108','Mario20181108-1'], \
			['Mario20181109', 'Mario20181109','Mario20181109-1'], \
			
			]


def running_mean(x, N):
	cumsum = np.nancumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 


##################################################################
# BEHAVIOR ONLY ANALYSIS
##################################################################
'''
Look at the average number of trials/min completed in each of the three blocks
of the task, as well as the average reaction time. Values are averaged
across the session.
'''

def ComputeRTandTrialsPerMin(hdf_list):
	'''
	Method to compute the average rt and trials completed/min per block (baseline, stress,
	treatment) for all days included in the hdf_list. The result of the method is a barplot
	with average values per condition (averaged across days) with error bars, as well as a 
	panda dataframe with all of the data which may be used in a statistical test (e.g. anova_lm)
	to test for significance.

	Input:
	- hdf_list: list; list of filenames for hdf files to be included.

	Output:
	- dta_behavior: panda dataframe; includes information on behavior condition (categorical
	variable; 0 = baseline, 1 = stress, 2 = treatment), reaction time (rt), and trials per min
	'''
	# Define variables
	rt_avg_baseline = np.zeros(len(hdf_list))
	rt_avg_stress = np.zeros(len(hdf_list))
	rt_avg_treatment = np.zeros(len(hdf_list))
	trial_rate_baseline = np.zeros(len(hdf_list))
	trial_rate_stress = np.zeros(len(hdf_list))
	trial_rate_treatment = np.zeros(len(hdf_list))
	dta_behavior = []

	# Compute metrics for each day
	for i,files in enumerate(hdf_list):
		# Get trials per min information
		baseline = StressBehaviorWithDrugs_CenterOut(files[0])
		stress = StressBehaviorWithDrugs_CenterOut(files[1])
		treatment = StressBehaviorWithDrugs_CenterOut(files[2])

		trial_rate_baseline[i] = baseline.trials_per_min
		trial_rate_stress[i] = stress.trials_per_min
		trial_rate_treatment[i] = treatment.trials_per_min

		# Get reaction time information
		if baseline.num_successful_trials > 0:
			rt_per_trial_baseline, total_vel = baseline.compute_rt_per_trial(range(0,baseline.num_successful_trials))
		else:
			rt_per_trial_baseline = np.nan
		rt_avg_baseline[i] = np.nanmean(rt_per_trial_baseline)
		if stress.num_successful_trials > 0:
			rt_per_trial_stress, total_vel = stress.compute_rt_per_trial(range(0,stress.num_successful_trials))
		else:
			rt_per_trial_stress = np.nan
		rt_avg_stress[i] = np.nanmean(rt_per_trial_stress)
		if treatment.num_successful_trials > 0:
			rt_per_trial_treatment, total_vel = treatment.compute_rt_per_trial(range(0,treatment.num_successful_trials))
		else:
			rt_per_trial_treatment = np.nan
		rt_avg_treatment[i] = np.nanmean(rt_per_trial_treatment)

		# Format data into pandas dataframe
		dta_behavior += [(0, rt_avg_baseline[i], trial_rate_baseline[i])]
		dta_behavior += [(1, rt_avg_stress[i], trial_rate_stress[i])]
		dta_behavior += [(2, rt_avg_treatment[i], trial_rate_treatment[i])]

	dta_behavior = pd.DataFrame(dta_behavior, columns=['behavior_condition','rt','trials_per_min'])

	return dta_behavior

hdf_locations_control_mario = []
for i in range(len(hdf_list_control_mario)):
	hdf_locations_control_mario += [[('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/'+hdf) for hdf in hdf_list_control_mario[i]]]

hdf_locations_stim_mario = []
for i in range(len(hdf_list_stim_mario)):
	hdf_locations_stim_mario += [[('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/'+hdf) for hdf in hdf_list_stim_mario[i]]]

hdf_locations_control_luigi = []
for i in range(len(hdf_list_control_luigi)):
	hdf_locations_control_luigi += [[('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/'+hdf) for hdf in hdf_list_control_luigi[i]]]

hdf_locations_stim_luigi = []
for i in range(len(hdf_list_stim_luigi)):
	hdf_locations_stim_luigi += [[('C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/'+hdf) for hdf in hdf_list_stim_luigi[i]]]

t = time.time()
dta_control_mario = ComputeRTandTrialsPerMin(hdf_locations_control_mario)
dta_stim_mario = ComputeRTandTrialsPerMin(hdf_locations_stim_mario)
dta_control_luigi = ComputeRTandTrialsPerMin(hdf_locations_control_luigi)
dta_stim_luigi = ComputeRTandTrialsPerMin(hdf_locations_stim_luigi)

# Separate one-way ANOVAs for now, could also two one-way MANOVA
formula1 = 'rt ~ C(behavior_condition)'
formula2 = 'trials_per_min ~ C(behavior_condition)'

modelSL1 = ols(formula1, dta_stim_luigi).fit()
modelSL2 = ols(formula2, dta_stim_luigi).fit()
aov_table_stim_luigi_rt = anova_lm(modelSL1, typ=2)
aov_table_stim_luigi_tpm = anova_lm(modelSL2, typ=2)

modelCL1 = ols(formula1, dta_control_luigi).fit()
modelCL2 = ols(formula2, dta_control_luigi).fit()
aov_table_control_luigi_rt = anova_lm(modelCL1, typ=2)
aov_table_control_luigi_tpm = anova_lm(modelCL2, typ=2)

modelSM1 = ols(formula1, dta_stim_mario).fit()
modelSM2 = ols(formula2, dta_stim_mario).fit()
aov_table_stim_mario_rt = anova_lm(modelSM1, typ=2)
aov_table_stim_mario_tpm = anova_lm(modelSM2, typ=2)

modelCM1 = sm.ols(formula1, dta_control_mario).fit()
modelCM2 = sm.ols(formula2, dta_control_mario).fit()
aov_table_control_mario_rt = anova_lm(modelCM1, typ=2)
aov_table_control_mario_tpm = anova_lm(modelCM2, typ=2)

elapsed = (time.time() - t)/60.
print('Took %f mins' % (elapsed))


'''
# Define variables
phys_dir = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/PowerFeatures/old/'
phys_stim = dict()
phys_control = dict()
ibi_md_stress = dict()
pupil_md_stress = dict()
ibi_md_treat = dict()
pupil_md_treat = dict()
pupil_time_base_mean_stim = np.zeros(len(filenames_stim_luigi))
pupil_time_stress_mean_stim = np.zeros(len(filenames_stim_luigi))
pupil_time_treat_mean_stim = np.zeros(len(filenames_stim_luigi))

pupil_time_stress_mean_early_stim = np.zeros(len(filenames_stim_luigi))
pupil_time_treat_mean_early_stim = np.zeros(len(filenames_stim_luigi))
pupil_time_stress_mean_late_stim = np.zeros(len(filenames_stim_luigi))
pupil_time_treat_mean_late_stim = np.zeros(len(filenames_stim_luigi))

ibi_time_base_mean_stim = np.zeros(len(filenames_stim_luigi))
ibi_time_stress_mean_stim = np.zeros(len(filenames_stim_luigi))
ibi_time_treat_mean_stim = np.zeros(len(filenames_stim_luigi))

ibi_time_stress_mean_early_stim = np.zeros(len(filenames_stim_luigi))
ibi_time_treat_mean_early_stim= np.zeros(len(filenames_stim_luigi))
ibi_time_stress_mean_late_stim = np.zeros(len(filenames_stim_luigi))
ibi_time_treat_mean_late_stim= np.zeros(len(filenames_stim_luigi))

pupil_time_base_mean_control = np.zeros(len(filenames_control_luigi))
pupil_time_stress_mean_control = np.zeros(len(filenames_control_luigi))
pupil_time_treat_mean_control = np.zeros(len(filenames_control_luigi))

pupil_time_stress_mean_early_control = np.zeros(len(filenames_control_luigi))
pupil_time_treat_mean_early_control = np.zeros(len(filenames_control_luigi))
pupil_time_stress_mean_late_control = np.zeros(len(filenames_control_luigi))
pupil_time_treat_mean_late_control = np.zeros(len(filenames_control_luigi))

ibi_time_base_mean_control = np.zeros(len(filenames_control_luigi))
ibi_time_stress_mean_control = np.zeros(len(filenames_control_luigi))
ibi_time_treat_mean_control = np.zeros(len(filenames_control_luigi))

ibi_time_stress_mean_early_control = np.zeros(len(filenames_control_luigi))
ibi_time_treat_mean_early_control = np.zeros(len(filenames_control_luigi))
ibi_time_stress_mean_late_control = np.zeros(len(filenames_control_luigi))
ibi_time_treat_mean_late_control = np.zeros(len(filenames_control_luigi))


for i,name in enumerate(filenames_stim_luigi):
	print(name)
	# Load data into class
	phys_filenames = [phys_dir + name[0] + '_b1_PhysFeatures.mat',  phys_dir + name[0] + '_b2_PhysFeatures.mat', phys_dir + name[0] + '_b3_PhysFeatures.mat']
	phys_stim[i] = StressTask_PhysData(phys_filenames)

	# Compute modulation depth
	#ibi_md_stress[i], pupil_md_stress[i], ibi_md_treat[i], pupil_md_treat[i] = phys[i].ModulationDepth('time', show_fig = False)

	# Compute average values per session
	ibi_time_base_mean_stim[i] = np.nanmean(phys_stim[i].ibi_time_base)
	pupil_time_base_mean_stim[i] = np.nanmean(phys_stim[i].pupil_time_base)
	ibi_time_stress_mean_stim[i] = np.nanmean(phys_stim[i].ibi_time_stress)
	pupil_time_stress_mean_stim[i] = np.nanmean(phys_stim[i].pupil_time_stress)
	ibi_time_treat_mean_stim[i] = np.nanmean(phys_stim[i].ibi_time_treat)
	pupil_time_treat_mean_stim[i] = np.nanmean(phys_stim[i].pupil_time_treat)

	# Compute average values for early half of stress and treatment sessions
	ibi_time_stress_mean_early_stim[i] = np.nanmean(phys_stim[i].ibi_time_stress[:int(len(phys_stim[i].ibi_time_stress)/2.)])
	pupil_time_stress_mean_early_stim[i] = np.nanmean(phys_stim[i].pupil_time_stress[:int(len(phys_stim[i].ibi_time_stress)/2.)])
	ibi_time_treat_mean_early_stim[i] = np.nanmean(phys_stim[i].ibi_time_treat[:int(len(phys_stim[i].ibi_time_stress)/2.)])
	pupil_time_treat_mean_early_stim[i] = np.nanmean(phys_stim[i].pupil_time_treat[:int(len(phys_stim[i].ibi_time_stress)/2.)])

	# Compute average values for late half of stress and treatment sessions
	ibi_time_stress_mean_late_stim[i] = np.nanmean(phys_stim[i].ibi_time_stress[int(len(phys_stim[i].ibi_time_stress)/2.):])
	pupil_time_stress_mean_late_stim[i] = np.nanmean(phys_stim[i].pupil_time_stress[int(len(phys_stim[i].ibi_time_stress)/2.):])
	ibi_time_treat_mean_late_stim[i] = np.nanmean(phys_stim[i].ibi_time_treat[int(len(phys_stim[i].ibi_time_stress)/2.):])
	pupil_time_treat_mean_late_stim[i] = np.nanmean(phys_stim[i].pupil_time_treat[int(len(phys_stim[i].ibi_time_stress)/2.):])

	# Compute smoothed ibi and pupil values
	ibi_base_smooth_stim = running_mean(phys_stim[i].ibi_time_base,100)
	ibi_stress_smooth_stim = running_mean(phys_stim[i].ibi_time_stress,100)
	ibi_treat_smooth_stim = running_mean(phys_stim[i].ibi_time_treat,100)
	pupil_base_smooth_stim = running_mean(phys_stim[i].pupil_time_base,100)
	pupil_stress_smooth_stim = running_mean(phys_stim[i].pupil_time_stress,100)
	pupil_treat_smooth_stim = running_mean(phys_stim[i].pupil_time_treat,100)

	"""
	plt.subplot(1,2,1)
	plt.plot(range(len(ibi_base_smooth_stim)), ibi_base_smooth_stim, 'b', label = 'baseline')
	plt.plot(range(len(ibi_base_smooth_stim), len(ibi_base_smooth_stim) + len(ibi_stress_smooth_stim)), ibi_stress_smooth_stim, 'r', label = 'stress')
	plt.plot(range(len(ibi_base_smooth_stim) + len(ibi_stress_smooth_stim), len(ibi_base_smooth_stim) + len(ibi_stress_smooth_stim) + len(ibi_treat_smooth_stim)), ibi_treat_smooth_stim, 'm', label = 'stim')
	plt.subplot(1,2,2)
	plt.plot(range(len(pupil_base_smooth_stim)), pupil_base_smooth_stim, 'b', label = 'baseline')
	plt.plot(range(len(pupil_base_smooth_stim), len(pupil_base_smooth_stim) + len(pupil_stress_smooth_stim)), pupil_stress_smooth_stim, 'r', label = 'stress')
	plt.plot(range(len(pupil_base_smooth_stim) + len(pupil_stress_smooth_stim), len(pupil_base_smooth_stim) + len(pupil_stress_smooth_stim) + len(pupil_treat_smooth_stim)), pupil_treat_smooth_stim, 'm', label = 'stim')
	plt.show()
	"""
	
for i,name in enumerate(filenames_control_luigi):
	print(name)
	# Load data into class
	phys_filenames = [phys_dir + name[0] + '_b1_PhysFeatures.mat',  phys_dir + name[0] + '_b2_PhysFeatures.mat', phys_dir + name[0] + '_b3_PhysFeatures.mat']
	phys_control[i] = StressTask_PhysData(phys_filenames)
	phys_control[i].IBIvsPup_ScatterPlot('time',save_fig = False)

	# Compute modulation depth
	#ibi_md_stress[i], pupil_md_stress[i], ibi_md_treat[i], pupil_md_treat[i] = phys[i].ModulationDepth('time', show_fig = False)

	# Compute average values per session
	ibi_time_base_mean_control[i] = np.nanmean(phys_control[i].ibi_time_base)
	pupil_time_base_mean_control[i] = np.nanmean(phys_control[i].pupil_time_base)
	ibi_time_stress_mean_control[i] = np.nanmean(phys_control[i].ibi_time_stress)
	pupil_time_stress_mean_control[i] = np.nanmean(phys_control[i].pupil_time_stress)
	ibi_time_treat_mean_control[i] = np.nanmean(phys_control[i].ibi_time_treat)
	pupil_time_treat_mean_control[i] = np.nanmean(phys_control[i].pupil_time_treat)

	# Compute average values for early half of stress and treatment sessions
	ibi_time_stress_mean_early_control[i] = np.nanmean(phys_control[i].ibi_time_stress[:int(len(phys_control[i].ibi_time_stress)/2.)])
	pupil_time_stress_mean_early_control[i] = np.nanmean(phys_control[i].pupil_time_stress[:int(len(phys_control[i].pupil_time_stress)/2.)])
	ibi_time_treat_mean_early_control[i] = np.nanmean(phys_control[i].ibi_time_treat[:int(len(phys_control[i].ibi_time_treat)/2.)])
	pupil_time_treat_mean_early_control[i] = np.nanmean(phys_control[i].pupil_time_treat[:int(len(phys_control[i].pupil_time_treat)/2.)])

	# Compute average values for late half of stress and treatment sessions
	ibi_time_stress_mean_late_control[i] = np.nanmean(phys_control[i].ibi_time_stress[int(len(phys_control[i].ibi_time_stress)/2.):])
	pupil_time_stress_mean_late_control[i] = np.nanmean(phys_control[i].pupil_time_stress[int(len(phys_control[i].pupil_time_stress)/2.):])
	ibi_time_treat_mean_late_control[i] = np.nanmean(phys_control[i].ibi_time_treat[int(len(phys_control[i].ibi_time_treat)/2.):])
	pupil_time_treat_mean_late_control[i] = np.nanmean(phys_control[i].pupil_time_treat[int(len(phys_control[i].pupil_time_treat)/2.):])


	# Compute smoothed ibi and pupil values
	ibi_base_smooth_control = running_mean(phys_control[i].ibi_time_base,100)
	ibi_stress_smooth_control = running_mean(phys_control[i].ibi_time_stress,100)
	ibi_treat_smooth_control = running_mean(phys_control[i].ibi_time_treat,100)
	pupil_base_smooth_control = running_mean(phys_control[i].pupil_time_base,100)
	pupil_stress_smooth_control = running_mean(phys_control[i].pupil_time_stress,100)
	pupil_treat_smooth_control = running_mean(phys_control[i].pupil_time_treat,100)
	
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(range(len(ibi_base_smooth_control)), ibi_base_smooth_control, 'b', label = 'baseline')
	plt.plot(range(len(ibi_base_smooth_control), len(ibi_base_smooth_control) + len(ibi_stress_smooth_control)), ibi_stress_smooth_control, 'r', label = 'stress')
	plt.plot(range(len(ibi_base_smooth_control) + len(ibi_stress_smooth_control), len(ibi_base_smooth_control) + len(ibi_stress_smooth_control) + len(ibi_treat_smooth_control)), ibi_treat_smooth_control, 'm', label = 'stim')
	plt.legend()
	plt.xlabel('Time epochs')
	plt.ylabel('IBI (s)')
	plt.subplot(1,2,2)
	plt.plot(range(len(pupil_base_smooth_control)), pupil_base_smooth_control, 'b', label = 'baseline')
	plt.plot(range(len(pupil_base_smooth_control), len(pupil_base_smooth_control) + len(pupil_stress_smooth_control)), pupil_stress_smooth_control, 'r', label = 'stress')
	plt.plot(range(len(pupil_base_smooth_control) + len(pupil_stress_smooth_control), len(pupil_base_smooth_control) + len(pupil_stress_smooth_control) + len(pupil_treat_smooth_control)), pupil_treat_smooth_control, 'm', label = 'stim')
	plt.xlabel('Time epochs')
	plt.ylabel('PD (a.u.)')
	plt.legend()
	plt.show()
	
	


#### Bar plot of average values (with error bars) per condition 
# compute mean for ibi
ibi_stress_mean_stim = np.nanmean(ibi_time_stress_mean_stim - ibi_time_base_mean_stim)
ibi_treat_mean_stim = np.nanmean(ibi_time_treat_mean_stim - ibi_time_base_mean_stim)
ibi_stress_mean_control = np.nanmean(ibi_time_stress_mean_control - ibi_time_base_mean_control)
ibi_treat_mean_control = np.nanmean(ibi_time_treat_mean_control - ibi_time_base_mean_control)
# compute error for ibi
ibi_stress_sem_stim = np.nanstd(ibi_time_stress_mean_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_stress_mean_stim))
ibi_treat_sem_stim = np.nanstd(ibi_time_treat_mean_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_treat_mean_stim))
ibi_stress_sem_control = np.nanstd(ibi_time_stress_mean_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_stress_mean_control))
ibi_treat_sem_control = np.nanstd(ibi_time_treat_mean_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_treat_mean_control))
# compute mean for pupil
pupil_stress_mean_stim = np.nanmean(pupil_time_stress_mean_stim - pupil_time_base_mean_stim)
pupil_treat_mean_stim = np.nanmean(pupil_time_treat_mean_stim - pupil_time_base_mean_stim)
pupil_stress_mean_control = np.nanmean(pupil_time_stress_mean_control - pupil_time_base_mean_control)
pupil_treat_mean_control = np.nanmean(pupil_time_treat_mean_control - pupil_time_base_mean_control)
# compute error for pupil
pupil_stress_sem_stim = np.nanstd(pupil_time_stress_mean_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_stress_mean_stim))
pupil_treat_sem_stim = np.nanstd(pupil_time_treat_mean_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_treat_mean_stim))
pupil_stress_sem_control = np.nanstd(pupil_time_stress_mean_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_stress_mean_control))
pupil_treat_sem_control = np.nanstd(pupil_time_treat_mean_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_treat_mean_control))


ibi_stim = [ibi_stress_mean_stim, ibi_treat_mean_stim]
ibi_control = [ibi_stress_mean_control, ibi_treat_mean_control]
ibi_stim_err = np.array([ibi_stress_sem_stim, ibi_treat_sem_stim])
ibi_control_err = np.array([ibi_stress_sem_control, ibi_treat_sem_control])

pupil_stim = [pupil_stress_mean_stim, pupil_treat_mean_stim]
pupil_control = [pupil_stress_mean_control, pupil_treat_mean_control]
pupil_stim_err = np.array([pupil_stress_sem_stim, pupil_treat_sem_stim])
pupil_control_err = np.array([pupil_stress_sem_control, pupil_treat_sem_control])

ind = np.arange(2)
width = 0.35

plt.figure()
plt.subplot(1,2,1)
plt.bar(ind, ibi_stim, width, color = 'b', yerr = 0.5*ibi_stim_err, label = 'Stim')
plt.bar(ind+0.35, ibi_control, width, color = 'c', yerr = 0.5*ibi_control_err, label = 'Control')
xticklabels = ['Stress - Baseline', 'Treatment - Baseline']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Values relative to Baseline')
plt.ylabel('Average IBI (s)')

plt.subplot(1,2,2)
plt.bar(ind, pupil_stim, width, color = 'b', yerr = 0.5*pupil_stim_err, label = 'Stim')
plt.bar(ind+0.35, pupil_control, width, color = 'c', yerr = 0.5*pupil_control_err, label = 'Control')
xticklabels = ['Stress - Baseline', 'Treatment - Baseline']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Values relative to Baseline')
plt.ylabel('Average PD (a.u.)')
plt.legend()
plt.show()


### Compute values for early vs late
# Baseline - all
ibi_base_mean_stim = np.nanmean(ibi_time_base_mean_stim)
ibi_base_mean_control = np.nanmean(ibi_time_base_mean_control)
ibi_base_sem_stim = np.nanstd(ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_base_mean_stim))
ibi_base_sem_control = np.nanstd(ibi_time_base_mean_control)/np.sqrt(len(ibi_time_base_mean_control))

pupil_base_mean_stim = np.nanmean(pupil_time_base_mean_stim)
pupil_base_mean_control = np.nanmean(pupil_time_base_mean_control)
pupil_base_sem_stim = np.nanstd(pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_base_mean_stim))
pupil_base_sem_control = np.nanstd(pupil_time_base_mean_control)/np.sqrt(len(pupil_time_base_mean_control))

# Stress - all
ibi_stress_mean_stim = np.nanmean(ibi_time_stress_mean_stim)
ibi_stress_mean_control = np.nanmean(ibi_time_stress_mean_control)
ibi_stress_sem_stim = np.nanstd(ibi_time_stress_mean_stim)/np.sqrt(len(ibi_time_stress_mean_stim))
ibi_stress_sem_control = np.nanstd(ibi_time_stress_mean_control)/np.sqrt(len(ibi_time_stress_mean_control))

pupil_stress_mean_stim = np.nanmean(pupil_time_stress_mean_stim)
pupil_stress_mean_control = np.nanmean(pupil_time_stress_mean_control)
pupil_stress_sem_stim = np.nanstd(pupil_time_stress_mean_stim)/np.sqrt(len(pupil_time_stress_mean_stim))
pupil_stress_sem_control = np.nanstd(pupil_time_stress_mean_control)/np.sqrt(len(pupil_time_stress_mean_control))


# Stress - early
ibi_stress_mean_early_stim = np.nanmean(ibi_time_stress_mean_early_stim)
ibi_stress_mean_early_control = np.nanmean(ibi_time_stress_mean_early_control)
ibi_stress_sem_early_stim = np.nanstd(ibi_time_stress_mean_early_stim)/np.sqrt(len(ibi_time_stress_mean_early_stim))
ibi_stress_sem_early_control = np.nanstd(ibi_time_stress_mean_early_control)/np.sqrt(len(ibi_time_stress_mean_early_control))

pupil_stress_mean_early_stim = np.nanmean(pupil_time_stress_mean_early_stim)
pupil_stress_mean_early_control = np.nanmean(pupil_time_stress_mean_early_control)
pupil_stress_sem_early_stim = np.nanstd(pupil_time_stress_mean_early_stim)/np.sqrt(len(pupil_time_stress_mean_early_stim))
pupil_stress_sem_early_control = np.nanstd(pupil_time_stress_mean_early_control)/np.sqrt(len(pupil_time_stress_mean_early_control))

# Stress - late
ibi_stress_mean_late_stim = np.nanmean(ibi_time_stress_mean_late_stim)
ibi_stress_mean_late_control = np.nanmean(ibi_time_stress_mean_late_control)
ibi_stress_sem_late_stim = np.nanstd(ibi_time_stress_mean_late_stim)/np.sqrt(len(ibi_time_stress_mean_late_stim))
ibi_stress_sem_late_control = np.nanstd(ibi_time_stress_mean_late_control)/np.sqrt(len(ibi_time_stress_mean_late_control))

pupil_stress_mean_late_stim = np.nanmean(pupil_time_stress_mean_late_stim)
pupil_stress_mean_late_control = np.nanmean(pupil_time_stress_mean_late_control)
pupil_stress_sem_late_stim = np.nanstd(pupil_time_stress_mean_late_stim)/np.sqrt(len(pupil_time_stress_mean_late_stim))
pupil_stress_sem_late_control = np.nanstd(pupil_time_stress_mean_late_control)/np.sqrt(len(pupil_time_stress_mean_late_control))

# Treatment - all
ibi_treat_mean_stim = np.nanmean(ibi_time_treat_mean_stim)
ibi_treat_mean_control = np.nanmean(ibi_time_treat_mean_control)
ibi_treat_sem_stim = np.nanstd(ibi_time_treat_mean_stim)/np.sqrt(len(ibi_time_treat_mean_stim))
ibi_treat_sem_control = np.nanstd(ibi_time_treat_mean_control)/np.sqrt(len(ibi_time_treat_mean_control))

pupil_treat_mean_stim = np.nanmean(pupil_time_treat_mean_stim)
pupil_treat_mean_control = np.nanmean(pupil_time_treat_mean_control)
pupil_treat_sem_stim = np.nanstd(pupil_time_treat_mean_stim)/np.sqrt(len(pupil_time_treat_mean_stim))
pupil_treat_sem_control = np.nanstd(pupil_time_treat_mean_control)/np.sqrt(len(pupil_time_treat_mean_control))


# Treat - early
ibi_treat_mean_early_stim = np.nanmean(ibi_time_treat_mean_early_stim)
ibi_treat_mean_early_control = np.nanmean(ibi_time_treat_mean_early_control)
ibi_treat_sem_early_stim = np.nanstd(ibi_time_treat_mean_early_stim)/np.sqrt(len(ibi_time_treat_mean_early_stim))
ibi_treat_sem_early_control = np.nanstd(ibi_time_treat_mean_early_control)/np.sqrt(len(ibi_time_treat_mean_early_control))

pupil_treat_mean_early_stim = np.nanmean(pupil_time_treat_mean_early_stim)
pupil_treat_mean_early_control = np.nanmean(pupil_time_treat_mean_early_control)
pupil_treat_sem_early_stim = np.nanstd(pupil_time_treat_mean_early_stim)/np.sqrt(len(pupil_time_treat_mean_early_stim))
pupil_treat_sem_early_control = np.nanstd(pupil_time_treat_mean_early_control)/np.sqrt(len(pupil_time_treat_mean_early_control))

# Treat - late
ibi_treat_mean_late_stim = np.nanmean(ibi_time_treat_mean_late_stim)
ibi_treat_mean_late_control = np.nanmean(ibi_time_treat_mean_late_control)
ibi_treat_sem_late_stim = np.nanstd(ibi_time_treat_mean_late_stim)/np.sqrt(len(ibi_time_treat_mean_late_stim))
ibi_treat_sem_late_control = np.nanstd(ibi_time_treat_mean_late_control)/np.sqrt(len(ibi_time_treat_mean_late_control))

pupil_treat_mean_late_stim = np.nanmean(pupil_time_treat_mean_late_stim)
pupil_treat_mean_late_control = np.nanmean(pupil_time_treat_mean_late_control)
pupil_treat_sem_late_stim = np.nanstd(pupil_time_treat_mean_late_stim)/np.sqrt(len(pupil_time_treat_mean_late_stim))
pupil_treat_sem_late_control = np.nanstd(pupil_time_treat_mean_late_control)/np.sqrt(len(pupil_time_treat_mean_late_control))


# Plot figure of individual block averages
ibi_stim = [ibi_base_mean_stim, ibi_stress_mean_stim, ibi_stress_mean_early_stim, ibi_stress_mean_late_stim,ibi_treat_mean_stim,ibi_treat_mean_early_stim, ibi_treat_mean_late_stim]
ibi_control = [ibi_base_mean_control, ibi_stress_mean_control, ibi_stress_mean_early_control, ibi_stress_mean_late_control,ibi_treat_mean_control,ibi_treat_mean_early_control, ibi_treat_mean_late_control]
ibi_stim_err = np.array([ibi_base_sem_stim, ibi_stress_sem_stim, ibi_stress_sem_early_stim, ibi_stress_sem_late_stim,ibi_treat_sem_stim,ibi_treat_sem_early_stim, ibi_treat_sem_late_stim])
ibi_control_err = np.array([ibi_base_sem_control, ibi_stress_sem_control, ibi_stress_sem_early_control, ibi_stress_sem_late_control,ibi_treat_sem_control,ibi_treat_sem_early_control, ibi_treat_sem_late_control])

pupil_stim = [pupil_base_mean_stim, pupil_stress_mean_stim, pupil_stress_mean_early_stim, pupil_stress_mean_late_stim,pupil_treat_mean_stim,pupil_treat_mean_early_stim, pupil_treat_mean_late_stim]
pupil_control = [pupil_base_mean_control, pupil_stress_mean_control, pupil_stress_mean_early_control, pupil_stress_mean_late_control,pupil_treat_mean_control,pupil_treat_mean_early_control, pupil_treat_mean_late_control]
pupil_stim_err = np.array([pupil_base_sem_stim, pupil_stress_sem_stim, pupil_stress_sem_early_stim, pupil_stress_sem_late_stim,pupil_treat_sem_stim,pupil_treat_sem_early_stim, pupil_treat_sem_late_stim])
pupil_control_err = np.array([pupil_base_sem_control, pupil_stress_sem_control, pupil_stress_sem_early_control, pupil_stress_sem_late_control,pupil_treat_sem_control,pupil_treat_sem_early_control, pupil_treat_sem_late_control])


print(ibi_stim)
print(ibi_stim_err)
ind = np.arange(7)
width = 0.35
plt.figure()
plt.subplot(1,2,1)
plt.bar(ind, ibi_stim, width, color = 'b', yerr = 0.5*ibi_stim_err, label = 'Stim')
plt.bar(ind+0.35, ibi_control, width, color = 'c', yerr = 0.5*ibi_control_err, label = 'Control')
xticklabels = ['Baseline', 'Stress', 'Stress-Early','Stress-Late','Treatment','Treatmeant-Early', 'Treatment-Late']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Blocks')
plt.ylabel('Average IBI (s)')

plt.subplot(1,2,2)
plt.bar(ind, pupil_stim, width, color = 'b', yerr = 0.5*pupil_stim_err, label = 'Stim')
plt.bar(ind+0.35, pupil_control, width, color = 'c', yerr = 0.5*pupil_control_err, label = 'Control')
xticklabels = ['Baseline', 'Stress', 'Stress-Early','Stress-Late','Treatment','Treatmeant-Early', 'Treatment-Late']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Blocks')
plt.ylabel('Average PD (a.u.)')
plt.legend()
plt.show()


# Plots relative to baseline
# Stress - all
ibi_stress_mean_stim = np.nanmean(ibi_time_stress_mean_stim - ibi_time_base_mean_stim)
ibi_stress_mean_control = np.nanmean(ibi_time_stress_mean_control - ibi_time_base_mean_control)
ibi_stress_sem_stim = np.nanstd(ibi_time_stress_mean_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_stress_mean_stim))
ibi_stress_sem_control = np.nanstd(ibi_time_stress_mean_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_stress_mean_control))

pupil_stress_mean_stim = np.nanmean(pupil_time_stress_mean_stim - pupil_time_base_mean_stim)
pupil_stress_mean_control = np.nanmean(pupil_time_stress_mean_control - pupil_time_base_mean_control)
pupil_stress_sem_stim = np.nanstd(pupil_time_stress_mean_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_stress_mean_stim))
pupil_stress_sem_control = np.nanstd(pupil_time_stress_mean_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_stress_mean_control))


# Stress - early
ibi_stress_mean_early_stim = np.nanmean(ibi_time_stress_mean_early_stim - ibi_time_base_mean_stim)
ibi_stress_mean_early_control = np.nanmean(ibi_time_stress_mean_early_control - ibi_time_base_mean_control)
ibi_stress_sem_early_stim = np.nanstd(ibi_time_stress_mean_early_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_stress_mean_early_stim))
ibi_stress_sem_early_control = np.nanstd(ibi_time_stress_mean_early_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_stress_mean_early_control))

pupil_stress_mean_early_stim = np.nanmean(pupil_time_stress_mean_early_stim - pupil_time_base_mean_stim)
pupil_stress_mean_early_control = np.nanmean(pupil_time_stress_mean_early_control - pupil_time_base_mean_control)
pupil_stress_sem_early_stim = np.nanstd(pupil_time_stress_mean_early_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_stress_mean_early_stim))
pupil_stress_sem_early_control = np.nanstd(pupil_time_stress_mean_early_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_stress_mean_early_control))

# Stress - late
ibi_stress_mean_late_stim = np.nanmean(ibi_time_stress_mean_late_stim - ibi_time_base_mean_stim)
ibi_stress_mean_late_control = np.nanmean(ibi_time_stress_mean_late_control - ibi_time_base_mean_control)
ibi_stress_sem_late_stim = np.nanstd(ibi_time_stress_mean_late_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_stress_mean_late_stim))
ibi_stress_sem_late_control = np.nanstd(ibi_time_stress_mean_late_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_stress_mean_late_control))

pupil_stress_mean_late_stim = np.nanmean(pupil_time_stress_mean_late_stim - pupil_time_stress_mean_stim)
pupil_stress_mean_late_control = np.nanmean(pupil_time_stress_mean_late_control - pupil_time_stress_mean_control)
pupil_stress_sem_late_stim = np.nanstd(pupil_time_stress_mean_late_stim - pupil_time_stress_mean_stim)/np.sqrt(len(pupil_time_stress_mean_late_stim))
pupil_stress_sem_late_control = np.nanstd(pupil_time_stress_mean_late_control - pupil_time_stress_mean_control)/np.sqrt(len(pupil_time_stress_mean_late_control))

# Treatment - all
ibi_treat_mean_stim = np.nanmean(ibi_time_treat_mean_stim - ibi_time_base_mean_stim)
ibi_treat_mean_control = np.nanmean(ibi_time_treat_mean_control - ibi_time_base_mean_control)
ibi_treat_sem_stim = np.nanstd(ibi_time_treat_mean_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_treat_mean_stim))
ibi_treat_sem_control = np.nanstd(ibi_time_treat_mean_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_treat_mean_control))

pupil_treat_mean_stim = np.nanmean(pupil_time_treat_mean_stim - pupil_time_base_mean_stim)
pupil_treat_mean_control = np.nanmean(pupil_time_treat_mean_control - pupil_time_base_mean_control)
pupil_treat_sem_stim = np.nanstd(pupil_time_treat_mean_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_treat_mean_stim))
pupil_treat_sem_control = np.nanstd(pupil_time_treat_mean_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_treat_mean_control))


# Treat - early
ibi_treat_mean_early_stim = np.nanmean(ibi_time_treat_mean_early_stim - ibi_time_base_mean_stim)
ibi_treat_mean_early_control = np.nanmean(ibi_time_treat_mean_early_control - ibi_time_base_mean_control)
ibi_treat_sem_early_stim = np.nanstd(ibi_time_treat_mean_early_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_treat_mean_early_stim))
ibi_treat_sem_early_control = np.nanstd(ibi_time_treat_mean_early_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_treat_mean_early_control))

pupil_treat_mean_early_stim = np.nanmean(pupil_time_treat_mean_early_stim - pupil_time_base_mean_stim)
pupil_treat_mean_early_control = np.nanmean(pupil_time_treat_mean_early_control - pupil_time_base_mean_control)
pupil_treat_sem_early_stim = np.nanstd(pupil_time_treat_mean_early_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_treat_mean_early_stim))
pupil_treat_sem_early_control = np.nanstd(pupil_time_treat_mean_early_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_treat_mean_early_control))

# Treat - late
ibi_treat_mean_late_stim = np.nanmean(ibi_time_treat_mean_late_stim - ibi_time_base_mean_stim)
ibi_treat_mean_late_control = np.nanmean(ibi_time_treat_mean_late_control - ibi_time_base_mean_control)
ibi_treat_sem_late_stim = np.nanstd(ibi_time_treat_mean_late_stim - ibi_time_base_mean_stim)/np.sqrt(len(ibi_time_treat_mean_late_stim))
ibi_treat_sem_late_control = np.nanstd(ibi_time_treat_mean_late_control - ibi_time_base_mean_control)/np.sqrt(len(ibi_time_treat_mean_late_control))

pupil_treat_mean_late_stim = np.nanmean(pupil_time_treat_mean_late_stim - pupil_time_base_mean_stim)
pupil_treat_mean_late_control = np.nanmean(pupil_time_treat_mean_late_control - pupil_time_base_mean_control)
pupil_treat_sem_late_stim = np.nanstd(pupil_time_treat_mean_late_stim - pupil_time_base_mean_stim)/np.sqrt(len(pupil_time_treat_mean_late_stim))
pupil_treat_sem_late_control = np.nanstd(pupil_time_treat_mean_late_control - pupil_time_base_mean_control)/np.sqrt(len(pupil_time_treat_mean_late_control))


# Plot figure of individual block averages
ibi_stim = [ibi_stress_mean_stim, ibi_stress_mean_early_stim, ibi_stress_mean_late_stim,ibi_treat_mean_stim,ibi_treat_mean_early_stim, ibi_treat_mean_late_stim]
ibi_control = [-ibi_stress_mean_control, -ibi_stress_mean_early_control, -ibi_stress_mean_late_control,ibi_treat_mean_control,ibi_treat_mean_early_control, ibi_treat_mean_late_control]
ibi_stim_err = np.array([ibi_stress_sem_stim, ibi_stress_sem_early_stim, ibi_stress_sem_late_stim,ibi_treat_sem_stim,ibi_treat_sem_early_stim, ibi_treat_sem_late_stim])
ibi_control_err = np.array([ibi_stress_sem_control, ibi_stress_sem_early_control, ibi_stress_sem_late_control,ibi_treat_sem_control,ibi_treat_sem_early_control, ibi_treat_sem_late_control])

pupil_stim = [-pupil_stress_mean_stim, -pupil_stress_mean_early_stim, -pupil_stress_mean_late_stim,pupil_treat_mean_stim,pupil_treat_mean_early_stim, pupil_treat_mean_late_stim]
pupil_control = [pupil_stress_mean_control, pupil_stress_mean_early_control, pupil_stress_mean_late_control,pupil_treat_mean_control,pupil_treat_mean_early_control, pupil_treat_mean_late_control]
pupil_stim_err = np.array([pupil_stress_sem_stim, pupil_stress_sem_early_stim, pupil_stress_sem_late_stim,pupil_treat_sem_stim,pupil_treat_sem_early_stim, pupil_treat_sem_late_stim])
pupil_control_err = np.array([pupil_stress_sem_control, pupil_stress_sem_early_control, pupil_stress_sem_late_control,pupil_treat_sem_control,pupil_treat_sem_early_control, pupil_treat_sem_late_control])


print(ibi_stim)
print(ibi_stim_err)
ind = np.arange(6)
width = 0.35
plt.figure()
plt.subplot(1,2,1)
plt.bar(ind, ibi_stim, width, color = 'b', yerr = 0.5*ibi_stim_err, label = 'Stim')
plt.bar(ind+0.35, ibi_control, width, color = 'c', yerr = 0.5*ibi_control_err, label = 'Control')
xticklabels = ['Stress', 'Stress-Early','Stress-Late','Treatment','Treatmeant-Early', 'Treatment-Late']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Blocks - Baseline')
plt.ylabel('Average IBI (s)')

plt.subplot(1,2,2)
plt.bar(ind, pupil_stim, width, color = 'b', yerr = 0.5*pupil_stim_err, label = 'Stim')
plt.bar(ind+0.35, pupil_control, width, color = 'c', yerr = 0.5*pupil_control_err, label = 'Control')
xticklabels = ['Stress', 'Stress-Early','Stress-Late','Treatment','Treatmeant-Early', 'Treatment-Late']
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.xlabel('Blocks - Baseline')
plt.ylabel('Average PD (a.u.)')
plt.legend()
plt.show()
'''