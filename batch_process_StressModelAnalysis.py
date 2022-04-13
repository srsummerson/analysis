from StressModelAndTreatmentAnalysis import StressTaskAnalysis_ComputePowerFeatures, StressTask_PhysData, StressTask_PowerFeatureData
from StressTaskBehavior import StressBehaviorWithDrugs_CenterOut
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from statsmodels.stats.anova import anova_lm
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.formula.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy as sp
import scipy.io as spio
from scipy import stats
import seaborn as sns
from heatmap import corrplot, heatmap
import pandas as pd

#TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/'
#os.chdir(TDT_tank)



dir = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/"
TDT_tank_luigi = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Luigi - Neural Data/"
TDT_tank_mario = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/"
block_nums_luigi = [1,2,3]
block_nums_mario = [1,2,1]
power_bands = [[8,13],[13,30],[40,70],[70,200]]

mario_ofc_indices = np.array([20, 4, 18, 2, 28, 12, 26, 10, 27, 11, 25, 9]) - 1
mario_vmpfc_indices = np.array([30, 14, 15, 29, 13, 6, 22]) - 1
mario_cd_indices = np.array([1, 3, 4, 17, 18, 20, 35, 37, 38, 40, 41, 51, 53, 54, 56, 57, 63, 64, 65, 67, 70, 72, 73, 75, 81, 83, 86, 88, 89, 96, 100, 112, 114, 126, 130-1, 140-2, 143-2, 146-3, 156-3, 157-3, 159-3]) - 1  # need to account for channels 129, 131, and 145 being deleted


luigi_ofc_indices = np.arange(33,65) - 1
luigi_vmpfc_indices = np.arange(32) - 1
luigi_cd_indices = np.arange(65,97) - 1


luigi_no_vmpfc = ['Luigi20181026', 'Luigi20181027']


hdf_list_control_luigi = [['luig20181026_08_te1489.hdf', 'luig20181026_09_te1490.hdf', 'luig20181026_10_te1491.hdf'],
			['luig20181028_07_te1506.hdf', 'luig20181028_08_te1507.hdf', 'luig20181028_09_te1508.hdf'], \
			['luig20181030_02_te1518.hdf', 'luig20181030_03_te1519.hdf', 'luig20181030_04_te1520.hdf'], \
			['luig20181112_03_te1561.hdf', 'luig20181112_04_te1562.hdf', 'luig20181112_05_te1563.hdf'], \
			['luig20181114_02_te1569.hdf', 'luig20181114_03_te1570.hdf', 'luig20181114_04_te1571.hdf'], \
			['luig20181115_02_te1573.hdf', 'luig20181115_03_te1574.hdf', 'luig20181115_04_te1575.hdf'], \
			['luig20181116_02_te1577.hdf', 'luig20181116_03_te1578.hdf', 'luig20181116_04_te1579.hdf'], \
			['luig20181119_03_te1591.hdf', 'luig20181119_04_te1592.hdf', 'luig20181119_05_te1593.hdf'], \
			]
hdf_list_stim_luigi = [['luig20181027_06_te1497.hdf', 'luig20181027_07_te1498.hdf', 'luig20181027_08_te1499.hdf'],
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
"""

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
	trial_rate_baseline_early = np.zeros(len(hdf_list))
	trial_rate_stress_early = np.zeros(len(hdf_list))
	trial_rate_treatment_early = np.zeros(len(hdf_list))
	trial_rate_baseline_late = np.zeros(len(hdf_list))
	trial_rate_stress_late = np.zeros(len(hdf_list))
	trial_rate_treatment_late = np.zeros(len(hdf_list))
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

		trial_rate_baseline_early[i] = baseline.trials_per_min_early
		trial_rate_stress_early[i] = stress.trials_per_min_early
		trial_rate_treatment_early[i] = treatment.trials_per_min_early

		trial_rate_baseline_late[i] = baseline.trials_per_min_late
		trial_rate_stress_late[i] = stress.trials_per_min_late
		trial_rate_treatment_late[i] = treatment.trials_per_min_late

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

	trial_rate = [trial_rate_baseline, trial_rate_stress, trial_rate_treatment, \
				trial_rate_baseline_early, trial_rate_stress_early, trial_rate_treatment_early, \
				trial_rate_baseline_late, trial_rate_stress_late, trial_rate_treatment_late]

	rt_avg = [rt_avg_baseline, rt_avg_stress, rt_avg_treatment]

	return dta_behavior, rt_avg, trial_rate

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
dta_control_mario, rt_avg_control_mario, trial_rate_control_mario = ComputeRTandTrialsPerMin(hdf_locations_control_mario)
[rt_avg_baseline_control_mario, rt_avg_stress_control_mario, rt_avg_treatment_control_mario] = rt_avg_control_mario
[trial_rate_baseline_control_mario, trial_rate_stress_control_mario, trial_rate_treatment_control_mario, \
trial_rate_baseline_control_early_mario, trial_rate_stress_control_early_mario, trial_rate_treatment_control_early_mario, \
trial_rate_baseline_control_late_mario, trial_rate_stress_control_late_mario, trial_rate_treatment_control_late_mario] = trial_rate_control_mario

dta_stim_mario, rt_avg_stim_mario, trial_rate_stim_mario = ComputeRTandTrialsPerMin(hdf_locations_stim_mario)
[rt_avg_baseline_stim_mario, rt_avg_stress_stim_mario, rt_avg_treatment_stim_mario] = rt_avg_stim_mario
[trial_rate_baseline_stim_mario, trial_rate_stress_stim_mario, trial_rate_treatment_stim_mario,\
trial_rate_baseline_stim_early_mario, trial_rate_stress_stim_early_mario, trial_rate_treatment_stim_early_mario,\
trial_rate_baseline_stim_late_mario, trial_rate_stress_stim_late_mario, trial_rate_treatment_stim_late_mario] = trial_rate_stim_mario

dta_control_luigi, rt_avg_control_luigi, trial_rate_control_luigi = ComputeRTandTrialsPerMin(hdf_locations_control_luigi)
[rt_avg_baseline_control_luigi, rt_avg_stress_control_luigi, rt_avg_treatment_control_luigi] = rt_avg_control_luigi 
[trial_rate_baseline_control_luigi, trial_rate_stress_control_luigi, trial_rate_treatment_control_luigi, \
trial_rate_baseline_control_early_luigi, trial_rate_stress_control_early_luigi, trial_rate_treatment_control_early_luigi, \
trial_rate_baseline_control_late_luigi, trial_rate_stress_control_late_luigi, trial_rate_treatment_control_late_luigi] = trial_rate_control_luigi

dta_stim_luigi, rt_avg_stim_luigi, trial_rate_stim_luigi = ComputeRTandTrialsPerMin(hdf_locations_stim_luigi)
[rt_avg_baseline_stim_luigi, rt_avg_stress_stim_luigi, rt_avg_treatment_stim_luigi] = rt_avg_stim_luigi 
[trial_rate_baseline_stim_luigi, trial_rate_stress_stim_luigi, trial_rate_treatment_stim_luigi, \
trial_rate_baseline_stim_early_luigi, trial_rate_stress_stim_early_luigi, trial_rate_treatment_stim_early_luigi, \
trial_rate_baseline_stim_late_luigi, trial_rate_stress_stim_late_luigi, trial_rate_treatment_stim_late_luigi] = trial_rate_stim_luigi

# Separate one-way ANOVAs for now, could also two one-way MANOVA
formula1 = 'rt ~ C(behavior_condition)'
formula2 = 'trials_per_min ~ C(behavior_condition)'

modelSL1 = sm.ols(formula1, dta_stim_luigi).fit()
modelSL2 = sm.ols(formula2, dta_stim_luigi).fit()
aov_table_stim_luigi_rt = anova_lm(modelSL1, typ=2)
aov_table_stim_luigi_tpm = anova_lm(modelSL2, typ=2)

modelCL1 = sm.ols(formula1, dta_control_luigi).fit()
modelCL2 = sm.ols(formula2, dta_control_luigi).fit()
aov_table_control_luigi_rt = anova_lm(modelCL1, typ=2)
aov_table_control_luigi_tpm = anova_lm(modelCL2, typ=2)

modelSM1 = sm.ols(formula1, dta_stim_mario).fit()
modelSM2 = sm.ols(formula2, dta_stim_mario).fit()
aov_table_stim_mario_rt = anova_lm(modelSM1, typ=2)
aov_table_stim_mario_tpm = anova_lm(modelSM2, typ=2)

modelCM1 = sm.ols(formula1, dta_control_mario).fit()
modelCM2 = sm.ols(formula2, dta_control_mario).fit()
aov_table_control_mario_rt = anova_lm(modelCM1, typ=2)
aov_table_control_mario_tpm = anova_lm(modelCM2, typ=2)

elapsed = (time.time() - t)/60.
print('Took %f mins' % (elapsed))

## Make plots of average rt and trials per min with p-vales reports
ind = np.arange(4)
width = 0.35/2.

# Combine baseline and stress data across days to make plots with baseline and stress data ONLY
rt_baseline_mario = np.append(rt_avg_baseline_control_mario, rt_avg_baseline_stim_mario)
rt_stress_mario = np.append(rt_avg_stress_control_mario, rt_avg_stress_stim_mario)
tpm_baseline_mario = np.append(trial_rate_baseline_control_mario, trial_rate_baseline_stim_mario)
tpm_stress_mario = np.append(trial_rate_stress_control_mario, trial_rate_stress_stim_mario)
tpm_baseline_early_mario = np.append(trial_rate_baseline_control_early_mario, trial_rate_baseline_stim_early_mario)
tpm_stress_early_mario = np.append(trial_rate_stress_control_early_mario, trial_rate_stress_stim_early_mario)
tpm_baseline_late_mario = np.append(trial_rate_baseline_control_late_mario, trial_rate_baseline_stim_late_mario)
tpm_stress_late_mario = np.append(trial_rate_stress_control_late_mario, trial_rate_stress_stim_late_mario)

rt_baseline_luigi = np.append(rt_avg_baseline_control_luigi, rt_avg_baseline_stim_luigi)
rt_stress_luigi = np.append(rt_avg_stress_control_luigi, rt_avg_stress_stim_luigi)
tpm_baseline_luigi = np.append(trial_rate_baseline_control_luigi, trial_rate_baseline_stim_luigi)
tpm_stress_luigi = np.append(trial_rate_stress_control_luigi, trial_rate_stress_stim_luigi)
tpm_baseline_early_luigi = np.append(trial_rate_baseline_control_early_luigi, trial_rate_baseline_stim_early_luigi)
tpm_stress_early_luigi = np.append(trial_rate_stress_control_early_luigi, trial_rate_stress_stim_early_luigi)
tpm_baseline_late_luigi = np.append(trial_rate_baseline_control_late_luigi, trial_rate_baseline_stim_late_luigi)
tpm_stress_late_luigi = np.append(trial_rate_stress_control_late_luigi, trial_rate_stress_stim_late_luigi)

rt_mario = [np.nanmean(rt_baseline_mario), np.nanmean(rt_stress_mario)]
rt_mario_sem = [np.nanstd(rt_baseline_mario)/np.sqrt(len(rt_baseline_mario)), np.nanstd(rt_stress_mario)/np.sqrt(len(rt_stress_mario))]
tpm_mario = [np.nanmean(tpm_baseline_mario), np.nanmean(tpm_stress_mario), np.nanmean(tpm_stress_early_mario), np.nanmean(tpm_stress_late_mario)]
tpm_mario_sem = [np.nanstd(tpm_baseline_mario)/np.sqrt(len(tpm_baseline_mario)), np.nanstd(tpm_stress_mario)/np.sqrt(len(tpm_stress_mario)), \
				np.nanstd(tpm_stress_early_mario)/np.sqrt(len(tpm_stress_early_mario)), np.nanstd(tpm_stress_late_mario)/np.sqrt(len(tpm_stress_late_mario))]

rt_luigi = [np.nanmean(rt_baseline_luigi), np.nanmean(rt_stress_luigi)]
rt_luigi_sem = [np.nanstd(rt_baseline_luigi)/np.sqrt(len(rt_baseline_luigi)), np.nanstd(rt_stress_luigi)/np.sqrt(len(rt_stress_luigi))]
tpm_luigi = [np.nanmean(tpm_baseline_luigi), np.nanmean(tpm_stress_luigi), np.nanmean(tpm_stress_early_luigi), np.nanmean(tpm_stress_late_luigi)]
tpm_luigi_sem = [np.nanstd(tpm_baseline_luigi)/np.sqrt(len(tpm_baseline_luigi)), np.nanstd(tpm_stress_luigi)/np.sqrt(len(tpm_stress_luigi)), \
				np.nanstd(tpm_stress_early_luigi)/np.sqrt(len(tpm_stress_early_luigi)), np.nanstd(tpm_stress_late_luigi)/np.sqrt(len(tpm_stress_late_luigi))]

plt.figure()
plt.subplot(1,2,1)
plt.bar(ind[:2],rt_mario,yerr=rt_mario_sem,color = 'm')
xticklabels = ['Baseline','Anxiety']
plt.xticks(ind[:2],xticklabels)
plt.ylabel('Reaction time (s)')
plt.title('Mario')
plt.legend()

plt.subplot(1,2,2)
plt.bar(ind[:2],rt_luigi,yerr=rt_luigi_sem,color = 'm')
xticklabels = ['Baseline','Anxiety']
plt.xticks(ind[:2],xticklabels)
plt.ylabel('Reaction time (s)')
plt.title('Luigi')
plt.legend()

plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.bar(ind,tpm_mario,yerr=tpm_mario_sem,color = 'b')
xticklabels = ['Baseline','Anxiety', 'Anxiety - Early', 'Anxiety - Late']
plt.xticks(ind,xticklabels)
plt.ylabel('Trials per min')
plt.ylim((0,17))
plt.title('Mario')
plt.legend()

plt.subplot(1,2,2)
plt.bar(ind,tpm_luigi,yerr=tpm_luigi_sem,color = 'b')
xticklabels = ['Baseline','Anxiety', 'Anxiety - Early', 'Anxiety - Late']
plt.xticks(ind,xticklabels)
plt.ylabel('Trials per min')
plt.title('Luigi')
plt.ylim((0,17))
plt.legend()

plt.show()

# Look at baseline, stress, and treatment separate for each day (stim day vs control day)
ind = np.arange(5)

rt_control_mario = [np.nanmean(rt_avg_baseline_control_mario), np.nanmean(rt_avg_stress_control_mario), np.nanmean(rt_avg_treatment_control_mario)]
rt_control_mario_sem = [np.nanstd(rt_avg_baseline_control_mario)/np.sqrt(len(rt_avg_baseline_control_mario)), np.nanstd(rt_avg_stress_control_mario)/np.sqrt(len(rt_avg_stress_control_mario)), np.nanstd(rt_avg_treatment_control_mario)/np.sqrt(len(rt_avg_treatment_control_mario))]

rt_stim_mario = [np.nanmean(rt_avg_baseline_stim_mario), np.nanmean(rt_avg_stress_stim_mario), np.nanmean(rt_avg_treatment_stim_mario)]
rt_stim_mario_sem = [np.nanstd(rt_avg_baseline_stim_mario)/np.sqrt(len(rt_avg_baseline_stim_mario)), np.nanstd(rt_avg_stress_stim_mario)/np.sqrt(len(rt_avg_stress_stim_mario)), np.nanstd(rt_avg_treatment_stim_mario)/np.sqrt(len(rt_avg_treatment_stim_mario))]

rt_control_luigi = [np.nanmean(rt_avg_baseline_control_luigi), np.nanmean(rt_avg_stress_control_luigi), np.nanmean(rt_avg_treatment_control_luigi)]
rt_control_luigi_sem = [np.nanstd(rt_avg_baseline_control_luigi)/np.sqrt(len(rt_avg_baseline_control_luigi)), np.nanstd(rt_avg_stress_control_luigi)/np.sqrt(len(rt_avg_stress_control_luigi)), np.nanstd(rt_avg_treatment_control_luigi)/np.sqrt(len(rt_avg_treatment_control_luigi))]

rt_stim_luigi = [np.nanmean(rt_avg_baseline_stim_luigi), np.nanmean(rt_avg_stress_stim_luigi), np.nanmean(rt_avg_treatment_stim_luigi)]
rt_stim_luigi_sem = [np.nanstd(rt_avg_baseline_stim_luigi)/np.sqrt(len(rt_avg_baseline_stim_luigi)), np.nanstd(rt_avg_stress_stim_luigi)/np.sqrt(len(rt_avg_stress_stim_luigi)), np.nanstd(rt_avg_treatment_stim_luigi)/np.sqrt(len(rt_avg_treatment_stim_luigi))]

tpm_control_mario = [np.nanmean(trial_rate_baseline_control_mario), np.nanmean(trial_rate_stress_control_mario), np.nanmean(trial_rate_treatment_control_mario), \
					np.nanmean(trial_rate_treatment_control_early_mario), np.nanmean(trial_rate_treatment_control_late_mario)]
tpm_control_mario_sem = [np.nanstd(trial_rate_baseline_control_mario)/np.sqrt(len(trial_rate_baseline_control_mario)), \
					np.nanstd(trial_rate_stress_control_mario)/np.sqrt(len(trial_rate_stress_control_mario)), \
					np.nanstd(trial_rate_treatment_control_mario)/np.sqrt(len(trial_rate_treatment_control_mario)), \
					np.nanstd(trial_rate_treatment_control_early_mario)/np.sqrt(len(trial_rate_treatment_control_early_mario)), \
					np.nanstd(trial_rate_treatment_control_late_mario)/np.sqrt(len(trial_rate_treatment_control_late_mario))]

tpm_stim_mario = [np.nanmean(trial_rate_baseline_stim_mario), np.nanmean(trial_rate_stress_stim_mario), np.nanmean(trial_rate_treatment_stim_mario), \
					np.nanmean(trial_rate_treatment_stim_early_mario), np.nanmean(trial_rate_treatment_stim_late_mario)]
tpm_stim_mario_sem = [np.nanstd(trial_rate_baseline_stim_mario)/np.sqrt(len(trial_rate_baseline_stim_mario)), \
					np.nanstd(trial_rate_stress_stim_mario)/np.sqrt(len(trial_rate_stress_stim_mario)), \
					np.nanstd(trial_rate_treatment_stim_mario)/np.sqrt(len(trial_rate_treatment_stim_mario)), 
					np.nanstd(trial_rate_treatment_stim_early_mario)/np.sqrt(len(trial_rate_treatment_stim_early_mario)), 
					np.nanstd(trial_rate_treatment_stim_late_mario)/np.sqrt(len(trial_rate_treatment_stim_late_mario))]

tpm_control_luigi = [np.nanmean(trial_rate_baseline_control_luigi), np.nanmean(trial_rate_stress_control_luigi), np.nanmean(trial_rate_treatment_control_luigi), \
					np.nanmean(trial_rate_treatment_control_early_luigi), np.nanmean(trial_rate_treatment_control_late_luigi)]
tpm_control_luigi_sem = [np.nanstd(trial_rate_baseline_control_luigi)/np.sqrt(len(trial_rate_baseline_control_luigi)), \
					np.nanstd(trial_rate_stress_control_luigi)/np.sqrt(len(trial_rate_stress_control_luigi)), \
					np.nanstd(trial_rate_treatment_control_luigi)/np.sqrt(len(trial_rate_treatment_control_luigi)), \
					np.nanstd(trial_rate_treatment_control_early_luigi)/np.sqrt(len(trial_rate_treatment_control_early_luigi)), \
					np.nanstd(trial_rate_treatment_control_late_luigi)/np.sqrt(len(trial_rate_treatment_control_late_luigi))]

tpm_stim_luigi = [np.nanmean(trial_rate_baseline_stim_luigi), np.nanmean(trial_rate_stress_stim_luigi), np.nanmean(trial_rate_treatment_stim_luigi), \
					np.nanmean(trial_rate_treatment_stim_early_luigi), np.nanmean(trial_rate_treatment_stim_late_luigi)]
tpm_stim_luigi_sem = [np.nanstd(trial_rate_baseline_stim_luigi)/np.sqrt(len(trial_rate_baseline_stim_luigi)), \
					np.nanstd(trial_rate_stress_stim_luigi)/np.sqrt(len(trial_rate_stress_stim_luigi)), \
					np.nanstd(trial_rate_treatment_stim_luigi)/np.sqrt(len(trial_rate_treatment_stim_luigi)), \
					np.nanstd(trial_rate_treatment_stim_early_luigi)/np.sqrt(len(trial_rate_treatment_stim_early_luigi)),\
					np.nanstd(trial_rate_treatment_stim_late_luigi)/np.sqrt(len(trial_rate_treatment_stim_late_luigi))]


plt.figure(1)
plt.subplot(1,2,1)
plt.bar(ind[:3],rt_control_mario,width,yerr=rt_control_mario_sem,color = 'b',label = 'Control days')
plt.bar(ind[:3]+width,rt_stim_mario,width,yerr=rt_stim_mario_sem,color = 'r',label = 'Stim days')
xticklabels = ['Baseline','Anxiety','Treatment']
plt.xticks(ind,xticklabels)
plt.ylabel('Reaction time (s)')
plt.title('Mario')
plt.legend()

plt.subplot(1,2,2)
plt.bar(ind[:3],rt_control_luigi,width,yerr=rt_control_luigi_sem,color = 'b',label = 'Control days')
plt.bar(ind[:3]+width,rt_stim_luigi,width,yerr=rt_stim_luigi_sem,color = 'r',label = 'Stim days')
xticklabels = ['Baseline','Anxiety','Treatment']
plt.xticks(ind,xticklabels)
plt.ylabel('Reaction time (s)')
plt.title('Luigi')
plt.legend()

plt.show()

plt.figure(2)
plt.subplot(1,2,1)
plt.bar(ind,tpm_control_mario,width,yerr=tpm_control_mario_sem,color = 'b',label = 'Control days')
plt.bar(ind+width,tpm_stim_mario,width,yerr=tpm_stim_mario_sem,color = 'r',label = 'Stim days')
xticklabels = ['Baseline','Anxiety','Treatment', 'Treatmeant-Early', 'Treatment-Late']
plt.xticks(ind,xticklabels)
plt.ylabel('Trials per min')
plt.ylim((0,17))
plt.title('Mario')
plt.legend()

plt.subplot(1,2,2)
plt.bar(ind,tpm_control_luigi,width,yerr=tpm_control_luigi_sem,color = 'b',label = 'Control days')
plt.bar(ind+width,tpm_stim_luigi,width,yerr=tpm_stim_luigi_sem,color = 'r',label = 'Stim days')
xticklabels = ['Baseline','Anxiety','Treatment','Treatmeant-Early', 'Treatment-Late']
plt.xticks(ind,xticklabels)
plt.ylabel('Trials per min')
plt.title('Luigi')
plt.ylim((0,17))
plt.legend()

plt.show()
"""
##################################################################
# PHYSIOLOGY ONLY ANALYSIS
##################################################################
'''
Look at the IBI and PD data across time, how it's changed from baseline,
and how it's distributed.
'''

# Define variables
phys_dir = 'C:/Users/ss45436/Box Sync/UC Berkeley/Stress Task/PowerFeatures/'
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

#avg_cv_trial = np.zeros(len(filenames_stim_luigi))
#avg_cv_time = np.zeros(len(filenames_stim_luigi))
#avg_cv_trial_shuffle = np.zeros(len(filenames_stim_luigi))
#avg_cv_time_shuffle = np.zeros(len(filenames_stim_luigi))

luigi_all_files = filenames_stim_luigi + filenames_control_luigi
mario_all_files = filenames_stim_mario + filenames_control_mario

luigi_beta_ibi_time = np.zeros(len(luigi_all_files))
luigi_beta_pd_time = np.zeros(len(luigi_all_files))
luigi_beta_ibi_trial = np.zeros(len(luigi_all_files))
luigi_beta_pd_trial = np.zeros(len(luigi_all_files))

luigi_p_ibi_time = np.zeros(len(luigi_all_files))
luigi_p_pd_time = np.zeros(len(luigi_all_files))
luigi_p_ibi_trial = np.zeros(len(luigi_all_files))
luigi_p_pd_trial = np.zeros(len(luigi_all_files))

mario_beta_ibi_time = np.zeros(len(mario_all_files))
mario_beta_pd_time = np.zeros(len(mario_all_files))
mario_beta_ibi_trial = np.zeros(len(mario_all_files))
mario_beta_pd_trial = np.zeros(len(mario_all_files))

mario_p_ibi_time = np.zeros(len(mario_all_files))
mario_p_pd_time = np.zeros(len(mario_all_files))
mario_p_ibi_trial = np.zeros(len(mario_all_files))
mario_p_pd_trial = np.zeros(len(mario_all_files))

luigi_ibi_trials_base = np.zeros(len(luigi_all_files))
luigi_ibi_trials_stress = np.zeros(len(luigi_all_files))
luigi_ibi_time_base = np.zeros(len(luigi_all_files))
luigi_ibi_time_stress = np.zeros(len(luigi_all_files))

luigi_pupil_trials_base = np.zeros(len(luigi_all_files))
luigi_pupil_trials_stress = np.zeros(len(luigi_all_files))
luigi_pupil_time_base = np.zeros(len(luigi_all_files))
luigi_pupil_time_stress = np.zeros(len(luigi_all_files))

mario_ibi_trials_base = np.zeros(len(mario_all_files))
mario_ibi_trials_stress = np.zeros(len(mario_all_files))
mario_ibi_time_base = np.zeros(len(mario_all_files))
mario_ibi_time_stress = np.zeros(len(mario_all_files))

mario_pupil_trials_base = np.zeros(len(mario_all_files))
mario_pupil_trials_stress = np.zeros(len(mario_all_files))
mario_pupil_time_base = np.zeros(len(mario_all_files))
mario_pupil_time_stress = np.zeros(len(mario_all_files))

def pvalue_pie_plots(p_ibi,p_pd):
	"""
	Make pie plot with describing for what percentage of sessions ibi and/or pd
	were significant in predicting the state of the animals
	"""
	# First, determine percentage of sessions with significant pvalues

	ibi_sig_only = np.sum((p_ibi < 0.05)&(p_pd > 0.05))/len(p_ibi)
	pd_sig_only = np.sum((p_pd < 0.05)&(p_ibi > 0.05))/len(p_ibi)
	ibi_and_pd_sig = np.sum((p_ibi < 0.05)&(p_pd < 0.05))/len(p_ibi)
	no_sig = 1 - ibi_sig_only - pd_sig_only - ibi_and_pd_sig

	fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

	percents = ["IBI only: %f" % (ibi_sig_only),
	          "PD only: %f" % (pd_sig_only),
	          "Both: %f" % (ibi_and_pd_sig),
	          "Neither: %f" % (no_sig)]

	data = [ibi_sig_only, pd_sig_only,ibi_and_pd_sig,no_sig]

	wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	kw = dict(arrowprops=dict(arrowstyle="-"),
	          bbox=bbox_props, zorder=0, va="center")

	for i, p in enumerate(wedges):
	    ang = (p.theta2 - p.theta1)/2. + p.theta1
	    y = np.sin(np.deg2rad(ang))
	    x = np.cos(np.deg2rad(ang))
	    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
	    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
	    kw["arrowprops"].update({"connectionstyle": connectionstyle})
	    ax.annotate(percents[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
	                horizontalalignment=horizontalalignment, **kw)

	ax.set_title("Significant Coefficients for Logistic Regression")

	plt.show()

	return

luigi_pred_treat_stress_trial = np.zeros(len(luigi_all_files))
luigi_pred_base_stress_trial = np.zeros(len(luigi_all_files))
luigi_pred_stress_stress_trial = np.zeros(len(luigi_all_files))

for i,name in enumerate(luigi_all_files):
	print(name)
	# Load data into class

	phys_filenames = [phys_dir + name[0] + '_b1_PhysFeatures.mat',  phys_dir + name[0] + '_b2_PhysFeatures.mat', phys_dir + name[0] + '_b3_PhysFeatures.mat']
	phys_stim[i] = StressTask_PhysData(phys_filenames)

	# Fit logistic regression

	fit = phys_stim[i].logistic_regression_baseline_vs_stress('trial')
	luigi_beta_ibi_trial[i] = fit.params[1]
	luigi_beta_pd_trial[i] = fit.params[2]
	luigi_p_ibi_trial[i] = fit.pvalues[1]
	luigi_p_pd_trial[i] = fit.pvalues[2]

	luigi_pred_treat_stress_trial[i] = np.sum(((fit.params[0] + fit.params[1]*(phys_stim[i].ibi_trials_treat - np.nanmean(phys_stim[i].ibi_trials_treat))/np.nanstd(phys_stim[i].ibi_trials_treat) \
											+ fit.params[2]*(phys_stim[i].pupil_trials_treat - np.nanmean(phys_stim[i].pupil_trials_treat))/np.nanstd(phys_stim[i].pupil_trials_treat)) > 0.5))/len(phys_stim[i].ibi_trials_treat)
	luigi_pred_base_stress_trial[i] = np.sum(((fit.params[0] + fit.params[1]*(phys_stim[i].ibi_trials_base - np.nanmean(phys_stim[i].ibi_trials_base))/np.nanstd(phys_stim[i].ibi_trials_base) \
											+ fit.params[2]*(phys_stim[i].pupil_trials_base - np.nanmean(phys_stim[i].pupil_trials_base))/np.nanstd(phys_stim[i].pupil_trials_base)) > 0.5))/len(phys_stim[i].ibi_trials_base)
	luigi_pred_stress_stress_trial[i] = np.sum(((fit.params[0] + fit.params[1]*(phys_stim[i].ibi_trials_stress - np.nanmean(phys_stim[i].ibi_trials_stress))/np.nanstd(phys_stim[i].ibi_trials_stress) \
											+ fit.params[2]*(phys_stim[i].pupil_trials_stress - np.nanmean(phys_stim[i].pupil_trials_stress))/np.nanstd(phys_stim[i].pupil_trials_stress)) > 0.5))/len(phys_stim[i].ibi_trials_stress)


	fit = phys_stim[i].logistic_regression_baseline_vs_stress('time')
	luigi_beta_ibi_time[i] = fit.params[1]
	luigi_beta_pd_time[i] = fit.params[2]
	luigi_p_ibi_time[i] = fit.pvalues[1]
	luigi_p_pd_time[i] = fit.pvalues[2]

	# Compute average values per session and find correlation
	luigi_ibi_trials_base[i] = np.nanmean(phys_stim[i].ibi_trials_base)
	luigi_ibi_trials_stress[i] = np.nanmean(phys_stim[i].ibi_trials_stress)
	luigi_ibi_time_base[i] = np.nanmean(phys_stim[i].ibi_time_base)
	luigi_ibi_time_stress[i] = np.nanmean(phys_stim[i].ibi_time_stress)

	luigi_pupil_trials_base[i] = np.nanmean(phys_stim[i].pupil_trials_base)
	luigi_pupil_trials_stress[i] = np.nanmean(phys_stim[i].pupil_trials_stress)
	luigi_pupil_time_base[i] = np.nanmean(phys_stim[i].pupil_time_base)
	luigi_pupil_time_stress[i] = np.nanmean(phys_stim[i].pupil_time_stress)


# Correlate physiological metrics
luigi_r_ibi_trials_ibi_time_base, luigi_p_ibi_trials_ibi_time_base = sp.stats.pearsonr(luigi_ibi_trials_base, luigi_ibi_time_base)
luigi_r_ibi_trials_ibi_time_stress, luigi_p_ibi_trials_ibi_time_stress = sp.stats.pearsonr(luigi_ibi_trials_stress, luigi_ibi_time_stress)

luigi_r_ibi_trials_pupil_trials_base, luigi_p_ibi_trials_pupil_trials_base = sp.stats.pearsonr(luigi_ibi_trials_base, luigi_pupil_trials_base)
luigi_r_ibi_trials_pupil_trials_stress, luigi_p_ibi_trials_pupil_trials_stress = sp.stats.pearsonr(luigi_ibi_trials_stress, luigi_pupil_trials_stress)

luigi_r_ibi_time_pupil_time_base, luigi_p_ibi_time_pupil_time_base = sp.stats.pearsonr(luigi_ibi_time_base, luigi_pupil_time_base)
luigi_r_ibi_time_pupil_time_stress, luigi_p_ibi_time_pupil_time_stress = sp.stats.pearsonr(luigi_ibi_time_stress, luigi_pupil_time_stress)

luigi_r_pupil_trials_pupil_time_base, luigi_p_pupil_trials_pupil_time_base = sp.stats.pearsonr(luigi_pupil_trials_base, luigi_pupil_time_base)
luigi_r_pupil_trials_pupil_time_stress, luigi_p_pupil_trials_pupil_time_stress = sp.stats.pearsonr(luigi_pupil_trials_stress, luigi_pupil_time_stress)

luigi_mat_base = np.array([[luigi_r_ibi_trials_ibi_time_base, luigi_r_ibi_trials_pupil_trials_base],\
					[luigi_r_ibi_time_pupil_time_base, luigi_r_pupil_trials_pupil_time_base]])
luigi_mat_stress = np.array([[luigi_r_ibi_trials_ibi_time_stress, luigi_r_ibi_trials_pupil_trials_stress],\
					[luigi_r_ibi_time_pupil_time_stress, luigi_r_pupil_trials_pupil_time_stress]])


# Look at fraction of sessions in which physiology is predictive of state
pvalue_pie_plots(luigi_p_ibi_trial, luigi_p_pd_trial)
pvalue_pie_plots(luigi_p_ibi_time, luigi_p_pd_time)

for i,name in enumerate(mario_all_files):
	print(name)
	# Load data into class
	phys_filenames = [phys_dir + name[0] + '_b1_PhysFeatures.mat',  phys_dir + name[1] + '_b2_PhysFeatures.mat', phys_dir + name[2] + '_b1_PhysFeatures.mat']
	phys_stim[i] = StressTask_PhysData(phys_filenames)

	#scores_logreg_trial, scores_logreg_shuffle_trial, X_mat = phys_stim[i].logistic_regression_baseline_vs_stress('trial')
	#scores_logreg_time, scores_logreg_shuffle_time, X_mat = phys_stim[i].logistic_regression_baseline_vs_stress('time')

	#avg_cv_trial[i] = scores_logreg_trial.mean()
	#avg_cv_time[i] = scores_logreg_time.mean()
	#avg_cv_trial_shuffle[i] = scores_logreg_shuffle_trial.mean()
	#avg_cv_time_shuffle[i] = scores_logreg_shuffle_time.mean()

	fit = phys_stim[i].logistic_regression_baseline_vs_stress('trial')
	mario_beta_ibi_trial[i] = fit.params[1]
	mario_beta_pd_trial[i] = fit.params[2]
	mario_p_ibi_trial[i] = fit.pvalues[1]
	mario_p_pd_trial[i] = fit.pvalues[2]

	fit = phys_stim[i].logistic_regression_baseline_vs_stress('time')
	mario_beta_ibi_time[i] = fit.params[1]
	mario_beta_pd_time[i] = fit.params[2]
	mario_p_ibi_time[i] = fit.pvalues[1]
	mario_p_pd_time[i] = fit.pvalues[2]

	# Compute average values per session and find correlation
	mario_ibi_trials_base[i] = np.nanmean(phys_stim[i].ibi_trials_base)
	mario_ibi_trials_stress[i] = np.nanmean(phys_stim[i].ibi_trials_stress)
	mario_ibi_time_base[i] = np.nanmean(phys_stim[i].ibi_time_base)
	mario_ibi_time_stress[i] = np.nanmean(phys_stim[i].ibi_time_stress)

	mario_pupil_trials_base[i] = np.nanmean(phys_stim[i].pupil_trials_base)
	mario_pupil_trials_stress[i] = np.nanmean(phys_stim[i].pupil_trials_stress)
	mario_pupil_time_base[i] = np.nanmean(phys_stim[i].pupil_time_base)
	mario_pupil_time_stress[i] = np.nanmean(phys_stim[i].pupil_time_stress)

# Correlate physiological metrics
mario_r_ibi_trials_ibi_time_base, mario_p_ibi_trials_ibi_time_base = sp.stats.pearsonr(mario_ibi_trials_base, mario_ibi_time_base)
mario_r_ibi_trials_ibi_time_stress, mario_p_ibi_trials_ibi_time_stress = sp.stats.pearsonr(mario_ibi_trials_stress, mario_ibi_time_stress)

mario_r_ibi_trials_pupil_trials_base, mario_p_ibi_trials_pupil_trials_base = sp.stats.pearsonr(mario_ibi_trials_base, mario_pupil_trials_base)
mario_r_ibi_trials_pupil_trials_stress, mario_p_ibi_trials_pupil_trials_stress = sp.stats.pearsonr(mario_ibi_trials_stress, mario_pupil_trials_stress)

mario_r_ibi_time_pupil_time_base, mario_p_ibi_time_pupil_time_base = sp.stats.pearsonr(mario_ibi_time_base, mario_pupil_time_base)
mario_r_ibi_time_pupil_time_stress, mario_p_ibi_time_pupil_time_stress = sp.stats.pearsonr(mario_ibi_time_stress, mario_pupil_time_stress)

mario_r_pupil_trials_pupil_time_base, mario_p_pupil_trials_pupil_time_base = sp.stats.pearsonr(mario_pupil_trials_base, mario_pupil_time_base)
mario_r_pupil_trials_pupil_time_stress, mario_p_pupil_trials_pupil_time_stress = sp.stats.pearsonr(mario_pupil_trials_stress, mario_pupil_time_stress)

mario_mat_base = np.array([[mario_r_ibi_trials_ibi_time_base, mario_r_ibi_trials_pupil_trials_base],\
					[mario_r_ibi_time_pupil_time_base, mario_r_pupil_trials_pupil_time_base]])
mario_mat_stress = np.array([[mario_r_ibi_trials_ibi_time_stress, mario_r_ibi_trials_pupil_trials_stress],\
					[mario_r_ibi_time_pupil_time_stress, mario_r_pupil_trials_pupil_time_stress]])

# Look at fraction of seccions with significant regressors
pvalue_pie_plots(mario_p_ibi_trial, mario_p_pd_trial)
pvalue_pie_plots(mario_p_ibi_time, mario_p_pd_time)

#phys_stim[len(filenames_stim_mario)].IBIvsPup_ScatterPlot('time',save_fig = False)
phys_stim[len(filenames_stim_mario)].IBIvsPup_ScatterPlot('trial',save_fig = False)


for i,name in enumerate(filenames_stim_luigi):
	print(name)
	# Load data into class
	phys_filenames = [phys_dir + name[0] + '_b1_PhysFeatures.mat',  phys_dir + name[0] + '_b2_PhysFeatures.mat', phys_dir + name[0] + '_b3_PhysFeatures.mat']
	phys_stim[i] = StressTask_PhysData(phys_filenames)
	phys_stim[i].IBIvsPup_ScatterPlot('time',save_fig = False)

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


##################################################################
## POWER FEATURE ANALYSIS
##################################################################
"""
colors = ['b','r','g','m']
channel_group_labels = ['ofc', 'vmpfc','cd']
ofc_power = np.empty((len(filenames_stim_luigi), 4, 3))
vmpfc_power = np.empty((len(filenames_stim_luigi), 4, 3))
cd_power = np.empty((len(filenames_stim_luigi), 4, 3))

def correlate_average_power(avg_ofc_power, avg_vmpfc_power, avg_cd_power, animal, day):
	'''
	Correlates power with bands across areas per condition. 

	Inputs:
	- avg_ofc_power: 3D array; session x band x behavioral condition average power values for OFC
	- avg_vmpfc_power: 3D array; session x band x behavioral condition average power values for vmPFC
	- avg_cd_power: 3D array; session x band x behavioral condition average power values for Cd
	- animal: string; string used to create the filename for saving
	- day: string; string used to indicate 'control' or 'stim' day
	'''
	#plt.figure(figsize = (3,3))
	for b in range(4): 	# loop over different bands
		for c in range(3): # loop over behavioral conditions
			ofc_data = avg_ofc_power[:,b,c].reshape((len(avg_ofc_power[:,b,c]),1))
			vmpfc_data = avg_vmpfc_power[:,b,c].reshape((len(avg_vmpfc_power[:,b,c]),1))
			cd_data = avg_cd_power[:,b,c].reshape((len(avg_cd_power[:,b,c]),1))

			data_mat = np.concatenate((ofc_data, vmpfc_data, cd_data), axis = 1)	# matrix should be sessions x 3 brain areas

			labels = ['ofc', 'vmpfc','cd']
			data = pd.DataFrame(data_mat, columns = labels)
			data_corr = data.corr()

			mask = np.zeros_like(data_corr, dtype=np.bool)
			mask[np.triu_indices_from(mask)] = True

			plt.figure(figsize = (2,2))
			#plt.subplot(4,3,3*b+c+1)
			corrplot(data_corr)
			cmap = sns.diverging_palette(220, 10, as_cmap=True)
			#heatmap(data_corr,cmap=cmap)
			#sns.heatmap(data_corr,mask = mask,cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
			#sns.heatmap(data_corr, cmap=cmap)
			plt.title('Band: %i, Condition: %i' % (b,c))

			plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + animal + "_power_correlation_band" + str(b) + "_block" + str(c) + "_" + day + ".svg")
			plt.close()	
	return data_corr

def correlation_difference_average_power(avg_ofc_power_stim, avg_vmpfc_power_stim, avg_cd_power_stim,avg_ofc_power_control, avg_vmpfc_power_control, avg_cd_power_control, animal):
	'''
	Correlates power with bands across areas per condition for stim days and control days, then computes the difference in average correlation.

	Inputs:
	- avg_ofc_power: 3D array; session x band x behavioral condition average power values for OFC
	- avg_vmpfc_power: 3D array; session x band x behavioral condition average power values for vmPFC
	- avg_cd_power: 3D array; session x band x behavioral condition average power values for Cd
	- animal: string; string used to create the filename for saving
	- day: string; string used to indicate 'control' or 'stim' day
	'''
	#plt.figure(figsize = (3,3))
	for b in range(4): 	# loop over different bands
		for c in range(3): # loop over behavioral conditions
			# Compute correlation for stim days
			ofc_data_stim = avg_ofc_power_stim[:,b,c].reshape((len(avg_ofc_power_stim[:,b,c]),1))
			vmpfc_data_stim = avg_vmpfc_power_stim[:,b,c].reshape((len(avg_vmpfc_power_stim[:,b,c]),1))
			cd_data_stim = avg_cd_power_stim[:,b,c].reshape((len(avg_cd_power_stim[:,b,c]),1))

			data_mat_stim = np.concatenate((ofc_data_stim, vmpfc_data_stim, cd_data_stim), axis = 1)	# matrix should be sessions x 3 brain areas

			labels = ['ofc', 'vmpfc','cd']
			data_stim = pd.DataFrame(data_mat_stim, columns = labels)
			data_corr_stim = data_stim.corr()

			#Compute correlation for control days
			ofc_data_control = avg_ofc_power_control[:,b,c].reshape((len(avg_ofc_power_control[:,b,c]),1))
			vmpfc_data_control = avg_vmpfc_power_control[:,b,c].reshape((len(avg_vmpfc_power_control[:,b,c]),1))
			cd_data_control = avg_cd_power_control[:,b,c].reshape((len(avg_cd_power_control[:,b,c]),1))

			data_mat_control = np.concatenate((ofc_data_control, vmpfc_data_control, cd_data_control), axis = 1)	# matrix should be sessions x 3 brain areas

			data_control = pd.DataFrame(data_mat_control, columns = labels)
			data_corr_control = data_control.corr()

			plt.figure(figsize = (2,2))

			corrplot(data_corr_stim - data_corr_control)
			cmap = sns.diverging_palette(220, 10, as_cmap=True)
			#heatmap(data_corr,cmap=cmap)
			#sns.heatmap(data_corr,mask = mask,cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
			#sns.heatmap(data_corr, cmap=cmap)
			plt.title('Band: %i, Condition: %i' % (b,c))

			#plt.show()
			plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + animal + "_power_correlation_difference_band" + str(b) + "_block" + str(c) + ".svg")
			plt.close()	

	return

def correlation_and_power_recovery_tuning_metric(avg_ofc_power, avg_vmpfc_power, avg_cd_power, animal, day):

	'''
	Correlates power with bands across areas per condition for base, stim days and control days separately, then computes the difference in average correlation between treatment and baseline and stress and baseline.
	Computes a correlation recovery tuning metric: (|stress - base| - |treat - base|)/(|treat - base| + |stress - base|). This metric is close to 1 if |treat - base| << |stress - base| (TREATMENT WORKING) and close to -1
	if |treat - base| >> |stress - base| (TREATMENT NOT WORKING). Values are close to zero if the differences are similar, either both small or both large.

	Inputs:
	- avg_ofc_power: 3D array; session x band x behavioral condition average power values for OFC
	- avg_vmpfc_power: 3D array; session x band x behavioral condition average power values for vmPFC
	- avg_cd_power: 3D array; session x band x behavioral condition average power values for Cd
	- animal: string; string used to create the filename for saving
	- day: string; string used to indicate 'control' or 'stim' day
	'''
	#plt.figure(figsize = (3,3))

	corr_mats = np.empty((3,3,3))		#  3 behaviors conditions, 3 x 3 correlation matrix across areas
	corr_tuning_metric = np.empty((4,3,3))	# 	4 bands, 3 x 3 tuning matrix
	power_tuning_metric = np.empty((4,3))	# 	4 bands, 3 areas

	for b in range(4): 	# loop over different bands
		for c in range(3): # loop over behavioral conditions
			# Compute correlation for stim days
			ofc_data = avg_ofc_power[:,b,c].reshape((len(avg_ofc_power[:,b,c]),1))				# array for a given band and condition, data points across sessions
			vmpfc_data = avg_vmpfc_power[:,b,c].reshape((len(avg_vmpfc_power[:,b,c]),1))
			cd_data = avg_cd_power[:,b,c].reshape((len(avg_cd_power[:,b,c]),1))

			data_mat = np.concatenate((ofc_data, vmpfc_data, cd_data), axis = 1)	# matrix should be sessions x 3 brain areas

			labels = ['ofc', 'vmpfc','cd']
			data = pd.DataFrame(data_mat, columns = labels)
			data_corr = data.corr()

			corr_mats[c,:,:] = data_corr

		# for a given band, compute the difference in correlation
		corr_tuning_metric[b,:,:] = (np.abs(corr_mats[1,:,:] - corr_mats[0,:,:]) - np.abs(corr_mats[2,:,:] - corr_mats[0,:,:]))/(np.abs(corr_mats[1,:,:] - corr_mats[0,:,:]) + np.abs(corr_mats[2,:,:] - corr_mats[0,:,:]))		# (|stress - base| - |treat - base|)/(|treat - base| + |stress - base|)

		plt.figure(1)
		plt.subplot(1,4,b+1)
		cmap = sns.diverging_palette(220, 10, as_cmap=True)
		cmap=sns.diverging_palette(20, 220, n=256)
		sns.heatmap(corr_tuning_metric[b,:,:],cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin = -1, vmax = 0.25)
		plt.title('Band: %i, Order: %s, %s, %s' % (b,labels[0], labels[1], labels[2]))

		
		power_tuning_metric[b,0] = np.mean((np.abs(avg_ofc_power[:,b,1] - avg_ofc_power[:,b,0]) - np.abs(avg_ofc_power[:,b,2] - avg_ofc_power[:,b,0]))/(np.abs(avg_ofc_power[:,b,1] - avg_ofc_power[:,b,0]) + np.abs(avg_ofc_power[:,b,2] - avg_ofc_power[:,b,0])))			# inside mean should be array of length = # sessions
		power_tuning_metric[b,1] = np.mean((np.abs(avg_vmpfc_power[:,b,1] - avg_vmpfc_power[:,b,0]) - np.abs(avg_vmpfc_power[:,b,2] - avg_vmpfc_power[:,b,0]))/(np.abs(avg_vmpfc_power[:,b,1] - avg_vmpfc_power[:,b,0]) + np.abs(avg_vmpfc_power[:,b,2] - avg_vmpfc_power[:,b,0])))			# inside mean should be array of length = # sessions
		power_tuning_metric[b,2] = np.mean((np.abs(avg_cd_power[:,b,1] - avg_cd_power[:,b,0]) - np.abs(avg_cd_power[:,b,2] - avg_cd_power[:,b,0]))/(np.abs(avg_cd_power[:,b,1] - avg_cd_power[:,b,0]) + np.abs(avg_cd_power[:,b,2] - avg_cd_power[:,b,0])))			# inside mean should be array of length = # sessions

	plt.figure(1)
	plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + animal + "_correlation_tuning_metric_"  + day + ".svg")
	plt.close()

	plt.figure(2)
	#plt.subplot(1,4,b+1)
	cmap=sns.diverging_palette(20, 220, n=256)
	sns.heatmap(power_tuning_metric,cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin = -1, vmax = 0)
	plt.title('Average Power tuning')

	plt.savefig("C:/Users/ss45436/Box/SfN 2019/OFC-vmPFC stim/Figures/" + animal + "_power_tuning_metric_" + day + ".png")
	plt.close()
	#plt.show()
	return power_tuning_metric

for i,name in enumerate(filenames_stim_luigi):
	print(name)
	# Load data into class

	power_filenames = [phys_dir + name[0] + '_b1_PowerFeatures.mat',  phys_dir + name[0] + '_b2_PowerFeatures.mat', phys_dir + name[0] + '_b3_PowerFeatures.mat']
	power = StressTask_PowerFeatureData(power_filenames, 96)
	#power.power_average_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'])
	#power.power_heatmap_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices])
	average_power_per_area = power.power_average_per_condition(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'], normalization = False)  # output matrices are brain areas x frequency band x behavioral condition
	#fit_glm = power.logistic_decoding_with_power()

	ofc_power[i,:,:] = average_power_per_area[0,:,:]
	vmpfc_power[i,:,:] = average_power_per_area[1,:,:]
	cd_power[i,:,:] = average_power_per_area[2,:,:] 

avg_ofc_power = np.nanmean(ofc_power,axis = 0)
sem_ofc_power = np.nanstd(ofc_power,axis = 0)/np.sqrt(len(filenames_stim_luigi))

avg_vmpfc_power = np.nanmean(vmpfc_power,axis = 0)
sem_vmpfc_power = np.nanstd(vmpfc_power,axis = 0)/np.sqrt(len(filenames_stim_luigi))

avg_cd_power = np.nanmean(cd_power,axis = 0)
sem_cd_power = np.nanstd(cd_power,axis = 0)/np.sqrt(len(filenames_stim_luigi))

avg_ofc_power_stim = ofc_power
avg_vmpfc_power_stim = vmpfc_power
avg_cd_power_stim = cd_power

#correlate_average_power(ofc_power, vmpfc_power, cd_power, 'Luigi', 'stim')
power_tuning_metric = correlation_and_power_recovery_tuning_metric(ofc_power, vmpfc_power, cd_power, 'Luigi', 'stim')

'''
plt.figure(1)

ind = np.arange(3)
width = 0.35*2
for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b+1)
	plt.bar(ind, avg_ofc_power[b,:],width = width,yerr = sem_ofc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Stim - Channel group: %s" % (channel_group_labels[0]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 2)
	plt.bar(ind, avg_vmpfc_power[b,:],width = width,yerr = sem_vmpfc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[1]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 3)
	plt.bar(ind, avg_cd_power[b,:],width = width,yerr = sem_cd_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[2]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()

plt.show()
'''

ofc_power = np.empty((len(filenames_control_luigi), 4, 3))
vmpfc_power = np.empty((len(filenames_control_luigi), 4, 3))
cd_power = np.empty((len(filenames_control_luigi), 4, 3))

for i,name in enumerate(filenames_control_luigi):
	print(name)
	# Load data into class

	power_filenames = [phys_dir + name[0] + '_b1_PowerFeatures.mat',  phys_dir + name[0] + '_b2_PowerFeatures.mat', phys_dir + name[0] + '_b3_PowerFeatures.mat']
	power = StressTask_PowerFeatureData(power_filenames, 96)
	#fit_glm = power.logistic_decoding_with_power()
	#power.power_average_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'])
	#power.power_heatmap_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices])
	average_power_per_area = power.power_average_per_condition(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'], normalization = False)  # output matrices are brain areas x frequency band x behavioral condition

	ofc_power[i,:,:] = average_power_per_area[0,:,:]
	vmpfc_power[i,:,:] = average_power_per_area[1,:,:]
	cd_power[i,:,:] = average_power_per_area[2,:,:] 

avg_ofc_power = np.nanmean(ofc_power,axis = 0)
sem_ofc_power = np.nanstd(ofc_power,axis = 0)/np.sqrt(len(filenames_control_luigi))

avg_vmpfc_power = np.nanmean(vmpfc_power,axis = 0)
sem_vmpfc_power = np.nanstd(vmpfc_power,axis = 0)/np.sqrt(len(filenames_control_luigi))

avg_cd_power = np.nanmean(cd_power,axis = 0)
sem_cd_power = np.nanstd(cd_power,axis = 0)/np.sqrt(len(filenames_control_luigi))

avg_ofc_power_control = ofc_power
avg_vmpfc_power_control = vmpfc_power
avg_cd_power_control = cd_power

#correlate_average_power(ofc_power, vmpfc_power, cd_power, 'Luigi', 'control')
#correlation_difference_average_power(avg_ofc_power_stim, avg_vmpfc_power_stim, avg_cd_power_stim,avg_ofc_power_control, avg_vmpfc_power_control, avg_cd_power_control, 'Luigi')
power_tuning_metric = correlation_and_power_recovery_tuning_metric(ofc_power, vmpfc_power, cd_power, 'Luigi', 'control')
'''

plt.figure(2)

ind = np.arange(3)
for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b+1)
	plt.bar(ind, avg_ofc_power[b,:],width = width,yerr = sem_ofc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Control - Channel group: %s" % (channel_group_labels[0]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 2)
	plt.bar(ind, avg_vmpfc_power[b,:],width = width,yerr = sem_vmpfc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[1]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 3)
	plt.bar(ind, avg_cd_power[b,:],width = width,yerr = sem_cd_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[2]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()

plt.show()
'''

ofc_power = np.empty((len(filenames_stim_mario), 4, 3))
vmpfc_power = np.empty((len(filenames_stim_mario), 4, 3))
cd_power = np.empty((len(filenames_stim_mario), 4, 3))

for i,name in enumerate(filenames_stim_mario):
	print(name)
	# Load data into class

	power_filenames = [phys_dir + name[0] + '_b1_PowerFeatures.mat',  phys_dir + name[0] + '_b2_PowerFeatures.mat', phys_dir + name[2] + '_b1_PowerFeatures.mat']
	power = StressTask_PowerFeatureData(power_filenames, 157)
	#fit_glm = power.logistic_decoding_with_power()
	#power.power_average_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'])
	#power.power_heatmap_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices])
	average_power_per_area = power.power_average_per_condition(channel_groups = [mario_ofc_indices, mario_vmpfc_indices, mario_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'], normalization = False)  # output matrices are brain areas x frequency band x behavioral condition

	ofc_power[i,:,:] = average_power_per_area[0,:,:]
	vmpfc_power[i,:,:] = average_power_per_area[1,:,:]
	cd_power[i,:,:] = average_power_per_area[2,:,:] 

avg_ofc_power = np.nanmean(ofc_power,axis = 0)
sem_ofc_power = np.nanstd(ofc_power,axis = 0)/np.sqrt(len(filenames_stim_mario))

avg_vmpfc_power = np.nanmean(vmpfc_power,axis = 0)
sem_vmpfc_power = np.nanstd(vmpfc_power,axis = 0)/np.sqrt(len(filenames_stim_mario))

avg_cd_power = np.nanmean(cd_power,axis = 0)
sem_cd_power = np.nanstd(cd_power,axis = 0)/np.sqrt(len(filenames_stim_mario))

avg_ofc_power_stim = ofc_power
avg_vmpfc_power_stim = vmpfc_power
avg_cd_power_stim = cd_power

#correlate_average_power(ofc_power, vmpfc_power, cd_power, 'Mario', 'stim')
#power_tuning_metric = correlation_and_power_recovery_tuning_metric(ofc_power, vmpfc_power, cd_power, 'Mario', 'stim')
'''

plt.figure(1)

ind = np.arange(3)
width = 0.35*2
for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b+1)
	plt.bar(ind, avg_ofc_power[b,:],width = width,yerr = sem_ofc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Stim - Channel group: %s" % (channel_group_labels[0]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 2)
	plt.bar(ind, avg_vmpfc_power[b,:],width = width,yerr = sem_vmpfc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[1]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 3)
	plt.bar(ind, avg_cd_power[b,:],width = width,yerr = sem_cd_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[2]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()

plt.show()
'''

ofc_power = np.empty((len(filenames_control_mario), 4, 3))
vmpfc_power = np.empty((len(filenames_control_mario), 4, 3))
cd_power = np.empty((len(filenames_control_mario), 4, 3))

for i,name in enumerate(filenames_control_mario):
	print(name)
	# Load data into class

	power_filenames = [phys_dir + name[0] + '_b1_PowerFeatures.mat',  phys_dir + name[0] + '_b2_PowerFeatures.mat', phys_dir + name[2] + '_b1_PowerFeatures.mat']
	power = StressTask_PowerFeatureData(power_filenames, 157)
	#fit_glm = power.logistic_decoding_with_power()
	#power.power_average_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'])
	#power.power_heatmap_overtime(channel_groups = [luigi_ofc_indices, luigi_vmpfc_indices, luigi_cd_indices])
	average_power_per_area = power.power_average_per_condition(channel_groups = [mario_ofc_indices, mario_vmpfc_indices, mario_cd_indices],channel_group_labels = ['ofc', 'vmpfc','cd'], normalization = False)  # output matrices are brain areas x frequency band x behavioral condition

	ofc_power[i,:,:] = average_power_per_area[0,:,:]
	vmpfc_power[i,:,:] = average_power_per_area[1,:,:]
	cd_power[i,:,:] = average_power_per_area[2,:,:] 

avg_ofc_power = np.nanmean(ofc_power,axis = 0)
sem_ofc_power = np.nanstd(ofc_power,axis = 0)/np.sqrt(len(filenames_control_mario))

avg_vmpfc_power = np.nanmean(vmpfc_power,axis = 0)
sem_vmpfc_power = np.nanstd(vmpfc_power,axis = 0)/np.sqrt(len(filenames_control_mario))

avg_cd_power = np.nanmean(cd_power,axis = 0)
sem_cd_power = np.nanstd(cd_power,axis = 0)/np.sqrt(len(filenames_control_mario))

avg_ofc_power_control = ofc_power
avg_vmpfc_power_control = vmpfc_power
avg_cd_power_control = cd_power

#correlate_average_power(ofc_power, vmpfc_power, cd_power, 'Mario','control')
#correlation_difference_average_power(avg_ofc_power_stim, avg_vmpfc_power_stim, avg_cd_power_stim,avg_ofc_power_control, avg_vmpfc_power_control, avg_cd_power_control, 'Mario')
#power_tuning_metric = correlation_and_power_recovery_tuning_metric(ofc_power, vmpfc_power, cd_power, 'Mario', 'control')
'''

plt.figure(2)

ind = np.arange(3)
for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b+1)
	plt.bar(ind, avg_ofc_power[b,:],width = width,yerr = sem_ofc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Control - Channel group: %s" % (channel_group_labels[0]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 2)
	plt.bar(ind, avg_vmpfc_power[b,:],width = width,yerr = sem_vmpfc_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[1]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()


for b,band in enumerate(power_bands):
	plt.subplot(4,3,3*b + 3)
	plt.bar(ind, avg_cd_power[b,:],width = width,yerr = sem_cd_power[b,:],color = colors[b], label = 'Band: %i - %i Hz' % (power_bands[b][0], power_bands[b][1]))
	plt.title("Channel group: %s" % (channel_group_labels[2]))
	#plt.yscale('log')
	#plt.ylim((-1,1))
	plt.legend()

plt.show()
'''
'''
for i,name in enumerate(mario_all_files):
	print(name)
	# Load data into class

	power_filenames = [phys_dir + name[0] + '_b1_PowerFeatures.mat',  phys_dir + name[0] + '_b2_PowerFeatures.mat', phys_dir + name[2] + '_b1_PowerFeatures.mat']
	power = StressTask_PowerFeatureData(power_filenames, 157)
	power.power_average_overtime(channel_groups = [mario_ofc_indices, mario_vmpfc_indices, mario_cd_indices], channel_group_labels = ['ofc', 'vmpfc','cd'])
	power.power_heatmap_overtime(channel_groups = [mario_ofc_indices, mario_vmpfc_indices, mario_cd_indices])
'''
"""