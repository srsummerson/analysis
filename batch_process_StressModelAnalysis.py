from StressModelAndTreatmentAnalysis import StressTaskAnalysis_ComputePowerFeatures, StressTask_PhysData
import os
import numpy as np
import matplotlib.pyplot as plt

TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/'
os.chdir(TDT_tank)

1026, 1027, 1028, 1029, 1030, 1031, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119

dir = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Hdf files/"
TDT_tank_luigi = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Luigi - Neural Data/"
block_nums = [1,2,3]
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
			['luig20181031_02_te1522.hdf', 'luig20181031_03_te1523.hdf', 'luig20181031_04_te1524.hdf'], \
			['luig20181111_03_te1556.hdf', 'luig20181111_04_te1557.hdf', 'luig20181111_05_te1558.hdf'], \
			['luig20181113_02_te1565.hdf', 'luig20181113_03_te1566.hdf', 'luig20181113_04_te1567.hdf'], \
			['luig20181117_02_te1581.hdf', 'luig20181117_03_te1582.hdf', 'luig20181117_04_te1583.hdf'], \
			['luig20181118_03_te1586.hdf', 'luig20181118_04_te1587.hdf', 'luig20181118_05_te1588.hdf'], \
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
			['Luigi20181031','Luigi20181031','Luigi20181031'], \
			['Luigi20181111','Luigi20181111','Luigi20181111'], \
			['Luigi20181113','Luigi20181113','Luigi20181113'], \
			['Luigi20181117','Luigi20181117','Luigi20181117'], \
			['Luigi20181118','Luigi20181118','Luigi20181118'], \
			]

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 

'''
for i in range(len(hdf_list_stim_luigi)):
	print(i)
	StressTaskAnalysis_ComputePowerFeatures(hdf_list_stim_luigi[i], filenames_stim_luigi[i], block_nums, TDT_tank_luigi, power_bands)

for i in range(len(hdf_list_control_luigi)):
	print(i+7)
	StressTaskAnalysis_ComputePowerFeatures(hdf_list_control_luigi[i], filenames_control_luigi[i], block_nums, TDT_tank_luigi, power_bands)

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
	ibi_time_stress_mean_early_stim[i] = np.nanmean(phys_stim[i].ibi_time_stress[:int(len(ibi_time_stress)/2.)])
	pupil_time_stress_mean_early_stim[i] = np.nanmean(phys_stim[i].pupil_time_stress[:int(len(ibi_time_stress)/2.)])
	ibi_time_treat_mean_early_stim[i] = np.nanmean(phys_stim[i].ibi_time_treat[:int(len(ibi_time_stress)/2.)])
	pupil_time_treat_mean_early_stim[i] = np.nanmean(phys_stim[i].pupil_time_treat[:int(len(ibi_time_stress)/2.)])

	# Compute average values for late half of stress and treatment sessions
	ibi_time_stress_mean_late_stim[i] = np.nanmean(phys_stim[i].ibi_time_stress[int(len(ibi_time_stress)/2.):])
	pupil_time_stress_mean_late_stim[i] = np.nanmean(phys_stim[i].pupil_time_stress[int(len(ibi_time_stress)/2.):])
	ibi_time_treat_mean_late_stim[i] = np.nanmean(phys_stim[i].ibi_time_treat[int(len(ibi_time_stress)/2.):])
	pupil_time_treat_mean_late_stim[i] = np.nanmean(phys_stim[i].pupil_time_treat[int(len(ibi_time_stress)/2.):])

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
	phys_control.IBIvsPup_ScatterPlot('time')

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
	ibi_time_stress_mean_early_control[i] = np.nanmean(phys_control[i].ibi_time_stress[:int(len(ibi_time_stress)/2.)])
	pupil_time_stress_mean_early_control[i] = np.nanmean(phys_control[i].pupil_time_stress[:int(len(pupil_time_stress)/2.)])
	ibi_time_treat_mean_early_control[i] = np.nanmean(phys_control[i].ibi_time_treat[:int(len(ibi_time_treat)/2.)])
	pupil_time_treat_mean_early_control[i] = np.nanmean(phys_control[i].pupil_time_treat[:int(len(pupil_time_treat)/2.)])

	# Compute average values for late half of stress and treatment sessions
	ibi_time_stress_mean_late_control[i] = np.nanmean(phys_control[i].ibi_time_stress[int(len(ibi_time_stress)/2.):])
	pupil_time_stress_mean_late_control[i] = np.nanmean(phys_control[i].pupil_time_stress[int(len(pupil_time_stress)/2.):])
	ibi_time_treat_mean_late_control[i] = np.nanmean(phys_control[i].ibi_time_treat[int(len(ibi_time_treat)/2.):])
	pupil_time_treat_mean_late_control[i] = np.nanmean(phys_control[i].pupil_time_treat[int(len(pupil_time_treat)/2.):])


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


ind = np.arange(6)
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