from probabilisticRewardTaskPerformance import FreeChoicePilotTask_LowValueChoiceProb
import matplotlib.pyplot as plt
import numpy as np

stim_days = ['luig20160204_15_te1382.hdf', 'luig20160208_07_te1401.hdf','luig20160212_08_te1429.hdf','luig20160217_06_te1451.hdf']

sham_days = ['luig20160213_05_te1434.hdf']


control_days = ['luig20160218_10_te1469.hdf']

all_days = stim_days + sham_days + control_days
counter_stim = 0
counter_sham = 0
counter_control = 0
avg_low_stim = 0
avg_low_sham = 0
avg_low_control = 0

plt.figure()

for item in all_days:
	hdf_location = '/storage/rawdata/hdf/'+item
	block1_prob_choose_low,block3_prob_choose_low = FreeChoicePilotTask_LowValueChoiceProb(hdf_location)
	if item in stim_days:
		plt.subplot(2,1,1)
		plt.plot(counter_stim,block1_prob_choose_low,'r*')
		plt.subplot(2,1,2)
		plt.plot(counter_stim,block3_prob_choose_low,'r*')
		counter_stim += 1
		avg_low_stim += block3_prob_choose_low
	if item in sham_days:
		plt.subplot(2,1,1)
		plt.plot(counter_sham,block1_prob_choose_low,'g*')
		plt.subplot(2,1,2)
		plt.plot(counter_sham,block3_prob_choose_low,'g*')
		counter_sham +=1
		avg_low_sham += block3_prob_choose_low
	if item in control_days:
		plt.subplot(2,1,1)
		plt.plot(counter_control,block1_prob_choose_low,'m*')
		plt.subplot(2,1,2)
		plt.plot(counter_control,block3_prob_choose_low,'m*')
		counter_control += 1
		avg_low_control += block3_prob_choose_low

avg_low_stim = float(avg_low_stim)/counter_stim
avg_low_sham = float(avg_low_sham)/counter_sham
avg_low_control = float(avg_low_control)/counter_control

plt.subplot(2,1,2)
max_days = np.max([counter_stim,counter_sham,counter_control])
plt.plot(range(0,max_days),avg_low_stim*np.ones(max_days),'r--')
plt.plot(range(0,max_days),avg_low_sham*np.ones(max_days),'g--')
plt.plot(range(0,max_days),avg_low_control*np.ones(max_days),'m--')

plt.show()
