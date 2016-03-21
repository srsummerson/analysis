from probabilisticRewardTaskPerformance import PeriStimulusFreeChoiceBehavior
import numpy as np
import matplotlib.pyplot as plt


stim_days = ['\luig20160204_15_te1382.hdf','\luig20160208_07_te1401.hdf','\luig20160212_08_te1429.hdf','\luig20160217_06_te1451.hdf',
                '\luig20160229_11_te1565.hdf','\luig20160301_07_te1572.hdf','\luig20160301_09_te1574.hdf', '\luig20160311_08_te1709.hdf',
                '\luig20160313_07_te1722.hdf', '\luig20160315_14_te1739.hdf']
sham_days = ['\luig20160213_05_te1434.hdf','\luig20160219_04_te1473.hdf','\luig20160221_05_te1478.hdf', '\luig20160305_26_te1617.hdf', \
                 '\luig20160306_11_te1628.hdf', '\luig20160307_13_te1641.hdf']

#hdf_list_hv = ['\luig20160218_10_te1469.hdf','\luig20160223_09_te1506.hdf','\luig20160223_11_te1508.hdf','\luig20160224_11_te1519.hdf',
#                '\luig20160224_15_te1523.hdf', '\luig20160302_06_te1580.hdf, '\luig20160303_09_te1589.hdf', '\luig20160302_06_te1580.hdf']

control_days = ['\luig20160218_10_te1469.hdf','\luig20160223_11_te1508.hdf','\luig20160224_15_te1523.hdf', \
                '\luig20160303_11_te1591.hdf', '\luig20160308_06_te1647.hdf','\luig20160309_25_te1672.hdf', '\luig20160316_13_te1752.hdf']
'''
sham_days = ['\papa20150217_05.hdf','\papa20150305_02.hdf',
    '\papa20150310_02.hdf',
    '\papa20150519_02.hdf','\papa20150519_04.hdf','\papa20150528_02.hdf']

stim_days = ['\papa20150211_11.hdf',
    '\papa20150218_04.hdf','\papa20150219_09.hdf','\papa20150223_02.hdf','\papa20150224_02.hdf','\papa20150303_03.hdf',
    '\papa20150306_07.hdf','\papa20150309_04.hdf']
control_days = ['\papa20150508_12.hdf','\papa20150508_13.hdf','\papa20150518_03.hdf',
    '\papa20150518_05.hdf','\papa20150518_06.hdf','\papa20150522_05.hdf','\papa20150522_06.hdf','\papa20150524_02.hdf',
    '\papa20150524_04.hdf','\papa20150525_01.hdf','\papa20150525_02.hdf',
    '\papa20150530_01.hdf','\papa20150530_02.hdf','\papa20150601_02.hdf','\papa20150602_03.hdf',
    '\papa20150602_04.hdf']
'''



all_days = stim_days + sham_days + control_days
counter_stim = 0
counter_sham = 0
counter_control = 0
avg_low_stim = 0
avg_low_sham = 0
avg_low_control = 0

num_stim_days = len(stim_days)
num_sham_days = len(sham_days)
num_control_days = len(control_days)

prob_low_aligned_stim = np.zeros((num_stim_days,5))
prob_low_aligned_rewarded_stim = np.zeros((num_stim_days,5))
prob_low_aligned_unrewarded_stim = np.zeros((num_stim_days,5))

prob_low_aligned_sham = np.zeros((num_sham_days,5))
prob_low_aligned_rewarded_sham = np.zeros((num_sham_days,5))
prob_low_aligned_unrewarded_sham = np.zeros((num_sham_days,5))

prob_low_aligned_control = np.zeros((num_control_days,5))
prob_low_aligned_rewarded_control = np.zeros((num_control_days,5))
prob_low_aligned_unrewarded_control = np.zeros((num_control_days,5))



'''
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
'''
for i in range(0,num_stim_days):
	name = stim_days[i]
	hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Luigi\hdf'+name
	prob_low_aligned_stim[i,:], prob_low_aligned_rewarded_stim[i,:], prob_low_aligned_unrewarded_stim[i,:] = PeriStimulusFreeChoiceBehavior(hdf_location)

for i in range(0,num_sham_days):
	name = sham_days[i]
	hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Luigi\hdf'+name
	prob_low_aligned_sham[i,:], prob_low_aligned_rewarded_sham[i,:], prob_low_aligned_unrewarded_sham[i,:] = PeriStimulusFreeChoiceBehavior(hdf_location)

for i in range(0,num_control_days):
	name = control_days[i]
	hdf_location = 'C:\Users\Samantha Summerson\Dropbox\Carmena Lab\Luigi\hdf'+name
	prob_low_aligned_control[i,:], prob_low_aligned_rewarded_control[i,:], prob_low_aligned_unrewarded_control[i,:] = PeriStimulusFreeChoiceBehavior(hdf_location)

avg_prob_low_aligned_stim = np.mean(prob_low_aligned_stim,axis=0)
std_prob_low_aligned_stim = np.std(prob_low_aligned_stim,axis=0)
sem_prob_low_aligned_stim = std_prob_low_aligned_stim/np.sqrt(num_stim_days)
avg_prob_low_aligned_rewarded_stim = np.mean(prob_low_aligned_rewarded_stim,axis=0)
std_prob_low_aligned_rewarded_stim = np.std(prob_low_aligned_rewarded_stim,axis=0)
sem_prob_low_aligned_rewarded_stim = std_prob_low_aligned_rewarded_stim/np.sqrt(num_stim_days)
avg_prob_low_aligned_unrewarded_stim = np.mean(prob_low_aligned_unrewarded_stim,axis=0)
std_prob_low_aligned_unrewarded_stim = np.std(prob_low_aligned_unrewarded_stim,axis=0)
sem_prob_low_aligned_unrewarded_stim = std_prob_low_aligned_unrewarded_stim/np.sqrt(num_stim_days)

avg_prob_low_aligned_sham = np.mean(prob_low_aligned_sham,axis=0)
std_prob_low_aligned_sham = np.std(prob_low_aligned_sham,axis=0)
sem_prob_low_aligned_sham = std_prob_low_aligned_sham/np.sqrt(num_sham_days)
avg_prob_low_aligned_rewarded_sham = np.mean(prob_low_aligned_rewarded_sham,axis=0)
std_prob_low_aligned_rewarded_sham = np.std(prob_low_aligned_rewarded_sham,axis=0)
sem_prob_low_aligned_rewarded_sham = std_prob_low_aligned_sham/np.sqrt(num_sham_days)
avg_prob_low_aligned_unrewarded_sham = np.mean(prob_low_aligned_unrewarded_sham,axis=0)
std_prob_low_aligned_unrewarded_sham = np.std(prob_low_aligned_unrewarded_sham,axis=0)
sem_prob_low_aligned_unrewarded_sham = std_prob_low_aligned_unrewarded_sham/np.sqrt(num_sham_days)

avg_prob_low_aligned_control = np.mean(prob_low_aligned_control,axis=0)
std_prob_low_aligned_control = np.std(prob_low_aligned_control,axis=0)
sem_prob_low_aligned_control = std_prob_low_aligned_control/np.sqrt(num_control_days)
avg_prob_low_aligned_rewarded_control = np.mean(prob_low_aligned_rewarded_control,axis=0)
std_prob_low_aligned_rewarded_control = np.std(prob_low_aligned_rewarded_control,axis=0)
sem_prob_low_aligned_rewarded_control = std_prob_low_aligned_rewarded_control/np.sqrt(num_control_days)
avg_prob_low_aligned_unrewarded_control = np.mean(prob_low_aligned_unrewarded_control,axis=0)
std_prob_low_aligned_unrewarded_control = np.std(prob_low_aligned_unrewarded_control,axis=0)
sem_prob_low_aligned_unrewarded_control = std_prob_low_aligned_unrewarded_control/np.sqrt(num_control_days)

width = float(0.35)
ind = np.arange(0,5)
# Get linear fit to total prob
m_stim,b_stim = np.polyfit(ind, avg_prob_low_aligned_stim, 1)
m_sham,b_sham = np.polyfit(ind, avg_prob_low_aligned_sham, 1)
m_control,b_control = np.polyfit(ind, avg_prob_low_aligned_control, 1)


plt.figure()
plt.bar(ind, avg_prob_low_aligned_rewarded_stim, width/2, color = 'c', hatch = '//', yerr = sem_prob_low_aligned_rewarded_stim/2)
plt.bar(ind, avg_prob_low_aligned_unrewarded_stim, width/2, color = 'c', bottom=avg_prob_low_aligned_rewarded_stim,yerr = sem_prob_low_aligned_unrewarded_stim/2, label='LV Stim')
plt.plot(ind,m_stim*ind + b_stim,'c--')
plt.bar(ind + width/2, avg_prob_low_aligned_rewarded_sham, width/2, color = 'm',hatch = '//', yerr = sem_prob_low_aligned_rewarded_sham/2)
plt.bar(ind + width/2, avg_prob_low_aligned_unrewarded_sham, width/2, color = 'm',bottom=avg_prob_low_aligned_rewarded_sham,yerr = sem_prob_low_aligned_unrewarded_sham/2, label='Sham')
plt.plot(ind+width/2, m_sham*ind + b_sham,'m--')
plt.bar(ind + width, avg_prob_low_aligned_rewarded_control, width/2, color = 'y',hatch = '//', yerr = sem_prob_low_aligned_rewarded_control/2)
plt.bar(ind + width, avg_prob_low_aligned_unrewarded_control, width/2, color = 'y',bottom=avg_prob_low_aligned_rewarded_control,yerr = sem_prob_low_aligned_unrewarded_control/2, label='HV Stim')
plt.plot(ind+width, m_control*ind + b_control,'y--')
plt.ylabel('P(Choose LV Target)')
plt.title('Target Selection')
plt.xticks(ind + width/2, ('1', '2', '3', '4','5'))
plt.xlabel('Trials post-stimulation')
plt.ylim([0.0,.35])
plt.xlim([-0.1,5.4])
plt.legend()
'''
plt.subplot(1,2,2)
plt.plot(range(0,5), avg_prob_low_aligned_stim,'c')
plt.plot(range(5,10), avg_prob_low_aligned_sham, 'm')
plt.plot(range(10,15), avg_prob_low_aligned_control, 'y')
plt.ylim([0.0,0.3])
'''
plt.show()