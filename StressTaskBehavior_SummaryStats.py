import numpy as np 
import scipy as sp
import pandas
from scipy import stats
import matplotlib as mpl
import tables
from matplotlib import pyplot as plt
from rt_calc import get_rt_change_deriv
from StressTaskBehavior import StressBehavior, TDTNeuralData
from scipy import fft, arange, signal
from matplotlib.colors import LogNorm


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

hdf_single = 'mari20160418_04_te2002.hdf'
tdt_prefix = 'C:/Users/Samantha Summerson/Documents/GitHub/analysis/'
tdt_single = 'Mario20160418'

cmap = mpl.cm.hsv

print 'Loaded all file names.'

def CompareRT_RegularVStress(stress_hdf_files, hdf_prefix): 

	all_reg_rt = []
	all_stress_rt = []

	avg_reg_rt = np.zeros(len(stress_hdf_files))
	avg_stress_rt = np.zeros(len(stress_hdf_files))

	for i, name in enumerate(stress_hdf_files):
		print 'File %i of %i' % (i+1, len(stress_hdf_files))
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

def CompareTrialRateandChoices_RegularVStress(stress_hdf_files, hdf_prefix):
	'''
	This method computes the average number of trials/min performed over each block, as well as the fraction
	of optimal choices for each block. Since learning may occur during the first block, we only consider the 
	choice behavior over the last 20 trials in each block.
	'''

	avg_trialspermin_reg = np.zeros(len(stress_hdf_files))
	avg_trialspermin_stress = np.zeros(len(stress_hdf_files))
	avg_fracoptchoice_reg = np.zeros(len(stress_hdf_files))
	avg_fracoptchoice_stress = np.zeros(len(stress_hdf_files))

	for i, name in enumerate(stress_hdf_files):
		print 'File %i of %i' % (i+1, len(stress_hdf_files))
		# load stress data into class object StressBehavior
		hdf_location = hdf_prefix +name
		stress_data = StressBehavior(hdf_location)

		# avg_trials_per_min is an array of length M, where M is the number of blocks, with elements equaling the 
		# average number of trials per min in that block. end_block_avg_hv_choices is an array of length M with
		# elements equaling the final sliding avg probability of choosing the HV option. 
		avg_trials_per_min, end_block_avg_hv_choices = stress_data.PlotTrialChoices(20)

		avg_trialspermin_reg[i] = np.nanmean(avg_trials_per_min[0::2]) 	# always start with block of regular trials
		avg_trialspermin_stress[i] = np.nanmean(avg_trials_per_min[1::2])	# stress block always comes second and alternates
		avg_fracoptchoice_reg[i] = np.nanmean(end_block_avg_hv_choices[0::2])
		avg_fracoptchoice_stress[i] = np.nanmean(end_block_avg_hv_choices[1::2])

	
	t, p = stats.ttest_ind(avg_fracoptchoice_reg, avg_fracoptchoice_stress)
	u_value, p_value = stats.mannwhitneyu(avg_trialspermin_reg, avg_trialspermin_stress)

	width = 0.5
	fig = plt.figure(1)
	ax = plt.subplot(121)
	plt.boxplot([avg_trialspermin_reg, avg_trialspermin_stress],0,'')
	plt.ylabel('Avg Trials/Min')
	plt.xticks(np.arange(2)+1., ('Regular', 'Stress'))
	plt.title('Avg. Trial Rate over %i Sessions: (u,p) = (%f,%f)' % (len(stress_hdf_files), u_value, p_value))
	ax.text(0.6, 6.5, 'Regular: med = %f, Stress: med = %f' % (np.median(avg_trialspermin_reg), np.median(avg_trialspermin_stress)))
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	ax = plt.subplot(122)
	plt.bar(0, [np.nanmean(avg_fracoptchoice_reg)], width, color = 'b', ecolor = 'k', yerr = [np.nanstd(avg_fracoptchoice_reg)/np.sqrt(len(avg_fracoptchoice_reg) - 1.)])
	plt.bar(1, [np.nanmean(avg_fracoptchoice_stress)], width, color = 'r', ecolor = 'k', yerr = [ np.nanstd(avg_fracoptchoice_stress)/np.sqrt(len(avg_fracoptchoice_stress) - 1.)])
	plt.ylabel('Avg Frac of Optimal Choices')
	plt.xticks(np.arange(2)+1., ('Regular', 'Stress'))
	plt.title('Avg. Optimal Choices over %i Sessions: (t,p) = (%f,%f)' % (len(stress_hdf_files), t, p))
	ax.text(0.6, 0.85, 'Regular: m = %f, Stress: m = %f' % (np.mean(avg_fracoptchoice_reg), np.mean(avg_fracoptchoice_stress)))
	ax.get_yaxis().set_tick_params(direction='out')
	ax.get_xaxis().set_tick_params(direction='out')
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	plt.show()
	return

#CompareRT_RegularVStress(stress_hdf_files_adjust, hdf_prefix)
#CompareTrialRateandChoices_RegularVStress(stress_hdf_files_adjust, hdf_prefix)

hdf_location = hdf_prefix +hdf_single
print "Loading behavior"
stress_data = StressBehavior(hdf_location)
channel = 10
NFFT = 1024
noverlap = 512

tdt_location = tdt_prefix + tdt_single
print "Loading neural data"
tdt_data = TDTNeuralData(tdt_location,1)
hdf_row_ind_gocue = stress_data.state_time[stress_data.ind_check_reward_states - 3]
trial_type = stress_data.stress_type[stress_data.state_time[stress_data.ind_check_reward_states]]

reg_trial_inds = np.array([ind for ind in range(len(trial_type)) if trial_type[ind] == 0])
stress_trial_inds = np.array([ind for ind in range(len(trial_type)) if trial_type[ind] == 1])
print "%i reg trials, %i stress trials" %(len(reg_trial_inds), len(stress_trial_inds))

pulse_ind, pupil_ind, lfp_ind = tdt_data.get_indices_for_behavior_events(hdf_row_ind_gocue)
Fs = tdt_data.lfp_samprate
lfp_single_chan = tdt_data.lfp[channel]

tot_Sxx_reg = 0
print "Computing spectrograms for regular trials"
for ind in lfp_ind[reg_trial_inds]:
	data = lfp_single_chan[ind - Fs:ind + 3*Fs]
	cf_list, time_points, Sxx = signal.spectrogram(data, fs = Fs, nperseg = NFFT, noverlap=noverlap)  # units are V**2/Hz
	Sxx = np.sqrt(Sxx)		# units are V/sqrt(Hz)
	tot_Sxx_reg += Sxx
	dur = len(data)/(Fs*60)  # duration in minutes

tot_Sxx_reg = tot_Sxx_reg/len(reg_trial_inds)

tot_Sxx_stress = 0
print "Computing spectrograms for stress trials"
for ind in lfp_ind[stress_trial_inds]:
	data = lfp_single_chan[ind - Fs:ind + 3*Fs]
	cf_list, time_points, Sxx = signal.spectrogram(data, fs = Fs, nperseg = NFFT, noverlap=noverlap)  # units are V**2/Hz
	Sxx = np.sqrt(Sxx)		# units are V/sqrt(Hz)
	tot_Sxx_stress += Sxx

tot_Sxx_stress = tot_Sxx_stress/len(stress_trial_inds)

num_time_points = len(time_points)
f_end = 8

min_val = np.min([np.min(tot_Sxx_reg), np.min(tot_Sxx_stress)])
max_val = np.max([np.max(tot_Sxx_reg), np.max(tot_Sxx_stress)])
### need to ensure colorbars are the same
fig = plt.figure()
ax = plt.subplot(121)
im = plt.imshow(tot_Sxx_reg[1:f_end*2,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0,dur,0, len(cf_list[1:f_end*2])], clim=(min_val, max_val))
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
yticks = np.arange(0, len(cf_list[1:f_end*2]), 10)
yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#yticks = np.arange(0, cf_list[f_end*2], 10)
#yticklabels = ['{0:.2f}'.format(i) for i in yticks]
#xticks = np.arange(0, num_time_points/Fs, Fs)  # ticks every second
#xticklabels = ['{0:.2f}'.format(x[i]) for i in xticks]
plt.yticks(yticks, yticklabels)
#plt.xticks(xticks, xticklabels)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (min)')
plt.title('Spectrogram')
fig.colorbar(im)

ax = plt.subplot(122)
im = plt.imshow(tot_Sxx_stress[1:f_end*2,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0,dur,0, len(cf_list[1:f_end*2])], clim=(min_val, max_val))
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
yticks = np.arange(0, len(cf_list[1:f_end*2]), 10)
yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#yticks = np.arange(0, cf_list[f_end*2], 10)
#yticklabels = ['{0:.2f}'.format(i) for i in yticks]
#xticks = np.arange(0, num_time_points/Fs, Fs)  # ticks every second
#xticklabels = ['{0:.2f}'.format(x[i]) for i in xticks]
plt.yticks(yticks, yticklabels)
#plt.xticks(xticks, xticklabels)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (min)')
plt.title('Spectrogram')
fig.colorbar(im)
plt.show()

# z-score across frequency
num_x, num_f = tot_Sxx_reg.shape
tot_Sxx_reg_zscore = tot_Sxx_reg
for i in range(num_f):
	tot_Sxx_reg_zscore[:,i] = (tot_Sxx_reg[:,i] - np.nanmean(tot_Sxx_reg[:,i]))/np.nanstd(tot_Sxx_reg[:,i])

fig = plt.figure()
ax = plt.subplot(111)
im = plt.imshow(tot_Sxx_reg_zscore[1:f_end*3,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0,dur,0, len(cf_list[1:f_end*2])])
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
yticks = np.arange(0, len(cf_list[1:f_end*2]), 10)
yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#yticks = np.arange(0, cf_list[f_end*2], 10)
#yticklabels = ['{0:.2f}'.format(i) for i in yticks]
#xticks = np.arange(0, num_time_points/Fs, Fs)  # ticks every second
#xticklabels = ['{0:.2f}'.format(x[i]) for i in xticks]
plt.yticks(yticks, yticklabels)
#plt.xticks(xticks, xticklabels)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (min)')
plt.title('Spectrogram')
fig.colorbar(im)

"""
ax = plt.subplot(133)
im = plt.imshow(tot_Sxx_stress[1:f_end*2,:] - tot_Sxx_reg[1:f_end*2,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0,dur,0, len(cf_list[1:f_end*2])])
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
yticks = np.arange(0, len(cf_list[1:f_end*2]), 10)
yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#yticks = np.arange(0, cf_list[f_end*2], 10)
#yticklabels = ['{0:.2f}'.format(i) for i in yticks]
#xticks = np.arange(0, num_time_points/Fs, Fs)  # ticks every second
#xticklabels = ['{0:.2f}'.format(x[i]) for i in xticks]
plt.yticks(yticks, yticklabels)
#plt.xticks(xticks, xticklabels)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (min)')
plt.title('Spectrogram')
fig.colorbar(im)
"""
plt.show()