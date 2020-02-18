from CaImagingAnalysis import CaData
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib as mpl


#dir = "C:/Users/ss45436/Box/CNPRC/Data/Left/New/New/"
dir = "C:/Users/ss45436/Documents/MATLAB/Left/New/New/"
filenames = ['2018-11-21-10-49-56.mat', \
			'2018-11-26-11-45-46.mat', \
			'2018-12-04-10-31-21.mat', \
			'2018-12-11-10-53-04.mat', \
			'2018-12-14-11-01-41.mat', \
			'2018-12-17-11-38-42.mat', \
			'2018-12-18-11-20-21.mat', \
			'2018-12-19-11-24-58.mat', \
			'2019-01-07-10-45-52.mat', \
			'2019-01-24-11-36-02.mat']

def unit_classification_by_zone(ca_data,**kwargs):
	'''
	Classify units as tuned to either Zone 1 or Zone 2. Classification is done by regressing the event rates of the units
	from a designated time bin as a function of the zone reached to. If the regressor is significant for either zone 1 or
	zone 2 reaches, the unit is labeled accordingly. 

	The default parameters are for decoding in the reach task for rand-hand reaches to either zone 1 or zone 2, but 
	optional input parameters can be used to adjust the hands and zones used in decoding so that this code generalizes
	to any behavioral conditions.

	Inputs:
	- ca_data: CaData object; data from one session formated using the CaData class
	- t_before: int; optional input to indicate when decoding time window should start, default is -1s relative to zone entry
	- t_after: int; optional input to indicate when decoding time window should end, default is 1 s relative to zone entry
	- event: string; optional input to indicate whether to use calcium ('ca'; default) or spike ('sp') events
	- hand1: string; either 'r' or 'l' to indicate which hand is used for first set of reaches considered, default is 'r'
	- hand2: string; either 'r' or 'l' to indicate which hand is used for second set of reaches considered, default is 'r'
	- zone1: int; either 1 or 2 to indicate whether first set of reaches are to zone 1 or zone 2, respectively, with a default of 1
	- zone2: int; either 1 or 2 to indicate whether second set of reaches are to zone 1 or zone 2, respectively, with a default of 2
	- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
	'''
	# Determine method parameters
	t_before = kwargs.get('t_before', 1)
	t_after = kwargs.get('t_after',1)
	event = kwargs.get('event', 'ca')
	hand1 = kwargs.get('hand1', 'r')
	hand2 = kwargs.get('hand2', 'r')
	zone1 = kwargs.get('zone1', 1)
	zone2 = kwargs.get('zone2', 2)
	chans = kwargs.get('chans', np.arange(0,ca_data.num_units))

	# Retrieve zone entry data for designated hand and zone
	if (hand1=='r') & (zone1==1):
		zentry1 = ca_data.RH_zone1_entry
	elif (hand1=='r') & (zone1==2):
		zentry1 = ca_data.RH_zone2_entry
	elif (hand1=='l') & (zone1==1):
		zentry1 = ca_data.LH_zone1_entry
	elif (hand1=='l') & (zone1==2):
		zentry1 = ca_data.LH_zone2_entry

	if (hand2=='r') & (zone2==1):
		zentry2 = ca_data.RH_zone1_entry
	elif (hand2=='r') & (zone2==2):
		zentry2 = ca_data.RH_zone2_entry
	elif (hand2=='l') & (zone2==1):
		zentry2 = ca_data.LH_zone1_entry
	elif (hand2=='l') & (zone2==2):
		zentry2 = ca_data.LH_zone2_entry

	# Compute event rates for all units for each zone entry
	er1 = ca_data.event_rates(event, zentry1, chans, t_before, t_after)
	er2 = ca_data.event_rates(event, zentry2, chans, t_before, t_after)

	# For each unit, do regression of event rate as a function of each zone
	zone_pvalues = np.zeros((ca_data.num_units, 3)) 		# placeholder for pvalues from regression, dim = 3 with first regressor for zone 1, seconds for zone 2, and third for intercept
	for i,chan in enumerate(chans):
		y = np.append(np.ravel(er1[chan]),np.ravel(er2[chan]))
		x = np.append(np.ones(len(np.ravel(er1[chan]))), np.zeros(len(np.ravel(er2[chan]))))
		x = np.transpose(x)
		x = np.hstack((x, np.append(np.zeros(len(np.ravel(er1[chan]))), np.ones(len(np.ravel(er2[chan]))))))
		x = np.hstack((x, np.ones([len(np.ravel(er1[chan]))+ len(np.ravel(er2[chan])),1]))) 	# use this in place of add_constant which doesn't work when constant Q values are used

		print("Regression for unit ", i)
		model_glm = sm.OLS(y,x)
		fit_glm = model_glm.fit()
		zone_pvalues[i,:] = fit_glm.pvalues

	return zone_pvalues

def logistic_regression_decoding_binsize(ca_data, **kwargs):
	'''
	Look at effect of bin size on decoding with logistic regression (using decoding_logisitic_regression method for 
	CaData class). Bins are centered around the zone entry. Output is a plot of the decoding accuracy over the different
	bin sizes.

	The default parameters are for decoding in the reach task for rand-hand reaches to either zone 1 or zone 2, but 
	optional input parameters can be used to adjust the hands and zones used in decoding so that this code generalizes
	to any behavioral conditions.

	Inputs:
	- ca_data: CaData object; data from one session formated using the CaData class
	- smallest_bin: int; optional input to indicate the smallest bin size in seconds to consider, default is 0.5 s
	- largest_bin: int; optional input to indicate the largest bin size in seconds to consider, default is 4 s
	- bin_step_size: int; optional input to indicate the step size in seconds for the time bins, default is 0.5 s
	- event: string; optional input to indicate whether to use calcium ('ca'; default) or spike ('sp') events
	- hand1: string; either 'r' or 'l' to indicate which hand is used for first set of reaches considered, default is 'r'
	- hand2: string; either 'r' or 'l' to indicate which hand is used for second set of reaches considered, default is 'r'
	- zone1: int; either 1 or 2 to indicate whether first set of reaches are to zone 1 or zone 2, respectively, with a default of 1
	- zone2: int; either 1 or 2 to indicate whether second set of reaches are to zone 1 or zone 2, respectively, with a default of 2
	'''
	
	# Determine method parameters
	smallest_bin = kwargs.get('smallest_bin', 0.5)
	largest_bin = kwargs.get('largest_bin', 4.)
	bin_step_size = kwargs.get('bin_step_size', 0.5)
	event = kwargs.get('event', 'ca')
	hand1 = kwargs.get('hand1', 'r')
	hand2 = kwargs.get('hand2', 'r')
	zone1 = kwargs.get('zone1', 1)
	zone2 = kwargs.get('zone2', 2)

	# Define new variables for storing cross-validated decoding accuracy for each fold
	bin_size = np.arange(smallest_bin,largest_bin+bin_step_size,bin_step_size)
	cv_binsize = np.zeros((len(bin_size),10))					# dimension 10 comes from 10-fold cross-validation
	cv_binsize_shuffle = np.zeros((len(bin_size),10))			# dimension 10 comes from 10-fold cross-validation

	# For each bin size, find the decoding accuracy over the data and shuffled data
	for i in range(len(bin_size)):
		t_before = bin_size[i]/2.
		t_after = bin_size[i]/2.
		cv_binsize[i,:], cv_binsize_shuffle[i,:] = ca_data.decoding_logistic_regression(event, hand1,hand2,zone1,zone2, t_before = t_before, t_after = t_after)

	# Compute average performance and the standard error of the mean
	cv_binsize_avg = np.nanmean(cv_binsize,axis = 1)
	cv_binsize_sem = np.nanstd(cv_binsize,axis = 1)/np.sqrt(10)
	cv_binsize_shuffle_avg = np.nanmean(cv_binsize_shuffle,axis = 1)
	cv_binsize_shuffle_sem = np.nanstd(cv_binsize_shuffle,axis = 1)/np.sqrt(10)

	# Plot the average performance with error bars
	plt.fill_between(bin_size, cv_binsize_avg-cv_binsize_sem,cv_binsize_avg+cv_binsize_sem, facecolor = 'b', alpha = 0.5, label = 'Observed outcomes')
	plt.fill_between(bin_size, cv_binsize_shuffle_avg-cv_binsize_shuffle_sem,cv_binsize_shuffle_avg+cv_binsize_shuffle_sem, facecolor = 'c', alpha = 0.5, label = "Shuffled outcomes")
	plt.plot(bin_size, cv_binsize_avg, 'b-o')
	plt.plot(bin_size, cv_binsize_shuffle_avg, 'c-o')
	plt.ylim((0,1))
	plt.xlabel('Bin size (s)')
	plt.ylabel('Decoding accuracy')
	plt.title('Logistic Regression')
	plt.legend()
	#plt.show()

	plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + ca_data.filename[:-4] + '_decoding_RH_reach_binsize.svg')
	plt.close()

	return

def logistic_regression_decoding_overtime(ca_data, fix_bin_size, **kwargs):
	'''
	Look at effect of decoding performance with logistic regression (using decoding_logisitic_regression method for 
	CaData class) over time using a fixed bin width. Output is a plot of the decoding accuracy over time points relative
	to zone entry.

	The default parameters are for decoding in the reach task for rand-hand reaches to either zone 1 or zone 2, but 
	optional input parameters can be used to adjust the hands and zones used in decoding so that this code generalizes
	to any behavioral conditions.

	Inputs:
	- ca_data: CaData object; data from one session formated using the CaData class
	- t_begin: int; optional input to indicate the earliest time that a decoding time window should start, default is -3 s relative to zone entry
	- t_end: int; optional input to indicate the latest time that a decoding time window should end, default is 1 s relative to zone entry
	- t_slide: int; optional input to indicate the amount that time bins should move between analyses, default is 0.1 s
	- event: string; optional input to indicate whether to use calcium ('ca'; default) or spike ('sp') events
	- hand1: string; either 'r' or 'l' to indicate which hand is used for first set of reaches considered, default is 'r'
	- hand2: string; either 'r' or 'l' to indicate which hand is used for second set of reaches considered, default is 'r'
	- zone1: int; either 1 or 2 to indicate whether first set of reaches are to zone 1 or zone 2, respectively, with a default of 1
	- zone2: int; either 1 or 2 to indicate whether second set of reaches are to zone 1 or zone 2, respectively, with a default of 2
	'''
	
	# Determine method parameters
	t_begin = kwargs.get('t_begin', -3)
	t_end = kwargs.get('t_end',1)
	t_slide = kwargs.get('t_slide', 0.1)
	event = kwargs.get('event', 'ca')
	hand1 = kwargs.get('hand1', 'r')
	hand2 = kwargs.get('hand2', 'r')
	zone1 = kwargs.get('zone1', 1)
	zone2 = kwargs.get('zone2', 2)
	
	# Define new variables for storing cross-validated decoding accuracy for each fold 
	t_start = np.arange(t_begin,t_end+t_slide,t_slide)
	cv_t_start = np.zeros((len(t_start),10))				# 10 comes from the fact that we're doing 10-fold cross-validation
	cv_t_start_shuffle = np.zeros((len(t_start),10))		# 10 comes from the fact that we're doing 10-fold cross-validation
	
	# Compute accuracy for fixed bin size beginning at different time points relative to zone entry
	for i in range(len(t_start)):
		t_before = -t_start[i]
		t_after = fix_bin_size - t_before
		cv_t_start[i,:], cv_t_start_shuffle[i,:] = ca_data.decoding_logistic_regression(event, hand1,hand2,zone1,zone2, t_before = t_before, t_after = t_after)

	# Compute average performance and the standard error of the mean
	cv_t_start_avg = np.nanmean(cv_t_start,axis = 1)
	cv_t_start_sem = np.nanstd(cv_t_start,axis = 1)/np.sqrt(10)
	cv_t_start_shuffle_avg = np.nanmean(cv_t_start_shuffle,axis = 1)
	cv_t_start_shuffle_sem = np.nanstd(cv_t_start_shuffle,axis = 1)/np.sqrt(10)

	# Plot the average performance with error bars
	plt.fill_between(t_start+fix_bin_size/2., cv_t_start_avg-cv_t_start_sem,cv_t_start_avg+cv_t_start_sem, facecolor = 'tab:orange', alpha = 0.5, label = 'Observed outcomes')
	plt.fill_between(t_start+fix_bin_size/2., cv_t_start_shuffle_avg-cv_t_start_shuffle_sem,cv_t_start_shuffle_avg+cv_t_start_shuffle_sem, facecolor = 'tab:gray', alpha = 0.5, label = "Shuffled outcomes")
	plt.plot(t_start+fix_bin_size/2., cv_t_start_avg, 'tab:orange','o')
	plt.plot(t_start+fix_bin_size/2., cv_t_start_shuffle_avg, 'tab:gray','o')
	plt.xlabel('Time relative to zone entry (s)')
	plt.ylabel('Decoding accuracy')
	plt.title('Logistic Regression - Bin size %i s' % (fix_bin_size))
	plt.ylim((0,1))
	plt.legend()
	#plt.show()

	plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + ca_data.filename[:-4] + '_decoding_RH_reach_overtime.svg')
	plt.close()

	return cv_t_start_avg, cv_t_start_sem, cv_t_start_shuffle_avg, cv_t_start_shuffle_sem

'''

ca_data = CaData(dir + filenames[0])
pvals = unit_classification_by_zone(ca_data)
'''
# Make the following plots:
# 1. Population rasters
# 2. Rasters aligned to zone 1 entry with RH
# 3. Rasters aligned to zone 2 entry with RH
# 4. PSTHs aligned to zone 1 and zone 2 entry with RH
# 5. Average calcium traces for zone 1 and zone 2 entry with RH
"""
for k,file in enumerate(filenames):
	print(file)
	ca_data = CaData(dir + file)
	#ca_data.population_raster('ca')
	#ca_data.trial_raster('ca','r',1)
	#ca_data.trial_raster('ca','r',2)
	#ca_data.psth_compare_zones('ca','r')
	ca_data.avg_trial_traces('r',1,zone2 = 2)
"""

# Make the following plots:
# 	1. Decoding with different bin sizes for each session
#	2. Decoding with fixed bin size over time for each session
#	3. Peak accuracy with a fixed bin size 
#	4. Decoding with random subsamplings of units

peak_avg_accuracy_all_units = np.zeros(len(filenames))
peak_sem_accuracy_all_units = np.zeros(len(filenames))
peak_avg_accuracy_shuffle_all_units = np.zeros(len(filenames))
peak_sem_accuracy_shuffle_all_units = np.zeros(len(filenames))
avg_accuracy_all_units = np.zeros(len(filenames))
sem_accuracy_all_units = np.zeros(len(filenames))
avg_accuracy_shuffle_all_units = np.zeros(len(filenames))
sem_accuracy_shuffle_all_units = np.zeros(len(filenames))

num_sub_units = [50,40,30,20,10,1]

avg_accuracy_subunits = np.zeros((len(filenames),len(num_sub_units),100))				# 10 comes from the fact that we're repeating the subsampling 10x
avg_accuracy_shuffle_subunits = np.zeros((len(filenames),len(num_sub_units),100))		# 10 comes from the fact that we're repeating the subsampling 10x

for k,file in enumerate(filenames):
	print(file)
	ca_data = CaData(dir + file)
	logistic_regression_decoding_binsize(ca_data)
	print('Completed decoding over bin sizes.')
	cv_t_start_avg, cv_t_start_sem, cv_t_start_shuffle_avg, cv_t_start_shuffle_sem = logistic_regression_decoding_overtime(ca_data, 2.)
	print('Completed decoding over time.')
	
	# Peak average accuracy for fixed bin size
	max_accuracy_ind = np.argmax(cv_t_start_avg)
	peak_avg_accuracy_all_units[k] = cv_t_start_avg[max_accuracy_ind]
	peak_sem_accuracy_all_units[k] = cv_t_start_sem[max_accuracy_ind]
	peak_avg_accuracy_shuffle_all_units[k] = cv_t_start_shuffle_avg[max_accuracy_ind]
	peak_sem_accuracy_shuffle_all_units[k] = cv_t_start_shuffle_sem[max_accuracy_ind]

	# Average accuracy at for bin (-1,1) around zone entry
	avg_accuracy_all_units[k] = cv_t_start_avg[20]
	sem_accuracy_all_units[k] = cv_t_start_sem[20]
	avg_accuracy_shuffle_all_units[k] = cv_t_start_shuffle_avg[20]
	sem_accuracy_shuffle_all_units[k] = cv_t_start_shuffle_sem[20]

	# DECODING FOR SUB-NUMBER OF UNITS, repeat random subsampling 100x
	for n,num_su in enumerate(num_sub_units):	
		for m in np.arange(100):
			channels = np.random.choice(ca_data.num_units,size = num_su, replace = False)
			cv_base, cv_base_shuffle = ca_data.decoding_logistic_regression('ca', 'r','r',1,2, t_before = 1, t_after = 1, chans = channels)
			avg_accuracy_subunits[k,n,m] = np.nanmean(cv_base)
			avg_accuracy_shuffle_subunits[k,n,m] = np.nanmean(cv_base_shuffle)
		print('Completed decoding over subsampling with %i unit.' % (num_su))

# Plot peak accuracy over days
ind = np.arange(len(filenames))
width = 0.35
plt.bar(ind, peak_avg_accuracy_all_units, width, color = 'tab:orange', yerr = peak_sem_accuracy_all_units, label = 'Observed data')
plt.bar(ind+0.35, peak_avg_accuracy_shuffle_all_units, width, color = 'tab:gray', yerr = peak_sem_accuracy_shuffle_all_units, label = 'Shuffled data')
plt.ylim((0,1))
xticklabels = ['Session %i' % (d+1) for d in np.arange(len(filenames))]
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.ylabel('Decoding Accuracy')
plt.title('Peak Decoding Accuracy Across Bins - Fixed bin size 2 s')
plt.legend()
plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/Decoding_RH_reaches_all_sessions_peak_decoding.svg")
plt.close()

# Plot average accuracy in (-1,1) bin over days
plt.bar(ind, avg_accuracy_all_units, width, color = 'tab:orange', yerr = sem_accuracy_all_units, label = 'Observed data')
plt.bar(ind+0.35, avg_accuracy_shuffle_all_units, width, color = 'tab:gray', yerr = sem_accuracy_shuffle_all_units, label = 'Shuffled data')
plt.ylim((0,1))
xticklabels = ['Session %i' % (d+1) for d in np.arange(len(filenames))]
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.ylabel('Decoding Accuracy')
plt.title('Avg Decoding Accuracy for Bin [-1,1] s around zone entry')
plt.legend()
plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/Decoding_RH_reaches_all_sessions_average_decoding.svg")
plt.close()

# Compute and plot decoding performance for subsampling units
cmap = mpl.cm.brg
avg_accuracy_subsampling = np.nanmean(avg_accuracy_subunits,axis=2)
sem_accuracy_subsampling = np.nanstd(avg_accuracy_subunits,axis=2)/np.sqrt(10)
width = 1./(len(num_sub_units) + 2)
plt.errorbar(ind, avg_accuracy_all_units, color = 'k', yerr = sem_accuracy_all_units, label = 'All units')
for n,num_su in enumerate(num_sub_units):
	plt.errorbar(ind, avg_accuracy_subsampling[:,n], color=cmap(n/float(len(num_sub_units))), yerr = sem_accuracy_subsampling[:,n], label = '%i units' % (num_su))
plt.errorbar(ind, avg_accuracy_shuffle_all_units, color = 'c', yerr = sem_accuracy_shuffle_all_units, label = 'Shuffled data -\n (all units)')
plt.ylim((0,1))
xticklabels = ['Session %i' % (d+1) for d in np.arange(len(filenames))]
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.ylabel('Decoding Accuracy')
plt.title('Avg Decoding Accuracy for Bin [-1,1] s around zone entry')
plt.legend()
plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/Decoding_RH_reaches_all_sessions_average_decoding_withsubsampling.svg")
plt.close()

avg_accuracy_per_number_subsampled = np.nanmean(avg_accuracy_subsampling,axis = 0)
sem_accuracy_per_number_subsampled = np.nanstd(avg_accuracy_subsampling,axis = 0)/np.sqrt(len(filenames))
ind1 = np.arange(len(num_sub_units))
plt.errorbar(0,np.nanmean(avg_accuracy_all_units), yerr = np.nanstd(avg_accuracy_all_units)/np.sqrt(len(filenames)), fmt = '-o', color = 'tab:orange', ecolor = 'k', label = 'Decoding with all units')
plt.errorbar(ind1+1, avg_accuracy_per_number_subsampled, yerr = sem_accuracy_per_number_subsampled, fmt = '-o', color = 'b', ecolor = 'tab:orange', label = 'Decoding with subsampled units')
plt.plot(0,np.nanmean(avg_accuracy_shuffle_all_units),'tab:gray', label = 'Shuffled performance (all units)')
#plt.ylim((0.5,1))
xticklabels1 = ['%i units' % (d) for d in num_sub_units]
xticklabels = ['All units'] + xticklabels1
xticks = np.arange(len(num_sub_units)+1)
plt.xticks(xticks, xticklabels)
plt.ylabel('Decoding Accuracy')
plt.xlabel('Units Used for Decoding')
plt.title('Avg Decoding Accuracy for Bin [-1,1] s around zone entry')
plt.legend()
plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/Decoding_RH_reaches_averaged_over_sessions_decoding_withsubsampling_.svg")
plt.close()

# Compute and plot decoding performance for subsampling units

cmap = mpl.cm.brg
avg_accuracy_subsampling = np.nanmean(avg_accuracy_subunits,axis=2)
sem_accuracy_subsampling = np.nanstd(avg_accuracy_subunits,axis=2)/np.sqrt(10)
width = 1./(len(num_sub_units) + 2)
plt.errorbar(ind, avg_accuracy_all_units, color = 'tab:orange', yerr = sem_accuracy_all_units, label = 'All units')
for n,num_su in enumerate(num_sub_units):
	plt.errorbar(ind, avg_accuracy_subsampling[:,n], color=cmap(n/float(len(num_sub_units))), yerr = sem_accuracy_subsampling[:,n], label = '%i units' % (num_su))
plt.errorbar(ind, avg_accuracy_shuffle_all_units, exitcolor = 'c', yerr = sem_accuracy_shuffle_all_units, label = 'Shuffled data -\n (all units)')
plt.ylim((0,1))
xticklabels = ['Session %i' % (d+1) for d in np.arange(len(filenames))]
xticks = ind + width/2
plt.xticks(xticks, xticklabels)
plt.ylabel('Decoding Accuracy')
plt.title('Avg Decoding Accuracy for Bin [-1,1] s around zone entry')
plt.legend()
plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/Decoding_RH_reaches_all_sessions_average_decoding_withsubsampling.svg")
plt.close()