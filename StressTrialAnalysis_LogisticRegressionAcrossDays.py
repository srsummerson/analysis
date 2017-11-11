import numpy as np
from StressTrialAnalysis_FeatureExtraction import SelectionAndClassification_PowerAndPhysFeatures
import matplotlib.pyplot as plt
from basicAnalysis import ElectrodeGridMat

## Initialize variables
data_dir = 'C:/Users/Samantha Summerson/Dropbox/Carmena Lab/Mario/Mood Bias Task/Data'
hdf_location_list = ['mari20160418_04_te2002.hdf', 'mari20160517_07_te2097.hdf', 'mari20160613_02_te2226.hdf', \
					'mari20160614_03_te2237.hdf', 'mari20160707_02_te2314.hdf', 'mari20160711_02_te2327.hdf', \
					'mari20160712_03_te2333.hdf', 'mari20160714_03_te2351.hdf', 'mari20160715_04_te2357.hdf', \
					'mari20160716_03_te2361.hdf', 'mari20161005_03_te2542.hdf', 'mari20161006_03_te2551.hdf', \
					'mari20161010_03_te2571.hdf', 'mari20161012_03_te2591.hdf', 'mari20161013_03_te2598.hdf', \
					'mari20161026_04_te2634.hdf']
hdf_location_list = ['mari20160418_04_te2002.hdf', 'mari20160517_07_te2097.hdf', 'mari20160613_02_te2226.hdf', \
					'mari20160614_03_te2237.hdf', 'mari20160707_02_te2314.hdf',  \
					'mari20160712_03_te2333.hdf',  \
					'mari20161005_03_te2542.hdf',  \
					'mari20161026_04_te2634.hdf']
hdf_location_list = [data_dir + '/' + item for item in hdf_location_list]

phys_filename_list = ['Mario20160418_b1_PhysFeatures.mat', 'Mario20160517_b1_PhysFeatures.mat', 'Mario20160613_b1_PhysFeatures.mat', \
					'Mario20160614_b1_PhysFeatures.mat', 'Mario20160707_b1_PhysFeatures.mat', 'Mario20160711_b1_PhysFeatures.mat', \
					'Mario20160712_b1_PhysFeatures.mat', 'Mario20160714_b1_PhysFeatures.mat', 'Mario20160715_b2_PhysFeatures.mat', \
					'Mario20160716_b1_PhysFeatures.mat', 'Mario20161005_b1_PhysFeatures.mat', 'Mario20161006_b1_PhysFeatures.mat', \
					'Mario20161010_b1_PhysFeatures.mat', 'Mario20161012_b1_PhysFeatures.mat', 'Mario20161013_b1_PhysFeatures.mat', \
					'Mario20161026_b1_PhysFeatures.mat']
phys_filename_list = ['Mario20160418_b1_PhysFeatures.mat', 'Mario20160517_b1_PhysFeatures.mat', 'Mario20160613_b1_PhysFeatures.mat', \
					'Mario20160614_b1_PhysFeatures.mat', 'Mario20160707_b1_PhysFeatures.mat',  \
					'Mario20160712_b1_PhysFeatures.mat',  \
					'Mario20161005_b1_PhysFeatures.mat',  \
					'Mario20161026_b1_PhysFeatures.mat']
phys_filename_list = [data_dir + '/' + item for item in phys_filename_list]

power_filename_list = ['Mario20160418_b1_PowerFeatures.mat', 'Mario20160517_b1_PowerFeatures.mat', 'Mario20160613_b1_PowerFeatures.mat', \
					'Mario20160614_b1_PowerFeatures.mat', 'Mario20160707_b1_PowerFeatures.mat', 'Mario20160711_b1_PowerFeatures.mat', \
					'Mario20160712_b1_PowerFeatures.mat', 'Mario20160714_b1_PowerFeatures.mat', 'Mario20160715_b2_PowerFeatures.mat', \
					'Mario20160716_b1_PowerFeatures.mat', 'Mario20161005_b1_PowerFeatures.mat', 'Mario20161006_b1_PowerFeatures.mat', \
					'Mario20161010_b1_PowerFeatures.mat', 'Mario20161012_b1_PowerFeatures.mat', 'Mario20161013_b1_PowerFeatures.mat', \
					'Mario20161026_b1_PowerFeatures.mat']
power_filename_list = ['Mario20160418_b1_PowerFeatures.mat', 'Mario20160517_b1_PowerFeatures.mat', 'Mario20160613_b1_PowerFeatures.mat', \
					'Mario20160614_b1_PowerFeatures.mat', 'Mario20160707_b1_PowerFeatures.mat',  \
					'Mario20160712_b1_PowerFeatures.mat',  \
					'Mario20161005_b1_PowerFeatures.mat',  \
					'Mario20161026_b1_PowerFeatures.mat']
power_filename_list = [data_dir + '/' + item for item in power_filename_list]

days = np.arange(len(hdf_location_list))
num_days = len(days)
num_conditions = 12
num_chan = 157
num_features = num_conditions*num_chan

feat_counter = np.zeros(num_features)
avg_coef = np.zeros(num_features)
accuracy = np.zeros(num_days)

all_channels = np.arange(0,161,dtype = int)					# channels 1 - 160
lfp_channels = np.delete(all_channels, [0, 1, 129, 131, 145])		# channels 129, 131, and 145 are open
power_labels = np.chararray([num_chan, num_conditions], itemsize = 13)
for i, chan in enumerate(lfp_channels):
	power_labels[i,0] = 'chan'+str(chan)+'_T1_B1'
	power_labels[i,1] = 'chan'+str(chan)+'_T1_B2'
	power_labels[i,2] = 'chan'+str(chan)+'_T1_B3'
	power_labels[i,3] = 'chan'+str(chan)+'_T1_B4'
	power_labels[i,4] = 'chan'+str(chan)+'_T2_B1'
	power_labels[i,5] = 'chan'+str(chan)+'_T2_B2'
	power_labels[i,6] = 'chan'+str(chan)+'_T2_B3'
	power_labels[i,7] = 'chan'+str(chan)+'_T2_B4'
	power_labels[i,8] = 'chan'+str(chan)+'_T3_B1'
	power_labels[i,9] = 'chan'+str(chan)+'_T3_B2'
	power_labels[i,10] = 'chan'+str(chan)+'_T3_B3'
	power_labels[i,11] = 'chan'+str(chan)+'_T3_B4'
power_labels = power_labels.flatten()

counter = 0
## Iterate across days
for j in days:
	output = SelectionAndClassification_PowerAndPhysFeatures(hdf_location_list[j], phys_filename_list[j], power_filename_list[j], var_threshold = 10**-10, plot_output = False)
	chosen_power_feat = output[0]
	coef = output[2]
	accuracy[counter] = output[-1]
	counter += 1

	print coef.shape
	
	feat_counter[chosen_power_feat] += 1
	avg_coef += coef
	
avg_coef = avg_coef/num_days				# average coefficient value for a given feature
feat_counter = feat_counter/num_days		# fraction of days feature was significant
avg_accuracy = np.nanmean(accuracy)			# accuracy of logistic regression across days

# pull out coefficients by channel for each condition and plot
# Set up matrix for plotting average coefficients for power features
dx, dy = 1, 1
y, x = np.mgrid[slice(0,15,dy), slice(0,14,dx)]

min_coef = np.nanmin(avg_coef)
max_coef = np.nanmax(avg_coef)
plt.figure()
for cond in range(num_conditions):
	power_array = np.zeros(len(all_channels))     # dummy array where entries corresponding to real channels will be updated
	for i, chan in enumerate(lfp_channels):
		power_array[chan] = avg_coef[i*num_conditions + cond]
	power_layout = ElectrodeGridMat(power_array)
	plt.subplot(num_conditions/3, num_conditions/3,cond+1)
	#plt.pcolormesh(x,y,power_layout,vmin=10**-7, vmax = 10**-4)  # do log just to pull out smaller differences better
	plt.pcolormesh(x,y,power_layout, vmin=min_coef, vmax = max_coef)  # do log just to pull out smaller differences better
	plt.title('condition %i' % (cond + 1))
	#plt.colorbar()
outline_array = np.ones(len(all_channels)) 		# dummy array for orienting values
outline_layout = ElectrodeGridMat(outline_array)
plt.subplot(num_conditions/3, num_conditions/3,cond + 2)
plt.pcolormesh(x,y,outline_layout,vmin=min_coef, vmax = max_coef)  # do log just to pull out smaller differences better
plt.colorbar()
plt.title('Average Regression Coefficients for Classification with Power Features')
plt.show()

min_feat = np.nanmin(feat_counter)
max_feat = np.nanmax(feat_counter)
plt.figure()
for cond in range(num_conditions):
	power_array = np.zeros(len(all_channels))     # dummy array where entries corresponding to real channels will be updated
	for i, chan in enumerate(lfp_channels):
		power_array[chan] = feat_counter[i*num_conditions + cond]
	power_layout = ElectrodeGridMat(power_array)
	plt.subplot(num_conditions/3, num_conditions/3,cond+1)
	#plt.pcolormesh(x,y,power_layout,vmin=10**-7, vmax = 10**-4)  # do log just to pull out smaller differences better
	plt.pcolormesh(x,y,power_layout, vmin=0, vmax = 1)  # do log just to pull out smaller differences better
	plt.title('condition %i' % (cond + 1))
	#plt.colorbar()
outline_array = np.ones(len(all_channels)) 		# dummy array for orienting values
outline_layout = ElectrodeGridMat(outline_array)
plt.subplot(num_conditions/3, num_conditions/3,cond + 2)
plt.pcolormesh(x,y,outline_layout,vmin=min_feat, vmax = max_feat)  # do log just to pull out smaller differences better
plt.colorbar()
plt.title('Fraction of Days Power Feature Regression Coefficients Are Significant')
plt.show()
