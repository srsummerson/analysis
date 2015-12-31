from neo import io
from scipy import stats
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def PopulationResponse(filename,*args):

	sma = [39, 55, 34, 50, 91, 77, 93, 79, 74, 90, 76, 92, 69, 85, 71, 87]
	presma = [43, 59, 45, 61, 42, 58, 44, 60]
	m1 = [95, 78, 66, 82, 68, 84, 70, 86, 72, 88, 109, 125, 111, 127, 65, 81, 67, 83, 
		120, 106, 122, 108, 124, 110, 126, 112, 128, 105, 121, 107, 123, 115, 101, 117,
		103, 119, 98, 114, 100, 116, 102, 118, 104, 160, 137, 153, 139, 155, 141, 157, 
		143, 159, 97, 113, 99, 134, 150, 136, 152, 138, 154, 140, 156, 142, 158, 144, 147,
		133, 149, 135, 151, 130, 146, 132, 148]
	pmd = [34, 47, 63, 44, 46, 62, 48, 64, 41, 57, 54, 36, 52, 38, 54, 40, 56, 33, 49, 35, 
		51, 37, 53, 94, 80, 96, 73, 89, 75]

	r = io.TdtIO(filename)
	bl = r.read_block(lazy=False,cascade=True)
	rate_data = dict()
	block_num = []
	"""
	if args is not None:
		block_num.append(*args -1)
		print block_num
	else:
		block_num = range(0,len(bl.segments))
	"""
	block_num = range(0,len(bl.segments))

	for block in block_num:
		spiketrains = bl.segments[block].spiketrains

		# get sample hpf signal to find stimulation times
		for sig in bl.segments[block].analogsignals:
			if (sig.name == 'pNe1 33'):
				hpf_sample = sig
				hpf_samplingrate = np.floor(sig.sampling_rate).item()	# round to lowest integer

		# get first 10 s of recording
		hpf_snippet = hpf_sample[0:10*hpf_samplingrate]
		hpf_mean = np.mean(hpf_snippet)
		hpf_std = np.std(hpf_snippet)
		stim_thres = hpf_mean + 3*hpf_std

		# find stim times
		sample = 0
		stim_start_times = []
		while sample < hpf_sample.size:
			if (hpf_sample[sample] > stim_thres):
				stim_start_times.append(sample)
				sample += 11*hpf_samplingrate # advance 11 s = 1 s stim + 10 s minimum pre-train delay
			else:
				sample += 1
		#num_epochs = len(stim_start_times)
		num_epochs = 30
		if len(stim_start_times) > 20:   # only update stim times if vector was repopulated, otherwise use old stim_times
			stim_times = stim_start_times[0:num_epochs]

		channel = 0
		second_channel_bank = 0
		bin_size = .1  # .1 s = 100 ms
		prestim_time = 5 
		poststim_time = 10
		stim_time = 1
		total_time = prestim_time + stim_time + poststim_time
		#num_bins = 10/bin_size
		num_bins = total_time/bin_size
		for train in spiketrains:
			epoch_rates = np.zeros([num_epochs,num_bins])
			if train.name[4:6] < channel:
				second_channel_bank = 96
			channel = float(train.name[4:6]) + second_channel_bank
			if (train.name[-5:]!='Code0')&(train.name[-6:0]!='Code31'):
				code = train.name[-1]
				train_name = str(channel) +'_'+str(code)
				epoch_counter = 0
				for train_start in stim_times:
					epoch_start = float(train_start)/hpf_sample.sampling_rate.item() # get stim train start time in seconds
					#epoch_start += 1  # add 1 second to account for duration of stimulation
					epoch_start = epoch_start - prestim_time   # epoch to include 5 s pre-stim data
					#epoch_end = epoch_start + 10 	# epoch to look in
					epoch_end = epoch_start + total_time  # epoch is 5 s pre-stim + 1 s stim + 10 s post-stim
					epoch_bins = np.arange(epoch_start,epoch_end+bin_size/2,bin_size) 
					counts, bins = np.histogram(train,epoch_bins)
					epoch_rates[epoch_counter][:] = counts/bin_size	# collect all rates into a N-dim array
					epoch_counter += 1
				rate_data[train_name] = epoch_rates
				background_epoch = np.concatenate((np.arange(0,int(prestim_time/bin_size)), np.arange(int((prestim_time+stim_time)/bin_size),len(epoch_bins)-1)), axis=0)
		# add up population responses,z-score and find significance
		
		population_sma = np.zeros([num_epochs,num_bins])
		population_presma = np.zeros([num_epochs,num_bins])
		population_m1 = np.zeros([num_epochs,num_bins])
		population_pmd = np.zeros([num_epochs,num_bins])

		sig_population_sma = []
		sig_population_presma = []
		sig_population_m1 = []
		sig_population_pmd = []

		average_zscored_sma = np.zeros(num_bins)
		average_zscored_presma = np.zeros(num_bins)
		average_zscored_m1 = np.zeros(num_bins)
		average_zscored_pmd = np.zeros(num_bins)

		std_zscored_sma = []
		std_zscored_presma = []
		std_zscored_m1 = []
		std_zscored_pmd = []

		n_sma = 0
		n_presma = 0
		n_m1 = 0
		n_pmd = 0

		for rates in rate_data:
			channel_num = float(rates[:-2])
			if (channel_num in sma):
				population_sma = population_sma+rate_data[rates]
				n_sma += 1
			if (channel_num in presma):
				population_presma = population_presma+rate_data[rates]
				n_presma += 1
			if (channel_num in m1):
				population_m1 = population_m1+rate_data[rates]
				n_m1 += 1
			if (channel_num in pmd):
				population_pmd = population_pmd+rate_data[rates]
				n_pmd += 1

		#z score data per epoch by baseline population firing rate
		for epoch in range(0,num_epochs):
			std_sma = np.std(population_sma[epoch][background_epoch])
			if (std_sma > 0):
				population_sma[epoch][:] = (population_sma[epoch][:] - np.mean(population_sma[epoch][background_epoch]))/std_sma
			else:
				population_sma[epoch][:] = population_sma[epoch][:] - np.mean(population_sma[epoch][background_epoch])
			average_zscored_sma += population_sma[epoch][:]
			
			std_presma = np.std(population_presma[epoch][background_epoch])
			if (std_presma > 0):
				population_presma[epoch][:] = (population_presma[epoch][:] - np.mean(population_presma[epoch][background_epoch]))/std_presma
			else:
				population_presma[epoch][:] = population_presma[epoch][:] - np.mean(population_presma[epoch][background_epoch])
			average_zscored_presma += population_presma[epoch][:]
			
			std_m1 = np.std(population_m1[epoch][background_epoch])
			if (std_m1 > 0):
				population_m1[epoch][:] = (population_m1[epoch][:] - np.mean(population_m1[epoch][background_epoch]))/std_m1
			else:
				population_m1[epoch][:] = population_m1[epoch][:] - np.mean(population_m1[epoch][background_epoch])
			average_zscored_m1 += population_m1[epoch][:]
			
			std_pmd = np.std(population_pmd[epoch][background_epoch])
			if (std_pmd > 0):
				population_pmd[epoch][:] = (population_pmd[epoch][:] - np.mean(population_pmd[epoch][background_epoch]))/std_pmd
			else:
				population_pmd[epoch][:] = population_pmd[epoch][:] - np.mean(population_pmd[epoch][background_epoch])
			average_zscored_pmd += population_pmd[epoch][:]


		average_zscored_sma = average_zscored_sma/float(num_epochs)
		average_zscored_presma = average_zscored_presma/float(num_epochs)
		average_zscored_m1 = average_zscored_m1/float(num_epochs)
		average_zscored_pmd = average_zscored_pmd/float(num_epochs)

		for bin in range(0,int(total_time/bin_size)):
			t, prob = sp.stats.ttest_1samp(population_sma[:,bin],0.0)
			sig_population_sma.append(prob)
			std_zscored_sma.append(stats.sem(population_sma[:,bin]))
			t, prob = sp.stats.ttest_1samp(population_presma[:,bin],0.0)
			sig_population_presma.append(prob)
			std_zscored_presma.append(stats.sem(population_presma[:,bin]))
			t, prob = sp.stats.ttest_1samp(population_m1[:,bin],0.0)
			sig_population_m1.append(prob)
			std_zscored_m1.append(stats.sem(population_m1[:,bin]))
			t, prob = sp.stats.ttest_1samp(population_pmd[:,bin],0.0)
			sig_population_pmd.append(prob)
			std_zscored_pmd.append(stats.sem(population_pmd[:,bin]))

		sig_population_presma = (sig_population_presma > 0.05*np.ones(len(sig_population_presma)))
		sig_population_sma = (sig_population_sma > 0.05*np.ones(len(sig_population_sma)))
		sig_population_m1 = (sig_population_m1 > 0.05*np.ones(len(sig_population_m1)))
		sig_population_pmd = (sig_population_pmd > 0.05*np.ones(len(sig_population_pmd)))

		time = np.arange(0,total_time,bin_size) - prestim_time
		plt.figure()
		plt.subplot(2,2,1)
		plt.plot(time,average_zscored_presma,'b')
		plt.fill_between(time,average_zscored_presma-std_zscored_presma,average_zscored_presma+std_zscored_presma,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time,max(average_zscored_presma[background_epoch])*sig_population_presma,'xr')
		plt.title('Pre-SMA: n = %i' % (n_presma))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,2)
		plt.plot(time,average_zscored_sma,'b')
		plt.fill_between(time,average_zscored_sma-std_zscored_sma,average_zscored_sma+std_zscored_sma,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time,max(average_zscored_sma)*sig_population_sma,'xr')
		plt.title('SMA: n = %i' % (n_sma))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,3)
		plt.plot(time,average_zscored_m1,'b')
		plt.fill_between(time,average_zscored_m1-std_zscored_m1,average_zscored_m1+std_zscored_m1,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time,max(average_zscored_m1)*sig_population_m1,'xr')
		plt.title('M1: n = %i' % (n_m1))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,4)
		plt.plot(time,average_zscored_pmd,'b')
		plt.fill_between(time,average_zscored_pmd-std_zscored_pmd,average_zscored_pmd+std_zscored_pmd,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time,max(average_zscored_pmd)*sig_population_pmd,'xr')
		plt.title('PMd: n = %i' % (n_pmd))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_PopulationResponse.svg')
		plt.close()

	return 







