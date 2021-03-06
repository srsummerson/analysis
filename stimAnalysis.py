from neo import io
from scipy import stats
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def PopulationResponse(filename,*args):
	'''
	sma = [39, 55, 34, 50, 91, 77, 93, 79, 74, 90, 76, 92, 69, 85, 71, 87]
	presma = [43, 59, 45, 61, 42, 58, 44, 60]
	m1 = [95, 78, 66, 82, 68, 84, 70, 86, 72, 88, 109, 125, 111, 127, 65, 81, 67, 83, 
		120, 106, 122, 108, 124, 110, 126, 112, 128, 105, 121, 107, 123, 115, 101, 117,
		103, 119, 98, 114, 100, 116, 102, 118, 104, 160, 137, 153, 139, 155, 141, 157, 
		143, 159, 97, 113, 99, 134, 150, 136, 152, 138, 154, 140, 156, 142, 158, 144, 147,
		133, 149, 135, 151, 130, 146, 132, 148]
	pmd = [47, 63, 46, 62, 48, 64, 41, 57, 54, 36, 52, 38, 54, 40, 56, 33, 49, 35, 
		51, 37, 53, 94, 80, 96, 73, 89, 75]
	
	# Dec 28
	sma = ['34_2','34_3','90_2','76_1']
	m1 = ['83_2','66_1','83_1','95_1','97_1','100_1','104_1','106_1','112_1','113_2',
		'119_1','126_1','132_2','133_1','138_1','149_2','160_1','149_1']
	presma = []
	pmd = []
	
	# Dec 29
	sma = ['34_2','34_3','69_1','76_1','90_2']
	m1 = ['66_1','81_1','83_1','83_2','95_1','97_1','100_1','102_1','104_1','106_1',
		'112_1','126_1','132_2','133_1','149_2']
	presma = []
	pmd = []
	
	# Dec 30
	sma = ['34_2','34_3','76_1','90_2']
	m1 = ['66_1','83_2','100_2','102_1','105_2','106_2','107_1','107_2','107_3','107_4',
		'111_2','115_2','122_2','125_1','137_2','154_2']
	presma = []
	pmd = []
	'''
	# Jan 1
	sma = ['34_2','90_2']
	pmd = ['40_2']
	m1 = ['66_1','83_2','84_2','104_2','110_2','111_2','149_2']
	presma = []
	

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
		stim_thres = hpf_mean + 100*hpf_std

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
			if float(train.name[4:6]) < channel:
				second_channel_bank = 96
			channel = int(train.name[4:6]) + second_channel_bank
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
			#channel_num = float(rates[:-2])
			channel_num = rates
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

		sig_population_presma = (sig_population_presma < 0.05*np.ones(len(sig_population_presma)))
		sig_population_presma_ind = np.nonzero(sig_population_presma)
		sig_population_sma = (sig_population_sma < 0.05*np.ones(len(sig_population_sma)))
		sig_population_sma_ind = np.nonzero(sig_population_sma)
		sig_population_m1 = (sig_population_m1 < 0.05*np.ones(len(sig_population_m1)))
		sig_population_m1_ind = np.nonzero(sig_population_m1)
		sig_population_pmd = (sig_population_pmd < 0.05*np.ones(len(sig_population_pmd)))
		sig_population_pmd_ind = np.nonzero(sig_population_pmd)

		time = np.arange(0,total_time,bin_size) - prestim_time
		plt.figure()
		plt.subplot(2,2,1)
		plt.plot(time,average_zscored_presma,'b')
		plt.fill_between(time,average_zscored_presma-std_zscored_presma,average_zscored_presma+std_zscored_presma,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_presma_ind],sig_population_presma[sig_population_presma_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('Pre-SMA: n = %i' % (n_presma))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,2)
		plt.plot(time,average_zscored_sma,'b')
		plt.fill_between(time,average_zscored_sma-std_zscored_sma,average_zscored_sma+std_zscored_sma,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_sma_ind],sig_population_sma[sig_population_sma_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('SMA: n = %i' % (n_sma))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,3)
		plt.plot(time,average_zscored_m1,'b')
		plt.fill_between(time,average_zscored_m1-std_zscored_m1,average_zscored_m1+std_zscored_m1,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_m1_ind],sig_population_m1[sig_population_m1_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('M1: n = %i' % (n_m1))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,4)
		plt.plot(time,average_zscored_pmd,'b')
		plt.fill_between(time,average_zscored_pmd-std_zscored_pmd,average_zscored_pmd+std_zscored_pmd,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_pmd_ind],sig_population_pmd[sig_population_pmd_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('PMd: n = %i' % (n_pmd))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_PopulationResponse.svg')
		plt.close()

	return 


def PopulationResponseSingleUnit(filename,*args):
	# 34 is in both
	sma = [39, 55, 34, 50, 91, 77, 93, 79, 74, 90, 76, 92, 69, 85, 71, 87]
	presma = [43, 59, 45, 61, 42, 58, 44, 60]
	m1 = [95, 78, 66, 82, 68, 84, 70, 86, 72, 88, 109, 125, 111, 127, 65, 81, 67, 83, 
		120, 106, 122, 108, 124, 110, 126, 112, 128, 105, 121, 107, 123, 115, 101, 117,
		103, 119, 98, 114, 100, 116, 102, 118, 104, 160, 137, 153, 139, 155, 141, 157, 
		143, 159, 97, 113, 99, 134, 150, 136, 152, 138, 154, 140, 156, 142, 158, 144, 147,
		133, 149, 135, 151, 130, 146, 132, 148]
	pmd = [47, 63, 46, 62, 48, 64, 41, 57, 54, 36, 52, 38, 54, 40, 56, 33, 49, 35, 
		51, 37, 53, 94, 80, 96, 73, 89, 75]

	r = io.TdtIO(filename)
	bl = r.read_block(lazy=False,cascade=True)
	rate_data_sma = dict()
	rate_data_presma = dict()
	rate_data_m1 = dict()
	rate_data_pmd = dict()
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
		count_sma = 0
		count_presma = 0
		count_m1 = 0
		count_pmd = 0
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
			if float(train.name[4:6]) < channel:
				second_channel_bank = 96
			channel = int(train.name[4:6]) + second_channel_bank
			if (train.name[-5:]!='Code0')&(train.name[-6:0]!='Code31'):
				code = train.name[-1]
				train_name = str(channel) +'_'+str(code)
				epoch_counter = 0
				averages_zscored = np.zeros(num_bins)
				sig_per_bin = []
				sig_per_bin_ind = []
				std_scored = []
				stim_waveform_indices = []
				train_times = [time.item() for time in train]
				for train_start in stim_times:
					epoch_start = float(train_start)/hpf_sample.sampling_rate.item() # get stim train start time in seconds
					# find train spike times and indices that occur during stim period
					find_stim_indices = np.greater_equal(train_times,epoch_start)&np.less_equal(train_times,(epoch_start + stim_time))
					stim_waveform_indices.append(np.nonzero(find_stim_indices))
					#epoch_start += 1  # add 1 second to account for duration of stimulation
					epoch_start = epoch_start - prestim_time   # epoch to include 5 s pre-stim data
					#epoch_end = epoch_start + 10 	# epoch to look in
					epoch_end = epoch_start + total_time  # epoch is 5 s pre-stim + 1 s stim + 10 s post-stim
					epoch_bins = np.arange(epoch_start,epoch_end+bin_size/2,bin_size)
					background_epoch = np.concatenate((np.arange(0,int(prestim_time/bin_size)), np.arange(int((prestim_time+stim_time)/bin_size),len(epoch_bins)-1)), axis=0) 
					counts, bins = np.histogram(train,epoch_bins)
					epoch_rates[epoch_counter][:] = counts/bin_size	# collect all rates into a N-dim array
					epoch_counter += 1
				# find train spike indices that occur not during the stim period
				waveform_indices = [ind for ind in range(0,len(train)) if ind not in stim_waveform_indices]
				avg_waveform = np.mean(train.waveforms[waveform_indices],axis=0).ravel()
				#avg_waveform = [val.item() for val in avg_waveform.ravel()]
				sem_waveform = np.std(train.waveforms[waveform_indices],axis=0).ravel()
				#sem_waveform = [val.item() for val in sem_waveform.ravel()]
				# z score data per epoch and then average over epochs
				for epoch in range(0,num_epochs):
					std_train = np.std(epoch_rates[epoch][background_epoch])
					if (std_train > 0):
						epoch_rates[epoch][:] = (epoch_rates[epoch][:] - np.mean(epoch_rates[epoch][background_epoch]))/std_train
					else:
						epoch_rates[epoch][:] = epoch_rates[epoch][:] - np.mean(epoch_rates[epoch][background_epoch])
					averages_zscored += epoch_rates[epoch][:]
				averages_zscored = averages_zscored/float(num_epochs)

				for bin in range(0,int(total_time/bin_size)):
					t, prob = sp.stats.ttest_1samp(epoch_rates[:,bin],0.0)
					sig_per_bin.append(prob)
					std_scored.append(stats.sem(epoch_rates[:,bin]))
				sig_per_bin = (sig_per_bin < 0.05*np.ones(len(sig_per_bin)))
				sig_per_bin_ind = np.nonzero(sig_per_bin)
				# save/plot associated with the nucleus the channel is in 
				time = np.arange(0,total_time,bin_size) - prestim_time + bin_size
				if (channel in sma):
					#rate_data_sma[train_name] = averages_zscored
					#sig_sma[train_name] = sig_per_bin_ind
					#std_scored_sma[train_name] = std_scored
					plt.figure()
					plt.subplot(1,2,1)
					plt.plot(time,averages_zscored,'b')
					plt.fill_between(time,averages_zscored-std_scored,averages_zscored+std_scored,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.plot(time[sig_per_bin_ind],sig_per_bin[sig_per_bin_ind],'xr')
					plt.plot(time,np.zeros(time.size),'k--')
					plt.title('SMA: Ch %i - Unit %i' % (int(channel),int(code)))
					plt.xlabel('Time (s)')
					plt.ylabel('Spike Rate Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
					plt.ylim((-1,2))
					plt.subplot(1,2,2)
					plt.plot(range(0,len(avg_waveform)),avg_waveform)
					plt.fill_between(range(0,len(avg_waveform)),avg_waveform-sem_waveform,avg_waveform+sem_waveform,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.xlabel('Samples')
					plt.ylabel('Magnitude (V)')
					plt.tight_layout()
					plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_'+ train_name +'_SMA_Single_Unit_Response.png')
					plt.close()
					count_sma += 1
				if (channel in presma):
					#rate_data_presma[train_name] = averages_zscored
					#sig_presma[train_name] = sig_per_bin_ind
					#std_scored_presma[train_name] = std_scored
					plt.figure()
					plt.subplot(1,2,1)
					plt.plot(time,averages_zscored,'b')
					plt.fill_between(time,averages_zscored-std_scored,averages_zscored+std_scored,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.plot(time[sig_per_bin_ind],sig_per_bin[sig_per_bin_ind],'xr')
					plt.plot(time,np.zeros(time.size),'k--')
					plt.title('Pre-SMA: Ch %i - Unit %i' % (int(channel),int(code)))
					plt.xlabel('Time (s)')
					plt.ylabel('Spike Rate Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
					plt.ylim((-1,2))
					plt.subplot(1,2,2)
					plt.plot(range(0,len(avg_waveform)),avg_waveform)
					plt.fill_between(range(0,len(avg_waveform)),avg_waveform-sem_waveform,avg_waveform+sem_waveform,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.xlabel('Samples')
					plt.ylabel('Magnitude (V)')
					plt.tight_layout()
					plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_'+ train_name +'_preSMA_Single_Unit_Response.png')
					plt.close()
					count_presma += 1
				if (channel in m1):
					#rate_data_m1[train_name] = averages_zscored
					#sig_m1[train_name] = sig_per_bin_ind
					#std_scored_m1[train_name] = std_scored
					plt.figure()
					plt.subplot(1,2,1)
					plt.plot(time,averages_zscored,'b')
					plt.fill_between(time,averages_zscored-std_scored,averages_zscored+std_scored,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.plot(time[sig_per_bin_ind],sig_per_bin[sig_per_bin_ind],'xr')
					plt.plot(time,np.zeros(time.size),'k--')
					plt.title('M1: Ch %i - Unit %i' % (int(channel),int(code)))
					plt.xlabel('Time (s)')
					plt.ylabel('Spike Rate Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
					plt.ylim((-1,2))
					plt.subplot(1,2,2)
					plt.plot(range(0,len(avg_waveform)),avg_waveform)
					plt.fill_between(range(0,len(avg_waveform)),avg_waveform-sem_waveform,avg_waveform+sem_waveform,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.xlabel('Samples')
					plt.ylabel('Magnitude (V)')
					plt.tight_layout()
					plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_'+ train_name +'_M1_Single_Unit_Response.png')
					plt.close()
					count_m1 += 1
				if (channel in pmd):
					#rate_data_pmd[train_name] = averages_zscored
					#sig_pmd[train_name] = sig_per_bin_ind
					#std_scored_pmd[train_name] = std_scored
					plt.figure()
					plt.subplot(1,2,1)
					plt.plot(time,averages_zscored,'b')
					plt.fill_between(time,averages_zscored-std_scored,averages_zscored+std_scored,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.plot(time[sig_per_bin_ind],sig_per_bin[sig_per_bin_ind],'xr')
					plt.plot(time,np.zeros(time.size),'k--')
					plt.title('PMd: Ch %i - Unit %i' % (int(channel),int(code)))
					plt.xlabel('Time (s)')
					plt.ylabel('Spike Rate Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
					plt.ylim((-1,2))
					plt.subplot(1,2,2)
					plt.plot(range(0,len(avg_waveform)),avg_waveform)
					plt.fill_between(range(0,len(avg_waveform)),avg_waveform-sem_waveform,avg_waveform+sem_waveform,facecolor='gray',alpha=0.5,linewidth=0.5)
					plt.xlabel('Samples')
					plt.ylabel('Magnitude (V)')
					plt.tight_layout()
					plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_'+ train_name +'_PMd_Single_Unit_Response.png')
					plt.close()
					count_pmd += 1

		"""				
		plt.figure(1)
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_SMA_Single_Unit_Response.svg')
		plt.close()
		plt.figure(2)
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_preSMA_Single_Unit_Response.svg')
		plt.close()
		plt.figure(3)
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_M1_Single_Unit_Response.svg')
		plt.close()
		plt.figure(4)
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block+1)+'_PMd_Single_Unit_Response.svg')
		plt.close()
		"""
	return 

def PopulationResponseSingleBlock(filename,tdt_neo,block,stim_thres,train_length,generate_figs):
	"""
	Inputs: filename (string) is the parent folder for all blocks, block is the block number (starting at 1), stim thres is the threshold value in uV for detecting when stimulation
	was administered, and train_length is the value in s for the stimulation pulse train length.

	12/28: B1 - 100, B2 - 50, B3 - 100, B4 - 100, B5 - 100, B6 - 100
	12/29: B1 - 2000, B2 - 1000, B3 - 2000, B4 - 1000, B5 - 1000, B6 - sham, B7 - 1000
	12/30: B1 - 1000, B2 - 500, B3 - 1000, B4 - 1000, B5 - 1000, B6 - sham, B7 - 1000
	12/31: B1 - 1000, B2 - 1000, B3 - 1000
	1/1: B1 - 1000, B2 - 1000, B3 - 1000, B4 - 1000, B5 - 1000
	1/6: B1 - 200, B2 - sham, B3 - 150, B4 - 100, B5 - can't tell (train too short)
	1/8: B1 - 500, B2 - 500, B3 - 200
	1/12: can't read with neo
	"""

	
	sma = [39, 55, 34, 50, 91, 77, 93, 79, 74, 90, 76, 92, 69, 85, 71, 87]
	presma = [43, 59, 45, 61, 42, 58, 44, 60]
	m1 = [95, 78, 66, 82, 68, 84, 70, 86, 72, 88, 109, 125, 111, 127, 65, 81, 67, 83, 
		120, 106, 122, 108, 124, 110, 126, 112, 128, 105, 121, 107, 123, 115, 101, 117,
		103, 119, 98, 114, 100, 116, 102, 118, 104, 160, 137, 153, 139, 155, 141, 157, 
		143, 159, 97, 113, 99, 134, 150, 136, 152, 138, 154, 140, 156, 142, 158, 144, 147,
		133, 149, 135, 151, 130, 146, 132, 148]
	pmd = [47, 63, 46, 62, 48, 64, 41, 57, 54, 36, 52, 38, 54, 40, 56, 33, 49, 35, 
		51, 37, 53, 94, 80, 96, 73, 89, 75]
	'''
	# Dec 28
	sma = ['34_2','34_3','90_2','76_1']
	m1 = ['83_2','66_1','83_1','95_1','97_1','100_1','104_1','106_1','112_1','113_2',
		'119_1','126_1','132_2','133_1','138_1','149_2','160_1','149_1']
	presma = []
	pmd = []
	
	# Dec 29
	sma = ['34_2','34_3','69_1','76_1','90_2']
	m1 = ['66_1','81_1','83_1','83_2','95_1','97_1','100_1','102_1','104_1','106_1',
		'112_1','126_1','132_2','133_1','149_2']
	presma = []
	pmd = []
	
	# Dec 30
	sma = ['34_2','34_3','76_1','90_2']
	m1 = ['66_1','83_2','100_2','102_1','105_2','106_2','107_1','107_2','107_3','107_4',
		'111_2','115_2','122_2','125_1','137_2','154_2']
	presma = []
	pmd = []
	
	# Jan 1
	sma = ['34_2','90_2']
	pmd = ['40_2']
	m1 = ['66_1','83_2','84_2','104_2','110_2','111_2','149_2']
	presma = []
	'''

	#r = io.TdtIO(filename)
	#bl = r.read_block(lazy=False,cascade=True)
	bl = tdt_neo   # data already read in using neo library
	rate_data = dict()
	

	spiketrains = bl.segments[block-1].spiketrains

	# get sample hpf signal to find stimulation times
	for sig in bl.segments[block-1].analogsignals:
		if (sig.name == 'pNe1 33'):
			hpf_sample = sig
			hpf_samplingrate = sig.sampling_rate.item()	# round to lowest integer

	# find stim times
	sample = 0
	stim_times = []
	while sample < hpf_sample.size:
		if (hpf_sample[sample] > stim_thres):
			stim_times.append(sample)
			sample += 11*np.floor(hpf_samplingrate) # advance 11 s = 1 s stim + 10 s minimum pre-train delay, make sure advancement is an integer value
		else:
			sample += 1
	num_epochs = len(stim_times)
	#num_epochs = 30

	channel = 0
	second_channel_bank = 0
	bin_size = .2  # .1 s = 100 ms
	prestim_time = 5 
	poststim_time = 10
	stim_time = train_length
	total_time = prestim_time + stim_time + poststim_time
	#num_bins = 10/bin_size
	num_bins = total_time/bin_size
	for train in spiketrains:
		epoch_rates = np.zeros([num_epochs,num_bins])
		if float(train.name[4:6]) < channel:
			second_channel_bank = 96
		channel = int(train.name[4:6]) + second_channel_bank
		if (train.name[-5:]!='Code0')&(train.name[-6:0]!='Code31'):
			code = train.name[-1]
			train_name = str(channel) +'_'+str(code)
			epoch_counter = 0
			for train_start in stim_times:
				epoch_start = float(train_start)/hpf_samplingrate # get stim train start time in seconds
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
		#channel_num = rates
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

	sig_population_presma = (sig_population_presma < 0.05*np.ones(len(sig_population_presma)))
	sig_population_presma_ind = np.nonzero(sig_population_presma)
	sig_population_sma = (sig_population_sma < 0.05*np.ones(len(sig_population_sma)))
	sig_population_sma_ind = np.nonzero(sig_population_sma)
	sig_population_m1 = (sig_population_m1 < 0.05*np.ones(len(sig_population_m1)))
	sig_population_m1_ind = np.nonzero(sig_population_m1)
	sig_population_pmd = (sig_population_pmd < 0.05*np.ones(len(sig_population_pmd)))
	sig_population_pmd_ind = np.nonzero(sig_population_pmd)

	if generate_figs==1:
		time = np.arange(0,total_time,bin_size) - prestim_time
		plt.figure()
		plt.subplot(2,2,1)
		plt.plot(time,average_zscored_presma,'b')
		plt.fill_between(time,average_zscored_presma-std_zscored_presma,average_zscored_presma+std_zscored_presma,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_presma_ind],sig_population_presma[sig_population_presma_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('Pre-SMA: n = %i' % (n_presma))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,2)
		plt.plot(time,average_zscored_sma,'b')
		plt.fill_between(time,average_zscored_sma-std_zscored_sma,average_zscored_sma+std_zscored_sma,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_sma_ind],sig_population_sma[sig_population_sma_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('SMA: n = %i' % (n_sma))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,3)
		plt.plot(time,average_zscored_m1,'b')
		plt.fill_between(time,average_zscored_m1-std_zscored_m1,average_zscored_m1+std_zscored_m1,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_m1_ind],sig_population_m1[sig_population_m1_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('M1: n = %i' % (n_m1))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.subplot(2,2,4)
		plt.plot(time,average_zscored_pmd,'b')
		plt.fill_between(time,average_zscored_pmd-std_zscored_pmd,average_zscored_pmd+std_zscored_pmd,facecolor='gray',alpha=0.5,linewidth=0.0)
		plt.plot(time[sig_population_pmd_ind],sig_population_pmd[sig_population_pmd_ind],'xr')
		plt.plot(time,np.zeros(time.size),'k--')
		plt.title('PMd: n = %i' % (n_pmd))
		plt.xlabel('Time (s)')
		plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
		plt.ylim((-1,2))
		plt.tight_layout()
		plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block)+'_PopulationResponse.svg')
		plt.close()

	return stim_times, population_presma, population_sma, population_pmd, population_m1