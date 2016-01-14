from neo import io
import numpy as np
from scipy import stats
from stimAnalysis import PopulationResponseSingleBlock
from PulseMonitorData import findPulseTimes

filename = 'Mario20151228_stim'
block = 1
stim_thres = 1000
train_length = 1

# Read in TDT data
r = io.TdtIO(filename)
bl = r.read_block(lazy=False,cascade=True)

# Compute z-scored population activity	
stim_times, population_presma, population_sma, population_pmd, population_m1 = PopulationResponseSingleBlock(bl,block,stim_thres,train_length)
average_zscored_presma = np.mean(population_presma)
average_zscored_sma = np.mean(population_sma)
average_zscored_pmd = np.mean(population_pmd)
average_zscored_m1 = np.mean(population_m1)

# Get pulse data
for sig in bl.segments[block-1].analogsignals:
	if (sig.name = 'HrtR'):
		pulse_data = sig
		pulse_data_samplingrate = sig.sampling_rate

# Get pulse times
pulse_times = findPulseTimes(pulse_data)

# Define parameters need for computing average pulse rate following stim
bin_size = .1  # .1 s = 100 ms
prestim_time = 5 
poststim_time = 10
stim_time = train_length
total_time = prestim_time + stim_time + poststim_time
num_bins = total_time/bin_size
num_epochs = len(stim_times)
epoch_rates = np.zeros([num_epochs,num_bins])

for train_start in stim_times:
	epoch_start = float(train_start)/pulse_data_samplingrate # get stim train start time in seconds
	epoch_start = epoch_start - prestim_time   # epoch to include 5 s pre-stim data
	epoch_end = epoch_start + total_time  # epoch is 5 s pre-stim + 1 s stim + 10 s post-stim
	epoch_bins = np.arange(epoch_start,epoch_end+bin_size/2,bin_size) 
	counts, bins = np.histogram(pulse_times,epoch_bins)
	epoch_rates[epoch_counter][:] = counts/bin_size	# collect all rates into a N-dim array
	epoch_counter += 1

background_epoch = np.concatenate((np.arange(0,int(prestim_time/bin_size)), np.arange(int((prestim_time+stim_time)/bin_size),len(epoch_bins)-1)), axis=0)

# Compute z-scored pulse rates
average_zscore_pulse = np.zeros(num_bins)
for epoch in range(0,num_epochs):
	std_pulse = np.std(epoch_rates[epoch][background_epoch])
	if (std_pulse > 0):
		epoch_rates[epoch][:] = (epoch_rates[epoch][:] - np.mean(epoch_rates[epoch][background_epoch]))/std_pulse
	else:
		epoch_rates[epoch][:] = epoch_rates[epoch][:] - np.mean(epoch_rates[epoch][background_epoch])
	average_zscored_pulse += epoch_rates[epoch][:]
average_zscored_pulse = average_zscored_pulse/float(num_epochs)

# Correlate zscored pulse with zscored population responses per bin (gives correlation as function of time relative to stimulation train)
for bin in range(0,num_bins):
	coeff_presma, p_presma = stats.pearsonr(population_presma[:,bin],epoch_rates[:,bin])
	coeff_sma, p_sma = stats.pearsonr(population_sma[:,bin],epoch_rates[:,bin])
	coeff_pmd, p_pmd = stats.pearsonr(population_pmd[:,bin],epoch_rates[:,bin])
	coeff_m1, p_m1 = stats.pearsonr(population_m1[:,bin],epoch_rates[:,bin])

sig_corr_presma = (p_presma < 0.05*np.ones(len(p_presma)))
sig_corr_presma_ind = np.nonzero(sig_corr_presma)
sig_corr_sma = (p_sma < 0.05*np.ones(len(p_sma)))
sig_corr_sma_ind = np.nonzero(sig_corr_sma)
sig_corr_pmd = (p_pmd < 0.05*np.ones(len(p_pmd)))
sig_corr_pmd_ind = np.nonzero(sig_corr_pmd)
sig_corr_m1 = (p_m1 < 0.05*np.ones(len(p_m1)))
sig_corr_m1_ind = np.nonzero(sig_corr_m1)

time = np.arange(0,total_time,bin_size) - prestim_time
plt.figure()
plt.subplot(2,2,1)
plt.plot(time,average_zscored_presma,'b',label='PreSMA')
plt.plot(time,average_zscored_pulse,'m',label='Pulse')
plt.plot(time[sig_corr_presma_ind],sig_corr_presma[sig_corr_presma_ind],'xr')
plt.plot(time,np.zeros(time.size),'k--')
plt.title('Pre-SMA and Pulse Rate')
plt.xlabel('Time (s)')
plt.ylabel('Mean Population Deviation from Baseline \n [zscore(rate - background)] (Hz)',fontsize=8)
plt.legend()
plt.ylim((-1,2))
plt.subplot(2,2,2)
plt.plot(time,average_zscored_sma,'b',label='SMA')
plt.plot(time,average_zscored_pulse,'m',label='Pulse')
plt.plot(time[sig_corr_sma_ind],sig_corr_presma[sig_corr_sma_ind],'xr')
plt.plot(time,np.zeros(time.size),'k--')
plt.title('SMA and Pulse Rate')
plt.xlabel('Time (s)')
plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
plt.ylim((-1,2))
plt.legend()
plt.subplot(2,2,3)
plt.plot(time,average_zscored_pmd,'b',label='PMd')
plt.plot(time,average_zscored_pulse,'m',label='Pulse')
plt.plot(time[sig_corr_pmd_ind],sig_corr_pmd[sig_corr_pmd_ind],'xr')
plt.plot(time,np.zeros(time.size),'k--')
plt.title('PMd and Pulse Rate')
plt.xlabel('Time (s)')
plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
plt.ylim((-1,2))
plt.legend()
plt.subplot(2,2,4)
plt.plot(time,average_zscored_m1,'b',label='M1')
plt.plot(time,average_zscored_pulse,'m',label='Pulse')
plt.plot(time[sig_corr_m1_ind],sig_corr_presma[sig_corr_m1_ind],'xr')
plt.plot(time,np.zeros(time.size),'k--')
plt.title('M1 and Pulse Rate')
plt.xlabel('Time (s)')
plt.ylabel('Mean Population Deviation from Baseline \n [zscore (rate - background)] (Hz)',fontsize=8)
plt.ylim((-1,2))
plt.legend()
plt.tight_layout()
plt.savefig('/home/srsummerson/code/analysis/StimData/'+filename+'_b'+str(block)+'_PopulationResponse.svg')
plt.close()


		


