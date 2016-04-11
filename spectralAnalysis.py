from neo import io
from numpy import sin, linspace, pi
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft, arange, signal
from pylab import specgram
from scipy import signal

def LFPSpectrumSingleChannel(tankname,channel):
	"""
	Adopted from: http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
	"""
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	tank = tankname[-13:]  # extracts last 13 values, which should be LuigiYYYYMMDD
	block_num = 0
	for block in bl.segments:
		block_num += 1
		for analogsig in block.analogsignals:
			if analogsig.name[:4]=='LFP2':
				analogsig.channel_index +=96
			if (analogsig.name[:3]=='LFP')&(analogsig.channel_index==channel):
				Fs = analogsig.sampling_rate.item()
				data = analogsig
				num_timedom_samples = data.size
				time = [float(t)/Fs for t in range(0,num_timedom_samples)]
 				freq, Pxx_den = signal.welch(data, Fs, nperseg=1024)

 				plt.figure()
 				plt.subplot(2,1,1)
 				plt.plot(freq,Pxx_den/np.sum(Pxx_den),'r') # plotting the spectrum
 				plt.xlim((0, 100))
 				plt.xlabel('Freq (Hz)')
 				plt.ylabel('PSD')
 				plt.title('Channel ' +str(channel))
 				
 				plt.subplot(2,1,2)
				plt.plot(time[0:np.int(Fs)*10],data[0:np.int(Fs)*10],'r') # plotting LFP snippet
				plt.xlabel('Time (s)')
				plt.ylabel('LFP (uv)')
				plt.title('LFP Snippet')
 				plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/PowerSpec_'+tank+'_'+str(block_num)+'_Ch'+str(channel)+'.png')
 				plt.close()
 	return 

def LFPSpectrumAllChannel(tankname,num_channels):
	"""
	Adopted from: http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
	"""
	tank = tankname[-13:]  # extracts last 13 values, which should be LuigiYYYYMMDD
	block_num = 0
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	
	matplotlib.rcParams.update({'font.size': 6})
	for block in bl.segments:
		block_num += 1
		for analogsig in block.analogsignals:
			if analogsig.name[:4]=='LFP2':
				analogsig.channel_index +=96
			if (analogsig.name[:3]=='LFP'):
				Fs = analogsig.sampling_rate
				data = analogsig
				
 				freq, Pxx_den = signal.welch(data, Fs, nperseg=1024)
 				plt.figure(2*block_num-1)
 				if num_channels==96:
 					ax1 = plt.subplot(8,12,analogsig.channel_index)
 				else:
 					ax1 = plt.subplot(10,16,analogsig.channel_index)
 					
 				plt.plot(freq,Pxx_den/np.sum(Pxx_den),'r')
 				ax1.set_xlim([0, 40])
 				ax1.set_xticklabels([])
				ax1.set_ylim([0, 0.8])
				ax1.set_yticklabels([])
				plt.title(str(analogsig.channel_index))
 				
 				plt.figure(2*block_num)
 				if num_channels==96:
 					ax2 = plt.subplot(8,12,analogsig.channel_index)
 				else:
 					ax2 = plt.subplot(10,16,analogsig.channel_index)
				plt.semilogy(freq,Pxx_den,'r')
 				ax2.set_xlim([0, 40])
 				ax2.set_xticklabels([])
				#ax2.set_ylim([0, 1.0e-8])
				ax2.set_yticklabels([])
				plt.title(str(analogsig.channel_index))
 		plt.figure(1)
 		plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/NormalizedPowerSpec_'+tank+'_'+str(block_num)+'.png')
 		plt.close()
 		plt.figure(2)
 		plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/PowerSpec_'+tank+'_'+str(block_num)+'.png')
 		plt.close()
 	return 

def gen_spcgrm(tankname,channel,cutoffs=(0,250),binsize=50):
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	for analogsig in bl.segments[0].analogsignals:
		if analogsig.name[:4]=='LFP2':
			analogsig.channel_index +=96
		if (analogsig.name[:3]=='LFP')&(analogsig.channel_index==channel):
			data = analogsig
			srate = analogsig.sampling_rate
			spec,freqs,bins,im=specgram(data,Fs=srate,NFFT=binsize,noverlap=0)
	return 

def TrialAveragedPSD(lfp_data, chann, Fs, lfp_ind, samples_lfp, row_ind, stim_freq):
	'''
	Computes PSD per channel, with data averaged over trials. 
	'''
	density_length = 30
	
	trial_power = np.zeros([density_length,len(row_ind)])
	for i in range(0,len(row_ind)):	
		lfp_snippet = lfp_data[chann][lfp_ind[i]:lfp_ind[i]+samples_lfp[i]]
		num_timedom_samples = lfp_snippet.size
		time = [float(t)/Fs for t in range(0,num_timedom_samples)]
 		freq, Pxx_den = signal.welch(lfp_snippet, Fs, nperseg=512, noverlap=256)
 		norm_freq = np.append(np.ravel(np.nonzero(np.less(freq,stim_freq-3))),np.ravel(np.nonzero(np.less(freq,stim_freq+3))))
 		total_power_Pxx_den = np.sum(Pxx_den[norm_freq])
 		Pxx_den = Pxx_den/total_power_Pxx_den
 		trial_power[:,i] = Pxx_den[0:density_length]

 	'''
 	z_min, z_max = -np.abs(trial_power).max(), np.abs(trial_power).max()
	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(low_power, cmap='RdBu')
	#plt.xticks(np.arange(0.5,density_length+0.5),freq[0:density_length])
	#plt.yticks(range(0,len(row_ind_successful_stress)))
	plt.title('Channel %i - Spectrogram: 0 - 10 Hz power' % (chann))
	plt.axis('auto')
	plt.colorbar()
	#z_min, z_max = -np.abs(beta_power).max(), np.abs(beta_power).max()
	plt.subplot(1, 2, 2)
	plt.imshow(beta_power, cmap='RdBu')
	#plt.xticks(np.arange(0.5,len(bins)+0.5),bins)
	#plt.yticks(range(0,len(row_ind_successful_stress)))
	plt.title('Spectrogram: 50 - 100 Hz power')
	plt.axis('auto')
	# set the limits of the plot to the limits of the data
	#plt.axis([x.min(), x.max(), y.min(), y.max()])
	plt.colorbar()
	'''
	
	return freq, trial_power