from neo import io
from numpy import sin, linspace, pi
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft, arange, signal
from pylab import specgram

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
				Fs = analogsig.sampling_rate
				data = analogsig
				Fs = analogsig.sampling_rate
				data = analogsig
				
 				freq, Pxx_den = signal.welch(data, Fs, nperseg=1024)
 				plt.figure()
 				plt.plot(freq,Pxx_den/np.sum(Pxx_den),'r') # plotting the spectrum
 				plt.xlim((0, 100))
 				plt.xlabel('Freq (Hz)')
 				plt.ylabel('PSD')
 				plt.title('Channel ' +str(channel))
 				plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/PowerSpec_'+tank+'_'+str(block_num)+'_Ch'+str(channel)+'.png')

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
 		plt.figure(2)
 		plt.savefig('/home/srsummerson/code/analysis/Mario_Spectrum_figs/PowerSpec_'+tank+'_'+str(block_num)+'.png')
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