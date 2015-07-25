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

	for block in bl.segments:
		for analogsig in block.analogsignals:
			if analogsig.name[:4]=='LFP2':
				analogsig.channel_index +=96
			if (analogsig.name[:3]=='LFP')&(analogsig.channel_index==channel):
				Fs = analogsig.sampling_rate
				data = analogsig
				data_times = analogsig.times
				n = len(data) # length of the signal
				k = arange(n)
 				T = n/Fs
 				frq = k/T # two sides frequency range
 				#frq = frq[range(n/2)] # one side frequency range
 				
 				frq = frq[:100000]
 				#Welch method, see: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
 				#freq, Pxx_den = signal.welch(data, Fs, nperseg=1024)
 				Y = fft(data)/n # fft computing and normalization
 				#Y = Y[range(n/2)]
 				Y = Y[:100000]
 				plt.figure()
 				plt.plot(frq,abs(Y),'r') # plotting the spectrum
 				#plt.yscale('log')
 				plt.xlabel('Freq (Hz)')
 				plt.ylabel('|Y(freq)|')
 				#title('Channel %f' %channel)
 				plt.savefig('PowerSpec_Channel'+str(channel)+'.png')

 	return 

def LFPSpectrumAllChannel(tankname,num_channels):
	"""
	Adopted from: http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
	"""
	tank = tankname[-13:]  # extracts last 13 values, which should be LuigiYYYYMMDD
	block_num = 0
	r = io.TdtIO(dirname=tankname)
	bl = r.read_block(lazy=False,cascade=True)
	plt.figure()
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
 				
 				if num_channels==96:
 					ax1 = plt.subplot(8,12,analogsig.channel_index)
 				else:
 					ax1 = plt.subplot(10,16,analogsig.channel_index)

 				plt.plot(freq,Pxx_den/np.sum(Pxx_den),'r')
 				ax1.set_xlim([0, 100])
 				ax1.set_xticklabels([])
				#ax1.set_ylim([0, 1.0e-8])
				ax1.set_yticklabels([])
				plt.title(str(analogsig.channel_index))
 				#plt.yscale('log')
 				#plt.xlabel('Freq (Hz)')
 				#plt.ylabel('|Y(freq)|')
 		plt.savefig('PowerSpec_'+tank+'_'+str(block_num)+'.png')
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