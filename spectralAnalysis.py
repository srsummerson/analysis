from neo import io
from numpy import sin, linspace, pi
from matplotlib import pyplot as plt
from scipy import fft, arange
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
 				frq = frq[range(n/2)] # one side frequency range

 				Y = fft(data)/n # fft computing and normalization
 				Y = Y[range(n/2)]
 				
 				plt.figure()
 				plt.plot(frq,abs(Y),'r') # plotting the spectrum

 				plt.xlabel('Freq (Hz)')
 				plt.ylabel('|Y(freq)|')
 				#title('Channel %f' %channel)
 				plt.savefig('PowerSpec.png')
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
    pxxx,freqs,bins,im=specgram(data,Fs=srate,NFFT=binsize,noverlap=0)
        if i==0:
            spec = pxxx.copy()
        spec = pxxx + spec
    
    return spec/lfpdat.shape[1],freqs,bins