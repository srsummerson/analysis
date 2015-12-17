from neo import io
import os
import numpy as np
import matplotlib.pyplot as plt

"""
filename = 'Luigi20151203_HDEGG'
r = io.TdtIO(dirname=filename)
bl = r.read_block(lazy=False,cascade=True)

for sig in bl.segments[block].analogsignals:
	if (sig.name == 'HrtR 1'):
		pulse_signal = sig

IBI = findIBIs(pulse_signal)

plt.figure()
plt.plot(pulse_signal.times[0:pulse_signal.sampling_rate*60],pulse_signal[0:pulse_signal.sampling_rate*60]) # plot 1 min of data
cd /home/srsummerson/code/analysis
plot.savefig('test_pulse_signal.svg')
"""

def plottingIBI(IBI,tankname,block_num):

	IBI_max = np.amax(IBI)
	IBI_min = np.amin(IBI)
	nbins = 50
	hist_bins = np.arange(IBI_min,IBI_max,(IBI_max-IBI_min)/(nbins))

	IBI_div = np.floor(float(IBI.size)/3)
	IBI_first = IBI[0:IBI_div]
	IBI_second = IBI[IBI_div:2*IBI_div]
	IBI_third = IBI[2*IBI_div:]

	IBI_hist_first, hist_bins = np.histogram(IBI_first,bins=hist_bins)
	IBI_hist_second, hist_bins = np.histogram(IBI_second,bins=hist_bins)
	IBI_hist_third, hist_bins = np.histogram(IBI_third,bins=hist_bins)

	max_count = np.amax([IBI_hist_first,IBI_hist_second,IBI_hist_third])
	max_count = np.amax(max_count)


	plt.figure()
	plt.subplot(3,1,1)
	plt.plot(hist_bins[:-1],IBI_hist_first)
	plt.ylim((0, max_count))
	plt.title('First third of time in task')
	plt.subplot(3,1,2)
	plt.plot(hist_bins[:-1],IBI_hist_second)
	plt.ylim((0, max_count))
	plt.title('Second third of time in task')
	plt.subplot(3,1,2)
	plt.plot(hist_bins[:-1],IBI_hist_third)
	plt.ylim((0, max_count))
	plt.title('Last third of time in task')
	plt.xlabel('IBI (s)')
	plt.ylabel('Frequency (#)')
	plt.savefig('/home/srsummerson/code/analysis/PulseData/'+tankname+'_b'+str(block_num)+'_IBI_hists.svg')
 	plt.close()

 	plt.figure()
 	plt.plot(IBI,'b')
 	plt.savefig('/home/srsummerson/code/analysis/PulseData/'+tankname+'_b'+str(block_num)+'_IBI_signal.svg')