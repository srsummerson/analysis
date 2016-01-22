import numpy as np 
import scipy as sp
from scipy import signal
from neo import io
from PulseMonitorData import findIBIs
import matplotlib.pyplot as plt

''''
Just do for raw data or zscored data as well?
Adjust so that only look at data during record window (DIOx 1)
'''

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 

# Set up code for particular day and block
filename = 'Luigi20151217_HDEEG'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
block_num = 1

# Defining variables for sliding avg and num histogram bins
savg_pulse = 10
savg_pupil = 100
nbins_pulse = np.arange(0.2,0.5,0.01)
nbins_zscored_pulse = np.arange(-5,5,0.33)
nbins_pupil = 20

r = io.TdtIO(TDT_tank)
bl = r.read_block(lazy=False,cascade=True)

for sig in bl.segments[block_num-1].analogsignals:
	if (sig.name == 'DIOx 1'):
		DIOx1 = sig
	if (sig.name == 'HrtR 1'):
		pulse_data = sig
		pulse_times = np.ravel(sig.times)
		pulse_samplingrate = sig.sampling_rate.item()
	if (sig.name == 'PupD 1'):
		pupil_data = np.ravel(sig)
		pupil_times = np.ravel(sig.times)
		pupil_samplingrate = sig.sampling_rate.item()

record = np.ravel(np.nonzero(DIOx1))
test_record_start = np.nonzero(np.greater(record[1:] - record[0:-1],1))
test_record_start = np.ravel(test_record_start)
if (len(test_record_start) > 0):
	record_on = record[test_record_start[0]+1]
else:
	record_on = record[0]
record_off = record[-1]

pulse_data = pulse_data[record_on:record_off]
pulse_times = pulse_times[record_on:record_off]
pupil_data = pupil_data[record_on:record_off]
pupil_times = pupil_times[record_on:record_off]

# Compute sliding average and IBI distribution
pulse_ibi = findIBIs(pulse_data)
zscored_pulse_ibi = (pulse_ibi - np.mean(pulse_ibi))/float(np.std(pulse_ibi))
sliding_avg_ibi = running_mean(pulse_ibi,savg_pulse)
sliding_avg_zscored_ibi = running_mean(zscored_pulse_ibi,savg_pulse)
ibi_hist, ibi_bins = np.histogram(pulse_ibi,bins=nbins_pulse)
zscored_ibi_hist, zscored_ibi_bins = np.histogram(zscored_pulse_ibi,bins=nbins_zscored_pulse)
ibi_hist = ibi_hist/float(len(pulse_ibi))
zscored_ibi_hist = zscored_ibi_hist/float(len(zscored_pulse_ibi))

# Filter pupil data
eyes_open = np.nonzero(np.greater(pupil_data,0.5))
eyes_open = np.ravel(eyes_open)
cutoff_f = 50
cutoff_f = float(cutoff_f)/(pupil_samplingrate/2)
num_taps = 100
lpf = signal.firwin(num_taps,cutoff_f,window='hamming')
pupil_data_filtered = signal.lfilter(lpf,1,pupil_data[eyes_open])
zscored_pupil_data_filtered = (pupil_data_filtered - np.mean(pupil_data_filtered))/float(np.std(pupil_data_filtered))


# Compute sliding average and pupil diameter distribution
sliding_avg_pupil_diameter = running_mean(pupil_data_filtered,savg_pupil)
sliding_avg_zscored_pupil_diameter = running_mean(zscored_pupil_data_filtered,savg_pupil)
pupil_hist, pupil_bins = np.histogram(pupil_data_filtered,bins=nbins_pupil)
zscored_pupil_hist, zscored_pupil_bins = np.histogram(zscored_pupil_data_filtered,bins=nbins_pupil)
pupil_hist = pupil_hist/float(len(pupil_data_filtered))
zscored_pupil_hist = zscored_pupil_hist/float(len(pupil_data_filtered))

# Plot results
plt.figure(1)
plt.subplot(4,1,1)
plt.plot(pulse_times[0:10*np.ceil(pulse_samplingrate)],pulse_data[0:10*np.ceil(pulse_samplingrate)])
plt.subplot(4,1,2)
plt.plot(pulse_ibi[0:100])
plt.subplot(4,1,3)
plt.plot(sliding_avg_ibi)
plt.subplot(4,1,4)
plt.plot(ibi_bins[1:],ibi_hist)
plt.show()

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(zscored_pulse_ibi[0:100])
plt.subplot(3,1,2)
plt.plot(sliding_avg_zscored_ibi)
plt.subplot(3,1,3)
plt.plot(zscored_ibi_bins[1:],zscored_ibi_hist)
plt.show()

plt.figure(3)
plt.subplot(3,1,1)
plt.plot(pupil_times[0:10*np.ceil(pupil_samplingrate)],pupil_data[0:10*np.ceil(pupil_samplingrate)])
plt.subplot(3,1,2)
plt.plot(sliding_avg_pupil_diameter)
plt.subplot(3,1,3)
plt.plot(pupil_bins[1:],pupil_hist)
plt.show()

plt.figure(3)
plt.subplot(3,1,1)
plt.plot(pupil_times[0:10*np.ceil(pupil_samplingrate)],zscored_pupil_data_filtered[0:10*np.ceil(pupil_samplingrate)])
plt.subplot(3,1,2)
plt.plot(sliding_avg_zscored_pupil_diameter)
plt.subplot(3,1,3)
plt.plot(zscored_pupil_bins[1:],zscored_pupil_hist)
plt.show()