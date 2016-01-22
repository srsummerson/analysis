import numpy as np 
import scipy as sp
from neo import io
from PulseMonitorData import findIBIs
import matplotlib.pyplot as plt

''''
Just do for raw data or zscored data as well?
Adjust so that only look at data during record window (DIOx 1)
'''

def running_mean(x, N):
	cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 

# Set up code for particular day and block
hdf_filename = 'luig20151204_05.hdf'
filename = 'Luigi20151204_HDEEG'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
block_num = 1

# Defining variables for sliding avg and num histogram bins
savg_pulse = 10
savg_pupil = 10
nbins_pulse = np.arange(0.2,0.5,0.01)
nbins_pupil = 10

r = io.Tdt(filename)
bl = r.read_block(lazy=False,cascade=True)

for sig in bl.segments[0].analogsignals:
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

record = np.ravel(np.nonzero(DIOx))
record_on = record[0]
record_off = record[-1]

pulse_data = pulse_data[record_on:record_off]
pulse_times = pulse_times[record_on:record_off]
pupil_data = pupil_data[record_on:record_off]
pupil_times = pupil_times[record_on:record_off]

# Compute sliding average and IBI distribution
pulse_ibi = findIBIs(pulse_data)
sliding_avg_ibi = running_mean(pulse_ibi,savg_pulse)
ibi_hist, ibi_bins = np.histogram(pulse_ibi,bins=nbins_pulse)
ibi_hist = ibi_hist/len(pulse_ibi)

# Compute sliding average and pupil diameter distribution
min_diameter = np.min(pupil_data)
eyes_open = np.nonzero(np.greater(pupil_data,min_diameter))
eyes_open = np.ravel(eyes_open)
sliding_avg_pupil_diameter = running_mean(pupil_data[eyes_open],savg_pupil)
pupil_hist, pupil_bins = np.histogram(pupil_data[eyes_open],bins=nbins_pupil)
pupil_hist = pupil_hist/len(pupil_data[eyes_open])

# Plot results
plt.figure(1)
plt.subplot(4,1,1)
plt.plot(pulse_times[0:10*np.ceil(pulse_samplingrate)],pulse_data[0:10*np.ceil(pulse_samplingrate)])
plt.subplot(4,1,2)
plt.plot(pulse_ibi[0:100])
plt.subplot(4,1,3)
plt.plot(sliding_avg_ibi)
plt.subplot(4,1,4)
plt.plot(ibi_bins,ibi_hist)


plt.figure(2)
plt.subplot(3,1,1)
plt.plot(pupil_times[0:10*np.ceil(pupil_samplingrate)],pupil_data[0:10*np.ceil(pupil_samplingrate)])
plt.subplot(3,1,2)
plt.plot(sliding_avg_pupil_diameter)
plt.subplot(3,1,3)
plt.plot(pupil_bins,pupil_hist)