import numpy as np 
import scipy as sp
from scipy import signal
from neo import io
from PulseMonitorData import findIBIs
import matplotlib.pyplot as plt


def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 

# Set up code for particular day and block
filename = 'Mario20160106'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
block_num = 1

# Defining variables for sliding avg and num histogram bins
savg_pulse = 10
savg_pupil = 100


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
mean_ibi = np.mean(pulse_ibi).item()
std_ibi = np.std(pulse_ibi).item()
mean_zscored_ibi = np.mean(zscored_pulse_ibi).item()
std_zscored_ibi = np.std(zscored_pulse_ibi).item()

nbins_pulse = np.arange(mean_ibi-10*std_ibi,mean_ibi+10*std_ibi,float(std_ibi)/2)
nbins_zscored_pulse = np.arange(mean_zscored_ibi-10*std_zscored_ibi,mean_zscored_ibi+10*std_zscored_ibi,float(std_zscored_ibi)/2)

sliding_avg_ibi = running_mean(pulse_ibi,savg_pulse)
sliding_avg_zscored_ibi = running_mean(zscored_pulse_ibi,savg_pulse)
ibi_hist, ibi_bins = np.histogram(pulse_ibi,bins=nbins_pulse)
zscored_ibi_hist, zscored_ibi_bins = np.histogram(zscored_pulse_ibi,bins=nbins_zscored_pulse)
ibi_hist = ibi_hist/float(len(pulse_ibi))
zscored_ibi_hist = zscored_ibi_hist/float(len(zscored_pulse_ibi))

#print ibi_bins
#print mean_ibi

# Filter pupil data
eyes_open = np.nonzero(np.greater(pupil_data,0.5))
eyes_open = np.ravel(eyes_open)
cutoff_f = 50
cutoff_f = float(cutoff_f)/(pupil_samplingrate/2)
num_taps = 100
lpf = signal.firwin(num_taps,cutoff_f,window='hamming')
pupil_data_filtered = signal.lfilter(lpf,1,pupil_data[eyes_open])
zscored_pupil_data_filtered = (pupil_data_filtered - np.mean(pupil_data_filtered))/float(np.std(pupil_data_filtered))

mean_pupil = np.mean(pupil_data_filtered)
std_pupil = np.std(pupil_data_filtered)
mean_zscored_pupil = np.mean(zscored_pupil_data_filtered)
std_zscored_pupil = np.std(zscored_pupil_data_filtered)

nbins_pupil = np.arange(mean_pupil-10*std_pupil,mean_pupil+10*std_pupil,float(std_pupil)/2)
nbins_zscored_pupil = np.arange(mean_zscored_pupil-10*std_zscored_pupil,mean_zscored_pupil+10*std_zscored_pupil,float(std_zscored_pupil)/2)

# Compute sliding average and pupil diameter distribution
sliding_avg_pupil_diameter = running_mean(pupil_data_filtered,savg_pupil)
sliding_avg_zscored_pupil_diameter = running_mean(zscored_pupil_data_filtered,savg_pupil)
pupil_hist, pupil_bins = np.histogram(pupil_data_filtered,bins=nbins_pupil)
zscored_pupil_hist, zscored_pupil_bins = np.histogram(zscored_pupil_data_filtered,bins=nbins_zscored_pupil)
pupil_hist = pupil_hist/float(len(pupil_data_filtered))
zscored_pupil_hist = zscored_pupil_hist/float(len(pupil_data_filtered))

ibi_bins = ibi_bins[1:]
zscored_ibi_bins = zscored_ibi_bins[1:]
pupil_bins = pupil_bins[1:]
zscored_pupil_bins = zscored_pupil_bins[1:]

# Plot results
plt.figure(1)
plt.subplot(4,1,1)
plt.plot(pulse_times[0:10*np.ceil(pulse_samplingrate)],pulse_data[0:10*np.ceil(pulse_samplingrate)])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (s)', fontsize=8)
plt.ylabel('Amplitude (V)',fontsize=8)
#plt.title('Pulse Data: %s Block %i' % (filename,block_num))
plt.subplot(4,1,2)
plt.plot(pulse_ibi[0:100])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Sample number',fontsize=8)
plt.ylabel('IBI (s)',fontsize=8)
plt.subplot(4,1,3)
plt.plot(sliding_avg_ibi)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Sample number',fontsize=8)
plt.ylabel('Average IBI (s)',fontsize=8)
plt.subplot(4,1,4)
plt.plot(ibi_bins,ibi_hist)
plt.autoscale(enable=True, axis='x', tight=True)
plt.fill_between(ibi_bins[17:22],ibi_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1)
plt.plot([ibi_bins[19],ibi_bins[19]],[0,ibi_hist[19]],'k--')
plt.text(ibi_bins[-5],np.max(ibi_hist)-0.1,'m=%f \n $\sigma$=%f' % (mean_ibi,std_ibi))
plt.xlabel('IBI (s)',fontsize=8)
plt.ylabel('Density',fontsize=8)
#plt.tight_layout()
plt.savefig('/home/srsummerson/code/analysis/PulseData/'+filename+'_b'+str(block_num)+'_SingleDayPulseData.pdf')
plt.show()

plt.figure(2)
plt.subplot(4,1,1)
plt.plot(pulse_times[0:10*np.ceil(pulse_samplingrate)],pulse_data[0:10*np.ceil(pulse_samplingrate)])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (s)', fontsize=8)
plt.ylabel('Amplitude (V)',fontsize=8)
#plt.title('Z-scored Pulse Data: %s Block %i' % (filename,block_num))
plt.subplot(4,1,2)
plt.plot(zscored_pulse_ibi[0:100])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Sample number',fontsize=8)
plt.ylabel('Z-scored IBI (s)',fontsize=8)
plt.subplot(4,1,3)
plt.plot(sliding_avg_zscored_ibi)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Sample number',fontsize=8)
plt.ylabel('Average Z-scored IBI (s)',fontsize=8)
plt.subplot(4,1,4)
plt.plot(zscored_ibi_bins,zscored_ibi_hist)
plt.autoscale(enable=True, axis='x', tight=True)
plt.fill_between(zscored_ibi_bins[17:22],zscored_ibi_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1)
plt.plot([zscored_ibi_bins[19],zscored_ibi_bins[19]],[0,zscored_ibi_hist[19]],'k--')
plt.text(zscored_ibi_bins[-2],np.max(zscored_ibi_hist)-0.1,'m=%f \n $\sigma$=%f' % (mean_zscored_ibi,std_zscored_ibi))
plt.xlabel('Z-scored IBI (s)',fontsize=8)
plt.ylabel('Density',fontsize=8)
#plt.tight_layout()
plt.savefig('/home/srsummerson/code/analysis/PulseData/'+filename+'_b'+str(block_num)+'_SingleDayPulseDataZscored.pdf')
plt.show()

plt.figure(3)
plt.subplot(3,1,1)
plt.plot(pupil_times[0:10*np.ceil(pupil_samplingrate)],pupil_data[0:10*np.ceil(pupil_samplingrate)])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Diameter (AU)',fontsize=8)
#plt.title('Pupil Data: %s Block %i' % (filename,block_num))
plt.subplot(3,1,2)
plt.plot(sliding_avg_pupil_diameter)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Average Diameter (AU)',fontsize=8)
plt.subplot(3,1,3)
plt.plot(pupil_bins,pupil_hist)
plt.autoscale(enable=True, axis='x', tight=True)
plt.fill_between(pupil_bins[17:22],pupil_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1)
plt.plot([pupil_bins[19],pupil_bins[19]],[0,pupil_hist[19]],'k--')
plt.text(pupil_bins[-2],np.max(pupil_hist)-0.1,'m=%f \n $\sigma$=%f' % (mean_pupil,std_pupil))
plt.xlabel('Diameter (AU)',fontsize=8)
plt.ylabel('Density',fontsize=8)
#plt.tight_layout()
plt.savefig('/home/srsummerson/code/analysis/PulseData/'+filename+'_b'+str(block_num)+'_SingleDayPupilData.pdf')
plt.show()

plt.figure(4)
plt.subplot(3,1,1)
plt.plot(pupil_times[0:10*np.ceil(pupil_samplingrate)],zscored_pupil_data_filtered[0:10*np.ceil(pupil_samplingrate)])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Z-scored Diameter (AU)',fontsize=8)
#plt.title('Pupil Data: %s Block %i' % (filename,block_num))
plt.subplot(3,1,2)
plt.plot(sliding_avg_zscored_pupil_diameter)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Time (s)',fontsize=8)
plt.ylabel('Average Z-scored Diameter (AU)',fontsize=8)
plt.subplot(3,1,3)
plt.plot(zscored_pupil_bins,zscored_pupil_hist)
plt.autoscale(enable=True, axis='x', tight=True)
plt.fill_between(zscored_pupil_bins[17:22],zscored_pupil_hist[17:22],np.zeros(5),facecolor='gray',linewidth=0.1)
plt.plot([zscored_pupil_bins[19],zscored_pupil_bins[19]],[0,zscored_pupil_hist[19]],'k--')
plt.text(zscored_pupil_bins[-2],np.max(zscored_pupil_hist)-0.1,'m=%f \n $\sigma$=%f' % (mean_zscored_pupil,std_zscored_pupil))
plt.xlabel('Z-scored Diameter (AU)',fontsize=8)
plt.ylabel('Density',fontsize=8)
#plt.tight_layout()
plt.savefig('/home/srsummerson/code/analysis/PulseData/'+filename+'_b'+str(block_num)+'_SingleDayPupilDataZscored.pdf')
plt.show()