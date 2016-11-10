import numpy as np
import scipy as sp
from scipy import signal
from scipy.ndimage import filters
from scipy import io
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import mlab
from scipy.interpolate import spline
from spectrogram.spectrogram_methods import make_spectrogram, make_single_spectrogram
import glob
from pylab import specgram
import time


"""
Also do high-gamma: should see high-gamma increase during movement

"""

blocks = [1,2,3]
channel = 86  		# picking subset of channels to analyze
fmax = 200.
fs = 1000

filename_prefix = 'C:/Users/Samantha Summerson/Documents/GitHub/analysis/'

tdt_filename = 'Mario20161026-OMNI'

mat_files = glob.glob(tdt_filename + "*.mat")
print "Mat files used are:"
print mat_files
# Load all data
for i in range(len(mat_files)):
	omnib1 = dict()
	mat_file = mat_files[i]
	print "Loading data for Block %i." % (i+1)
	sp.io.loadmat(mat_file,omnib1)
	data = omnib1['corrected_data']
	if i==0:
		total_data = data
	else:
		total_data = np.vstack([total_data,data])

num_samps, num_chann = total_data.shape
print "Data is %f minutes" % (num_samps/(fs*60))
	
print "Computing spectrogram"
t = time.time()
#powers, cf_list = make_single_spectrogram(total_data[:fs*60*3, channel], fs, fmax, 0, fs, fs)
#powers, cf_list, time_points, fig = specgram(total_data[:fs*60*3, channel],Fs=fs)

data_window = total_data[fs*60*25:fs*60*35, channel]
cf_list, time_points, powers = signal.spectrogram(data_window, fs = fs, nperseg = 1024, noverlap = 512)
check_powers = powers
powers = 10*np.log10(powers)
dur = len(data_window)/(fs*60)  # duration in minutes


# theta power: 6 - 10 Hz
theta_ind = np.ravel(np.nonzero(np.logical_and(np.less(cf_list,8), np.greater(cf_list,4))))
theta_power = np.sum(powers[theta_ind,:], axis = 0)
# delta power: 1 - 4 Hz
delta_ind = np.ravel(np.nonzero(np.logical_and(np.less(cf_list,3), np.greater(cf_list,0.9))))
delta_power = np.sum(powers[delta_ind,:], axis = 0)
#theta/delta ratio: high values during REM, low values during SWS
theta_delta_ratio = theta_power/delta_power

# Gaussian smoothing
b = signal.gaussian(39,4)
theta_delta_ratio_smooth = filters.convolve1d(theta_delta_ratio, b/b.sum())
theta_smooth = filters.convolve1d(theta_power, b/b.sum())
delta_smooth = filters.convolve1d(delta_power, b/b.sum())

x = np.linspace(0,dur,len(theta_delta_ratio))
xnew = np.linspace(0,dur,100)
theta_delta_ratio_spline = spline(x,theta_delta_ratio,xnew)
theta_spline = spline(x,theta_power,xnew)
delta_spline = spline(x,delta_power,xnew)

num_powers, num_time_points = powers.shape

freq_mat = np.tile(cf_list, (num_time_points,1)).T 	# should be same size as powers 
norm_power = powers/np.sqrt(freq_mat)

f_end = 53

print "Spectrogram took %f secs to generate" % (time.time() - t)
fig = plt.figure()
plt.subplot(211)
ax = plt.imshow(powers[1:f_end*2,:],interpolation = 'bicubic', aspect='auto', origin='lower', 
	extent = [0,dur,0, len(cf_list[1:f_end*2])])
yticks = np.arange(0, len(cf_list[:f_end*2]), 10)
yticklabels = ['{0:.2f}'.format(cf_list[i]) for i in yticks]
#xticks = np.arange(0, num_time_points/fs, fs*60*10)  # ticks every 10 minutes
#xticklabels = ['{0:.2f}'.format(x[i]) for i in xticks]
plt.yticks(yticks, yticklabels)
#plt.xticks(xticks, xticklabels)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (min)')
plt.title('Spectrogram')
fig.colorbar(ax)


plt.subplot(212)
#plt.plot(x,theta_delta_ratio,'b',label='TDR')
plt.plot(x,theta_delta_ratio_smooth,'r',label='TDR Smooth')
#plt.plot(xnew,theta_delta_ratio_spline,'g',label='TDR Spline')
#plt.plot(theta_spline,'c',label = 'Theta')
#plt.plot(delta_spline,'m',label = 'Delta')
plt.ylabel('Theta/Delta Ratio')
plt.legend()
plt.show()