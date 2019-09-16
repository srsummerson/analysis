import numpy as np 
import matplotlib.pyplot as plt
import scipy
import scipy.io as spio
from matplotlib.pyplot import cm
import seaborn as sns
from heatmap import corrplot
import pandas as pd
from scipy import signal
from scipy.ndimage import filters

"""
ut_orange = {
0, 48, 99, 0
248, 151, 31
#f8971f
}
ut_yellow = {
0, 14, 100, 0
255, 214, 0
#ffd600
}
ut_lightgreen = {
40, 0, 89, 0
166, 205, 87
#a6cd57
}
ut_green = {
63, 0, 97, 20
87, 157, 66
#579d42
}
ut_blue = {
96, 0, 31, 2
0, 169, 183
#00a9b7
}
ut_darkblue = {
100, 31, 8, 42
0, 95, 134
#005f86
}
ut_gray = {
24, 9, 8, 22
156, 173, 183
#9cadb7
}
ut_beige = {
3, 4, 14, 8
214, 210, 196
#d6d2c4
}
"""
"""
Figure 2 Decoding of natural motor behavior from ensemble calcium dynamics (within session imaging with behavior)
B. Example Session (individual cell examples and all cells that show the diversity of tuning - multiple examples of each cell type):
Rasters (calcium events or ‘spikes’) with behavioral events (moving vs. not, ipsi vs. contra arm, left vs. right reach direction - same plot, different colors) marked. 
PSTHs showing same thing.
- rasters with rows by trial for individual units, aligned to end of reach, organized by type of reach, colored by type of reach
- from rasters, bin events and compute smoothed PSTH for each type of type
- plot PSTHs above raster in smooth curve form colored by type of reach
- automate tuning metric to make it easy to identify tuning specificity

Calcium traces (individual and mean) time locked to behavioral events. - try repeating for all cells used in rasters
Heat map averaged across trials for all cells time locked to behavioral event (time x cell x average firing rate - essentially an averaged raster)
C. Tuning index/metric based on event/spike modulation and/or calcium trace modulation that can then be applied for every cell in the example session and for all cells in all sessions. Report distributions of index/metric as histogram. 
D. Classify cells as left or right arm or direction of reach preferring, or movement-preferring, or non selective based on index/metric and report proportions of population in pie chart.
E. Color code the tuning index/metric and apply to cell map to show whether there is any spatial clustering of these tuning properties across the FOV. If nothing, could be moved to supplement.
F. Decoding performance (left vs. right arm, left vs. right direction) using calcium traces or events/spikes with respect to entry into reward zone (or other event times: start, end of reach) - try doing continuous decoding of hand position/velocity using calcium traces
G. Correlation or coherence across units based on the traces, could sort them based on their tuning classification. Maybe we could look at some metric of correlation between cells as a function of time relative to behavioral event. https://www.ncbi.nlm.nih.gov/pubmed/29720658

Figure 2 Supplement:
1. Example movie clip with simultaneous behavior: imaging FOV, behavior video, a few example cell traces and events marked, decoder decision output L/R.
2. Decoding performance as a function of time (vary window location and size) relative to reward zone entry
3. Decoding performance as a function of number of cells included
4. Variability in decoding performance across sessions
5. Stability of tuning across recording session: break a single day’s session into the first 10-15 minutes and the last 10-15 minutes and report scatter plot of early vs late tuning index/metric. If stable most data points should fall along the diagonal line of unity. 

Additional Supplement:
If the main figure only shows one of the behavioral tasks, similar analyses/results can be presented for the other behavioral task here.

Additional Supplement:
If the main figure only shows one of the hemispheres, similar analyses/results can be presented for the other hemisphere here.

"""

class CaUnit():
	'''
	Class for a unit captured in the Ca2+ imaging data. The format for the input 
	should be a single row in the C structure with seven fields: trace, trace_ts, 
	cell_contour, centroid, ca_evt_s, ca_evt_a, sp_evt_s. 

	trace: array, float values of delta F values (change from baseline)
	trace_ts: array, float values of time stamps (in seconds) for trace array
	cell_contour: 2D-array, integer x and y pixel coordinates for contour of unit
	centroid: array, integer x and y pixel coordinate for center of unit
	ca_evt_s: array, float values of time stamps (in seconds) for calcium events
	ca_evt_a: array, float values of amplitude corresponding to calcium events
	sp_evt_s: array, float values of time stamps (in seconds) for calcium events
	'''
	def __init__(self, unit):
		self.trace = unit['trace']
		self.trace_ts = unit['trace_ts']
		self.cell_contour = unit['cell_contour']
		self.centroid = unit['centroid']
		self.ca_evt_s = unit['ca_evt_s']
		self.ca_evt_a = unit['ca_evt_a']
		self.sp_evt_s = unit['sp_evt_s']


class CaData():
	def __init__(self, mat_filename):
		self.filename =  mat_filename
		# Read  data from .mat file
		self.raw_data = spio.loadmat(mat_filename, squeeze_me=True)
		self.B_mat = self.raw_data['B']
		self.B_timestamp = self.B_mat[:,0]		# Behavior time stamps
		self.RH_x = self.B_mat[:,2]				# Right hand x-position: X1_1_pix
		self.RH_y = self.B_mat[:,3]				# Right hand y-position: Y1_1_pix
		self.LH_x = self.B_mat[:,4]				# Left hand x-position: X2_1_pix
		self.LH_y = self.B_mat[:,5]				# Left hand y-position: Y2_1_pix
		self.RH_zone1 = self.B_mat[:,6]			# Right hand present in Zone 1: EV1_1
		self.RH_zone2 = self.B_mat[:,7]			# Right hand present in Zone 2: EV1_3
		self.LH_zone1 = self.B_mat[:,8]			# Left hand present in Zone 1: EV1_7
		self.LH_zone2 = self.B_mat[:,9]			# Left hand present in Zone 2: EV1_9
		self.RH_zone1_entry = self.B_mat[:,11]	# Right hand entry in Zone 1: EV1_1_unique
		self.RH_zone2_entry = self.B_mat[:,12]	# Right hand entry in Zone 2: EV1_3_unique
		self.LH_zone1_entry = self.B_mat[:,13]	# Left hand entry in Zone 1: EV1_7_unique
		self.LH_zone2_entry = self.B_mat[:,14]	# Left hand entry in Zone 2: EV1_9_unique
		self.C_struct = self.raw_data['C']
		self.num_units = len(self.C_struct)
		self.unit_dict = dict()
		self.t_end = self.B_timestamp[-1]

		for unit in range(len(self.C_struct)):
			self.unit_dict[unit] = CaUnit(self.C_struct[unit])

	'''
	Methods for analyzing neural and behavioral data.
	'''
	def zone_reach_timestamps(self, zone_logicals, hand_y):
		'''
		Return timestamps corresponding to entry into zone by hand.
		NOTE: this method is made redundant by updated data structure that includes the unique
		zone entries by each hand.

		Inputs:
		- zone_logicals: array; logical values indicating when hand is present in zone, should be one of the following: RH_zone1, RH_zone2, LH_zone1, LH_zone2
		- hand_y: array; y-values of hand position

		Outputs:
		- inds: array; integers indicating the indices of the entries when zone entry was achieved
		- ts: array; floats indicating times (in seconds) when zone entry was achieved
		'''
		# 1. Get rid of times when hand is in zone for < 250 ms (= 15 samples)
		enter_zone = np.array([ind for ind in range(len(zone_logicals))[1:] if ((zone_logicals[ind] - zone_logicals[ind-1])==1)])
		exit_zone = np.array([ind for ind in range(len(zone_logicals))[1:] if ((zone_logicals[ind] - zone_logicals[ind-1])==-1)])
		dur_zone = exit_zone - enter_zone
		enter_zone = enter_zone[np.greater(dur_zone,15)]

		# 2. Only keep zone entries if hand returned to "home zone" after previous zone entry
		check_return = np.greater(hand_y,np.nanmean(hand_y[np.nonzero(hand_y)])).astype(int)														# Define the "home zone" to be when the hand moves such that it is more than halfway towards the body (exclude noisy zero values)
		enter_home = [ind for ind in range(len(check_return))[1:] if ((check_return[ind] - check_return[ind-1])==1)]								# Find indices when hand returns towards the "home zone"
		unique_enter_zone = [int(enter_zone[i]) for i in range(len(enter_zone))[1:] if (enter_home[np.searchsorted(enter_home,enter_zone[i-1])] < enter_zone[i])]					# get rid of times where hand moves in and out of zone multiple times in short time frame (1 s = 60 samples)
		inds = [int(enter_zone[0])] + unique_enter_zone
		ts = self.B_timestamp[inds]

		return inds, ts

	def population_raster(self,event,**kwargs):
		'''
		Plot population raster for indicated channels of either calcium ('ca') or spike ('sp') events.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- chans: array; integers indicating which channels to use in the population raster, default is all channels
		- t_window: array; two float values in the form (t_start,t_stop) indicating the start and stop times (in seconds) of the time window to use, default is all tine
		
		Output:

		'''
		t_window = kwargs.get('t_window', np.array([0,self.t_end]))
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		num_chans = len(chans)
		color=iter(cm.rainbow(np.linspace(0,1,num_chans)))
		ax = plt.gca()
		ith = 0

		if event == 'ca':
			
			for chan in chans:
				c=next(color)
				event_times_list = self.unit_dict[chan].ca_evt_s
				if np.isscalar(event_times_list):
					event_times_list = np.array([event_times_list])
				event_times_list_twindow = np.array([t for t in event_times_list if (t >= t_window[0])&(t <= t_window[1])])
				
				for trial in event_times_list_twindow:
					plt.vlines(trial, ith, ith + 1.0, color=c)

				ith += 1
			
			plt.ylim(0, ith)
			plt.xlim(t_window)
			plt.xticks(np.arange(t_window[0],t_window[1],step=60), np.arange(0,(t_window[1]-t_window[0])/60,step=1))
			plt.xlabel('mins')
			plt.ylabel('Units')
			plt.show()
		
		elif event == 'sp':

			for chan in chans:
				c=next(color)
				event_times_list = self.unit_dict[chan].sp_evt_s
				if np.isscalar(event_times_list):
					event_times_list = np.array([event_times_list])
				event_times_list_twindow = np.array([t for t in event_times_list if (t >= t_window[0])&(t <= t_window[1])])
				
				for trial in event_times_list_twindow:
					plt.vlines(trial, ith, ith + 1.0, color=c)

				ith += 1
			
			plt.ylim(0, ith)
			plt.xlim(t_window)
			plt.xticks(np.arange(t_window[0],t_window[1],step=60), np.arange(0,(t_window[1]-t_window[0])/60,step=1))
			plt.xlabel('mins')
			plt.ylabel('Units')
			plt.show()

		else:
			print('Error - event type was not properly entered.\nPlease use either "ca" or "sp" to indicate event type.')

		return 

	def correlation_matrix(self,event,**kwargs):
		'''
		Correlation of calcium or spike events across indicated channels.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_resolution: float; the resolution (in seconds) of the time bins to use for computing correlation
		
		Output:
		- data_corr: 2D array; correlation matrix filled with floats corresponding to pairwise correlation values
		- chans: array; integers indicating which channels were used 

		'''
		t_resolution = kwargs.get('t_resolution', 1.)
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		num_chans = len(chans)
		tbins = np.arange(0,self.t_end + t_resolution,t_resolution)
		i = 0
		data_corr = np.array([])


		if event == 'ca':

			for chan in chans:
				event_times_list = self.unit_dict[chan].ca_evt_s
				hist, bin_edges = np.histogram(event_times_list, tbins)
				if i==0:
					event_mat = hist
					i += 1
				else:
					event_mat = np.vstack((event_mat,hist))
			event_mat = event_mat.T
			
			data = pd.DataFrame(event_mat, columns = chans)
			data_corr = data.corr()
			'''
			print(data.corr().shape)
			plt.figure()
			corrplot(data.corr())
			'''
			plt.figure()
			cmap = sns.color_palette("Blues", 256)
			sns.heatmap(data_corr, cmap=cmap)
			plt.xlabel('Units')
			plt.title('Correlation Matrix Heatmap for Ca Events')

		elif event == 'sp':

			for chan in chans:
				event_times_list = self.unit_dict[chan].sp_evt_s
				hist, bin_edges = np.histogram(event_times_list, tbins)
				if i==0:
					event_mat = hist
					i += 1
				else:
					event_mat = np.vstack((event_mat,hist))
			event_mat = event_mat.T
			
			data = pd.DataFrame(event_mat, columns = chans)
			data_corr = data.corr()
			'''
			print(data.corr().shape)
			plt.figure()
			corrplot(data.corr())
			'''
			plt.figure()
			cmap = sns.color_palette("Blues", 256)
			sns.heatmap(data_corr, cmap=cmap)
			plt.xlabel('Units')
			plt.title('Correlation Matrix Heatmap for Spike Events')

		else:
			print('Error - event type was not properly entered.\nPlease use either "ca" or "sp" to indicate event type.')
		
		return data_corr, chans

	def psth():
		

		return

	def population_psth(self,event,hand,zone,**kwargs):
		'''
		Make grid of psth plots for indicated channels of either calcium ('ca') or spike ('sp') events.
		activity is aligned to timestamps related to zone entry by indicated hand. A Gaussian kernel is
		used for smoothing the psth.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- hand: string; either 'l' or 'r' to indicate left or right hand, respectively
		- zone: int; either 1 or 2 to indicate zone 1 or zone 2, respectively
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 4 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 1 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 0.2 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.1 seconds

		Outputs:
		- smooth_avg_psth: dict; keys are the unit numbers and elements are an array containing the smoothed psth values, also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries
		- avg_psth: dict; keys are the unit numbers and elements are an array containing the smooth psth values, also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries

		'''
		# Determine PSTH parameters
		t_before = kwargs.get('t_before', 4.)
		t_after = kwargs.get('t_after', 1.)
		t_resolution = kwargs.get('t_resolution', 0.2)
		t_overlap = kwargs.get('t_overlap', 0.1)
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		# Retrieve zone entry data for designated hand and zone
		if (hand=='r') & (zone==1):
			zentry = self.RH_zone1
		elif (hand=='r') & (zone==2):
			zentry = self.RH_zone2
		elif (hand=='l') & (zone==1):
			zentry = self.LH_zone1
		elif (hand=='l') & (zone==2):
			zentry = self.LH_zone2

		# Determine unique zone entry times to align neural data to
		inds, times_align = self.zone_reach_timestamps(zentry)

		# Define variables for PSTH calculation
		psth_length = int(np.rint((t_before + t_after - t_overlap)/(t_resolution - t_overlap)))
		num_timepoints = len(times_align)
		xticklabels = np.arange(-t_before,t_after,(t_before	+ t_after)/psth_length)
		
		b = signal.gaussian(39, 1)
		
		# Compute PSTH per unit
		smooth_avg_psth = dict()
		avg_psth = dict()
		for chan in chans:
			psth = np.zeros((num_timepoints, psth_length))
			if event == 'ca':
				data = self.unit_dict[chan].ca_evt_s
			if event == 'sp':
				data = self.unit_dict[chan].sp_evt_s

			for i, tp in enumerate(times_align):
				t_start = tp - t_before
				t_end = tp + t_after
				for k in range(psth_length):
					data_window = np.ravel(np.greater(data, t_start + k*t_overlap)&np.less(data, t_start + k*t_overlap + t_resolution))
					psth[i,k] = np.sum(data_window)/t_resolution

			smooth_avg_psth[chan] = filters.convolve1d(np.nanmean(psth,axis=0), b/b.sum())
			avg_psth[chan] = np.nanmean(psth,axis=0)
			
		smooth_avg_psth['times'] = xticklabels
		avg_psth['times'] = xticklabels

		# Make grid plot
		num_plots = len(chans)
		num_rows = int(np.ceil(num_plots/8))
		num_cols = int(np.min([8,num_plots]))

		plt.figure()
		for i,c in enumerate(chans):
			plt.subplot(num_rows,num_cols,i+1)
			plt.title('Unit %i' % (c))
			plt.plot(xticklabels,avg_psth[c],'r')
			plt.plot(xticklabels,smooth_avg_psth[c],'b')

		plt.show()
		return smooth_avg_psth, avg_psth

	def psth_compare_zones(self,event,hand,**kwargs):
		'''
		Make psth plots of zone 1 versus zone 2 reaches with the same hand for comparison.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- hand: string; either 'l' or 'r' to indicate left or right hand, respectively
		- zone: int; either 1 or 2 to indicate zone 1 or zone 2, respectively
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 4 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 1 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 0.2 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.1 seconds
		- smoothing: logical; indicate if smoothed PSTH should be plotted, default is True

		'''
		# Determine PSTH parameters
		t_before = kwargs.get('t_before', 4.)
		t_after = kwargs.get('t_after', 1.)
		t_resolution = kwargs.get('t_resolution', 0.2)
		t_overlap = kwargs.get('t_overlap', 0.1)
		chans = kwargs.get('chans', np.arange(0,self.num_units))
		smoothing = kwargs.get('smoothing',True)

		# Compute PSTH for zones 1 and 2
		smooth_avg_psth_z1, avg_psth_z1 = self.population_psth(event,hand,1, t_before = t_before, t_after= t_after, t_resolution= t_resolution, t_overlap= t_overlap, chans= chans)
		smooth_avg_psth_z2, avg_psth_z2 = self.population_psth(event,hand,2, t_before = t_before, t_after= t_after, t_resolution= t_resolution, t_overlap= t_overlap, chans= chans)

		# Choose whether to use smooth PSTHs or unsmoothed for plotting
		if smoothing:
			psth_z1 = smooth_avg_psth_z1
			psth_z2 = smooth_avg_psth_z2
		else:
			psth_z1 = avg_psth_z1
			psth_z2 = avg_psth_z2

		# Make grid plot
		num_plots = len(chans)
		num_rows = int(np.ceil(num_plots/8))
		num_cols = int(np.min([8,num_plots]))

		plt.figure()
		for i,c in enumerate(chans):
			plt.subplot(num_rows,num_cols,i+1)
			plt.title('Unit %i' % (c))
			plt.plot(psth_z1['times'],psth_z1[c],'r',label='z1')
			plt.plot(psth_z2['times'],psth_z2[c],'b',label='z2')
			plt.legend()

		plt.show()

		return

