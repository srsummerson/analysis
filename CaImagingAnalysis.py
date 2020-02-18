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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score



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
		self.name = unit['name']
		self.trace = unit['trace']
		self.trace_ts = unit['trace_ts']
		self.cell_contour = unit['cell_contour']
		self.centroid = unit['centroid']
		self.ca_evt_s = unit['ca_evt_s']
		self.ca_evt_a = unit['ca_evt_a']
		self.sp_evt_s = unit['sp_evt_s']


class CaData():
	def __init__(self, mat_filename):
		self.filename =  mat_filename[-23:]		# For the filename, only store the .mat name rather than the entire address
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
		self.RH_zoneH = self.B_mat[:,8]			# Right hand present in home zone: EV1_5
		self.LH_zone1 = self.B_mat[:,9]			# Left hand present in Zone 1: EV1_7
		self.LH_zone2 = self.B_mat[:,10]		# Left hand present in Zone 2: EV1_9
		self.LH_zoneH = self.B_mat[:,11]		# Left hand present in home zone: EV1_11
		self.RH_zone1_entry = self.B_mat[:,12]	# Right hand entry in Zone 1: EV1_1_unique
		self.RH_zone2_entry = self.B_mat[:,13]	# Right hand entry in Zone 2: EV1_3_unique
		self.LH_zone1_entry = self.B_mat[:,14]	# Left hand entry in Zone 1: EV1_7_unique
		self.LH_zone2_entry = self.B_mat[:,15]	# Left hand entry in Zone 2: EV1_9_unique
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
			plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] +"_" + event +"_population_raster.svg")
			plt.close()
		
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
			plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_population_raster.svg")
			plt.close()

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

	def trial_raster(self,event,hand,zone,**kwargs):
		'''
		Make grid of raster plots for indicated channels of either calcium ('ca') or spike ('sp') events.
		activity is aligned to timestamps related to zone entry by indicated hand. A Gaussian kernel is
		used for smoothing the psth.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- hand: string; either 'l' or 'r' to indicate left or right hand, respectively
		- zone: int; either 1 or 2 to indicate zone 1 or zone 2, respectively
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 3 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 3 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 1 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.5 seconds

		Outputs:
		- smooth_avg_psth: dict; keys are the unit numbers and elements are an array containing the smoothed psth values, also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries
		- avg_psth: dict; keys are the unit numbers and elements are an array containing the smooth psth values, also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries

		'''
		'''
		Plot population raster for indicated channels of either calcium ('ca') or spike ('sp') events.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- chans: array; integers indicating which channels to use in the population raster, default is all channels
		- t_window: array; two float values in the form (t_start,t_stop) indicating the start and stop times (in seconds) of the time window to use, default is all tine
		
		Output:

		'''
		# Determine PSTH parameters
		t_before = kwargs.get('t_before', 3.)
		t_after = kwargs.get('t_after', 3.)
		t_resolution = kwargs.get('t_resolution', 1.0)
		t_overlap = kwargs.get('t_overlap', 0.5)
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		# Retrieve zone entry data for designated hand and zone
		if (hand=='r') & (zone==1):
			zentry = self.RH_zone1_entry
		elif (hand=='r') & (zone==2):
			zentry = self.RH_zone2_entry
		elif (hand=='l') & (zone==1):
			zentry = self.LH_zone1_entry
		elif (hand=='l') & (zone==2):
			zentry = self.LH_zone2_entry

		num_chans = len(chans)
		ax = plt.gca()

		# Make grid plot
		num_plots = len(chans)
		num_rows = int(np.ceil(num_plots/8))
		num_cols = int(np.min([8,num_plots]))
		

		# Determine unique zone entry times to align neural data to
		inds = np.ravel(np.nonzero(zentry))			# indices for unique zone entries
		times_align = self.B_timestamp[inds]
		

		if event == 'ca':
			fig = plt.figure()
			for k,chan in enumerate(chans):
				ith = 0
				for i, tp in enumerate(times_align):
					t_start = tp - t_before
					t_end = tp + t_after 
					
					event_times_list = self.unit_dict[chan].ca_evt_s
					if np.isscalar(event_times_list):
						event_times_list = np.array([event_times_list])
						#print(len(event_times_list))
					event_times_list_twindow = np.array([t for t in event_times_list if (t >= t_start)&(t <= t_end)])
					
					
					plt.subplot(num_rows,num_cols,k+1)
					for trial in event_times_list_twindow:
						plt.vlines(trial-t_start, ith, ith + 1.0)
					plt.title('Unit %i' % (chan))
					plt.ylim(0, ith+1)
					plt.xlim((0,t_after + t_before))
					plt.xticks(np.arange(0,(t_after+t_before)/1.,step=1),np.arange(-t_before,t_after,step=1))
					plt.xlabel('secs')
					plt.ylabel('Trials')
						

					ith += 1
				
			fig.set_size_inches((40, 50), forward=False)	
			plt.savefig("C:/Users/ss45436/Box Sync/CNPRC/Figures/" + self.filename[:-4] + "_" + hand+ "h_z" + str(zone)+ "_" + event +  "_event_raster_tbefore_" + str(t_before) + "_tafter_" + str(t_after) + ".svg", dpi = 500)
			plt.close()
		
		elif event == 'sp':

			
			for k,chan in enumerate(chans):
				ith = 0
				for i, tp in enumerate(times_align):
					t_start = tp - t_before
					t_end = tp + t_after 
					
					event_times_list = self.unit_dict[chan].sp_evt_s
					if np.isscalar(event_times_list):
						event_times_list = np.array([event_times_list])
					event_times_list_twindow = np.array([t for t in event_times_list if (t >= t_start)&(t <= t_end)])
					
					for trial in event_times_list_twindow:
						plt.vlines(trial-t_start, ith, ith + 1.0, color=k)

					ith += 1
				
				plt.subplot(num_rows,num_cols,k+1)
				plt.title('Unit %i' % (chan))
				plt.ylim(0, ith)
				#plt.xlim(t_window)
				plt.xticks(np.arange(t_before,t_after,step=1), np.arange(0,(t_after-t_before)/1.,step=1))
				plt.xlabel('secs')
				plt.ylabel('Trials')

			
			plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_" + hand+ "h_z" + str(zone)+ "_" + event +  "_event_raster.svg", dpi = 500)
			plt.close()	

		else:
			print('Error - event type was not properly entered.\nPlease use either "ca" or "sp" to indicate event type.')

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
		- tuning: string; either 'max' or 'avg' if using tuning to sort entries
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 3 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 3 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 1 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.5 seconds
		- show_fig: Boolean; indicate if figure should be shown

		Outputs:
		- smooth_avg_psth: dict; keys are the unit numbers and elements are an array containing the smoothed psth values, also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries
		- avg_psth: dict; keys are the unit numbers and elements are an array containing the smooth psth values, also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries

		'''
		# Determine PSTH parameters
		t_before = kwargs.get('t_before', 3.)
		t_after = kwargs.get('t_after', 3.)
		t_resolution = kwargs.get('t_resolution', 1.0)
		t_overlap = kwargs.get('t_overlap', 0.5)
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		show_fig = kwargs.get('show_fig', False)

		# Retrieve zone entry data for designated hand and zone
		if (hand=='r') & (zone==1):
			zentry = self.RH_zone1_entry
		elif (hand=='r') & (zone==2):
			zentry = self.RH_zone2_entry
		elif (hand=='l') & (zone==1):
			zentry = self.LH_zone1_entry
		elif (hand=='l') & (zone==2):
			zentry = self.LH_zone2_entry

		# Determine unique zone entry times to align neural data to
		inds = np.ravel(np.nonzero(zentry))			# indices for unique zone entries
		times_align = self.B_timestamp[inds]

		# Define variables for PSTH calculation
		psth_length = int(np.rint((t_before + t_after - t_overlap)/(t_resolution - t_overlap)))
		num_timepoints = len(times_align)
		xticklabels = np.arange(-t_before,t_after,(t_before	+ t_after)/psth_length)
		xticklabels = np.linspace(-t_before,t_after,psth_length)
		b = signal.gaussian(5, 1)
		
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

		max_val = 0
		psth_heatmap = np.empty([len(chans), len(xticklabels)])

		fig = plt.figure()
		for i,c in enumerate(chans):
			plt.subplot(num_rows,num_cols,i+1)
			plt.title('Unit %i' % (c))
			plt.xlabel('sec')
			plt.ylabel('event rate (Hz)')
			plt.step(xticklabels,avg_psth[c],'r')
			plt.plot(xticklabels,smooth_avg_psth[c],'b')
			psth_heatmap[i,:] = smooth_avg_psth[c]

		fig.set_size_inches((40, 50), forward=False)
		plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_" + hand+ "h_z" + str(zone)+ "_" + event +  "_psth.svg")
		plt.close()	

			
			
		if show_fig:
			plt.show()
		max_val = np.max(psth_heatmap)
		time_inds = np.array([i for i,t in enumerate(xticklabels) if (t>-0.5)&(t<0.5)])
		max_inds = np.argsort(np.max(psth_heatmap[:,time_inds],axis=1))							# sort units by max event rate in the (-0.5,0.5) time window around the home zone entry

		chan_order = kwargs.get('chan_order', 'none')
		if chan_order == 'none':
			chan_order = chans
		elif chan_order == 'max':
			tuning = self.tuning_metric_RZ1_RZ2_max(event = event, hand='r', hand2='l', chans = chans)
			chan_order = np.argsort(tuning)
		elif chan_order == 'avg':
			tuning = self.tuning_metric_RZ1_RZ2_avg(event = event, hand='r', hand2='l', chans = chans)
			chan_order = np.argsort(tuning)

		# create heatmap of activity
		plt.figure()
		plt.subplot(1,2,1)
		plt.pcolormesh(xticklabels,chans,psth_heatmap[chan_order,])
		plt.ylim((0,chans[-1]))
		plt.xlabel('Time (s)')
		plt.ylabel('Unit')
		plt.title('Units sorted by tuning')
		
		plt.subplot(1,2,2)
		plt.pcolormesh(xticklabels,chans,psth_heatmap)
		plt.ylim((0,chans[-1]))
		plt.xlabel('Time (s)')
		plt.ylabel('Unit')
		plt.title('Units unsorted')
		cbar = plt.colorbar()
		cbar.set_label('Event rate (Hz)')
		if show_fig:
			plt.show()
		plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_" + hand+ "h_z" + str(zone)+ "_" + event +  "_psth_heatmap.svg")
		plt.close()	

		return smooth_avg_psth, avg_psth, chan_order

	def psth_compare_zones(self,event,hand,**kwargs):
		'''
		Make psth plots of zone 1 versus zone 2 reaches with the same hand for comparison.

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- hand: string; either 'l' or 'r' to indicate left or right hand, respectively
		- hand2: string; either 'l' or 'r', used to indicate if comparison should be made with other hand
		- zone: int; either 1 or 2 to indicate zone 1 or zone 2, respectively
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 3 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 3 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 1 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.5 seconds
		- smoothing: logical; indicate if smoothed PSTH should be plotted, default is True

		'''
		# Determine PSTH parameters
		hand2 = kwargs.get('hand2', hand)
		t_before = kwargs.get('t_before', 3.)
		t_after = kwargs.get('t_after', 3.)
		t_resolution = kwargs.get('t_resolution', 1)
		t_overlap = kwargs.get('t_overlap', 0.5)
		chans = kwargs.get('chans', np.arange(0,self.num_units))
		smoothing = kwargs.get('smoothing',True)

		# Compute PSTH for zones 1 and 2
		smooth_avg_psth_z1, avg_psth_z1, chan_order = self.population_psth(event,hand,1, t_before = t_before, t_after= t_after, t_resolution= t_resolution, t_overlap= t_overlap, chans= chans, show_fig = False)
		smooth_avg_psth_z2, avg_psth_z2, chan_order = self.population_psth(event,hand2,2, t_before = t_before, t_after= t_after, t_resolution= t_resolution, t_overlap= t_overlap, chans= chans, show_fig = False)

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

		
		'''
		for i,c in enumerate(chans):
			plt.subplot(num_rows,num_cols,i+1)
			plt.title('Unit %i' % (c))
			plt.plot(psth_z1['times'],psth_z1[c],'r',label='%s - z1' % (hand))
			plt.plot(psth_z2['times'],psth_z2[c],'b',label='%s - z2' % (hand2))
			plt.legend()
		'''
		ts = np.linspace(-t_before,t_after,len(psth_z1['times']))
		fig = plt.figure()
		for i,c in enumerate(chans):
			plt.subplot(num_rows,num_cols,i+1)
			plt.title('Unit %i' % (c))
			#plt.bar(psth_z1['times'],psth_z1[c],width = 0.5, color = 'r',label='%s - z1' % (hand))
			plt.step(ts,psth_z1[c], color = 'm', where = 'post',label='%s - z1' % (hand))
			#plt.bar(psth_z2['times'],psth_z2[c], width = 0.5, color = 'b',label='%s - z2' % (hand2))
			plt.step(ts,psth_z2[c], color = 'g', where = 'post',label='%s - z2' % (hand2))
			plt.xlabel('secs')
			plt.ylabel('event rate (Hz)')
			plt.legend()

		fig.set_size_inches((40, 50), forward=False)
		plt.savefig("C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_" + hand+ "h_" + event +  "_psth_compare_zones_tbefore_" + str(t_before) + "_tafter_" + str(t_after) + ".svg")
		plt.close()

		return psth_z1, psth_z2

	def tuning_metric_RZ1_LZ2_max(self, event, hand='r', hand2='l',**kwargs):

		chans = kwargs.get('chans', np.arange(0,self.num_units))

		psth_rh_z1, psth_lh_z2 = self.psth_compare_zones(event,hand,hand2=hand2)

		times = np.ravel(psth_rh_z1['times'])
		time_inds = np.array([i for i,t in enumerate(times) if (t>-1)&(t<2)])

		peak_l_z2 = np.zeros(len(chans))
		peak_r_z1 = np.zeros(len(chans))
		tuning = np.zeros(len(chans))

		for i,c in enumerate(chans):
			peak_l_z2[i] = np.nanmax(psth_lh_z2[c][time_inds])
			peak_r_z1[i] = np.nanmax(psth_rh_z1[c][time_inds])

		peak_l_z2[np.isnan(peak_l_z2)] = 0
		peak_r_z1[np.isnan(peak_r_z1)] = 0
		tuning = (peak_l_z2 - peak_r_z1)/(peak_l_z2 + peak_r_z1)
			
		tuning[np.isnan(tuning)] = 0
		return tuning

	def tuning_metric_RZ1_RZ2_max(self, event, hand='r', hand2='r',**kwargs):

		chans = kwargs.get('chans', np.arange(0,self.num_units))

		psth_rh_z1, psth_lh_z2 = self.psth_compare_zones(event,hand,hand2=hand2)

		times = np.ravel(psth_rh_z1['times'])
		time_inds = np.array([i for i,t in enumerate(times) if (t>-1)&(t<2)])

		peak_l_z2 = np.zeros(len(chans))
		peak_r_z1 = np.zeros(len(chans))
		tuning = np.zeros(len(chans))

		for i,c in enumerate(chans):
			peak_l_z2[i] = np.nanmax(psth_lh_z2[c][time_inds])
			peak_r_z1[i] = np.nanmax(psth_rh_z1[c][time_inds])

		peak_l_z2[np.isnan(peak_l_z2)] = 0
		peak_r_z1[np.isnan(peak_r_z1)] = 0
		tuning = (peak_l_z2 - peak_r_z1)/(peak_l_z2 + peak_r_z1)
			
		tuning[np.isnan(tuning)] = 0
		return tuning

	def tuning_metric_RZ1_RZ2_overtime(self, event, hand='r', hand2='r',**kwargs):

		chans = kwargs.get('chans', np.arange(0,self.num_units))

		psth_rh_z1, psth_rh_z2 = self.psth_compare_zones(event,hand,hand2=hand2)

		times = np.ravel(psth_rh_z1['times'])
		#time_inds = np.array([i for i,t in enumerate(times) if (t>-1)&(t<2)])

		tuning = np.zeros((len(chans),len(times)))

		for i,c in enumerate(chans):
			peak_r_z2 = psth_rh_z2[c]
			peak_r_z1 = psth_rh_z1[c]
			tuning[i,:] = (peak_r_z2 - peak_r_z1)/(peak_r_z2 + peak_r_z1)
			tuning[i, np.isnan(tuning[i,:])] = 0

		cmap = sns.color_palette("RdBu_r", 256)
		plt.close()
		plt.figure()
		plt.pcolormesh(times,chans,tuning, cmap='RdBu_r')
		plt.xticks(times)
		plt.ylim((0,chans[-1]))
		plt.xlabel('Time (s)')
		plt.ylabel('Unit')
		#plt.title('Tuning Metric')
		cbar = plt.colorbar()
		cbar.set_label('Tuning Metric (AU)')
		plt.show()

		
		return tuning

	def tuning_metric_RZ1_LZ2_avg(self, event, hand='r', hand2='l',**kwargs):

		chans = kwargs.get('chans', np.arange(0,self.num_units))

		psth_rh_z1, psth_lh_z2 = self.psth_compare_zones(event,hand,hand2=hand2)

		times = np.ravel(psth_rh_z1['times'])
		time_inds = np.array([i for i,t in enumerate(times) if (t>-1)&(t<2)])

		er_l_z2 = np.zeros(len(chans))
		er_r_z1 = np.zeros(len(chans))
		tuning = np.zeros(len(chans))

		for i,c in enumerate(chans):
			er_l_z2[i] = np.nanmean(psth_lh_z2[c][time_inds])
			er_r_z1[i] = np.nanmean(psth_rh_z1[c][time_inds])
		
		er_l_z2[np.isnan(er_l_z2)] = 0
		er_r_z1[np.isnan(er_r_z1)] = 0
		tuning[i] = (er_l_z2[i] - er_r_z1[i])/(er_l_z2[i] + er_r_z1[i])
			
		#tuning[np.isnan(tuning)] = 0
		return tuning

	def tuning_metric_RZ1_RZ2_avg(self, event, hand='r', hand2='r',**kwargs):

		chans = kwargs.get('chans', np.arange(0,self.num_units))

		psth_rh_z1, psth_lh_z2 = self.psth_compare_zones(event,hand,hand2=hand2)

		times = np.ravel(psth_rh_z1['times'])
		time_inds = np.array([i for i,t in enumerate(times) if (t>-1)&(t<2)])

		er_l_z2 = np.zeros(len(chans))
		er_r_z1 = np.zeros(len(chans))
		tuning = np.zeros(len(chans))

		for i,c in enumerate(chans):
			er_l_z2[i] = np.nanmean(psth_lh_z2[c][time_inds])
			er_r_z1[i] = np.nanmean(psth_rh_z1[c][time_inds])
		
		er_l_z2[np.isnan(er_l_z2)] = 0
		er_r_z1[np.isnan(er_r_z1)] = 0
		tuning[i] = (er_l_z2[i] - er_r_z1[i])/(er_l_z2[i] + er_r_z1[i])
			
		#tuning[np.isnan(tuning)] = 0
		return tuning

	def trial_traces(self,hand,zone,**kwargs):
		'''
		Make dictionary of Ca2+ trace data for indicated channels aligned to reaches to indicated zone.
		Activity is aligned to timestamps related to zone entry by indicated hand. 

		Inputs:
		- hand: string; either 'l' or 'r' to indicate left or right hand, respectively
		- zone: int; either 1 or 2 to indicate zone 1 or zone 2, respectively
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 3 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 3 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 1 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.5 seconds

		Outputs:
		- trial_traces: dict; keys are the unit numbers and elements are a matrix containing the trace values across trials where each row corresponds to one trials's data; also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries
		
		'''
		
		# Determine trace parameters
		t_before = kwargs.get('t_before', 3.)
		t_after = kwargs.get('t_after', 3.)
		t_resolution = kwargs.get('t_resolution', 1.0)
		t_overlap = kwargs.get('t_overlap', 0.5)
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		avg_trace = dict()

		# Retrieve zone entry data for designated hand and zone
		if (hand=='r') & (zone==1):
			zentry = self.RH_zone1_entry
		elif (hand=='r') & (zone==2):
			zentry = self.RH_zone2_entry
		elif (hand=='l') & (zone==1):
			zentry = self.LH_zone1_entry
		elif (hand=='l') & (zone==2):
			zentry = self.LH_zone2_entry

		num_chans = len(chans)
		ax = plt.gca()

		# Make grid plot
		num_plots = len(chans)
		num_rows = int(np.ceil(num_plots/8))
		num_cols = int(np.min([8,num_plots]))
		

		# Determine unique zone entry times to align neural data to
		inds = np.ravel(np.nonzero(zentry))			# indices for unique zone entries
		times_align = self.B_timestamp[inds]
		
		# For each channel, pull out relevant trace data
		for k,chan in enumerate(chans):
			ith = 0
			trace_data = self.unit_dict[chan].trace
			trace_data_ts = self.unit_dict[chan].trace_ts

			trace_mean = np.nanmean(trace_data)
			trace_sd = np.nanstd(trace_data)	

			trace_mat = np.zeros((len(times_align),(int(t_before)+ int(t_after))*10)) 		# data is subsampled to 10 Hz
			for i, tp in enumerate(times_align):
				t_start = tp - t_before
				t_end = tp + t_after 
				
				event_times_list_twindow = np.array([ind for ind,t in enumerate(trace_data_ts) if (t >= t_start)&(t <= t_end)])
				trace_snippet = trace_data[event_times_list_twindow]
				trace_mat[i,:len(trace_snippet)] = trace_snippet

				ith += 1
			
			avg_trace[chan] = (trace_mat - trace_mean)/trace_sd
		
		avg_trace['times'] = np.arange(-t_before, t_after, 1./10)
		

		return avg_trace

	def avg_trial_traces(self,hand,zone,**kwargs):
		'''
		Make grid of Ca2+ average trace plots for indicated channels aligned to reaches to indicated zone.
		Activity is aligned to timestamps related to zone entry by indicated hand. 

		Inputs:
		- hand: string; either 'l' or 'r' to indicate left or right hand, respectively
		- hand2: string; either 'l' or 'r', used to indicate if comparison should be made with other hand
		- zone: int; either 1 or 2 to indicate zone 1 or zone 2, respectively
		- zone2: int; either 1 or 2, if want to do comparison between zones
		- chans: array; integers indicating which channels to use to do pairwise correlations, default is all channels
		- t_before: float; time (in seconds) before zone entry to include in psth, default is 3 seconds
		- t_after: float; time (in seconds) after zone entry to include in psth, default is 3 second
		- t_resolution: float; bin size (in seconds) of bins used for psth, default is 1 seconds
		- t_overlap: float; time (in seconds) to overlap bins, default is 0.5 seconds

		Outputs:
		- trial_traces: dict; keys are the unit numbers and elements are a matrix containing the trace values across trials where each row corresponds to one trials's data; also a key called 'times' which has the timepoints (in seconds) corresponding to the psth array entries
		
		'''
		# Determine trace parameters
		hand2 = kwargs.get('hand2', hand)
		t_before = kwargs.get('t_before', 3.)
		t_after = kwargs.get('t_after', 3.)
		t_resolution = kwargs.get('t_resolution', 1.0)
		t_overlap = kwargs.get('t_overlap', 0.5)
		chans = kwargs.get('chans', np.arange(0,self.num_units))
		zone2 = kwargs.get('zone2', 0)		# if second zone was not indicated, set to 0 and disregard later

		# Get dictionary of trace values over time window for all indicated channels
		avg_trace = self.trial_traces(hand,zone,chans = chans, t_before	= t_before, t_after = t_after, t_resolution	= t_resolution, t_overlap = t_overlap)
		avg_trace_mat = np.zeros((len(chans),avg_trace[chans[0]].shape[1]))
		sem_trace_mat = avg_trace_mat
		times = avg_trace['times']

		if zone2 > 0:
			avg_trace2 = self.trial_traces(hand2,zone2,chans = chans, t_before	= t_before, t_after = t_after, t_resolution	= t_resolution, t_overlap = t_overlap)
			avg_trace_mat2 = np.zeros((len(chans),avg_trace2[chans[0]].shape[1]))
			sem_trace_mat2 = avg_trace_mat2
			times2 = avg_trace2['times']

		# Make grid plot
		num_plots = len(chans)
		num_rows = int(np.ceil(num_plots/8))
		num_cols = int(np.min([8,num_plots]))

		
		# Compute averages and SEMs per channel
		fig = plt.figure()

		for i,c in enumerate(chans):
			trace_mat = avg_trace[c]
			trace_avg = np.nanmean(trace_mat,axis = 0)
			trace_sem = np.nanstd(trace_mat,axis = 0)/np.sqrt(trace_mat.shape[0])
			avg_trace_mat[i,:] = trace_avg
			sem_trace_mat[i,:] = trace_sem

			if zone2 > 0:
				trace_mat2 = avg_trace2[c]
				trace_avg2 = np.nanmean(trace_mat2,axis = 0)
				trace_sem2 = np.nanstd(trace_mat2,axis = 0)/np.sqrt(trace_mat2.shape[0])
				avg_trace_mat2[i,:] = trace_avg2
				sem_trace_mat2[i,:] = trace_sem2

			plt.subplot(num_rows,num_cols,i+1)
			plt.title('Unit %i - (%s)' % (c, self.unit_dict[c].name), fontsize = 24)
			plt.plot(times,trace_avg, color = 'm')
			plt.fill_between(times,trace_avg - trace_sem, trace_avg	+ trace_sem, facecolor = 'm', alpha = 0.25, label='%s - z%i' % (hand, zone))
			figname = "C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_" + hand+ "h_z" + str(zone)+ "_avg_traces_tbefore_" + str(t_before) + "_tafter_" + str(t_after) + ".svg"
			if zone2 > 0:
				plt.plot(times2,trace_avg2, color = 'g')
				plt.fill_between(times2,trace_avg2 - trace_sem2, trace_avg2	+ trace_sem2, facecolor = 'g', alpha = 0.25, label='%s - z%i' % (hand2, zone2))
				figname = "C:/Users/ss45436/Box/CNPRC/Figures/" + self.filename[:-4] + "_" + hand +"h" + hand2+ "h_both_zones_avg_traces_tbefore_" + str(t_before) + "_tafter_" + str(t_after) + ".svg"
			plt.ylim((-0.5,2))
			plt.xlim((-t_before,t_after))
			plt.legend(fontsize = 20)
			plt.tick_params(labelsize = 20)
			plt.xlabel('Time relative to zone entry (s)', fontsize = 24)
			plt.ylabel('Average z-scored' + r'$\Delta$' +'F', fontsize = 24)
		fig.set_size_inches((40, 20), forward=False)	
		plt.savefig(figname, dpi = 500)
		plt.close()
		
		return 

	def decoding_logistic_regression(self, event, hand1,hand2,zone1,zone2, **kwargs):
		'''
		Method that uses event activity from a specified window to decode reaches of hand1 to zone1 and hand2 to zone2.
		'''
		# Determine method parameters
		t_before = kwargs.get('t_before', 1.)
		t_after = kwargs.get('t_after', 2.)
		chans = kwargs.get('chans', np.arange(0,self.num_units))

		# Retrieve zone entry data for designated hands and zones
		if (hand1=='r') & (zone1==1):
			zentry1 = self.RH_zone1_entry
		elif (hand1=='r') & (zone1==2):
			zentry1 = self.RH_zone2_entry
		elif (hand1=='l') & (zone1==1):
			zentry1 = self.LH_zone1_entry
		elif (hand1=='l') & (zone1==2):
			zentry1 = self.LH_zone2_entry
		if (hand2=='r') & (zone2==1):
			zentry2 = self.RH_zone1_entry
		elif (hand2=='r') & (zone2==2):
			zentry2 = self.RH_zone2_entry
		elif (hand2=='l') & (zone2==1):
			zentry2 = self.LH_zone1_entry
		elif (hand2=='l') & (zone2==2):
			zentry2 = self.LH_zone2_entry

		# Determine unique zone entry times to align neural data to
		inds1 = np.ravel(np.nonzero(zentry1))			# indices for unique zone entries
		times_align1 = self.B_timestamp[inds1]
		inds2 = np.ravel(np.nonzero(zentry2))			# indices for unique zone entries
		times_align2 = self.B_timestamp[inds2]
		
		event_rate1 = self.event_rates(event = event, times_align = times_align1, chans = chans, t_before = t_before, t_after = t_after)
		event_rate2 = self.event_rates(event = event, times_align = times_align2, chans = chans, t_before = t_before, t_after = t_after)

		# organize data into trial x channel matrix
		trials1 = len(inds1)
		trials2 = len(inds2)
		X_mat = np.zeros((trials1 + trials2,len(chans)))
		for i,c in enumerate(chans):
			trial1_data = event_rate1[c]
			trial2_data = event_rate2[c]
			X_mat[:trials1,i] = trial1_data
			X_mat[trials1:,i] = trial2_data

		y = np.zeros(trials1 + trials2)
		y[trials1:] = 1

		# Model evaluation using 10-fold cross-validation
		scores_logreg = cross_val_score(LogisticRegression(),X_mat,y,scoring='accuracy',cv=10)
		np.random.shuffle(X_mat)
		mc_sim = 2
		scores_shuffle = np.zeros((mc_sim,10))
		for j in range(mc_sim):
			scores_shuffle[j,:] = cross_val_score(LogisticRegression(),X_mat,y,scoring='accuracy',cv=10)
		scores_logreg_shuffle = np.nanmean(scores_shuffle, axis = 0)
		#print("CV scores:", scores_logreg)
		#ca_print("Avg CV score:", scores_logreg.mean())
    	

		return scores_logreg, scores_logreg_shuffle

	def event_rates(self, event, times_align, chans, t_before, t_after):
		'''
		Compute event rate per unit across all units indicated for all of the indicated time points.
		
		Inputs:

		Outputs:
		- event_rate: dict; dictionary with a key for every channel and stored for each channel an array
			of event rates corresponding to each of the time points in times_align
		'''

		event_rate = dict()

		for chan in chans:
			er = np.zeros(len(times_align))
			if event == 'ca':
				data = self.unit_dict[chan].ca_evt_s
			if event == 'sp':
				data = self.unit_dict[chan].sp_evt_s

			for i, tp in enumerate(times_align):
				t_start = tp - t_before
				t_end = tp + t_after
				data_window = np.ravel(np.greater(data, t_start)&np.less(data, t_end))
				er[i] = np.sum(data_window)/(t_before + t_after)						# gives final measure in Hz
			event_rate[chan] = er

		return event_rate