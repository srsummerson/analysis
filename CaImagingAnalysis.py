import numpy as np 
import matplotlib.pyplot as plt
import scipy
import scipy.io as spio
from matplotlib.pyplot import cm
import seaborn as sns
from heatmap import corrplot
import pandas as pd


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

	def psth(self,event,ts):
		'''
		make psth for either calcium ('ca') or spike ('sp') events.
		activity is aligned to timestamps in ts array.
		'''
		return



class CaData():
	def __init__(self, mat_filename):
		self.filename =  mat_filename
		# Read  data from .mat file
		self.raw_data = spio.loadmat(mat_filename, squeeze_me=True)
		self.B_mat = self.raw_data['B']
		self.B_timestamp = self.B_mat[:,0]
		self.X1_1_pix = self.B_mat[:,2]
		self.Y1_1_pix = self.B_mat[:,3]
		self.X2_1_pix = self.B_mat[:,4]
		self.Y2_1_pix = self.B_mat[:,5]
		self.EV1_1 = self.B_mat[:,6]
		self.EV1_3 = self.B_mat[:,7]
		self.EV1_5 = self.B_mat[:,8]
		self.EV1_7 = self.B_mat[:,9]
		self.C_struct = self.raw_data['C']
		self.num_units = len(self.C_struct)
		self.unit_dict = dict()
		self.t_end = self.B_timestamp[-1]

		for unit in range(len(self.C_struct)):
			self.unit_dict[unit] = CaUnit(self.C_struct[unit])

	'''
	functions to be written
	'''
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
		correlation of spike events across indicated channels

		Inputs:
		- event: string; either 'ca' or 'sp' to indicate calcium or spike events, respectively
		- chans: array; integers indicating which channels to use in the population raster, default is all channels
		- t_resolution: float; the resolution (in seconds) of the time bins to use for computing correlation
		
		Output:

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
		
		return data_corr

	def population_psth(self,chans,event,ts):
		'''
		make grid of psth plots for indicated channels of either calcium ('ca') or spike ('sp') events.
		activity is aligned to timestamps in ts array.
		'''
		return

