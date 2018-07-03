import numpy as np 
import scipy as sp
import pandas as pd
import matplotlib as mpl
import tables
import sys
import statsmodels.api as sm
from neo import io
from PulseMonitorData import findIBIs, getIBIandPuilDilation
from scipy import signal
from scipy import stats
from matplotlib import mlab
import matplotlib.pyplot as plt
from basicAnalysis import plot_cov_ellipse, LDAforFeatureSelection
from csv_processing import get_csv_data_singlechannel
from probabilisticRewardTaskPerformance import FreeChoiceBehavior_withStressTrials
from spectralAnalysis import TrialAveragedPSD, computePowerFeatures, computeAllCoherenceFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
import os.path
import time

from StressTaskBehavior import StressBehavior
from StressTrialAnalysis_LDA_PowerFeatures import StressTrialAnalysis_ComputePowerFeatures

behavior_sheet = 'Mario_Behavior_Log.xlsx'
session_type_sheet = 5

xls = pd.ExcelFile(behavior_sheet)
sheetX = xls.parse(session_type_sheet)
var1 = sheetX['Session ']
var2 = sheetX['Hdf filename']
var3 = sheetX['TDT filename']

session_renumber = np.zeros(len(var1))
session_renumber = np.array([(var1[n-1] if np.isnan(var1[n]) else var1[n]) for n in range(len(var1))])
sessions, len_sessions = np.unique(session_renumber, return_counts = True) 		# extract session numbers

start_session_num = 9
last_session_num = 9

for k in range(start_session_num, last_session_num+1):
	curr_session = k
	len_curr_session = len_sessions[k-1]
	row_num = np.ravel(np.argwhere(var1==curr_session))
	
	hdf_filename = np.ravel(var2[row_num])[0]
	filename = np.ravel(var3[row_num])[0][:-7]
	block_num = int(np.ravel(var3[row_num])[0][-1])
	
	if len_sessions[k-1] == 2:
		hdf_filename_stim = np.ravel(var2[row_num + 1])[0]
		filename2 = np.ravel(var3[row_num + 1])[0][:-7]
		block_num_stim = int(np.ravel(var3[row_num + 1])[0][-1])
	else:
		hdf_filename_stim = ''
		filename2 = filename
		block_num_stim = block_num

	print hdf_filename
	StressTrialAnalysis_ComputePowerFeatures(hdf_filename, hdf_filename_stim, filename, filename2, block_num, block_num_stim)


