from neo import io
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import signal
from basicAnalysis import bandpassFilterData
import tables
from DecisionMakingBehavior import ChoiceBehavior_ThreeTargets_Stimulation

# Set up code for particular day and block
hdf_filename = 'mari20170204_03_te2996.hdf'
filename = 'Mario20170204'
TDT_tank = '/home/srsummerson/storage/tdt/'+filename
hdf_location = '/storage/rawdata/hdf/'+hdf_filename
syncHDF_file = 

block_num = 1

vmPFC_chans = [13, 27, 11, 14, 28, 12]
OFC_chans = [25, 9, 26, 10]
Cd_chans = [20, 4, 3, 17, 57, 41, 56, 40, 54, 53, 37, 51]

###########################
# Load behavior data
###########################

# Unpack behavioral data
cb = ChoiceBehavior_ThreeTargets_Stimulation(hdf_files, 150, 100)

# Load syncing data for hdf file and TDT recording
hold_times = cb.get_state_TDT_LFPvalues(ind_check_reward_states-2,syncHDF_file)  # only consider holds from successful trials
reward_times = cb.get_state_TDT_LFPvalues(ind_check_reward_states,syncHDF_file)

targets_on, chosen_target, rewards, instructed_or_freechoice = cb.GetChoicesAndRewards()
hold_times_chooseL = hold_times[np.where(chosen_target==0)[0]]
hold_times_chooseM = hold_times[np.where(chosen_target==1)[0]]
hold_times_chooseH = hold_times[np.where(chosen_target==2)[0]]
reward_times_yes = reward_times[np.nonzero(rewards)[0]]
reward_times_no = reward_times[np.where(rewards==0)[0]]

# Loading LFP data
r = io.TdtIO(TDT_tank)
bl = r.read_block(lazy=False,cascade=True)
print "File read."
lfp = dict()

for sig in bl.segments[block_num-1].analogsignals:
	if (sig.name[0:4] == 'LFP1'):
		lfp_samprate = sig.sampling_rate.item()
		channel = sig.channel_index
		lfp[channel] = np.ravel(sig)


bandpassFilterData(data, Fs, cutoff_low, cutoff_high):