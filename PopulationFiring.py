from neo import io
import os
import numpy as np
import matplotlib.pyplot as plt


def PopulationResponseToStimulation(filename,**kwargs):

	# Reading Data
	r = io.TdtIO(filename)
	bl = r.read_block(lazy=False,cascade=True)

	if kwargs is not None:
		block = kwargs - 1
	else:
		block = range(0,len(bl.segments))


	for block_num in block:
		spiketrains = bl.segments[block_num].spiketrains
		
		# Use one lfp channel (33 in this case) to detect stimulation times
		for sig in bl.segments[block_num].analogsignals:
			if (sig.name == 'LFP1 33'): # third channel indicates message type
				sample_lfp = sig

		lfp_samplingrate = sample_lfp.sampling_rate.item()
		lfp_snippet = sample_lfp[0:lfp_samplingrate]	# take 1 sec of activity as sample
		lfp_snippet_max = np.amax(lfp_snippet).item()
		stim_thres = 2*lfp_snippet_max

		threshold_sample_lfp = (sample_lfp > stim_thres)

		sample = 0
		stim_start_indices = []
		while sample < sample_lfp.size:
			if threshold_sample_lfp[sample]==1:
				stim_start_indices.append(sample)
				sample += 11*lfp_samplingrate # skip ahead 11 s since that's the duration of stimulation train + pre-train delay (10s)
			else:
				sample += 1

		



