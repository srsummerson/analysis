from CaImagingAnalysis import CaData
import numpy as np
from os import listdir

dir = "C:/Users/ss45436/Box/CNPRC/Data/Left/New/"
filenames = listdir(dir)

no_reach_files = []
len_files = []

for name in filenames:
	path = dir + name
	print(path)
	ca_data = CaData(path)
	lz1 = np.sum(ca_data.LH_zone1)
	lz2 = np.sum(ca_data.LH_zone2)
	rz1 = np.sum(ca_data.RH_zone1)
	rz2 = np.sum(ca_data.RH_zone2)
	rec_length = ca_data.t_end/60.

	reaches = lz1 + lz2 + rz1 + rz2

	if reaches == 0:
		no_reach_files += [name]
		len_files += [rec_length]
