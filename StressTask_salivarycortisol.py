'''
File for loading and analyzing salivary cortisol levels
'''
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

dir = "C:/Users/ss45436/Box/UC Berkeley/Stress Task/Salivary cortisol data/"
File1 = dir + 'Export1-122118.xlsx'
File2 = dir + 'Export2-122118.xlsx'
File3 = dir + 'Export3-122118.xlsx'
File4 = dir + 'Export4-122118.xlsx'
File1_legend = dir + 'Plate1_legend.xlsx'
File2_legend = dir + 'Plate2_legend.xlsx'
File3_legend = dir + 'Plate3_legend.xlsx'
File4_legend = dir + 'Plate4_legend.xlsx'

# Imports data as pandas dataframe
df1 = pd.read_excel(File1)
df2 = pd.read_excel(File2)
df3 = pd.read_excel(File3)
df4 = pd.read_excel(File4)

df1_legend = pd.read_excel(File1_legend)
df2_legend = pd.read_excel(File2_legend)
df3_legend = pd.read_excel(File3_legend)
df4_legend = pd.read_excel(File4_legend)

plate1 = df1.values[1:,1:]
plate2 = df2.values[1:,1:]
plate3 = df3.values[1:,1:]
plate4 = df4.values[1:,1:]

Plate1_legend = df1_legend.values
Plate2_legend = df2_legend.values
Plate3_legend = df3_legend.values
Plate4_legend = df4_legend.values


def logistic4(x, A, B, C, D):
    """4PL lgoistic equation.
	x is the measured percent bound B/Bo
	A is the minimum asymptote
	B is the steepness
	C is the inflection point
	D is the maximum asymptote
    """
    val = ((A-D)/(1.0+((x/C)**B))) + D
    return val

def residuals(p, y, x):
    """Deviations of data from fitted 4PL curve
	p is the set the 4 parameters for the logistic function.
	y is the measured B/Bo values
	x is the known concentrations
    """
    A,B,C,D = p
    err = y-logistic4(x, A, B, C, D)
    return err

def peval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C,D = p
    return logistic4(x, A, B, C, D)


def process_plate(plate, plate_legend):
	'''
	Compute cortisol density from samples. Return these values along with their labels.

	1. Compute average optical density (OD) for duplicate wells
	2. Subtract the average NSB well from the average of the zero well, average of the standards, controls, and saliva samples
	3. Divide the OD of each well by the average OD of the zero to get the percent bound B/Bo
	4. Use 4-parameter non-linear regression curve fit to determine concentrations of the controls and saliva samples
	'''
	cortisol_vals = []
	cortisol_legend = []

	# Should be 8 rows, 12 columns: duplicates are in adjacent columns
	shapex, shapey = plate.shape
	data = []
	data_legend = []

	# 1. Average duplicate entries
	for i in np.arange(shapex):
		for j in np.arange(0,shapey,2):
			entry = plate_legend[i,j:j+2]
			vals = plate[i,j:j+2]
			if (entry[0]==entry[1])&(entry[0] != 'x'):
				data.append(np.mean(vals))
				data_legend.append(entry[0])
				
			elif (entry[0] != 'x'):
				data.append(vals[0])
				data_legend.append(entry[0])

	# 2. Subtract NSB well value from the others
	data = np.array(data)
	nsb_ind = data_legend.index('NSB')
	data = data - data[nsb_ind]

	# 3. Divide by the zero well value to get the pecentage bound B/Bo
	zero_ind = data_legend.index('zero')
	data /= data[zero_ind]

	# 4. Use 4 - parameter linear regression curve fit to determine concentrations of controls and saliva samples

	# y is cortisol ug/dL
	y = np.array([3., 1., 0.333, 0.111, 0.037, 0.012])
	# x is B/Bo
	x = np.array([data[data_legend.index('3.000_std')],data[data_legend.index('1.000_std')],data[data_legend.index('0.333_std')],data[data_legend.index('0.111_std')],data[data_legend.index('0.037_std')],data[data_legend.index('0.012_std')]])
	
	# Initial guess for parameters
	p0 = [0, 1, 1, 1]

	# Fit equation using least squares optimization
	plsq = leastsq(residuals, p0, args=(y, x))
	cortisol_vals = peval(data,plsq[0])

	xval = np.linspace(0.0,1.000,20)
	#plt.plot(peval(xval,plsq[0]),x,y,'o')
	plt.plot(peval(xval,plsq[0]), xval,'k-')
	plt.plot(cortisol_vals,data,'bo')
	plt.plot(y,x,'ro')
	plt.xlabel('cortisol (ug/dL)')
	plt.ylabel('B/Bo')
	#plt.xlim((0,3.1))
	#plt.ylim((0,1.1))
	plt.show()
	
	return cortisol_vals, data_legend
			

cort_vals1, data1_legend = process_plate(plate1, Plate1_legend)
cort_vals2, data2_legend = process_plate(plate2, Plate2_legend)
cort_vals3, data3_legend = process_plate(plate3, Plate3_legend)
cort_vals4, data4_legend = process_plate(plate4, Plate4_legend)

###Need to go through to extract the four data points for each session and make comparisons