import numpy as np

airport_rot = np.array([74.40, -107.06, 23.45])	# values are in degrees, [AP, ML, DV]
test_rot = np.array([0,45,0])

'''
This code says that if we are at (0,0,0) and lower 10 mm, the final position
is: (-8.50, -0.37, -5.26)
'''

def compute_electrode_movement(rot_array, dist_traveled):
	"""Method to compute how much the electrode has moved relative to its starting point.
	Currently the starting point is treated as (0,0,0) and the distance traveled is the 
	amount moved in the z-axis (DV). 

	Parameters:
	rot_array -- (3,) array with values of rotations in degrees
	dist_traveled -- int represented the total amount the electrode was moved in mm
	Output:
	pos_array -- (3,) array with values in mm of new position
	"""

	# Convert degrees to radians
	rot_array = rot_array * np.pi/180

	# Define rotation matrices
	Rz = np.array([[np.cos(rot_array[2]), -np.sin(rot_array[2]), 0],
				   [np.sin(rot_array[2]), np.cos(rot_array[2]), 0],
				   [0, 0, 1]])
	Ry = np.array([[np.cos(rot_array[1]), 0, np.sin(rot_array[1])],
				   [0 , 1, 0],
				   [-np.sin(rot_array[1]), 0, np.cos(rot_array[1])]])
	Rx = np.array([[1, 0, 0],
				   [0 , np.cos(rot_array[0]), -np.sin(rot_array[0])],
				   [0, np.sin(rot_array[0]), np.cos(rot_array[0])]])
	print('R:', np.matmul(Rz, np.matmul(Ry, Rx)))

	# Define point to be translated
	pos = np.array([0, 0, dist_traveled])

	pos_array = np.matmul(np.matmul(Rz, np.matmul(Ry, Rx)),pos)

	return Rx, Ry, Rz, pos_array

def compute_electrode_position(starting_position, rot_array, dist_traveled):
	"""Method to compute how much the electrode has moved relative to its starting point.
	The starting point is given as an input (starting_position) and the distance traveled is the 
	amount moved in the z-axis (DV). 

	Parameters:
	starting_position -- (3,) array with values in mm for AP, ML, and DV starting coordinates
	rot_array -- (3,) array with values of rotations in degrees
	dist_traveled -- int represented the total amount the electrode was moved in mm
	Output:
	pos_array -- (3,) array with values in mm of new position
	"""

	# Compute the amount moved in each axis
	pos_array = compute_electrode_movement(rot_array, dist_traveled)
	# Update position from starting position
	pos_array += starting_position

	return pos_array