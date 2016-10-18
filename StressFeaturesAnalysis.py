import scipy as sp
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from basicAnalysis import computeFisherScore

filename = ['Mario20161013']
block_num = [1]

num_top_scores = 50

Ftop_scores = np.zeros([len(filename), num_top_scores])

for i, name in enumerate(filename):
	print filename
	#TDT_tank = '/backup/subnetsrig/storage/PowerFeatures/'+filename
	TDT_tank = '/home/srsummerson/storage/PowerFeatures/'
	pf_filename = TDT_tank + nam e+'_b'+str(block_num[i])+'_PowerFeatures.mat'

	'''
	Load data.
	Note that power feat is a dictionary with one entry per trial. Each entry is a matrix
	of C x K entries, where C is the number of channels and K is the number of features.
	'''
	power_feat = dict()
	sp.io.loadmat(pf_filename, power_feat)
	print "Loaded data."
	power_feat_keys = power_feat.keys()
	num_trials = len(power_feat_keys) - 3
	C, K = power_feat['0'].shape

	'''
	Arrange data into feature matrix. 
	Matrix will be of size N x C*K, where N is the number of trials and C*K is the
	total number of features. Features are organized such that the first K features are
	from channel one, the next K features are from channel two, and so on.
	'''
	features_reg = np.zeros([100, C*K])
	features_stress = np.zeros([num_trials - 100, C*K])
	features_all = np.zeros([num_trials, C*K])

	for trial in range(num_trials):
		if trial < 100:
			features_reg[trial,:] = power_feat[str(trial)].flatten()
		else:
			features_stress[trial - 100,:] = power_feat[str(trial)].flatten()
		features_all[trial, :] = power_feat[str(trial)].flatten()

	'''
	Compute basic statistics
	'''
	feat_reg_avg = np.nanmean(features_reg, axis = 0)
	feat_reg_std = np.nanstd(features_reg, axis = 0)
	feat_stress_avg = np.nanmean(features_stress, axis = 0)
	feat_stress_std = np.nanstd(features_stress, axis = 0)

	plt.figure()
	plt.plot(range(C*K), feat_reg_avg, 'b')
	plt.plot(range(C*K), feat_reg_avg - feat_reg_std, 'b--')
	plt.plot(range(C*K), feat_reg_avg + feat_reg_std, 'b--')
	plt.plot(range(C*K), feat_stress_avg, 'r')
	plt.plot(range(C*K), feat_stress_avg - feat_stress_std, 'r--')
	plt.plot(range(C*K), feat_stress_avg + feat_stress_std, 'r--')
	#plt.show()


	'''
	Compute Fisher scores
	'''
	print "Compute Fisher score."
	class_ass = np.zeros(num_trials)
	class_ass[100:] = 1
	nb_classes = 2
	Fscores = computeFisherScore(features_all, class_ass, nb_classes)
	Fscores = np.ravel(Fscores)
	top_scores = np.argsort(Fscores)[-num_top_scores:]
	Ftop_scores[i,:] = top_scores

	plt.figure()
	plt.subplot(211)
	plt.plot(range(C*K), Fscores,'b')
	plt.plot(top_scores, Fscores[top_scores], linewidth=0, marker = '*', color = 'm')
	plt.subplot(212)
	Fscores_sorted = sorted(Fscores, reverse = True)  # sort largest to smallest
	plt.plot(range(C*K), Fscores_sorted, 'b')
	#plt.show()


	'''
	Compute correlation between features 
	# Change imshow so that range is always [-1, 1]
	'''
	print "Compting correlation"
	R_reg = np.corrcoef(features_reg.T)
	R_stress = np.corrcoef(features_stress.T)
	delta_R = R_stress - R_reg
	fig = plt.figure()
	plt.subplot(121)
	plt.title('Regular')
	ax = plt.imshow(R_reg, aspect='auto', vmin = -1.0, vmax = 1.0, 
				extent = [0,C*K,0, C*K])
	yticks = np.arange(0, C*K, 100)
	yticklabels = ['{0:.2f}'.format(range(C*K)[i]) for i in yticks]
	plt.yticks(yticks, yticklabels)
	fig.colorbar(ax)

	plt.subplot(122)
	plt.title('Stress')
	ax = plt.imshow(R_stress, aspect='auto', vmin = -1.0, vmax = 1.0, 
				extent = [0,C*K,0, C*K])
	yticks = np.arange(0, C*K, 100)
	yticklabels = ['{0:.2f}'.format(range(C*K)[i]) for i in yticks]
	plt.yticks(yticks, yticklabels)
	fig.colorbar(ax)
	"""
	plt.subplot(133)
	plt.title('Difference (Stress - Regular')
	ax = plt.imshow(delta_R, aspect='auto', origin='lower', 
				extent = [0,C*K,0, C*K])
	yticks = np.arange(0, C*K, 100)
	yticklabels = ['{0:.2f}'.format(range(C*K)[i]) for i in yticks]
	plt.yticks(yticks, yticklabels)
	fig.colorbar(ax)
	"""
	#plt.show()


max_top_score = np.max(Ftop_scores)
Count_top_scores = np.zeros([len(filename), max_top_score])
for j in range(len(filename)):
	Count_top_scores[j,Ftop_scores[j,:]] = 1

counts_top_scores = np.sum(Count_top_scores, axis = 0)
"""
plt.figure()
plt.plot(range(max_top_score), counts_top_scores)
plt.show()
"""