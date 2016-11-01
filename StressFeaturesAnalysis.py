import scipy as sp
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from basicAnalysis import computeFisherScore
import sys


'''
Features are organized as 8 features per channel, so that the first 8 features are for channel 1, the next 8 features are for channel 2, etc.
For the 8 features, the first 4 features are for the center state and the next 4 features are for the reward state.
For each 4 state features, they are the power in the first power band, second power band, third power band, and fourth power band. 

For example, the features are organized as [Ch1-Center-B1, Ch1-Center-B2, Ch1-Center-B3, Ch-Center-B4, Ch1-Reward-B1, Ch1-Reward-B2, etc.]

'''

filename = [['Mario20160613', 1], 
			['Mario20160614', 1], 
			#['Mario20160707', 1], 
			#['Mario20160709', 1],
			['Mario20160712', 1], 
			['Mario20160714', 1], 
			['Mario20160715', 2], 
			['Mario20160716', 1], 
			['Mario20161006', 1],
			['Mario20161010', 1],
			['Mario20161012', 1],  
			['Mario20161013', 1]]

OFC_indices = np.array([20, 4, 18, 2, 28, 12, 26, 10, 27, 11, 25, 9]) - 1
vmPFC_indices = np.array([30, 14, 15, 29, 13, 6, 22]) - 1
preSMA_indices = np.array([60, 44, 58, 61, 45, 59, 7, 21, 5, 24, 8, 22, 32, 16, 30, 31]) - 1

num_top_scores = 200

Ftop_scores = np.zeros([len(filename), num_top_scores])

for i, name in enumerate(filename):
	print name[0]
	#TDT_tank = '/backup/subnetsrig/storage/PowerFeatures/'+filename
	TDT_tank = '/home/srsummerson/storage/PowerFeatures/'
	pf_filename = TDT_tank + name[0] +'_b'+str(name[1])+'_PowerFeatures.mat'

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
	features_reg_OFC = np.zeros([100,len(OFC_indices)*K])
	features_stress = np.zeros([num_trials - 100, C*K])
	features_all = np.zeros([num_trials, C*K])

	for trial in range(num_trials):
		if trial < 100:
			features_reg[trial,:] = power_feat[str(trial)].flatten()
			features_reg_OFC[trial,:] = power_feat[str(trial)][OFC_indices,:].flatten()
		else:
			features_stress[trial - 100,:] = power_feat[str(trial)].flatten()
		features_all[trial, :] = power_feat[str(trial)].flatten()

	'''
	Load stim data.
	Note that power feat is a dictionary with one entry per trial. Each entry is a matrix
	of C x K entries, where C is the number of channels and K is the number of features.
	'''
	power_feat = dict()
	sp.io.loadmat(pf_filename, power_feat)
	print "Loaded stim data."
	power_feat_keys = power_feat.keys()
	num_trials = len(power_feat_keys) - 3
	C, K = power_feat['0'].shape

	features_stim = np.zeros([num_trials, C*K])

	for trial in range(num_trials):
		features_stim[trial, :] = power_feat[str(trial)].flatten()

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

	Fscores_hist, bins = np.histogram(Fscores, 25)

	plt.figure()
	plt.subplot(211)
	plt.plot(range(C*K), Fscores,'b')
	plt.plot(top_scores, Fscores[top_scores], linewidth=0, marker = '*', color = 'm')
	plt.xlim((0,C*K))
	plt.ylabel('F-score')
	plt.xlabel('Feature Number')
	plt.subplot(212)
	plt.plot(bins[:-1], Fscores_hist, 'b')
	plt.xlabel('F-score')
	plt.ylabel('Frequency')
	plt.show()

	plt.figure()
	plt.subplot(211)
	plt.plot(range(C*K), Fscores,'b')
	plt.plot(top_scores, Fscores[top_scores], linewidth=0, marker = '*', color = 'm')
	plt.xlim((0,C*K))
	plt.ylabel('F-score')
	plt.xlabel('Feature Number')
	plt.subplot(212)
	Fscores_sorted = sorted(Fscores, reverse = True)  # sort largest to smallest
	plt.plot(range(C*K), Fscores_sorted, 'b')
	plt.show()


	'''
	Compute correlation between features 
	# Change imshow so that range is always [-1, 1]
	'''
	print "Computing correlation"
	R_reg = np.corrcoef(features_reg.T)
	R_stress = np.corrcoef(features_stress.T)
	R_stim = np.corrcoef(features_stim.T)
	delta_R = R_stress - R_reg
	fig = plt.figure()
	plt.subplot(131)
	plt.title('Regular')
	ax = plt.imshow(R_reg, aspect='auto', vmin = -1.0, vmax = 1.0, 
				extent = [0,C*K,C*K,0])
	yticks = np.arange(0, C*K, 100)
	yticklabels = ['{0:.2f}'.format(range(C*K)[i]) for i in yticks]
	plt.yticks(yticks, yticklabels)
	fig.colorbar(ax)

	plt.subplot(132)
	plt.title('Stress')
	ax = plt.imshow(R_stress, aspect='auto', vmin = -1.0, vmax = 1.0, 
				extent = [0,C*K,C*K,0])
	yticks = np.arange(0, C*K, 100)
	yticklabels = ['{0:.2f}'.format(range(C*K)[i]) for i in yticks]
	plt.yticks(yticks, yticklabels)
	fig.colorbar(ax)

	plt.subplot(133)
	plt.title('Stim')
	ax = plt.imshow(R_stim, aspect='auto', vmin = -1.0, vmax = 1.0, 
				extent = [0,C*K,C*K,0])
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
	plt.show()



plt.close('all')
max_top_score = int(np.max(Ftop_scores))
Count_top_scores = np.zeros([len(filename), max_top_score + 1])
for j in range(len(filename)):
	scores = [int(score) for score in Ftop_scores[j,:]]
	Count_top_scores[j,scores] = 1

counts_top_scores = np.sum(Count_top_scores, axis = 0)


plt.figure()
plt.plot(np.arange(max_top_score+1), counts_top_scores)
plt.show()

# features that were top scoring F-scores for at least 4 days
common_features = np.ravel(np.nonzero(np.greater(counts_top_scores, 3.5)))


# create feature list
chan = np.arange(160) + 1
feat_labels = []
for ch in chan:
	feat_labels.append('CH'+str(ch)+'_Center_B1')
	feat_labels.append('CH'+str(ch)+'_Center_B2')
	feat_labels.append('CH'+str(ch)+'_Center_B3')
	feat_labels.append('CH'+str(ch)+'_Center_B4')
	feat_labels.append('CH'+str(ch)+'_Reward_B1')
	feat_labels.append('CH'+str(ch)+'_Reward_B2')
	feat_labels.append('CH'+str(ch)+'_Reward_B3')
	feat_labels.append('CH'+str(ch)+'_Reward_B4')

best_features = [feat_labels[f] for f in common_features]
print best_features

orig_stdout = sys.stdout
f = file('/home/srsummerson/code/analysis/StressPlots/best_features.txt', 'w')
sys.stdout = f

print best_features

sys.stdout = orig_stdout
f.close()