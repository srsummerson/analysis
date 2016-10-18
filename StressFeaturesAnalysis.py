import scipy as sp
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from basicAnalysis import computeFisherScore

filename = 'Mario20161013'
block_num = 1
print filename
#TDT_tank = '/backup/subnetsrig/storage/PowerFeatures/'+filename
TDT_tank = '/home/srsummerson/storage/PowerFeatures/'
pf_filename = TDT_tank + filename+'_b'+str(block_num)+'_PowerFeatures.mat'

'''
Load data.
Note that power feat is a dictionary with one entry per trial. Each entry is a matrix
of C x K entries, where C is the number of channels and K is the number of features.
'''
power_feat = dict()
sp.io.loadmat(pf_filename, power_feat)

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
feat_stress_std = np.nanmean(features_stress, axis = 0)

plt.figure()
plt.plot(range(C*K), feat_reg_avg, 'b')
plt.plot(range(C*K), feat_reg_avg - feat_reg_std, 'b--')
plt.plot(range(C*K), feat_reg_avg + feat_reg_std, 'b--')
plt.plot(range(C*K), feat_stress_avg, 'r')
plt.plot(range(C*K), feat_stress_avg - feat_stress_std, 'r--')
plt.plot(range(C*K), feat_stress_avg + feat_stress_std, 'r--')
plt.show()


'''
Compute Fisher scores
'''
class_ass = np.zeros(num_trials)
class_ass[100:] = 1
nb_classes = 2
Fscores = computeFisherScore(features_all, class_ass, nb_classes)
Fscores = np.ravel(Fscores)
top_scores = np.argsort(Fscores)[-50:]

plt.figure()
plt.subplot(211)
plt.plot(range(C*K), Fscores,'b')
plt.plot(top_scores, Fscores[top_scores],'m*', linestyle=None)
plt.subplot(212)
plt.plot(range(C*K), Fscores.sort())
plt.show()



