import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

class bme335_exam1():
	'''
	Class for CSV file of offline-sorted units recorded with TDT system. Units are offline-sorted with OpenSorter and then exported to a CSV files
	using OpenBrowser. Each entry is separated by a comma and new rows are indicated with a return character.
	'''

	def __init__(self, csv_file):
		self.filename =  csv_file
		# Read offline sorted data into pandas dataframe. Note that first row in csv file contains the columns headers.
		self.df = pd.read_csv(self.filename, sep=',', header = 0)
		self.student = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'Student'])))[2:72]
		self.prob1 = dict()
		self.prob2 = dict()
		self.prob3 = dict()
		self.prob4 = dict()
		self.prob1['a'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'1a'])))[2:72]
		self.prob1['b'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'1b'])))[2:72]
		self.prob1['c'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'1c'])))[2:72]
		self.prob1['d'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'1d'])))[2:72]
		self.prob1['e'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'1e'])))[2:72]
		self.prob1['f'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'1f'])))[2:72]
		self.prob2['a'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'2a'])))[2:72]
		self.prob2['b'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'2b'])))[2:72]
		self.prob2['c'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'2c'])))[2:72]
		self.prob2['d'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'2d'])))[2:72]
		self.prob2['e'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'2e'])))[2:72]
		self.prob3['a'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'3a'])))[2:72]
		self.prob3['b'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'3b'])))[2:72]
		self.prob3['c'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'3c'])))[2:72]
		self.prob3['d'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'3d'])))[2:72]
		self.prob3['e'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'3e'])))[2:72]
		self.prob4['a'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'4a'])))[2:72]
		self.prob4['b'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'4b'])))[2:72]
		self.prob4['c'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'4c'])))[2:72]
		self.prob4['d'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'4d'])))[2:72]
		self.prob4['e'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'4e'])))[2:72]
		self.prob4['f'] = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'4f'])))[2:72]
		self.total = np.ravel(np.array(pd.DataFrame(self.df, columns = [u'Total'])))[2:72]

	def subproblem_score_hist(self,problem):
		'''
		Compute the histogram of scores for each subproblem of the indicated problem, as well as
		a histogram of the total score for the problem.

		Input:
		- problem: integer corresponding to problem of interest
		'''
		if problem == 1:
			problem_dict = self.prob1
		elif problem == 2:
			problem_dict = self.prob2
		elif problem == 3:
			problem_dict = self.prob3
		elif problem == 4:
			problem_dict = self.prob4

		total_prob_score = np.zeros(len(self.total))
		bins = np.arange(-0.5,9,1)			# max score for any subproblem is 8
		bin_centers = np.arange(0,9,1)
		bins_all = np.arange(-0.5,26,1)		# max score for total problem is 25
		bin_all_centers = np.arange(0,26,1)

		plt.figure()
		for i,key in enumerate(problem_dict.keys()):
			scores = problem_dict[key]
			mean_score = np.nanmean(scores)
			max_score = np.nanmax(scores)
			total_prob_score += scores
			counts, bs = np.histogram(scores, bins)
			max_counts = np.nanmax(counts)
			plt.subplot(2,4,i+2 + int((i+1)/4))
			plt.bar(bin_centers,counts)
			plt.plot([mean_score, mean_score],[0, max_counts], 'r--', linewidth = 3,label = 'Mean = %f' % (mean_score))
			plt.plot([max_score, max_score], [0, max_counts], 'b--', linewidth = 3,label = 'Max score = %i' % (max_score))
			plt.title('Problem %i - %s' % (problem,key))
			plt.legend()
			plt.xlabel('Score')
			plt.ylabel('Count')

		counts, bs = np.histogram(total_prob_score,bins_all)
		mean_score = np.nanmean(total_prob_score)
		max_counts = np.nanmax(counts)
		plt.subplot(2,4,1)
		plt.bar(bin_all_centers,counts)
		plt.plot([mean_score, mean_score],[0, max_counts], 'r--', linewidth = 3,label = 'Mean = %f' % (mean_score))
		plt.legend()
		plt.title('Problem %i' % (problem))
		plt.xlabel('Total Score')
		plt.ylabel('Count')

		plt.show()


		return

	def adjust_scores(self):
		'''
		Method that compares scores for all 4 problems versus selecting the best 3 of 4 problems.
		'''
		scores = np.zeros((len(self.total),4))
		

		for problem in np.arange(1,5):

			total_prob_score = np.zeros(len(self.total))

			if problem == 1:
				problem_dict = self.prob1
			elif problem == 2:
				problem_dict = self.prob2
			elif problem == 3:
				problem_dict = self.prob3
			elif problem == 4:
				problem_dict = self.prob4


			for i,key in enumerate(problem_dict.keys()):
				scores_p = problem_dict[key]
				total_prob_score = total_prob_score + np.array(scores_p)

			scores[:,problem-1] = total_prob_score

			

		return 


def regression_example(n):
	'''
	Method for creating an example of linear regression for class. Fake data is used that relates
	undergrad GPA to salary post-graduation.

	Input:
	- n: int; sample size of the fake data used

	Output:

	'''
	fig_address = 'C:/Users/ss45436/Box/UT Austin/Courses/BME 335 - Fall 2019/Practice/Week 11/'

	gpa = 3 + np.random.rand(n)
	salary = (gpa + np.random.rand(n))*22000	

	# Plot all data
	plt.plot(gpa,salary,'o',markersize = 8, color = 'r')
	plt.xlabel('GPA', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.ylabel('Salary', fontsize = 24)
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary.png',bbox_inches='tight',dpi = 500)	
	plt.show()

	# Plot small subsampling of data with all data
	subn = int(n/10)
	plt.plot(gpa,salary,'o',markersize = 8, color = 'r')
	plt.plot(gpa[:subn],salary[:subn],'o',markersize = 8, color = 'b')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_subsample1.png',bbox_inches='tight',dpi = 500)
	plt.show()

	# Plot subsampling of data alone
	plt.plot(gpa[:subn],salary[:subn],'o',markersize = 8, color = 'b')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_subsample1_only.png',bbox_inches='tight',dpi = 500)
	plt.show()

	# Plot subsample with it's linear regression line
	gpa_sub = gpa[:subn].reshape(subn,1)
	salary_sub = salary[:subn].reshape(subn,1)
	salary_model = linear_model.LinearRegression()
	salary_model.fit(gpa_sub,salary_sub)
	gpa = gpa.reshape(n,1)
	salary_pred = salary_model.predict(gpa)	# do prediction with all gpa data so that prediction line spans the x-axis

	plt.plot(gpa_sub,salary_sub,'o',markersize = 8, color = 'b')
	plt.plot(gpa,salary_pred, linewidth = 3, color = 'b')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_subsample1_lr.png',bbox_inches='tight',dpi = 500)
	plt.show()

	# Plot subsample with it's linear regression line, plus the original population added back
	plt.plot(gpa,salary,'o',markersize = 8, color = (1,0,0,0.2))
	plt.plot(gpa_sub,salary_sub,'o',markersize = 8, color = 'b')
	plt.plot(gpa,salary_pred, linewidth = 3, color = 'b')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_all_subsample1_lr.png',bbox_inches='tight',dpi = 500)
	plt.show()

	# Plot subsample with linear regression line, but highlight just one data point
	# in order to go through how the line should be determine

	plt.plot(gpa_sub,salary_sub,'o',markersize = 8, color = (0,0,1,0.2))
	plt.plot(gpa_sub[0],salary_sub[0],'o',markersize = 8, color = (0,0,1,1))
	plt.plot(gpa,salary_pred, linewidth = 3, color = 'b')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_subsample1_single_lr.png',bbox_inches='tight',dpi = 500)
	plt.show()

	# Plot second subsample with it's linear regression line, plus the original population added back
	plt.plot(gpa,salary,'o',markersize = 8, color = (1,0,0,0.2))
	plt.plot(gpa[subn:2*subn],salary[subn:2*subn],'o',markersize = 8, color = 'g')
	plt.plot(gpa_sub,salary_sub,'o',markersize = 8, color = 'b')
	plt.plot(gpa,salary_pred, linewidth = 3, color = 'b')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_subsample2.png',bbox_inches='tight',dpi = 500)
	plt.show()

	# Plot second subsample with it's linear regression line
	gpa_sub2 = gpa[subn:2*subn].reshape(subn,1)
	salary_sub2 = salary[subn:2*subn].reshape(subn,1)
	salary_model2 = linear_model.LinearRegression()
	salary_model2.fit(gpa_sub2,salary_sub2)
	salary_pred2 = salary_model2.predict(gpa)	# do prediction with all gpa data so that prediction line spans the x-axis

	# Plot second subsample with it's linear regression line, plus the original population added back
	plt.plot(gpa,salary,'o',markersize = 8, color = (1,0,0,0.2))
	plt.plot(gpa_sub2,salary_sub2,'o',markersize = 8, color = 'g')
	plt.plot(gpa_sub,salary_sub,'o',markersize = 8, color = 'b')
	plt.plot(gpa,salary_pred, linewidth = 3, color = 'b')
	plt.plot(gpa,salary_pred2, linewidth = 3, color = 'g')
	plt.xlabel('GPA', fontsize = 24)
	plt.ylabel('Salary', fontsize = 24)
	plt.xlim((3,4))
	plt.ylim((3*22000,5*22000))
	plt.tick_params(labelsize = 20)
	plt.savefig(fig_address + 'GPAvsSalary_subsample2_lr.png',bbox_inches='tight',dpi = 500)
	plt.show()
	return