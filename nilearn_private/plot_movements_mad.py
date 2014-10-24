import copy
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from more_datasets import fetch_abide_movements

plt.close('all')
plt.style.use('ggplot')


def eval_against_dumm(FS, aut_target, myclf, folder):
	real_acc = []
	dummy1_acc, dummy2_acc, dummy3_acc = [], [], []
	clf = copy.deepcopy(myclf)
	for train_index, test_index in folder:
		clf.fit(FS[train_index, :],aut_target[train_index])
		labels = np.asarray(clf.predict(FS[test_index, :]))
		acc = np.mean(aut_target[test_index] == labels)
		real_acc.append(acc)

		clf = DummyClassifier("stratified")
		clf.fit(FS[train_index, :], aut_target[train_index])
		labels = np.asarray(clf.predict(FS[test_index, :]))
		acc = np.mean(aut_target[test_index] == labels)
		dummy1_acc.append(acc)

		clf = DummyClassifier("most_frequent")
		clf.fit(FS[train_index, :], aut_target[train_index])
		labels = np.asarray(clf.predict(FS[test_index, :]))
		acc = np.mean(aut_target[test_index] == labels)
		dummy2_acc.append(acc)

		clf = DummyClassifier("uniform")
		clf.fit(FS[train_index, :], aut_target[train_index])
		labels = np.asarray(clf.predict(FS[test_index, :]))
		acc = np.mean(aut_target[test_index] == labels)
		dummy3_acc.append(acc)

	return np.mean(real_acc), np.mean(dummy1_acc), np.mean(dummy2_acc),\
		np.mean(dummy3_acc)

bunch = fetch_abide_movements()
aut_target = bunch.pheno["DX_GROUP"]
nparticipants = len(aut_target)

FS = np.zeros((nparticipants,24))
for i in xrange(6):
	values = [np.mean(mov[:,i]) for mov in bunch.movement]
	FS[:, i] = values
	mins = [np.min(mov[:,i]) for mov in bunch.movement]
	FS[:, i + 6] = mins
	maxs = [np.max(mov[:,i]) for mov in bunch.movement]
	FS[:, i + 12] = maxs
	stds = [np.std(mov[:,i]) for mov in bunch.movement]
	FS[:, i + 18] = stds

inds_kids = bunch.pheno['AGE_AT_SCAN'] < 18.
inds_adults = np.logical_not(inds_kids)
inds_males = bunch.pheno['SEX'] == 1
inds_females = np.logical_not(inds_males)
inds_eyesopen = bunch.pheno['EYE_STATUS_AT_SCAN'] == 1
inds_eyesclosed = bunch.pheno['EYE_STATUS_AT_SCAN'] == 2
inds_nomeds = bunch.pheno['CURRENT_MED_STATUS'] == '0'
inds_meds = bunch.pheno['CURRENT_MED_STATUS'] ==' 1'
inds_left = bunch.pheno['HANDEDNESS_CATEGORY'] == 'L'
inds_right = np.logical_not(inds_left)

i = np.ones(nparticipants, dtype=np.bool)
FS = FS[i, :]
aut_target = aut_target[i]

aut_target[aut_target == 2] = 0
nparticipants = np.sum(i)
naut = np.sum(aut_target)
print "#participants: %i" % nparticipants
print "autist:non-autist = %i/%i" % (naut, nparticipants - naut)

n_folds = 10
myclf = Pipeline([('SS', StandardScaler()), ('LR', LogisticRegression(penalty='l1'))])

bar_width = .25
n_clf = 4
bar_block_width = ((n_clf + 1) * bar_width)
group_width = (n_clf + 1) * bar_width
bool_labeled = False
tick_position = 0
plt.figure()
xlabels_pos = []
xlabels = []
for i_ana in range(14):

	if i_ana == 0:
		cur_FS = FS
		cur_target = aut_target[i]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('MAIN EFFECT %i%%' % (a * 100))
	elif i_ana == 1:
		cur_FS = FS[:, 0:6]
		cur_target = aut_target[i]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only mean() %i%%' % (a * 100))
	elif i_ana == 2:
		cur_FS = FS[:, 6:12]
		cur_target = aut_target[i]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only min() %i%%' % (a * 100))
	elif i_ana == 3:
		cur_FS = FS[:, 12:18]
		cur_target = aut_target[i]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only max() %i%%' % (a * 100))
	elif i_ana == 4:
		cur_FS = FS[:, 18:24]
		cur_target = aut_target[i]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only std() %i%%' % (a * 100))
	elif i_ana == 5:
		cur_FS = FS[inds_nomeds, :]
		cur_target = aut_target.copy()[inds_nomeds]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('no medis %i%%' % (a * 100))
	elif i_ana == 6:
		cur_FS = FS[inds_kids, :]
		cur_target = aut_target.copy()[inds_kids]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only kids %i%%' % (a * 100))
	elif i_ana == 7:
		cur_FS = FS[inds_adults, :]
		cur_target = aut_target.copy()[inds_adults]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only adults %i%%' % (a * 100))
	elif i_ana == 8:
		cur_FS = FS[inds_males, :]
		cur_target = aut_target.copy()[inds_males]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only males %i%%' % (a * 100))
	elif i_ana == 9:
		cur_FS = FS[inds_females, :]
		cur_target = aut_target.copy()[inds_females]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('only females %i%%' % (a * 100))
	elif i_ana == 10:
		cur_FS = FS[inds_eyesopen, :]
		cur_target = aut_target.copy()[inds_eyesopen]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('eyes closed %i%%' % (a * 100))
	elif i_ana == 11:
		cur_FS = FS[inds_eyesclosed, :]
		cur_target = aut_target.copy()[inds_eyesclosed]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('eyes closed %i%%' % (a * 100))
	elif i_ana == 12:
		cur_FS = FS[inds_left, :]
		cur_target = aut_target.copy()[inds_left]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('left-handed %i%%' % (a * 100))
	elif i_ana == 13:
		cur_FS = FS[inds_right, :]
		cur_target = aut_target.copy()[inds_right]
		folder = StratifiedKFold(cur_target, n_folds=n_folds)
		a, b, c, d = eval_against_dumm(cur_FS, cur_target, myclf, folder)
		score_list = [a, b, c, d]
		xlabels.append('right-handed %i%%' % (a * 100))
	else:
		pass

	from scipy.stats import ttest_ind
	sign_string = ''
	for imov in range(6):
		values_a = cur_FS[cur_target == 0, imov]
		values_b = cur_FS[cur_target == 1, imov]
		maxlen = min(len(values_a), len(values_b))
		values_a = values_a[:maxlen]
		values_b = values_b[:maxlen]
		assert len(values_a) == len(values_b)
		_, p = ttest_ind(values_a, values_b)
		print 'p-value: %.2f' % p
		if p < 0.05:
			sign_string += '*'
		else:
			sign_string += '-'
	print sign_string
	plt.text(0.1 + i_ana * 1.27, 0.90, sign_string, fontsize=10)

	# print plot
	print "REAL ACCURACY MEAN: %.2f" % a
	print "DUMMY1 ACCURACY MEAN: %.2f" % b
	print "DUMMY2 ACCURACY MEAN: %.2f" % c
	print "DUMMY3 ACCURACY MEAN: %.2f" % d

	# bar plot
	mycolors = ['#002645', '#0f5765', '#568681', '#8c9d88']  # Adobe kooler
	for ii, (key, color) in enumerate(zip(['LogRegr(L1)', 'Dummy1', 'Dummy2', 'Dummy3'], mycolors)):
		handle = plt.bar(tick_position, score_list[ii], label=key,
			width=bar_width, color=color)
		tick_position += bar_width
	tick_position += bar_width

	if not bool_labeled:
		plt.legend(loc='lower right')
		bool_labeled = True

	if xlabels_pos == []:
		xlabels_pos = [2 * bar_width]
	else:
		xlabels_pos.append(xlabels_pos[-1] + group_width)	

x1,x2,y1,y2 = plt.axis()
plt.axis((x1, x2, .0, 1.0))
xlabels_pos = [item - (group_width + 0.25) for item in xlabels_pos]  # readjust cosmetics
plt.xticks(list(xlabels_pos), list(xlabels), rotation=25)
plt.text(2.8, 0.85, '(-/*: (non-)significant t-test of each movement parameter)')
plt.yticks(np.linspace(.0, 1.0, 11))
plt.title(
	'ABIDE: classifying autism/healthy PURELY based on 6 movement parameters\n'
	'Michael Eickenberg, Alexandre Abraham, Danilo Bzdok\n'
	'n=%i participants: %i versus %i (autism versus healthy)' % (nparticipants,
		naut, nparticipants - naut))
plt.tight_layout()
plt.show()
plt.savefig('autism_mov.png', dpi=400)



