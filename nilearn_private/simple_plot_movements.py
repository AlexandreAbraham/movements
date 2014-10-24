import numpy as np
from more_datasets import fetch_abide_movements

bunch = fetch_abide_movements()

aut_target = bunch.pheno["DX_GROUP"]

color = {1 : "r", 2 : "b"}

import pylab as pl
pl.figure()
pl.subplot(2, 3, 1)
for mov, targ in zip(bunch.movement, aut_target):
    #pl.plot(mov[:, :3], color[targ])
    pl.plot(mov[:, 0], color[targ])
    #pl.title("xyz movement")
    pl.title("x")
    #pl.xlabel("red: aut, blue: control")
pl.subplot(2, 3, 2)
for mov, targ in zip(bunch.movement, aut_target):
    #pl.plot(mov[:, :3], color[targ])
    pl.plot(mov[:, 1], color[targ])
    #pl.title("xyz movement")
    pl.title("y")
    #pl.xlabel("red: aut, blue: control")
pl.subplot(2, 3, 3)
for mov, targ in zip(bunch.movement, aut_target):
    #pl.plot(mov[:, :3], color[targ])
    pl.plot(mov[:, 2], color[targ])
    #pl.title("xyz movement")
    pl.title("z")
    #pl.xlabel("red: aut, blue: control")
pl.subplot(2, 3, 4)
for mov, targ in zip(bunch.movement, aut_target):
    #pl.plot(mov[:, :3], color[targ])
    pl.plot(mov[:, 3], color[targ])
    #pl.title("xyz movement")
    pl.title("roll?")
    #pl.xlabel("red: aut, blue: control")
pl.subplot(2, 3, 5)
for mov, targ in zip(bunch.movement, aut_target):
    #pl.plot(mov[:, :3], color[targ])
    pl.plot(mov[:, 4], color[targ])
    #pl.title("xyz movement")
    pl.title("pitch?")
    #pl.xlabel("red: aut, blue: control")
pl.subplot(2, 3, 6)
for mov, targ in zip(bunch.movement, aut_target):
    #pl.plot(mov[:, :3], color[targ])
    pl.plot(mov[:, 5], color[targ])
    #pl.title("xyz movement")
    pl.title("jaw?")
    pl.xlabel("red: aut, blue: control")

nparticipants = len(aut_target)



FS = np.zeros((nparticipants,24))
for i in xrange(6):
	values = [np.mean(mov[:,i]) for mov in bunch.movement]
	FS[:,i] = values
	mins = [np.min(mov[:,i]) for mov in bunch.movement]
	FS[:,i+6] = mins
	maxs = [np.max(mov[:,i]) for mov in bunch.movement]
	FS[:,i+12] = maxs
	stds = [np.std(mov[:,i]) for mov in bunch.movement]
	FS[:,i+18] = stds


inds_kids = bunch.pheno['AGE_AT_SCAN']<18.
inds_adults = np.logical_not(inds_kids)
inds_males = bunch.pheno['SEX']==1
inds_females = np.logical_not(inds_males)
inds_eyesopen = bunch.pheno['EYE_STATUS_AT_SCAN']==1
inds_eyesclosed = bunch.pheno['EYE_STATUS_AT_SCAN']==2
inds_nomeds = bunch.pheno['CURRENT_MED_STATUS']=='0'
inds_meds = bunch.pheno['CURRENT_MED_STATUS']=='1'
inds_right = bunch.pheno['HANDEDNESS_CATEGORY']=='L'


i = inds_right
FS = FS[i,:]
aut_target = aut_target[i,:]


FS = np.log10(np.abs(FS)+0.1)
from sklearn.preprocessing import StandardScaler

FS = StandardScaler().fit_transform(FS)

from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(aut_target, n_folds=5)

foldacc = cross_val_score(LinearSVC(),X=FS,y=aut_target,cv=cv,n_jobs=1)
acc = np.mean(foldacc)
print(acc)


# Also look at first 2 PCs of detrend coefs
"""
acquisition_lengths = [len(mov) for mov in bunch.movement]
min_acq_length = np.min(acquisition_lengths)
from features import trend_coef
trend_coefs = np.array([trend_coef(mov, axis=0).T.ravel() 
                        for mov in bunch.movement])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform((trend_coefs - trend_coefs.mean(0)) /
                        trend_coefs.std(0))

colors = [color[targ] for targ in aut_target]
pl.figure()
pl.scatter(pcs.T[0], pcs.T[1], c=colors, s=5, lw=0)
"""

#pl.show()

