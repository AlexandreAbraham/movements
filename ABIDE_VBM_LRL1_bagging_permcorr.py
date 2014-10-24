import os
import nibabel as nib
import numpy as np
import scipy
import datetime
import time
import sys
from numpy.random import shuffle

# sklearn imports
from sklearn.svm import LinearSVC, SVC
from sklearn import feature_selection
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from nisl.decoding import SearchLight
from nisl.utils import *
from sklearn.dummy import DummyClassifier
from nilearn import datasets, input_data
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# scikits
from scikits import bootstrap

"""
Constant section for user defined analysis definition
-----------------------------------------------------
"""
#sys.path.append('/data/Bzdok2/mypylibs/')
TARPATH = r"/Volumes/ABIDE/ABIDE"
VBM_PREF = "sm0wrp1"
TAR_ABIDE_DIRS = ['CALTECH', 'CMU', 'LEUVEN_1', 'MAX_MUN', 'USM']
n_bootstrap = 100
n_permutation = 1000
"""
-----------------------------------------------------
"""

f = open('ABIDE.csv','r')
header = f.readline()
niilabels = []
niipaths = []
for i in xrange(1112):
    l = f.readline()
    lineitems = l.split(';')
    sitename = lineitems[0]
    subnum = lineitems[1]
    subdiag = int(lineitems[2]) # 1=Autism, 2=control

    if (sitename in TAR_ABIDE_DIRS)==False:
        continue

    if sitename=='CALTECH':
        sitename = 'Caltech'
    elif sitename=='LEUVEN_1':
        sitename = 'Leuven'
    elif sitename=='MAX_MUN':
        sitename = 'MaxMun'

    base = sitename + '_' + subnum
    fname = VBM_PREF + 'mprage.nii'
    vbmpath = os.path.join(TARPATH, base, base, 'scans', 'anat', \
        'resources', 'NIfTI', 'files', fname)

    if os.path.exists(vbmpath)==False:
        print "Couldn't find: %s" % vbmpath
        continue

    niilabels.append(subdiag)
    niipaths.append(vbmpath)

#a = b
niilabels = np.array(niilabels)
niipaths = np.array(niipaths)

nsamples = len(niilabels)
print "total VBM images: %i" % nsamples
print "autist/control: %i/%i" % (np.sum(niilabels==1),np.sum(niilabels==2))

# construct feature space
MASK_TMP = "Grey10_VBM_ABIDE.nii"
mask_nifti = nib.load(MASK_TMP)
mask_data = mask_nifti.get_data()
maskshape = mask_data.shape
nx, ny, nz = maskshape[0], maskshape[1], maskshape[2]
mask = mask_data>0
mask_flat = mask.flatten()
nmask = int(mask_flat.sum())
#nmask = len(np.where(mask==True)[0])
nmask_total = nx*ny*nz
print "%i of %i total voxels in mask!" % \
    (nmask,nmask_total)
y = niilabels

X = np.zeros((nsamples,nmask))
print "Loading images..."
for s in xrange(nsamples):
    print "Loading %s..." % niipaths[s]
    cur_vbm = nib.load(niipaths[s])
    X[s,:] = np.nan_to_num(cur_vbm.get_data().flatten()[mask_flat])
print "done!"

# choose current penalty term
curC = 0.2

# shrink/classif
# C: The smaller it is, the bigger is the regularization.
print "Starting classification..."
LR = LogisticRegression(penalty='l1',dual=False,C=curC)

# first row will contain the ACTUAL results
coef_results = np.zeros((n_permutation+1,nmask))

PERM_DIR = '/f/projekt_ABIDE_VBM/PERM'
try:
    os.mkdir(PERM_DIR)
except:
    pass

for iperm in xrange(n_permutation):
    print("---------------------")
    print("Permutation: %i" % (iperm))
    for ibs in xrange(n_bootstrap):
        print "Classifying bootstrap sample %i/%i..." % (ibs+1,n_bootstrap)

        # create new bootstrap sample
        inds = np.random.randint(0,nsamples, size=nsamples)
        bsX = X[inds,:]
        bsy = y[inds]
        ss = StandardScaler()
        bsX = ss.fit_transform(bsX)

        # this is the permuation here
        if iperm>0:
            print('Labels are permuted!')
            np.random.shuffle(bsy)

        LR.fit(bsX, bsy)
        # add now, divide later to obtain the mean
        coef_results[iperm,:] += LR.coef_[0]

    coef_results[iperm,:] /= n_bootstrap # model-averaging

    # save support of current permutation
    space = np.zeros((nmask_total))
    space[mask_flat] = coef_results[iperm,:]
    space[space!=0] = 1
    space = space.reshape((nx,ny,nz))

    n = nib.Nifti1Image(space,mask_nifti.get_affine(),\
        header=mask_nifti.get_header())
    fname = "averagesupport_perm" + str(iperm+1) + ".nii"
    n.to_filename(os.path.join(PERM_DIR,fname))

# voxel-wise significance testing
print('Hypothesis-testing...')
alpha = 5.
sign_voxels = np.zeros((nmask))
for v in xrange(nmask): # for each voxel/coefficient
    realvalue = coef_results[0,v]
    permvalues = coef_results[1:,v]
    #sinds = np.argsort(permvalues)[::-1]
    alpha_half = alpha/2.
    up_cutoff = np.percentile(permvalues,100-alpha_half)
    lo_cutoff = np.percentile(permvalues,alpha_half)
    if (realvalue>up_cutoff) | (realvalue<lo_cutoff):
        sign_voxels[v] = 1
    else:
        sign_voxels[v] = 0
print('%i of %i initial non-zero coefficients were significant!' %\
    (sign_voxels.sum(),(coef_results[0,:]!=0).sum()))

# dump significance corrected map to disk
space = np.zeros((nmask_total))
space[mask_flat] = sign_voxels
space = space.reshape((nx,ny,nz))
n = nib.Nifti1Image(space,mask_nifti.get_affine(),\
    header=mask_nifti.get_header())
fname = "ABIDE_pred_voxels_alpha" + str(alpha) + ".nii"
n.to_filename(fname)

print "READY!"

