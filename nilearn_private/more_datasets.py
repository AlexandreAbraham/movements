""" Additional datasets for Nilearn

Function to fetch abide movements
"""

import os
import collections

import numpy as np
from sklearn.datasets.base import Bunch

from nilearn.datasets import _fetch_files, _get_dataset_dir


def _filter_column(array, col, criteria):
    """ Return index array matching criteria

    Parameters
    ----------

    array: numpy array with columns
        Array in which data will be filtered

    col: string
        Name of the column

    criteria: integer (or float), pair of integers, string or list of these
        if integer, select elements in column matching integer
        if a tuple, select elements between the limits given by the tuple
        if a string, select elements that match the string
    """
    # Raise an error if the column does not exist
    array[col]

    if not isinstance(criteria, basestring) and \
            not isinstance(criteria, tuple) and \
            isinstance(criteria, collections.Iterable):
        filter = np.zeros(array.shape, dtype=np.bool)
        for criterion in criteria:
            filter = np.logical_or(filter,
                        _filter_column(array, col, criterion))
        return filter

    if isinstance(criteria, tuple):
        if len(criteria) != 2:
            raise ValueError("An interval must have 2 values")
        if criteria[0] is None:
            return array[col] <= criteria[1]
        if criteria[1] is None:
            return array[col] >= criteria[0]
        filter = array[col] <= criteria[1]
        return np.logical_and(filter, array[col] >= criteria[0])

    return array[col] == criteria


def _filter_columns(array, filters):
    filter = np.ones(array.shape, dtype=np.bool)
    for column in filters:
        filter = np.logical_and(filter,
                _filter_column(array, column, filters[column]))
    return filter


def fetch_abide_movements(data_dir=None, n_subjects=None, sort=True, verbose=0,
        **kwargs):
    """ Load ABIDE dataset

    The ABIDE dataset must be installed in the data_dir (or NILEARN_DATA env)
    into an 'ABIDE' folder. The Phenotypic information file should be in this
    folder too.

    Parameters
    ----------

    SUB_ID: list of integers in [50001, 50607], optional
        Ids of the subjects to be loaded.

    DX_GROUP: integer in {1, 2}, optional
        1 is autism, 2 is control

    DSM_IV_TR: integer in [0, 4], optional
        O is control, 1 is autism, 2 is Asperger, 3 is PPD-NOS,
        4 is Asperger or PPD-NOS

    AGE_AT_SCAN: float in [6.47, 64], optional
        Age of the subject

    SEX: integer in {1, 2}, optional
        1 is male, 2 is female

    HANDEDNESS_CATEGORY: string in {'R', 'L', 'Mixed', 'Ambi'}, optional
        R = Right, L = Left, Ambi = Ambidextrous

    HANDEDNESS_SCORE: integer in [-100, 100], optional
        Positive = Right, Negative = Left, 0 = Ambidextrous
    """

    name_csv = 'Phenotypic_V1_0b.csv'
    dataset_dir = _get_dataset_dir('abide_movements', data_dir=data_dir)
    #path_csv = _fetch_files('abide_movements', [(name_csv,
    #    'file:' + os.path.join('dataset', name_csv), {})],
    #                        data_dir=data_dir)[0]

    path_csv = _fetch_files('abide_movements', [(name_csv,
        'file:' + os.path.join('dataset', name_csv), {})])[0]

    # The situation is a bit complicated here as we will load movements
    # depending on whether they are provided or not. We load a file just to
    # download the movements files.

    sort_csv = _fetch_files('abide_movements', [('sort.csv',
        'file:' + os.path.join('dataset', 'abide_movements.tgz'), {'uncompress':
            True})])[0]

    sort_csv = np.genfromtxt(sort_csv, delimiter=',', dtype=None)
    pheno = np.genfromtxt(path_csv, names=True, delimiter=',', dtype=None)

    if sort:
        pheno = pheno[_filter_columns(pheno, {
            'SUB_ID': sort_csv[sort_csv['f2'] == 1]['f1']})]

    filter = _filter_columns(pheno, kwargs)
    pheno = pheno[filter]

    site_id_to_path = {
            'CALTECH': 'Caltech',
            'CMU': 'CMU',
            'KKI': 'KKI',
            'LEUVEN_1': 'Leuven',
            'LEUVEN_2': 'Leuven',
            'MAX_MUN': 'MaxMun',
            'NYU': 'NYU',
            'OHSU': 'OHSU',
            'OLIN': 'Olin',
            'PITT': 'Pitt',
            'SBL': 'SBL',
            'SDSU': 'SDSU',
            'STANFORD': 'Stanford',
            'TRINITY': 'Trinity',
            'UCLA_1': 'UCLA',
            'UCLA_2': 'UCLA',
            'UM_1': 'UM',
            'UM_2': 'UM',
            'USM': 'USM',
            'YALE': 'Yale'
    }

    # Get the files for all remaining subjects
    movement = []
    filter = np.zeros(pheno.shape, dtype=np.bool)
    for i, (site, id) in enumerate(pheno[['SITE_ID', 'SUB_ID']]):
        folder = site_id_to_path[site] + '_' + str(id)
        base = os.path.join(dataset_dir, folder)
        mov = os.path.join(base, 'rp_deleteorient_rest.txt')
        if os.path.exists(mov):
            movement.append(np.loadtxt(mov))
            filter[i] = True
        else:
            filter[i] = False

    pheno = pheno[filter]
    # Crop subjects if needed
    if n_subjects is not None:
        pheno = pheno[:n_subjects]
        movement = movement[:n_subjects]

    return Bunch(pheno=pheno, movement=movement)
