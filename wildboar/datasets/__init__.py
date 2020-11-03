try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

import numpy as np

from . import _resources

__all__ = ["load_dataset", "load_all_datasets", "load_two_lead_ecg", "load_synthetic_control", "load_gun_point"]

_dataset_mapping = {
    'synthetic_control': 'synthetic_control',
    'two_lead_ecg': 'TwoLeadECG',
    'gun_point': "Gun_Point",
}


def load_synthetic_control(**kvargs):
    """Load the Synthetic_Control dataset"""
    return load_dataset("synthetic_control", **kvargs)


def load_two_lead_ecg(**kwargs):
    """Load the TwoLeadECG dataset"""
    return load_dataset('two_lead_ecg', **kwargs)


def load_gun_point(**kwargs):
    """Load the GunPoint dataset"""
    return load_dataset('gun_point', **kwargs)


def load_all_datasets(**kwargs):
    """Load all datasets as a generator

    Parameters
    ----------
    dtype : dtype, optional, default=np.float64
        The dtype of the returned data

    contiguous : bool, optional
        Ensure that the returned dataset is memory contiguous

    merge_train_test : bool, optional
        Merge the existing training and testing partitions

    Yields
    ------
    x : array-like
        Data samples

    y : array-like
        Data labels

    Examples
    --------

    >>> from wildboar.datasets import load_all_datasets
    >>> for dataset, (x, y) in load_all_datasets(dtype=np.float64):
    >>>     pass # use x and y

    """
    for dataset in _dataset_mapping.keys():
        yield dataset, load_dataset(dataset, **kwargs)


def load_dataset(name, dtype=None, contiguous=True, merge_train_test=True):
    """
    Load a dataset

    Parameters
    ----------
    name : str
        - 'synthetic_control'
        - 'two_lead_ecg'

    dtype : dtype, optional, default=np.float64
        The dtype of the returned data

    contiguous : bool, optional
        Ensure that the returned dataset is memory contiguous

    merge_train_test : bool, optional
        Merge the existing training and testing partitions

    Returns
    -------
    x : ndarray
        The data samples

    y : ndarray
        The labels
    """
    if name not in _dataset_mapping:
        raise ValueError("dataset (%s) does not exist" % name)

    dtype = dtype or np.float64
    name = _dataset_mapping[name]
    train_file = pkg_resources.open_text(_resources, "%s_TRAIN" % name)
    test_file = pkg_resources.open_text(_resources, "%s_TEST" % name)
    train = np.loadtxt(train_file, delimiter=",")
    test = np.loadtxt(test_file, delimiter=",")
    train = np.vstack([train, test])
    y = train[:, 0].astype(dtype)
    x = train[:, 1:].astype(dtype)
    if contiguous:
        y = np.ascontiguousarray(y)
        x = np.ascontiguousarray(x)
    return x, y
