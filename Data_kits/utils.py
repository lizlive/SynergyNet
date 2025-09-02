import scipy
from scipy import io as spio
import numpy as np
from torch.utils.data import Dataset
import xlwt
import torch
from Neural_kits.MyDataLoader import *
import torchvision.transforms as transforms
import logging
logger = logging.getLogger()
def set_style(name,height,bold=False):
	style = xlwt.XFStyle()
	font = xlwt.Font()
	font.name = name
	font.bold = bold
	font.color_index = 4
	font.height = height
	style.font = font
	return style

def check_keys(data):
    def _check_keys(d):
        r"""Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries.
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        # if isinstance(d, spio.matlab.mio5_params.mat_struct):
        #     d = _todict(d)
        return d

    def _todict(matobj):
        r"""A recursive function which constructs from matobjects nested dictionaries."""
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        r"""A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    return _check_keys(data)


def load_mat(filename):
    r"""This function should be called instead of direct spio.loadmat
       as it cures the problem of not properly recovering python dictionaries
       from mat files. It calls the function check keys to cure all entries
       which are still mat-objects.
       """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    data_checked = check_keys(data)
    return data_checked

def visulize_dataloader(dataloader):
    num_sample = len(dataloader)
    data_shape= dataloader[0]["X_data"].shape
    logger.info("train data:{}_{}".format(num_sample, data_shape))



def data_spilt(config, data, y):
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    if config.data_split == "LOO":
        num_sample = np.shape(data)[0]
        for i in range(num_sample):
            index_test = i
            index_train = np.delete(range(num_sample), i)
            train_X = data[index_train, :]
            train_Y = y[index_train, :]
            test_X = np.expand_dims(data[index_test, :],axis= 0)
            test_Y = np.expand_dims(y[index_test, :], axis=0)

            train_X_list.append(train_X)
            train_Y_list.append(train_Y)
            test_X_list.append(test_X)
            test_Y_list.append(test_Y)
    elif config.data_split == "K_fold":
        # 随机K折数据
        if config.Kfold_random:
            from sklearn.model_selection import StratifiedShuffleSplit
            skf = StratifiedShuffleSplit(n_splits=10)
        else:
            from sklearn.model_selection import StratifiedKFold, KFold
            skf = StratifiedKFold(n_splits=10)

        for train, test in skf.split(data, y):
            train_X = data[train, :]
            train_Y = y[train, :]
            test_X = data[test, :]
            test_Y = y[test, :]

            train_X_list.append(train_X)
            train_Y_list.append(train_Y)
            test_X_list.append(test_X)
            test_Y_list.append(test_Y)
    return train_X_list, train_Y_list, test_X_list, test_Y_list


class MyDataLoader(Dataset):

    def __init__(self, X_data, Y_data, transform = None):
        self.X_data = np.asarray(X_data)
        self.Y_data = np.asarray(Y_data,dtype=np.int64)
        self.transform  = transform

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_data = self.X_data[idx,:]
        Y_data = self.Y_data[idx,:]
        # Y_data = self.Y_data[idx]
        sample = {'X_data': X_data, 'Y_data': Y_data}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        X_data, Y_data = data['X_data'], data['Y_data']
        X_data = torch.from_numpy(X_data).float()
        Y_data = torch.from_numpy(Y_data)
        return {'X_data': X_data,
                'Y_data': Y_data}








