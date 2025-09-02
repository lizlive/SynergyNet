import torch
from torch.utils.data import Dataset
import numpy as np




class MyDataLoader_regression(Dataset):

    def __init__(self, X_data, Y_data, Label_data, Trial_data, TrialTimeIdx = [], transform = None):
        self.X_data = np.asarray(X_data)
        self.Y_data = np.asarray(Y_data)
        self.Label_data = np.asarray(Label_data)
        self.Trial_data = np.asarray(Trial_data)
        self.TrialTimeIdx = np.asarray(TrialTimeIdx)
        self.transform  = transform

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_data = self.X_data[idx,:]
        Y_data = self.Y_data[idx,:]
        Label_data = self.Label_data[idx,:]
        Trial_data = self.Trial_data[idx,:]
        if len(self.TrialTimeIdx) == 0:
            TrialTimeIdx = self.TrialTimeIdx
        else:
            TrialTimeIdx = self.TrialTimeIdx[idx, :]
        sample = {'X_data': X_data, 'Y_data': Y_data, 'Label_data': Label_data, 'Trial_data': Trial_data, "TrialTimeIdx": TrialTimeIdx}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor_regression(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        X_data, Y_data, Label_data, Trial_data, TrialTimeIdx = data['X_data'], data['Y_data'], data['Label_data'], data['Trial_data'], data['TrialTimeIdx']
        X_data = torch.from_numpy(X_data).float()
        Y_data = torch.from_numpy(Y_data).float()
        Label_data = torch.from_numpy(Label_data).float()
        Trial_data = torch.from_numpy(Trial_data).float()
        TrialTimeIdx = torch.from_numpy(TrialTimeIdx).float()
        return {'X_data': X_data, 'Y_data': Y_data, 'Label_data': Label_data, 'Trial_data': Trial_data, 'TrialTimeIdx':TrialTimeIdx}


