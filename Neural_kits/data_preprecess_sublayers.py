from sklearn.decomposition import PCA
import numpy as np
from Neural_kits.data_stand import *
from sklearn.preprocessing import LabelEncoder
import glob
from Data_kits.utils import *
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold, KFold

def data_Standardization_cv1session(config, data_fold):
    modality_index = data_fold["modalityIndex"]
    trainX, trainY, testX, testY = data_fold["trainX"], data_fold["trainY"], data_fold["testX"], data_fold["testY"]
    trainLens, testLens = data_fold["trainLens"], data_fold["testLens"]

    if config.valid:
        validX, validY = data_fold["validX"], data_fold["validY"]
        validLens = data_fold["validLens"]

    modalityClass = np.unique(modality_index)
    for label_modality in modalityClass:
        # get feature from modality
        trainX_temp = trainX
        testX_temp = testX
        trainLens_temp = trainLens
        testLens_temp = testLens
        if config.valid:
            validX_temp = validX
            validLens_temp = validLens

        data_temp = {"trainX": trainX_temp, "testX": testX_temp, "trainLens":trainLens_temp, "testLens":testLens_temp}
        if config.valid:
            data_temp = {"trainX": trainX_temp, "testX": testX_temp, "validX": validX_temp, "trainLens":trainLens_temp, "testLens":testLens_temp, "validLens":validLens_temp}
        if label_modality == 0:
            if config.SPIKE.Standardization:
                if config.SPIKE.mean_sub:
                    data_temp_trans = data_stand_submean_channel(config, data_temp)
                else:
                    data_temp_trans = data_stand_channel_cv1session(config, data_temp)

            else:
                data_temp_trans = data_temp

        if label_modality == 1:
            if config.LFP.Standardization:
                if config.dataset == 'LFP_LMP':
                    data_temp_trans = data_stand_channel(config, data_temp)
                else:
                    data_temp_trans = data_stand_point(config, data_temp)
            else:
                data_temp_trans = data_temp

        # write back the Neural_kits after standardization
        trainX_temp, testX_temp, trainLens_temp, testLens_temp = data_temp_trans["trainX"], data_temp_trans["testX"], data_temp_trans["trainLens"], data_temp_trans["testLens"]

        if config.valid:
            validX_temp, validLens_temp = data_temp_trans["validX"], data_temp_trans["validLens"]

        trainX = trainX_temp
        testX = testX_temp
        trainLens = trainLens_temp
        testLens = testLens_temp
        if config.valid:
            validX = validX_temp
            validLens = validLens_temp

    data_fold["trainX"] = trainX
    data_fold["trainY"] = trainY
    data_fold["trainLens"] = trainLens
    data_fold["testX"] = testX
    data_fold["testY"] = testY
    data_fold["testLens"] = testLens

    # dataFold = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY}
    if config.valid:
        data_fold["validY"] = validY
        data_fold["validX"] = validX
        data_fold["validLens"] = validLens
        # dataFold = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY, "validX": validX,
        #             "validY": validY}
    data_fold["modalityIndex"] = modality_index
    # dataFold["modalityIndex"] = modality_index

    return data_fold

def data_Standardization(config, data_fold):
    modality_index = data_fold["modalityIndex"]
    trainX, trainY, testX, testY = data_fold["trainX"], data_fold["trainY"], data_fold["testX"], data_fold["testY"]
    trainLens, testLens = data_fold["trainLens"], data_fold["testLens"]

    if config.valid:
        validX, validY = data_fold["validX"], data_fold["validY"]
        validLens = data_fold["validLens"]

    modalityClass = np.unique(modality_index)
    for label_modality in modalityClass:
        # get feature from modality
        trainX_temp = trainX
        testX_temp = testX
        if config.valid:
            validX_temp = validX

        data_temp = {"trainX": trainX_temp, "testX": testX_temp}
        if config.valid:
            data_temp = {"trainX": trainX_temp, "testX": testX_temp, "validX": validX_temp}
        if label_modality == 0:
            if config.SPIKE.Standardization:
                if config.SPIKE.mean_sub:
                    data_temp_trans = data_stand_submean_channel(config, data_temp)
                else:
                    data_temp_trans = data_stand_channel(config, data_temp)
                    if config.SPIKE.clip_abnormal:
                        data_temp_trans = data_clip_abnormal_channel(config, data_temp_trans)
            else:
                data_temp_trans = data_temp



        if label_modality == 1:
            if config.LFP.Standardization:
                if config.dataset == 'LFP_LMP':
                    data_temp_trans = data_stand_channel(config, data_temp)
                else:
                    data_temp_trans = data_stand_point(config, data_temp)
            else:
                data_temp_trans = data_temp

        # write back the Neural_kits after standardization
        trainX_temp, testX_temp = data_temp_trans["trainX"], data_temp_trans["testX"]

        if config.valid:
            validX_temp = data_temp_trans["validX"]

        trainX = trainX_temp
        testX = testX_temp
        if config.valid:
            validX = validX_temp

    data_fold["trainX"] = trainX
    data_fold["trainY"] = trainY
    data_fold["testX"] = testX
    data_fold["testY"] = testY
    # dataFold = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY}
    if config.valid:
        data_fold["validY"] = validY
        data_fold["validX"] = validX
        # dataFold = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY, "validX": validX,
        #             "validY": validY}
    data_fold["modalityIndex"] = modality_index
    # dataFold["modalityIndex"] = modality_index

    return data_fold



def select_channel_index(data):
    [n_sample, n_channel, n_time] = np.shape(data)
    data = data.transpose(0, 2, 1).reshape([-1, n_channel])
    fr = np.average(data / 0.02, axis=0)
    index = np.where(fr > 0.5)[0]
    return index


def downsample(config, data):
    [n_sample, n_channel, n_time] = np.shape(data)
    win_len = config.SPIKE.win_len
    step_len = config.SPIKE.step_len
    num_time = (n_time - win_len) // step_len + 1
    data_new = np.zeros((n_sample, n_channel, num_time))
    index_1 = 0
    index_2 = index_1 + win_len
    for i in range(0, num_time):
        data_new[:, :, i] = np.sum(data[:, :, index_1:index_2], axis=2)
        # data_new[:, :, i] = np.mean(Neural_kits[:, :, index_1:index_2], axis=2)
        # data_new = data_new / win_len * 1000
        index_1 = index_1 + step_len
        index_2 = index_1 + win_len
    data = data_new

    return data

def downsample_mean(config, data):
    [n_sample, n_channel, n_time] = np.shape(data)
    win_len = config.SPIKE.win_len
    step_len = config.SPIKE.step_len
    num_time = (n_time - win_len) // step_len + 1
    data_new = np.zeros((n_sample, n_channel, num_time))
    index_1 = 0
    index_2 = index_1 + win_len
    for i in range(0, num_time):
        data_new[:, :, i] = np.mean(data[:, :, index_1:index_2], axis=2)
        index_1 = index_1 + step_len
        index_2 = index_1 + win_len
    data = data_new

    return data

def smooth(data, win_len):
    [n_sample, n_channel, n_time] = np.shape(data)
    window_len = win_len
    box = np.ones(window_len) / window_len
    win_len = win_len - 1 + np.mod(win_len, 2) #force it  to  be  odd
    data_new = np.zeros((n_sample, n_channel, n_time))
    for i in range(win_len//2, n_time - win_len//2):
        index_1_clip = i - win_len//2
        index_2_clip = i + win_len//2 + 1
        # print(index_2_clip - index_1_clip)
        data_new[:, :, i] = np.mean(data[:, :, index_1_clip:index_2_clip], axis=2)

    data_begin = np.cumsum(data[:,:,0:win_len-2], 2)
    data_begin = data_begin[:,:,list(range(0, np.shape(data_begin)[-1], 2))] / np.array(list(range(1, win_len -1, 2)))
    data_end = np.cumsum(data[:, :, list(range( n_time-1,n_time-win_len+1, -1))], 2)
    data_end = data_end[:, :, list(range(-1, -np.shape(data_end)[-1]-1, -2))] / np.array(list(range(win_len-2, 0, -2)))
    data_new = np.concatenate([data_begin, data_new[:,:,win_len//2 : - win_len//2 + 1], data_end], -1)
    # for i in range(n_sample):
    #     for j in range(n_channel):
    #         Neural_kits[i, j, :] = np.convolve(Neural_kits[i, j, :], box, mode='same')
    return data_new


def data_prepare_before_std(config, data_fold):
    modality_index = data_fold["modalityIndex"]
    trainX, trainY, testX, testY = data_fold["trainX"], data_fold["trainY"], data_fold["testX"], data_fold["testY"]

    if config.valid:
        validX, validY = data_fold["validX"], data_fold["validY"]

    modalityClass = np.unique(modality_index)
    for label_modality in modalityClass:
        trainX_temp = trainX
        testX_temp = testX
        if config.valid:
            validX_temp = validX

        if label_modality == 0:
            if config.SPIKE.selectchannel:
                index = select_channel_index(trainX_temp)
                trainX_temp = trainX_temp[:, index, :]
                testX_temp = testX_temp[:, index, :]
                if config.valid:
                    validX_temp = validX_temp[:, index, :]

            if config.SPIKE.smooth:
                trainX_temp = smooth(trainX_temp, 20)
                testX_temp = smooth(testX_temp, 20)
                if config.valid:
                    validX_temp = smooth(validX_temp, 20)

            if config.SPIKE.downsample:
                trainX_temp = downsample(config, trainX_temp)
                testX_temp = downsample(config, testX_temp)
                if config.valid:
                    validX_temp = downsample(config, validX_temp)

            if config.SPIKE.smooth_after:
                trainX_temp = smooth(trainX_temp, 5)
                testX_temp = smooth(testX_temp, 5)
                if config.valid:
                    validX_temp = smooth(validX_temp, 5)
            if config.SPIKE.PCA:
                [n_tr_x,  n_channel, n_points] = np.shape(trainX_temp)
                [n_te_x,  n_channel, n_points] = np.shape(testX_temp)
                trainX_temp = trainX_temp.transpose(0, 2, 1).reshape([-1, n_channel])
                testX_temp = testX_temp.transpose(0, 2, 1).reshape([-1, n_channel])
                pca = PCA()
                pca.fit(trainX_temp)
                explained_variance_ratio_ = pca.explained_variance_ratio_
                # explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio_)
                # n_componets = np.where(explained_variance_ratio_cumsum > 0.95)[0][0] + 1
                # pca = PCA(n_components=n_componets)
                # pca.n_components = n_componets
                trainX_temp = pca.fit_transform(trainX_temp)
                testX_temp = pca.transform(testX_temp)

                trainX_temp = trainX_temp.reshape([n_tr_x, n_points, n_channel]).transpose(0, 2, 1)
                testX_temp = testX_temp.reshape([n_te_x, n_points, n_channel]).transpose(0, 2, 1)
                if config.valid:
                    n_va_x, _, _ = validX_temp.shape
                    validX_temp = validX_temp.transpose(0, 2, 1).reshape([-1, n_channel])
                    validX_temp = pca.transform(validX_temp)
                    validX_temp = validX_temp.reshape([n_va_x, n_points, -1]).transpose(0, 2, 1)

        if label_modality == 1:
            data_shape = trainX_temp.shape
            if len(data_shape) == 3:
                n_tr_x, n_channel, n_points = trainX_temp.shape
                n_te_x, _, _ = testX_temp.shape
                if config.LFP.PCA:
                    pass

        trainX = trainX_temp
        testX = testX_temp
        if config.valid:
            validX = validX_temp

    data_fold["trainX"] = trainX
    data_fold["trainY"] = trainY
    data_fold["testX"] = testX
    data_fold["testY"] = testY
    if config.valid:
        data_fold["validY"] = validY
        data_fold["validX"] = validX
    data_fold["modalityIndex"] = modality_index
    if config.SPIKE.PCA:
        data_fold["explained_variance_ratio_"] = explained_variance_ratio_
    return data_fold


def data2pcaRecover(data_fold, config):
    trainX, trainY, testX, testY = data_fold["trainX"], data_fold["trainY"], data_fold["testX"], data_fold["testY"]

    if config.valid:
        validX, validY = data_fold["validX"], data_fold["validY"]

    [n_tr_x, n_channel, n_points] = np.shape(trainX)
    [n_te_x, n_channel, n_points] = np.shape(testX)
    trainX= trainX.transpose(0, 2, 1).reshape([-1, n_channel])
    testX = testX.transpose(0, 2, 1).reshape([-1, n_channel])
    if config.PC_cumsum:
        pca = PCA(n_components= config.n_componets)
        trainX = pca.fit_transform(trainX)
        testX = pca.transform(testX)
        trainX = pca.inverse_transform(trainX)
        testX = pca.inverse_transform(testX)
        trainX = trainX.reshape([n_tr_x, n_points, n_channel]).transpose(0, 2, 1)
        testX = testX.reshape([n_te_x, n_points, n_channel]).transpose(0, 2, 1)
    else:
        pca = PCA(n_components=config.n_componets)
        trainX = pca.fit_transform(trainX)
        testX = pca.transform(testX)
        trainX = np.expand_dims(trainX[:,-1],1)
        testX = np.expand_dims(testX[:,-1],1)
        trainX = np.dot(trainX, np.expand_dims(pca.components_[-1,:],0)) + pca.mean_
        testX = np.dot(testX, np.expand_dims(pca.components_[-1,:],0)) + pca.mean_
        trainX = trainX.reshape([n_tr_x, n_points, n_channel]).transpose(0, 2, 1)
        testX = testX.reshape([n_te_x, n_points, n_channel]).transpose(0, 2, 1)
    if config.valid:
        n_va_x, _, _ = validX.shape
        validX = validX.transpose(0, 2, 1).reshape([-1, n_channel])
        validX = pca.transform(validX)
        if config.PC_cumsum:
            validX = pca.inverse_transform(validX)
        else:
            validX = np.expand_dims(validX[:, -1], 1)
            validX = np.dot(validX, np.expand_dims(pca.components_[-1, :], 0)) + pca.mean_
        validX = validX.reshape([n_va_x, n_points, -1]).transpose(0, 2, 1)
    data_fold["trainX"] = trainX
    data_fold["trainY"] = trainY
    data_fold["testX"] = testX
    data_fold["testY"] = testY
    if config.valid:
        data_fold["validY"] = validY
        data_fold["validX"] = validX
    return data_fold


def data_prepare_before_std_kin_vel(config, Data):
    Data_recover = {}
    KinData = Data["KinData"]
    AxisData = Data["AxisData"]
    Data_recover["angle_base"] = Data["angle_base"]
    Data_recover["KinLabel"] = Data["KinLabel"]


    if config.KIN.kin_type == "position":
        KinData = KinData.copy()
    elif config.KIN.kin_type == "velocity":
        KinData_vel = KinData.copy()
        KinData_vel[:, :, 1:] = KinData[:, :, 1:] - KinData[:, :, 0:-1]
        KinData_vel = KinData_vel * config.SR
        KinData_vel[:, :, 0] = KinData_vel[:, :, 1]
        KinData = KinData_vel
    elif config.KIN.kin_type == "velocity_position":
        KinData_pos = KinData.copy()
        KinData_vel = KinData.copy()
        KinData_vel[:, :, 1:] = KinData[:, :, 1:] - KinData[:, :, 0:-1]
        KinData_vel = KinData_vel * config.SR
        KinData_vel[:, :, 0] = KinData_vel[:, :, 1]
        KinData = np.concatenate((KinData_vel, KinData_pos), 1)

    Data_recover["Originial_data"] = KinData

    if config.SPIKE.smooth:
       KinData = smooth(KinData, 20)
       AxisData = smooth(AxisData, 20)

    if config.SPIKE.downsample:
        KinData = downsample_mean(config, KinData)
        AxisData = downsample_mean(config, AxisData)

    if config.SPIKE.smooth_after:
        KinData = smooth(KinData, 5)
        AxisData = smooth(AxisData, 5)
    Data_recover["AfterDS_data"] = KinData

    Data["KinData"] = KinData
    Data["AxisData"] = AxisData
    Data_recover["AxisData"] = AxisData
    Data_recover["AfterSM"] = KinData

    return Data, Data_recover

def data_prepare_before_std_kin(config, Data):
    if config.cross_vel:
        Data_vels = {}
        Data_recover_vels = {}
        for key in Data.keys():
            Data_temp = Data[key]
            Data_vel, Data_recover_vel = data_prepare_before_std_kin_vel(config, Data_temp)
            Data_vels[key] = Data_vel
            Data_recover_vels[key] = Data_recover_vel
        Data = Data_vels
        Data_recover = Data_recover_vels
    else:
        Data, Data_recover = data_prepare_before_std_kin_vel(config, Data)

    return Data, Data_recover



def data_prepare_before_std_kin_reduction(config, Data):
    Data_recover = {}
    KinData = Data["KinData"]
    AxisData = Data["AxisData"]
    Data_recover["angle_base"] = Data["angle_base"]
    Data_recover["KinLabel"] = Data["KinLabel"]

    if config.KIN.kin_type == "position":
        KinData = KinData.copy()
    elif config.KIN.kin_type == "velocity":
        KinData_vel = KinData.copy()
        KinData_vel[:, :, 1:] = KinData[:, :, 1:] - KinData[:, :, 0:-1]
        KinData_vel = KinData_vel * config.SR
        KinData_vel[:, :, 0] = KinData_vel[:, :, 1]
        KinData = KinData_vel
    elif config.KIN.kin_type == "velocity_position":
        KinData_pos = KinData.copy()
        KinData_vel = KinData.copy()
        KinData_vel[:, :, 1:] = KinData[:, :, 1:] - KinData[:, :, 0:-1]
        KinData_vel = KinData_vel * config.SR
        KinData_vel[:, :, 0] = KinData_vel[:, :, 1]
        KinData = np.concatenate((KinData_vel, KinData_pos), 1)

    Data_recover["Originial_data"] = KinData


    if config.SPIKE.smooth:
        KinData = smooth(KinData, 20)
        AxisData = smooth(AxisData, 20)

    if config.SPIKE.downsample:
        KinData = downsample_mean(config, KinData)
        AxisData = downsample_mean(config, AxisData)

    Data_recover["AfterDS_data"] = KinData
    if config.kin_reduction.reduction_kin:
        if config.kin_reduction.reduction_method == "PCA":
            [n_move, n_dim, n_time] = np.shape(KinData)
            KinData = KinData.transpose(0, 2, 1).reshape([-1, n_dim])

            data_train_mean = np.mean(KinData, 0)
            KinData = KinData - data_train_mean
            config.data_train_mean = data_train_mean

            Data_recover["data_mean_before_PCA"] = data_train_mean

            pca = PCA(n_components = config.kin_reduction.reduction_dim)
            pca.fit(KinData)
            KinData = pca.fit_transform(KinData)

            Data_recover["model"] = pca

            KinData = KinData.reshape([n_move, n_time, -1]).transpose(0, 2, 1)

            Data_recover["AfterPCA_data"] = KinData
        if config.kin_reduction.reduction_method == "MLP_end2end":

            KinData = KinData
        if config.kin_reduction.reduction_method == "NMF":
            from sklearn import decomposition
            [n_move, n_dim, n_time] = np.shape(KinData)
            KinData = KinData.transpose(0, 2, 1).reshape([-1, n_dim])

            data_train_mean = np.mean(KinData, 0)
            KinData = KinData - data_train_mean
            config.data_train_mean = data_train_mean

            Data_recover["data_mean_before_NMF"] = data_train_mean
            # shift Neural_kits to be non-negative
            data_min = np.min(KinData, 0)
            index_negative = np.where(data_min < 0)
            data_shift = np.zeros(np.shape(data_min))
            data_shift[index_negative] = -data_min[index_negative]
            KinData = KinData + data_shift
            Data_recover["data_shift_NMF"] = data_shift

            NMF= decomposition.NMF(n_components=config.kin_reduction.reduction_dim, init="nndsvda", tol=5e-3)
            NMF.fit(KinData)
            KinData = NMF.fit_transform(KinData)

            Data_recover["model"] = NMF

            KinData = KinData.reshape([n_move, n_time, -1]).transpose(0, 2, 1)

            Data_recover["AfterNMF_data"] = KinData
    if config.SPIKE.smooth_after:
        KinData = smooth(KinData, 5)
        AxisData = smooth(AxisData, 5)

    Data["KinData"] = KinData
    Data["AxisData"] = AxisData
    Data_recover["AxisData"] = AxisData
    Data_recover["AfterSM"] = KinData
    return Data, Data_recover

def win_clip(config, data, y):
    num_sample = np.shape(data)[0]
    win_len = config.win_len
    index_1 = 0
    index_2 = index_1 + win_len
    data_new = np.expand_dims(np.transpose(data[index_1: index_2, :]), 0)
    y_new = np.expand_dims(y[index_2 - 1, :], 0)

    while index_2 < num_sample:
        index_1 = index_1 + 1
        index_2 = index_1 + win_len
        data_temp = np.expand_dims(np.transpose(data[index_1: index_2, :]), 0)
        data_new = np.concatenate((data_new, data_temp), 0)
        y_temp = np.expand_dims(y[index_2 - 1, :], 0)
        y_new = np.concatenate((y_new, y_temp), 0)

    data_shape = np.shape(data)
    if data_shape == 4:
        data_new = np.transpose(data_new, (0, 2, 1, 3))
    return data_new, y_new





# get Neural_kits according to split index
def get_data_fold(config, indexSplitFold, Data):
    trainIndex = indexSplitFold["trainIndex"]
    testIndex = indexSplitFold["testIndex"]
    modalityIndex = Data["modalityIndex"]
    if config.valid:
        validIndex = indexSplitFold["validIndex"]

    trainX = Data["dataX"][trainIndex, :, :]
    trainY = Data["dataY"][trainIndex, :]
    trainLens = Data["len_trials"][trainIndex, :]
    testX = Data["dataX"][testIndex, :, :]
    testY = Data["dataY"][testIndex, :]
    testLens = Data["len_trials"][testIndex, :]
    trainTrial = Data["dataTrial"][trainIndex, :]
    testTrial = Data["dataTrial"][testIndex, :]
    if config.valid:
        validX = Data["dataX"][validIndex, :, :]
        validY = Data["dataY"][validIndex, :]
        validTrial = Data["dataTrial"][validIndex, :]
        validLens = Data["len_trials"][validIndex, :]

    dataFold = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY, "trainTrial": trainTrial,
                "testTrial": testTrial, "trainLens":trainLens, "testLens":testLens}
    if config.valid:
        dataFold = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY, "validX": validX,
                    "validY": validY, "trainTrial": trainTrial, "testTrial": testTrial, "validTrial": validTrial, "trainLens":trainLens, "testLens":testLens, "validLens":validLens}
    dataFold["modalityIndex"] = modalityIndex
    return dataFold



# get split Neural_kits index for [trainIndex, testIndex] or [trainIndex, testIndex, validIndex]
def data_spilt(config, Data):
    data = Data["dataX"]
    y = Data["dataY"]

    indexSplitFolds = []
    # split train index and test index
    if not config.valid:
        print("-" * 30)
        print("split Neural_kits into train and test by {}".format(config.data_split))
        index_train_list, index_test_list = [], []
        if config.data_split == "LeaveOneOut":
            y_list = np.unique(y)
            n_label = np.shape(y_list)[0]
            n_samle = np.shape(y)[0]
            for i in range(n_label):
                idx_label = np.where(y == y_list[i])
                index_test = i
                index_train = np.delete(range(n_samle), i)
                index_temp = {"trainIndex": index_train, "testIndex": index_test}
                indexSplitFolds.append(index_temp)

        elif config.data_split == "K_fold":
            # 随机K折数据
            if config.Kfold_random:
                from sklearn.model_selection import StratifiedShuffleSplit
                for i in range(config.K_fold):
                    skf = StratifiedShuffleSplit(n_splits=config.K_fold, random_state=i)
                    for train, test in skf.split(data, y):
                        index_temp = {"trainIndex": train, "testIndex": test}
                        indexSplitFolds.append(index_temp)


            else:
                from sklearn.model_selection import StratifiedKFold, KFold
                skf = StratifiedKFold(n_splits=config.K_fold)

                for train, test in skf.split(data, y):
                    index_temp = {"trainIndex": train, "testIndex": test}
                    indexSplitFolds.append(index_temp)

        elif config.data_split == "LeaveOneBlockOut":
            from sklearn.model_selection import StratifiedKFold, KFold
            skf = StratifiedKFold(n_splits = config.SPIKE.num_block)

            for train, test in skf.split(data, y):
                index_temp = {"trainIndex": train, "testIndex": test}
                indexSplitFolds.append(index_temp)


    if config.valid:
        print("-" * 30)
        print("split Neural_kits into train valid and test by {}".format(config.data_split))
        if config.data_split == "LeaveOneOut":
            y_list = np.unique(y)
            n_label = np.shape(y_list)[0]
            n_samle = np.shape(y)[0]

            for i in range(n_label):
                [idx_label, idx_col]= np.where(y == y_list[i])
                index_test = idx_label
                index_train = np.delete(range(n_samle), idx_label)
                train_X = data[index_train, :]
                train_Y = y[index_train, :]
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits= np.shape(idx_label)[0])
                n_repeat_all =  np.shape(idx_label)[0]
                n_repeat = n_repeat_all / (config.num_block)

                idx_valid = []
                idx_fold_valid = 0
                for train_1, valid_1 in skf.split(train_X, train_Y):
                    idx_fold_valid = idx_fold_valid + 1
                    if 0 == np.mod(idx_fold_valid, n_repeat):
                        idx_valid.extend(valid_1)

                idx_all = np.array(list(range(len(train_Y))))
                train_1 = np.delete(idx_all, idx_valid)
                train_new = index_train[train_1]
                valid = index_train[idx_valid]

                # idx_fold_valid = 1
                # for train_1, valid_1 in skf.split(train_X, train_Y):
                #     valid = index_train[valid_1]
                #     train_new = index_train[train_1]
                #     idx_fold_valid = idx_fold_valid + 1
                #     # if idx_fold_valid == config.SPIKE.num_repeat // 2:
                #     if idx_fold_valid == 1:
                #         break
                index_temp = {"trainIndex": train_new, "testIndex": index_test, "validIndex":valid}
                print(valid)
                print(idx_valid)
                print(np.transpose(y[valid]))
                indexSplitFolds.append(index_temp)

        elif config.data_split == "K_fold":
            # 随机K折数据
            if config.Kfold_random:
                from sklearn.model_selection import StratifiedShuffleSplit
                skf = StratifiedShuffleSplit(n_splits=config.K_fold, random_state=1)
            else:
                from sklearn.model_selection import StratifiedKFold, KFold
                skf = StratifiedKFold(n_splits=config.K_fold)

            idx_fold = 0
            # obtain the nearest behind Neural_kits for valid dataset(1->2, 2->3, but 10->9)
            for train, test in skf.split(data, y):
                idx_fold = idx_fold + 1
                train_X = data[train, :]
                train_Y = y[train, :]
                # skf2 = StratifiedKFold(n_splits=config.K_fold - 1)
                skf2 = StratifiedKFold(n_splits=np.sum(train_Y == 0))
                n_repeat_all = np.sum(train_Y == 0)
                n_repeat = n_repeat_all / (config.num_block - 1)

                idx_valid = []
                idx_fold_valid = 0
                for train_1, valid_1 in skf2.split(train_X, train_Y):
                    idx_fold_valid = idx_fold_valid + 1
                    if 0 == np.mod(idx_fold_valid, n_repeat):
                        idx_valid.extend(valid_1)
                idx_all = np.array(list(range(len(train_Y))))
                train_1 = np.delete(idx_all, idx_valid)
                train_new = train[train_1]
                valid = train[idx_valid]

                index_temp = {"trainIndex": train_new, "testIndex": test, "validIndex": valid}
                indexSplitFolds.append(index_temp)

        elif config.data_split == "LeaveOneBlockOut":
            from sklearn.model_selection import StratifiedKFold, KFold
            skf = StratifiedKFold(n_splits = config.SPIKE.num_block)

            idx_fold = 0
            # obtain the nearest behind Neural_kits for valid dataset(1->2, 2->3, but 10->9)
            for train, test in skf.split(data, y):
                idx_fold = idx_fold + 1
                train_X = data[train, :]
                train_Y = y[train, :]
                # skf2 = StratifiedKFold(n_splits=config.K_fold - 1)
                skf2 = StratifiedKFold(n_splits=config.SPIKE.num_block - 1)

                idx_fold_valid = 0
                for train_1, valid_1 in skf2.split(train_X, train_Y):
                    valid = train[valid_1]
                    train_new = train[train_1]
                    idx_fold_valid = idx_fold_valid + 1
                    if idx_fold_valid == idx_fold:
                        break

                index_temp = {"trainIndex": train_new, "testIndex": test, "validIndex": valid}
                indexSplitFolds.append(index_temp)



    return indexSplitFolds



def get_data_mat(config):
    Data = load_mat(config.data_path)
    Data = get_fea(Data, config)
    data = Data["dataX"]
    data[np.isnan(data)] = 0
    y = Data["dataY"]
    len_trials = Data["len_trials"]
    if config.delete_1st_trial:
        index_delete = np.array(range(0, np.shape(data)[0], config.num_repeat))
        index = np.array(range(0, np.shape(data)[0]))
        index = np.delete(index, index_delete)
        data = data[index,:,:]
        y = y[index,:]
        len_trials = len_trials[index,:]
    # y = np.expand_dims(y,1)
    y = np.array(LabelEncoder().fit_transform(y.ravel()))
    y = np.expand_dims(y, 1)
    len_trials = np.expand_dims(len_trials, 1)
    modality_index = Data["modalityIndex"]

    Data = {"dataX": data, "dataY": y, "modalityIndex": modality_index, "len_trials":len_trials}
    Data["dataTrial"] = np.expand_dims(np.array(range(np.shape(data)[0])), 1)

    return Data


def get_data_mat_kin_vel(filename, config):
    files = sorted(glob.glob(os.path.join(config.KIN.root_dir, filename)))
    Data = {}
    KinLabel = []
    KinData = []
    KinAxis = []
    angle_base = []
    for file in files:
        Data_temp = load_mat(file)
        filename = file.split('/')[-1]
        KinLabel_tmp = filename.split('_')[3]
        KinLabel.append(KinLabel_tmp)
        KinData_tmp = Data_temp[config.KIN.kin_space]
        KinAxis_tmp = Data_temp["rotation_axis"]
        KinData.append(KinData_tmp)
        KinAxis.append(KinAxis_tmp)
        angle_base.append(Data_temp["angle_base"])
    KinLabel = np.array(KinLabel)
    KinAxis = np.array(KinAxis)
    angle_base = np.array(angle_base)
    idx = np.argsort(KinLabel)
    Data["KinLabel"] = KinLabel[idx]
    KinData = np.array(KinData).transpose(0, 2, 1)
    KinAxis = np.array(KinAxis).transpose(0, 2, 1)
    if not config.KIN.dim_0:
        KinData = KinData[:, 1:, :]
    Data["KinData"] = KinData[idx, :, :]
    Data["AxisData"] = KinAxis[idx, :, :]
    Data["angle_base"] = angle_base[idx, :]
    return Data


def get_data_mat_kin(config):
    if config.cross_vel:
        Data = {}
        for kin_data_pattern in config.kin_data:
            pattern = kin_data_pattern
            filename = "gesture_data_*_{}.mat".format(pattern)
            Data_tmp = get_data_mat_kin_vel(filename, config)
            Data[str(pattern)] = Data_tmp


    else:
        pattern = config.kin_data
        filename = "gesture_data_*_{}.mat".format(pattern)
        Data = get_data_mat_kin_vel(filename, config)

    return Data


# get according Neural_kits modality from Neural_kits which contain all spikes and LFPs
# Data -> Data = {"dataX": Neural_kits, "dataY": y, "modalityIndex":modality_index}
def get_fea(Data, config):
    if config.spike_flag:
        # if config.dataset == 'NeuronSpike':
        #     Neural_kits = np.array(Data["final_bin_spike_unit"])
        #     Neural_kits = Neural_kits.transpose(0, 2, 1)
        #     Neural_kits = Neural_kits[:, :, config.SPIKE.strat_bin:config.SPIKE.end_bin]
        #
        # if config.dataset == "SpikeBandPower":
        #     Neural_kits = np.array(Data["data_SBP"])
        #     Neural_kits = Neural_kits.transpose(0, 2, 1)
        #     Neural_kits = Neural_kits[:, :, config.SPIKE.strat_bin:config.SPIKE.end_bin]



        data = np.array(Data["data"])
        data = data.transpose(0, 2, 1)


        if config.cross_vel:
            idx_tmp = np.argmax(np.array(config.SPIKE.end_bin))
            strat_bin = config.strat_bin[idx_tmp]
            end_bin = config.end_bin[idx_tmp]
        else:
            strat_bin = config.start_bin
            end_bin = config.end_bin

        data = data[:, :, strat_bin: end_bin]

        data_Spike = data
        modality_index_spike = 0
        modality_index = modality_index_spike

    if config.LFP_flag:
        if config.get('noise_add') == None:

            if config.dataset == 'LFP':
                data = np.array(Data['LFP'])[0, 0]
                data = np.squeeze(np.mean(data, axis=2))
            elif config.dataset == 'LFP_FE':
                data = np.array(Data['LFP_FE_3'])[0, 0]
            elif config.dataset == 'LFP_FE_2':
                data = np.array(Data['LFP_FE_2'])[0, 0]
            elif config.dataset == "LFP_Spike":
                data = np.array(Data['LFP_FE_3'])[0, 0]
            elif config.dataset == "LFP_8_Spike":
                data = np.array(Data['LFP_FE_2'])[0, 0]
            elif config.dataset == "LFP_LMP":
                data = np.array(Data['LFP'])[0, 0]
                threshold = 500
                data = np.clip(data, -threshold, threshold)
                data = np.squeeze(np.mean(data, axis=2))
                # downsample
                [n_sample, n_channel, n_time] = np.shape(data)
                time_step = 20
                num_time = n_time // time_step
                data_new = np.zeros((n_sample, n_channel, num_time))
                for i in range(0, num_time):
                    data_new[:, :, i] = np.mean(data[:, :, i * time_step:(i + 1) * time_step], axis=2)
                data = data_new
            elif config.dataset == "LFP_FE_patch":
                data = np.array(Data['LFP_FE_patch'])[0, 0]
                # Neural_kits = Neural_kits.reshape(Neural_kits.shape[0], Neural_kits.shape[1], -1)
        else:
            if config.noise_add.count("shift") > 0:
                if config.noise_type.count("noise_10") > 0:
                    if config.dataset == 'LFP':
                        data = np.array(Data['LFP_shift_10'])[0, 0]
                        data = np.squeeze(np.mean(data, axis=2))
                    elif config.dataset == 'LFP_FE':
                        data = np.array(Data['LFP_FE_3_shift_10'])[0, 0]
                    elif config.dataset == 'LFP_FE_2':
                        data = np.array(Data['LFP_FE_2_shift_10'])[0, 0]
                    elif config.dataset == "LFP_Spike":
                        data = np.array(Data['LFP_FE_3_shift_10'])[0, 0]
                    elif config.dataset == "LFP_8_Spike":
                        data = np.array(Data['LFP_FE_2_shift_10'])[0, 0]
                    elif config.dataset == "LFP_LMP":
                        data = np.array(Data['LFP_shift_10'])[0, 0]
                        threshold = 500
                        data = np.clip(data, -threshold, threshold)
                        data = np.squeeze(np.mean(data, axis=2))
                        # downsample
                        [n_sample, n_channel, n_time] = np.shape(data)
                        time_step = 20
                        num_time = n_time // time_step
                        data_new = np.zeros((n_sample, n_channel, num_time))
                        for i in range(0, num_time):
                            data_new[:, :, i] = np.mean(data[:, :, i * time_step:(i + 1) * time_step], axis=2)
                        data = data_new
                if config.noise_type.count("noise_20") > 0:
                    if config.dataset == 'LFP':
                        data = np.array(Data['LFP_shift_20'])[0, 0]
                        data = np.squeeze(np.mean(data, axis=2))
                    elif config.dataset == 'LFP_FE':
                        data = np.array(Data['LFP_FE_3_shift_20'])[0, 0]
                    elif config.dataset == 'LFP_FE_2':
                        data = np.array(Data['LFP_FE_2_shift_20'])[0, 0]
                    elif config.dataset == "LFP_Spike":
                        data = np.array(Data['LFP_FE_3_shift_20'])[0, 0]
                    elif config.dataset == "LFP_8_Spike":
                        data = np.array(Data['LFP_FE_2_shift_20'])[0, 0]
                    elif config.dataset == "LFP_LMP":
                        data = np.array(Data['LFP_shift_20'])[0, 0]
                        threshold = 500
                        data = np.clip(data, -threshold, threshold)
                        data = np.squeeze(np.mean(data, axis=2))
                        # downsample
                        [n_sample, n_channel, n_time] = np.shape(data)
                        time_step = 20
                        num_time = n_time // time_step
                        data_new = np.zeros((n_sample, n_channel, num_time))
                        for i in range(0, num_time):
                            data_new[:, :, i] = np.mean(data[:, :, i * time_step:(i + 1) * time_step], axis=2)
                        data = data_new
                if config.noise_type.count("noise_30") > 0:
                    if config.dataset == 'LFP':
                        data = np.array(Data['LFP_shift_30'])[0, 0]
                        data = np.squeeze(np.mean(data, axis=2))
                    elif config.dataset == 'LFP_FE':
                        data = np.array(Data['LFP_FE_3_shift_30'])[0, 0]
                    elif config.dataset == 'LFP_FE_2':
                        data = np.array(Data['LFP_FE_2_shift_30'])[0, 0]
                    elif config.dataset == "LFP_Spike":
                        data = np.array(Data['LFP_FE_3_shift_30'])[0, 0]
                    elif config.dataset == "LFP_8_Spike":
                        data = np.array(Data['LFP_FE_2_shift_30'])[0, 0]
                    elif config.dataset == "LFP_LMP":
                        data = np.array(Data['LFP_shift_30'])[0, 0]
                        threshold = 500
                        data = np.clip(data, -threshold, threshold)
                        data = np.squeeze(np.mean(data, axis=2))
                        # downsample
                        [n_sample, n_channel, n_time] = np.shape(data)
                        time_step = 20
                        num_time = n_time // time_step
                        data_new = np.zeros((n_sample, n_channel, num_time))
                        for i in range(0, num_time):
                            data_new[:, :, i] = np.mean(data[:, :, i * time_step:(i + 1) * time_step], axis=2)
                        data = data_new
                if config.noise_type.count("noise_0") > 0:
                    if config.dataset == 'LFP':
                        data = np.array(Data['LFP'])[0, 0]
                        data = np.squeeze(np.mean(data, axis=2))
                    elif config.dataset == 'LFP_FE':
                        data = np.array(Data['LFP_FE_3'])[0, 0]
                    elif config.dataset == 'LFP_FE_2':
                        data = np.array(Data['LFP_FE_2'])[0, 0]
                    elif config.dataset == "LFP_Spike":
                        data = np.array(Data['LFP_FE_3'])[0, 0]
                    elif config.dataset == "LFP_8_Spike":
                        data = np.array(Data['LFP_FE_2'])[0, 0]
                    elif config.dataset == "LFP_LMP":
                        data = np.array(Data['LFP'])[0, 0]
                        threshold = 500
                        data = np.clip(data, -threshold, threshold)
                        data = np.squeeze(np.mean(data, axis=2))
                        # downsample
                        [n_sample, n_channel, n_time] = np.shape(data)
                        time_step = 20
                        num_time = n_time // time_step
                        data_new = np.zeros((n_sample, n_channel, num_time))
                        for i in range(0, num_time):
                            data_new[:, :, i] = np.mean(data[:, :, i * time_step:(i + 1) * time_step], axis=2)
                        data = data_new
            else:
                if config.dataset == 'LFP':
                    data = np.array(Data['LFP'])[0, 0]
                    data = np.squeeze(np.mean(data, axis=2))
                elif config.dataset == 'LFP_FE':
                    data = np.array(Data['LFP_FE_3'])[0, 0]
                elif config.dataset == 'LFP_FE_2':
                    data = np.array(Data['LFP_FE_2'])[0, 0]
                elif config.dataset == "LFP_Spike":
                    data = np.array(Data['LFP_FE_3'])[0, 0]
                elif config.dataset == "LFP_8_Spike":
                    data = np.array(Data['LFP_FE_2'])[0, 0]
                elif config.dataset == "LFP_LMP":
                    data = np.array(Data['LFP'])[0, 0]
                    threshold = 500
                    data = np.clip(data, -threshold, threshold)
                    data = np.squeeze(np.mean(data, axis=2))
                    # downsample
                    [n_sample, n_channel, n_time] = np.shape(data)
                    time_step = 20
                    num_time = n_time // time_step
                    data_new = np.zeros((n_sample, n_channel, num_time))
                    for i in range(0, num_time):
                        data_new[:, :, i] = np.mean(data[:, :, i * time_step:(i + 1) * time_step], axis=2)
                    data = data_new

        data_LFP = data
        modality_index_LFP = 1
        modality_index = modality_index_LFP

    if config.LFP_flag and config.Spike_flag:
        data = np.concatenate((data_Spike, data_LFP), axis=2)
        modality_index = np.concatenate((modality_index_spike, modality_index_LFP), axis=1)

    y = np.array(Data["target_no"])
    y = np.expand_dims(y.ravel(), 1)
    len_trials = np.array(Data["len_trials"])
    len_trials = fix_lens(config.end_bin, len_trials)

    Data = {"dataX": data, "dataY": y, "modalityIndex": modality_index, "len_trials":len_trials}
    return Data


# fix the trial length with the minimize difference from the length set
def fix_lens(len_set, len_trials):
    if isinstance(len_set, int):
        len_trials[range(len(len_trials))] = len_set
        len_trials_fix = len_trials
    else:
        len_set = np.array(len_set)
        len_trials = np.array(len_trials)
        len_trials = np.expand_dims(len_trials.ravel(), 1)
        len_trials_set = np.repeat(len_trials, len(len_set), axis=1)
        len_set = np.expand_dims(len_set.ravel(), 1)
        len_set_trials = np.repeat(len_set, len(len_trials), axis=1)
        len_set_trials = np.transpose(len_set_trials)
        diff_len = np.abs(len_set_trials - len_trials_set)
        idx_trials = np.argmin(diff_len, 1)
        len_trials_fix = len_set_trials[range(len(len_trials)),idx_trials ]
    len_trials_fix = np.expand_dims(len_trials_fix.ravel(), 1)
    return len_trials_fix

def prepare_kin_data_S2S(config, dataNeuron, dataLabel, dataTrial, KinData):
    KinData = KinData["KinData"]
    KinData_align = KinData[np.squeeze(dataLabel), :, :]
    [num_sample, num_dim, num_time] = np.shape(dataNeuron)
    step_len = config.regression.step_len
    X = []
    Y = []
    Label = []
    Trial = []
    time_idx = []
    if config.regression.flatten_matrix:

        X = dataNeuron.transpose(1, 0, 2)
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        X = np.transpose(X)
        Y = KinData_align.transpose(1, 0, 2)
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
        Y = np.transpose(Y)
        Label = dataLabel.repeat(num_time, 1)
        Label = Label.reshape(Label.shape[0] * Label.shape[1], 1)
        Trial = dataTrial.repeat(num_time, 1)
        Trial = Trial.reshape(Trial.shape[0] * Trial.shape[1], 1)
        time_idx = np.arange(num_time)
        time_idx = time_idx.repeat(num_sample, 0)
        a = np.reshape(time_idx, [num_sample, num_time], 'F')
        time_idx = a.reshape(a.shape[0] * a.shape[1], 1)
    else:
        for i_sample in range(num_sample):
            X_tmp = dataNeuron[i_sample, :, :]
            Y_tmp = KinData_align[i_sample, :, :]
            Label_tmp = dataLabel[i_sample, :]
            Trial_tmp = dataTrial[i_sample, :]

            if config.regression.padding:
                X_padding = np.zeros([np.shape(X_tmp)[0], step_len-1])
                Y_padding = np.zeros([np.shape(Y_tmp)[0], step_len-1])
                X_tmp = np.concatenate((X_padding, X_tmp), -1)
                Y_tmp = np.concatenate((Y_padding, Y_tmp), -1)
                num_time = np.shape(X_tmp)[-1]

            for i_time in range(step_len, num_time + 1):

                X.append(np.array(X_tmp[:, i_time - step_len: i_time]))
                Y.append(np.array(Y_tmp[:, i_time - step_len: i_time]))
                Label.append(Label_tmp)
                Trial.append(Trial_tmp)
                time_idx.append(np.array([i_time - step_len]))

        X = np.array(X)
        Y = np.array(Y)
        Label = np.array(Label)
        Trial = np.array(Trial)
        time_idx = np.array(time_idx)

        if not config.regression.RNN:
            X = X.reshape(X.shape[0], X.shapec[1] * X.shape[2])

    return X, Y, Label, Trial, time_idx

def prepare_kin_data(config, dataNeuron, dataLabel, dataTrial, KinData):
    KinData = KinData["KinData"]
    KinData_align = KinData[np.squeeze(dataLabel), :, :]
    [num_sample, num_dim, num_time] = np.shape(dataNeuron)
    step_len = config.regression.step_len
    X = []
    Y = []
    Label = []
    Trial = []
    time_idx = []
    if config.regression.flatten_matrix:

        X = dataNeuron.transpose(1, 0, 2)
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        X = np.transpose(X)
        Y = KinData_align.transpose(1, 0, 2)
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
        Y = np.transpose(Y)
        Label = dataLabel.repeat(num_time, 1)
        Label = Label.reshape(Label.shape[0] * Label.shape[1], 1)
        Trial = dataTrial.repeat(num_time, 1)
        Trial = Trial.reshape(Trial.shape[0] * Trial.shape[1], 1)
        time_idx = np.arange(num_time)
        time_idx = time_idx.repeat(num_sample, 0)
        a = np.reshape(time_idx, [num_sample, num_time], 'F')
        time_idx = a.reshape(a.shape[0] * a.shape[1], 1)
    else:
        for i_sample in range(num_sample):
            X_tmp = dataNeuron[i_sample, :, :]
            Y_tmp = KinData_align[i_sample, :, :]
            Label_tmp = dataLabel[i_sample, :]
            Trial_tmp = dataTrial[i_sample, :]

            if config.regression.padding:
                X_padding = np.zeros([np.shape(X_tmp)[0], step_len-1])
                Y_padding = np.zeros([np.shape(Y_tmp)[0], step_len-1])
                X_tmp = np.concatenate((X_padding, X_tmp), -1)
                Y_tmp = np.concatenate((Y_padding, Y_tmp), -1)
                num_time = np.shape(X_tmp)[-1]

            for i_time in range(step_len, num_time + 1):

                X.append(np.array(X_tmp[:, i_time - step_len: i_time]))
                Y.append(Y_tmp[:, i_time - 1])
                Label.append(Label_tmp)
                Trial.append(Trial_tmp)
                time_idx.append(np.array([i_time - step_len]))

        X = np.array(X)
        Y = np.array(Y)
        Label = np.array(Label)
        Trial = np.array(Trial)
        time_idx = np.array(time_idx)

        if not config.regression.RNN:
            X = X.reshape(X.shape[0], X.shapec[1] * X.shape[2])

    return X, Y, Label, Trial, time_idx

def prepare_kin_data_S2S_trial(config, dataNeuron, dataLabel, dataTrial, KinData):
    KinData = KinData["KinData"]
    KinData_align = KinData[np.squeeze(dataLabel), :, :]
    X = dataNeuron
    Y = KinData_align
    Label = dataLabel
    Trial = dataTrial
    time_idx = np.ones_like(Label)
    return X, Y, Label, Trial, time_idx

def prepare_kin_data_cv1session(config, dataNeuron, dataLabel, dataTrial, KinData, dataLens):


    [num_sample, num_dim, num_time] = np.shape(dataNeuron)
    step_len = config.regression.step_len
    X = []
    Y = []
    Label = []
    Trial = []
    if config.regression.flatten_matrix:
        X = dataNeuron[0, :, 0:dataLens[0]]
        Label = dataLabel[0].repeat(dataLens[0], 1)
        Trial = dataTrial[0].repeat(dataLens[0], 1)
        Y = obtain_data_Kindata(KinData, dataLens[0], dataLabel[0], config)
        for i_trial in range(1, num_sample):
            X = np.concatenate((X, dataNeuron[i_trial, :, 0:dataLens[i_trial]]), axis=1)
            Y = np.concatenate((Y, obtain_data_Kindata(KinData, dataLens[i_trial], dataLabel[i_trial], config), 1), axis=1)
            Label = np.concatenate((Label, dataLabel[i_trial].repeat(dataLens[i_trial], 1)), axis=1)
            Trial = np.concatenate((Trial, dataTrial[i_trial].repeat(dataLens[i_trial], 1)), axis=1)

        # X = dataNeuron.transpose(1, 0, 2)
        # X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        # X = np.transpose(X)
        # Y = KinData_align.transpose(1, 0, 2)
        # Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
        # Y = np.transpose(Y)
        # Label = dataLabel.repeat(num_time, 1)
        # Label = Label.reshape(Label.shape[0] * Label.shape[1], 1)
        # Trial = dataTrial.repeat(num_time, 1)
        # Trial = Trial.reshape(Trial.shape[0] * Trial.shape[1], 1)


    else:
        for i_sample in range(num_sample):
            X_tmp = dataNeuron[i_sample, :, 0:dataLens[i_sample]]
            Y_tmp = obtain_data_Kindata(KinData, dataLens[i_sample], dataLabel[i_sample], config)
            Label_tmp = dataLabel[i_sample, :]
            Trial_tmp = dataTrial[i_sample, :]

            if config.regression.padding:
                X_padding = np.zeros([np.shape(X_tmp)[0], step_len-1])
                Y_padding = np.zeros([np.shape(Y_tmp)[0], step_len-1])
                X_tmp = np.concatenate((X_padding, X_tmp), -1)
                Y_tmp = np.concatenate((Y_padding, Y_tmp), -1)
                num_time = np.shape(X_tmp)[-1]
            for i_time in range(step_len, num_time + 1):

                X.append(np.array(X_tmp[:, i_time - step_len: i_time]))
                Y.append(Y_tmp[:, i_time - 1])
                Label.append(Label_tmp)
                Trial.append(Trial_tmp)

        X = np.array(X)
        Y = np.array(Y)
        Label = np.array(Label)
        Trial = np.array(Trial)

        # if not config.regression.RNN:
        #     X = X.reshape(X.shape[0], X.shapec[1] * X.shape[2])

    return X, Y, Label, Trial



def obtain_data_Kindata(KinData, dataLen, dataLabel, config):
    # 1600, 2400, 3000 or  trainLens = (trainLens - win_len) // step_len + 1
    win_len = config.SPIKE.win_len
    step_len = config.SPIKE.step_len
    dataLen_ori = (dataLen - 1) * step_len + win_len
    dataLenPattern = np.array([1600, 2400, 3000])
    dataLenDiff = dataLenPattern - dataLen_ori
    idx = np.argmin(np.abs(dataLenDiff))
    for key in KinData.keys():
        if str(dataLenPattern[idx]) in key or str(dataLen) in key:
            kinPattern = key
            break

    return KinData[kinPattern]["KinData"][dataLabel[0],:,:]


