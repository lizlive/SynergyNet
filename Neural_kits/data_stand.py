import numpy as np
from sklearn.preprocessing import StandardScaler

def data_stand_channel_cv1session(config, data):
    trainX, testX = data["trainX"], data["testX"]
    trainLens, testLens = np.squeeze(data["trainLens"]), np.squeeze(data["testLens"])
    win_len = config.SPIKE.win_len
    step_len = config.SPIKE.step_len
    trainLens = (trainLens - win_len) // step_len + 1
    testLens = (testLens - win_len) // step_len + 1

    if config.valid:
        validX, validLens = data["validX"], np.squeeze(data["validLens"])
        validLens = (validLens - win_len) // step_len + 1

    n_tr_x, n_channel, n_points = trainX.shape
    trainX_fla = trainX[0,:,0:trainLens[0]]
    for i_trial in range(1, n_tr_x):
        trainX_fla = np.concatenate((trainX_fla, trainX[i_trial,:,0:trainLens[i_trial]]), axis=1)
    trainX_fla = np.transpose(trainX_fla)

    n_te_x, _, _ = testX.shape
    testX_fla = testX[0, :, 0:testLens[0]]
    for i_trial in range(1, n_te_x):
        testX_fla = np.concatenate((testX_fla, testX[i_trial, :, 0:testLens[i_trial]]), axis=1)
    testX_fla = np.transpose(testX_fla)

    if config.valid:
        n_va_x, _, _ = validX.shape
        validX_fla = validX[0, :, 0:validLens[0]]
        for i_trial in range(1, n_va_x):
            validX_fla = np.concatenate((validX_fla, validX[i_trial, :, 0:validLens[i_trial]]), axis=1)
        validX_fla = np.transpose(validX_fla)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(trainX_fla)
    trainX_fla = scaler.transform(trainX_fla)
    testX_fla = scaler.transform(testX_fla)

    if config.valid:
        validX_fla = scaler.transform(validX_fla)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(trainX_fla)
    if config.SPIKE.clip_abnormal:
        limit_p = scaler.mean_ + 3 * scaler.var_
        limit_n = scaler.mean_ - 3 * scaler.var_
        for i in range(0, n_channel):
            index_temp = np.where(trainX_fla[:, i] > limit_p[i])[0]
            trainX_fla[index_temp, i] = limit_p[i]
            index_temp = np.where(trainX_fla[:, i] < limit_n[i])[0]
            trainX_fla[index_temp, i] = limit_n[i]
            index_temp = np.where(testX_fla[:, i] > limit_p[i])[0]
            testX_fla[index_temp, i] = limit_p[i]
            index_temp = np.where(testX_fla[:, i] < limit_n[i])[0]
            testX_fla[index_temp, i] = limit_n[i]
            if config.valid:
                index_temp = np.where(validX_fla[:, i] > limit_p[i])[0]
                validX_fla[index_temp, i] = limit_p[i]
                index_temp = np.where(validX_fla[:, i] < limit_n[i])[0]
                validX_fla[index_temp, i] = limit_n[i]

    trainX_fla = np.transpose(trainX_fla)
    testX_fla = np.transpose(testX_fla)
    if config.valid:
        validX_fla = np.transpose(validX_fla)

    trainX_trans = np.zeros(np.shape(trainX))
    testX_trans = np.zeros(np.shape(testX))
    if config.valid:
        validX_trans = np.zeros(np.shape(validX))

    idx_2 = 0
    for i_trial in range(n_tr_x):
        idx_1 = idx_2
        idx_2 = idx_1 + trainLens[i_trial]
        trainX_trans[i_trial, :, 0:trainLens[i_trial]] = trainX_fla[:, idx_1:idx_2]


    idx_2 = 0
    for i_trial in range(n_te_x):
        idx_1 = idx_2
        idx_2 = idx_1 + testLens[i_trial]
        testX_trans[i_trial, :, 0:testLens[i_trial]] = testX_fla[:, idx_1:idx_2]


    if config.valid:
        idx_2 = 0
        for i_trial in range(n_va_x):
            idx_1 = idx_2
            idx_2 = idx_1 + validLens[i_trial]
            validX_trans[i_trial, :, 0:validLens[i_trial]] = validX_fla[:, idx_1:idx_2]



    dataTrans = {"trainX": trainX_trans, "testX": testX_trans,"trainLens":trainLens, "testLens":testLens}
    if config.valid:
        dataTrans["validX"] = validX_trans
        dataTrans["validLens"] = validLens
    return dataTrans

def data_stand_channel(config, data):
    trainX, testX = data["trainX"], data["testX"]

    if config.valid:
        validX = data["validX"]

    if len(trainX.shape) == 3:
        n_tr_x, n_channel, n_points = trainX.shape
        n_te_x, _, _ = testX.shape
        if config.valid:
            n_va_x, _, _ = validX.shape

        trainX = trainX.transpose(0, 2, 1).reshape([-1, n_channel])
        testX = testX.transpose(0, 2, 1).reshape([-1, n_channel])

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
        trainX = trainX.reshape([n_tr_x, n_points, -1]).transpose(0, 2, 1)
        testX = testX.reshape([n_te_x, n_points, -1]).transpose(0, 2, 1)
        if config.valid:
            validX = validX.transpose(0, 2, 1).reshape([-1, n_channel])
            validX = scaler.transform(validX)
            validX = validX.reshape([n_va_x, n_points, -1]).transpose(0, 2, 1)
    else:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
        if config.valid:
            validX = scaler.transform(validX)
    dataTrans = {"trainX": trainX, "testX": testX}
    if config.valid:
        dataTrans = {"trainX": trainX, "testX": testX, "validX": validX}
    return dataTrans

def data_stand_channel_data_vels(Data, Data_recover):
    data = []
    for key in Data.keys():
        data_tmp = Data[key]
        data_tmp = data_tmp["KinData"]
        data.append(data_tmp)


    data_shape = [data_tmp.shape for data_tmp in data]

    data_cat = data[0]
    for i_data in range(1, len(data)):
        data_cat = np.concatenate((data_cat, data[i_data]), axis=2)
    data = data_cat

    n_tr_x, n_channel, n_points = data.shape

    data = data.transpose(0, 2, 1).reshape([-1, n_channel])

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(data)
    data = scaler.transform(data)

    data = data.reshape([n_tr_x, n_points, -1]).transpose(0, 2, 1)

    idx_1 = 0
    for i_key, key in enumerate(Data.keys()):
        idx_2= idx_1 + data_shape[i_key][2]
        Data[key]["KinData"] = data[:,:, idx_1:idx_2]
        idx_1 = idx_2

    Data_recover["STD_mean"] = scaler.mean_
    Data_recover["STD_std"] = scaler.var_
    Data_recover["AfterSTD_data"] = data

    return Data, Data_recover

def data_stand_channel_data(Data, Data_recover):
    data = Data["KinData"]
    if len( data.shape) == 3:
        n_tr_x, n_channel, n_points = data.shape

        data = data.transpose(0, 2, 1).reshape([-1, n_channel])

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(data)
        data = scaler.transform(data)

        data = data.reshape([n_tr_x, n_points, -1]).transpose(0, 2, 1)
    else:
       scaler = StandardScaler(with_mean=True, with_std=True)
       scaler.fit(data)
       data = scaler.transform(data)

    Data_recover["STD_mean"] = scaler.mean_
    Data_recover["STD_std"] = scaler.var_
    Data_recover["AfterSTD_data"] = data
    Data["KinData"] = data
    return Data, Data_recover

def data_stand_channel_data_reduction(data, Data_recover):

    n_tr_x, n_channel, n_points = data.shape

    data = data.transpose(0, 2, 1).reshape([-1, n_channel])

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(data)
    data = scaler.transform(data)

    data = data.reshape([n_tr_x, n_points, -1]).transpose(0, 2, 1)

    Data_recover["STD_mean"] = scaler.mean_
    Data_recover["STD_std"] = scaler.var_
    Data_recover["AfterSTD_data"] = data
    return data, Data_recover

def data_clip_abnormal_channel(config, data):
    trainX, testX = data["trainX"], data["testX"]

    if config.valid:
        validX = data["validX"]

    if len(trainX.shape) == 3:
        n_tr_x, n_channel, n_points = trainX.shape
        n_te_x, _, _ = testX.shape
        if config.valid:
            n_va_x, _, _ = validX.shape

        trainX = trainX.transpose(0, 2, 1).reshape([-1, n_channel])
        testX = testX.transpose(0, 2, 1).reshape([-1, n_channel])
        if config.valid:
            validX = validX.transpose(0, 2, 1).reshape([-1, n_channel])

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(trainX)
        # scaler.mean_ = Neural_kits["mean_"]
        # scaler.var_ = Neural_kits["var_"]
        limit_p = scaler.mean_ + 3 * scaler.var_
        limit_n = scaler.mean_ - 3 * scaler.var_
        for i in range(0, n_channel):
            index_temp = np.where(trainX[:, i] > limit_p[i])[0]
            trainX[index_temp, i] = limit_p[i]
            index_temp = np.where(trainX[:, i] < limit_n[i])[0]
            trainX[index_temp, i] = limit_n[i]
            index_temp = np.where(testX[:, i] > limit_p[i])[0]
            testX[index_temp, i] = limit_p[i]
            index_temp = np.where(testX[:, i] < limit_n[i])[0]
            testX[index_temp, i] = limit_n[i]
            if config.valid:
                index_temp = np.where(validX[:, i] > limit_p[i])[0]
                validX[index_temp, i] = limit_p[i]
                index_temp = np.where(validX[:, i] < limit_n[i])[0]
                validX[index_temp, i] = limit_n[i]

            # index_nromal = np.where(scaler.var_ != 0)[0]
            # trainX = trainX[:, index_nromal]
            # testX = testX[:,index_nromal]
        trainX = trainX.reshape([n_tr_x, n_points, -1]).transpose(0, 2, 1)
        testX = testX.reshape([n_te_x, n_points, -1]).transpose(0, 2, 1)
        if config.valid:
            validX = validX.reshape([n_va_x, n_points, -1]).transpose(0, 2, 1)
    else:
        n_points, n_channel = trainX.shape
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(trainX)
        limit_p = scaler.mean_ + 3 * scaler.var_
        limit_n = scaler.mean_ - 3 * scaler.var_
        for i in range(0, n_channel):
            index_temp = np.where(trainX[:, i] > limit_p[i])[0]
            trainX[index_temp, i] = limit_p[i]
            index_temp = np.where(trainX[:, i] < limit_n[i])[0]
            trainX[index_temp, i] = limit_n[i]
            index_temp = np.where(testX[:, i] > limit_p[i])[0]
            testX[index_temp, i] = limit_p[i]
            index_temp = np.where(testX[:, i] < limit_n[i])[0]
            testX[index_temp, i] = limit_n[i]
            if config.valid:
                index_temp = np.where(validX[:, i] > limit_p[i])[0]
                validX[index_temp, i] = limit_p[i]
                index_temp = np.where(validX[:, i] < limit_n[i])[0]
                validX[index_temp, i] = limit_n[i]
    dataTrans = {"trainX": trainX, "testX": testX}
    if config.valid:
        dataTrans = {"trainX": trainX, "testX": testX, "validX": validX}
    return dataTrans


def data_stand_point(config, data):
    trainX, testX = data["trainX"], data["testX"]

    if config.valid:
        validX = data["validX"]
    data_shape = trainX.shape
    if len(data_shape) == 3:
        n_tr_x, n_channel, n_points = trainX.shape
    else:
        n_tr_x, n_channel, n_fre, n_points = trainX.shape

    n_te_x = testX.shape[0]
    if config.valid:
        n_va_x = validX.shape[0]

    trainX = trainX.reshape([n_tr_x, -1])
    testX = testX.reshape([n_te_x, -1])

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)

    if len(data_shape) == 3:
        trainX = trainX.reshape([n_tr_x, n_channel, n_points])
        testX = testX.reshape([n_te_x, n_channel, n_points])
    else:
        trainX = trainX.reshape([n_tr_x, n_channel, n_fre, n_points])
        testX = testX.reshape([n_te_x, n_channel, n_fre, n_points])

    if config.valid:
        validX = validX.reshape([n_va_x, -1])
        validX = scaler.transform(validX)
        if len(data_shape) == 3:
            validX = validX.reshape([n_va_x, n_channel, n_points])
        else:validX = validX.reshape([n_va_x, n_channel, n_fre, n_points])
    dataTrans = {"trainX": trainX, "testX": testX}
    if config.valid:
        dataTrans = {"trainX": trainX, "testX": testX, "validX": validX}
    return dataTrans


def data_stand_submean_channel(config, data):
    trainX, testX = data["trainX"], data["testX"]

    if config.valid:
        validX = data["validX"]

    if len(trainX.shape) == 3:
        n_tr_x, n_channel, n_points = trainX.shape
        n_te_x, _, _ = testX.shape
        if config.valid:
            n_va_x, _, _ = validX.shape

        trainX = trainX.transpose(0, 2, 1).reshape([-1, n_channel])
        testX = testX.transpose(0, 2, 1).reshape([-1, n_channel])

        mean_ = np.mean(trainX, 0)
        trainX = trainX - mean_
        testX = testX - mean_
        trainX = trainX.reshape([n_tr_x, n_points, -1]).transpose(0, 2, 1)
        testX = testX.reshape([n_te_x, n_points, -1]).transpose(0, 2, 1)
        if config.valid:
            validX = validX.transpose(0, 2, 1).reshape([-1, n_channel])
            validX = validX - mean_
            validX = validX.reshape([n_va_x, n_points, -1]).transpose(0, 2, 1)
        else:
            mean_ = np.mean(trainX, 0)
            trainX = trainX - mean_
            testX = testX - mean_
            if config.valid:
                validX = validX - mean_
        dataTrans = {"trainX": trainX, "testX": testX}
        if config.valid:
            dataTrans = {"trainX": trainX, "testX": testX, "validX": validX}


    return dataTrans





