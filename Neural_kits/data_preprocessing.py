from Neural_kits.data_preprecess_sublayers import *
import os


def get_dataLoader(cfg):
    file_path = "{}/{}_{}_{}_{}.npz".format(cfg.root_dir, cfg.date, cfg.session,
                                            cfg.dataset,
                                            cfg.des)
    if os.path.exists(file_path):
        print(f"The file {file_path} exists.")
        loaded_data = np.load(file_path, allow_pickle=True)
        data_loaders = loaded_data['data_loaders']  # Access the first array
        Data_recover = loaded_data['Data_recover']
    else:
        print(f"The file {file_path} does not exist.")
        if not os.path.exists(cfg.root_dir):
            os.makedirs(cfg.root_dir)
        print("load dataset.........................")
        # 加载数据
        data_loaders, Data_recover = get_data_K_fold_regression_cv1session(cfg)
        np.savez(file_path, data_loaders=data_loaders, Data_recover=Data_recover)
        print(f"save file at {file_path}")
    return data_loaders, Data_recover


def get_data_K_fold_regression_cv1session(config):
    # load Neural_kits from mat
    Data = get_data_mat(config)
    print("-" * 30)
    print("original X Neural_kits shape:", np.shape(Data["dataX"]))
    print("original y Neural_kits shape:", np.shape(Data["dataY"]))


    KinData = get_data_mat_kin(config)
    KinData, Data_recover = data_prepare_before_std_kin(config, KinData)
    if config.KIN.Standardization:
        if config.cross_vel:
            KinData, Data_recover = data_stand_channel_data_vels(KinData, Data_recover)
        else:
            KinData, Data_recover = data_stand_channel_data(KinData, Data_recover)

    data_loaders = []


    indexSplitFolds = data_spilt(config, Data)
    for i in range(len(indexSplitFolds)):

        data_fold = get_data_fold(config, indexSplitFolds[i], Data)


        data_fold_trans = data_prepare_before_std(config, data_fold)
        if config.cross_vel:
            data_fold_trans = data_Standardization_cv1session(config, data_fold_trans)
        else:
            data_fold_trans = data_Standardization(config, data_fold_trans)


        modality_index = data_fold_trans["modalityIndex"]
        trainX, trainY, testX, testY, trainTrial, testTrial= data_fold_trans["trainX"], data_fold_trans["trainY"], data_fold_trans["testX"], \
                                       data_fold_trans["testY"], data_fold_trans["trainTrial"], data_fold_trans["testTrial"]

        if config.valid:
            validX, validY, validTrial = data_fold_trans["validX"], data_fold_trans["validY"], data_fold_trans["validTrial"]


        if not config.cross_vel:
            if hasattr(config, 'S2S') and config.S2S:
                trainX, trainY, trainLabel, trainTrial, trainTimeIdx = prepare_kin_data_S2S(config, trainX, trainY,
                                                                                        trainTrial, KinData)
                testX, testY, testLabel, testTrial, testTimeIdx = prepare_kin_data_S2S(config, testX, testY, testTrial,
                                                                                   KinData)
                if config.valid:
                    validX, validY, validLabel, validTrial, validTimeIdx = prepare_kin_data_S2S(config, validX, validY,
                                                                                            validTrial, KinData)
            else:
                trainX, trainY, trainLabel, trainTrial, trainTimeIdx = prepare_kin_data(config, trainX, trainY, trainTrial, KinData)
                testX, testY, testLabel, testTrial, testTimeIdx = prepare_kin_data(config, testX, testY, testTrial, KinData)
                if config.valid:
                    validX, validY, validLabel, validTrial, validTimeIdx = prepare_kin_data(config, validX, validY, validTrial, KinData)
        else:
            trainTimeIdx, validTimeIdx, testTimeIdx = [], [], []# TimeIdx is not done in the prepare_kin_data_cv1session
            testLens = data_fold_trans["testLens"]
            trainLens = data_fold_trans["trainLens"]
            if config.valid:
                validLens = data_fold_trans["validLens"]

            trainX, trainY, trainLabel, trainTrial = prepare_kin_data_cv1session(config, trainX, trainY, trainTrial,
                                                                                 KinData, trainLens)
            testX, testY, testLabel, testTrial = prepare_kin_data_cv1session(config, testX, testY, testTrial, KinData, testLens)
            if config.valid:
                validX, validY, validLabel, validTrial = prepare_kin_data_cv1session(config, validX, validY, validTrial,
                                                                                     KinData, validLens)


        if hasattr(config, 'one_array'):
            if config.one_array:
                [_, n_channel, _] = np.shape(trainX)
                n_channel_array = int(n_channel/2)
                if config.array_idx == 0:
                    trainX = trainX[:, 0:n_channel_array,:]
                    testX = testX[:, 0:n_channel_array,:]
                    if config.valid:
                        validX = validX[:, 0:n_channel_array,:]
                if config.array_idx == 1:
                    trainX = trainX[:, n_channel_array:,:]
                    testX = testX[:, n_channel_array:,:]
                    if config.valid:
                        validX = validX[:, n_channel_array:,:]

        # 转为tensor loader
        train_loader = MyDataLoader_regression(trainX, trainY, trainLabel, trainTrial, trainTimeIdx, transform=transforms.Compose([ToTensor_regression()]))
        test_loader = MyDataLoader_regression(testX, testY, testLabel, testTrial, testTimeIdx, transform=transforms.Compose([ToTensor_regression()]))
        data_loader = {"trainLoader": train_loader, "testLoader": test_loader}
        if config.valid:
            valid_loader = MyDataLoader_regression(validX, validY, validLabel, validTrial, validTimeIdx, transform=transforms.Compose([ToTensor_regression()]))
            data_loader["validLoader"] = valid_loader

        data_loaders.append(data_loader)



    return data_loaders, Data_recover

















