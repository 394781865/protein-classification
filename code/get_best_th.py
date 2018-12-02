# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score as off1

def getOptimalT_helper(model, ValDataGen, valid_nums):

    lastFullValPred = np.empty((0,28))
    lastFullValLabels = np.empty((0,28))
    for i in tqdm(range(valid_nums)):
        img, lbl = next(ValDataGen)
        scores = model.predict(img)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
    print(lastFullValPred.shape, lastFullValLabels.shape)

    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:,i]>t, dtype=np.int8)
            scoref1 = off1(lastFullValLabels[:,i], p, average='binary')
            f1s[j, i] = scoref1

    print(np.max(f1s, axis=0))
    print(np.mean(np.max(f1s, axis=0)))

    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]

    print(T)

    return T, np.mean(np.max(f1s, axis=0))

def getOptimalT_main(model, input_shape):

    from helper import load_csv, data_generator
    from my_keras_model_zoo.resnet import preprocess_input

    # get valid generator
    csv_root_path = '../AllData'
    pic_root_path = '../AllData/train'
    dataset_info = load_csv(pic_root_path, csv_root_path)

    file = open('./train_valid_indexes.txt', 'r')
    contents = file.readlines()
    valid_content = contents[1].strip().split(' ')
    valid_indexes = [int(i) for i in valid_content]

    dataGen = data_generator()
    ValidDataGen = dataGen.create_train(
        dataset_info[valid_indexes],
        1,
        input_shape,
        augument=False,
        preprocessing_function=preprocess_input)

    # start optimal
    valid_nums = len(valid_indexes)
    T, _ = getOptimalT_helper(model, ValidDataGen, valid_nums)
    return T

if __name__ == '__main__':

    from main_keras import create_model

    # get best model
    input_shape = (224, 224, 4)
    n_out = 28
    model, _, _ = create_model(input_shape, n_out)
    weights_path = '../models/1543563875/weights.95-0.344.hdf5'
    model.load_weights(weights_path)

    getOptimalT_main(model, input_shape)


