# coding=utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

from helper import *
from loss_func import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

from my_keras_model_zoo.resnet import ResNet, preprocess_input

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Model
import keras
import tensorflow as tf
from read_h5 import get_weights

# 获得模型
def create_model(input_shape, n_out, lr=1e-04):
    weight_path = '../pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    resnet = ResNet()
    base_model = resnet.resnet50(
        input_shape=input_shape,
        weights=weight_path
    )
    if input_shape[2] == 4:
        print('reload')
        init_weights = list()
        name_list = ['conv1/conv1_W_1:0', 'conv1/conv1_b_1:0']
        for name in name_list:
            init_weight = get_weights(weight_path, name)
            init_weights.append(init_weight)
        layer = base_model.get_layer('conv1_4')
        layer.set_weights(init_weights)
    x = base_model.output
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    #x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(n_out, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)

    graph = tf.get_default_graph()
    return model, base_model, graph

# 训练
def train(train_dataset_info, params, train_indexes, valid_indexes):
    '''
    输入:
        train_dataset_info : 训练数据信息
        args : 训练相关参数字典
    输出：
        history : 训练信息
    '''

    # get params
    check_name = params.check_name
    check_root = params.check_root
    check_path = os.path.join(check_root, check_name)
    epochs = params.epochs
    batch_size = params.batch_size
    C = params.channel
    H = params.height
    W = params.width
    input_shape = (H, W, C)
    n_out = params.n_out
    lr = params.learning_rate

    stage = params.stage

    # create model
    keras.backend.clear_session()
    model, base_model, graph = create_model(input_shape=input_shape, n_out=n_out, lr=lr)

    # if stage large than 1 , load weights
    if stage > 1:
        print('=====================================================')
        print('load weights in stage {}'.format(stage))
        check_path_ = get_check_path(check_root)
        model.load_weights(check_path_)
        shutil.rmtree(check_root)
        os.mkdir(check_root)
        print('=====================================================')
        # set base lr, middle lr/5, head lr/10

    else:
        # freeze base
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(loss=f1_loss,#loss="binary_crossentropy",
                  optimizer=SGD(lr, momentum=0.9), #clip_norm=1.0),
                  metrics=[acc(params.predict_threshold), precision(params.predict_threshold), recall(params.predict_threshold), f1(params.predict_threshold)])
    model.summary()

    # check
    checkpointer = ModelCheckpoint(
            check_path,
            verbose=2,
            monitor="val_f1_fix",
            mode='max',
            save_best_only=True
    )
    lr_reduce = ReduceLROnPlateau(
        monitor='val_f1_fix',
        mode='max',
        factor=0.5,
        epsilon=1e-5,
        patience=3,
        verbose=1,
        min_lr=0.00001
    )

    # create train and valid datagens
    dataGen = data_generator()
    train_gen = dataGen.create_train_ohem(
        train_dataset_info[train_indexes],
        batch_size,
        input_shape,
        augument=True,
        preprocessing_function=preprocess_input,
        model=model,
        graph=graph
    )
    valid_gen = dataGen.create_train(
        train_dataset_info[valid_indexes],
        batch_size,
        input_shape,
        augument=False,
        preprocessing_function=preprocess_input
    )

    # train model
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=1*(len(train_indexes) // batch_size + 1),
        validation_data=valid_gen,
        validation_steps=1*(len(valid_indexes) // batch_size + 1),
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer, lr_reduce]
    )

    return history

# 预测
def predict(params):

    # set predict
    K.set_learning_phase(0)

    # get params
    C = params.channel
    H = params.height
    W = params.width
    input_shape = (H, W, C)
    n_out = params.n_out
    predict_threshold = params.predict_threshold
    lr = params.learning_rate
    dataGen = data_generator()
    check_path = get_check_path(params.check_root)
    #check_path = get_check_path('../models/1542636132')
    #check_path = './weights.02-0.03.hdf5'
    #print(check_path)

    # load model
    model, _, _ = create_model(input_shape=input_shape, n_out=n_out, lr=lr)
    model.compile(loss=focal_loss_binary,#"binary_crossentropy",
                  optimizer=SGD(lr, momentum=0.9),
                  metrics=[acc(predict_threshold), precision(predict_threshold), recall(predict_threshold), f1(predict_threshold)])
    model.load_weights(check_path)

    # predict
    submit = pd.read_csv('../AllData/sample_submission.csv')
    predicted = list()
    for name in tqdm(submit["Id"]):
        path = os.path.join('../AllData/test/', name)
        image = dataGen.load_image(path,
                                   input_shape,
                                   preprocessing_function=preprocess_input)
        #show_pic(image)
        score_predict = model.predict(image[np.newaxis])[0]
        #print(score_predict)
        label_predict = np.arange(n_out)[score_predict >= predict_threshold]
        if len(label_predict) == 0:
            label = np.argmax(score_predict)
            label_predict = list()
            label_predict.append(label)
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    # save
    submit['Predicted'] = predicted
    submit.to_csv('submission.csv', index=False)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--channel', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_out', type=int, default=28)
    parser.add_argument('--predict_threshold', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=1e-04)
    parser.add_argument('--check_root', type=str, default='../models')
    parser.add_argument('--check_name', type=str, default='1')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_pic_root', type=str, default='../train_plot')
    parser.add_argument('--stage', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # 获得相关参数
    args = get_args()

    csv_root_path = '../AllData'
    pic_root_path = '../AllData/train'
    train_dataset_info = load_csv(pic_root_path, csv_root_path)

    # 拆分训练集
    if args.stage == 1:
        np.random.seed(2018)
        indexes = np.arange(train_dataset_info.shape[0])
        np.random.shuffle(indexes)
        split_samples = int(len(indexes)*0.9)
        train_indexes = indexes[:split_samples]
        valid_indexes = indexes[split_samples:]
        file = open('./train_valid_indexes.txt','w')
        for i in train_indexes:
            file.write(str(i)+' ')
        file.write('\n')
        for i in valid_indexes:
            file.write(str(i)+' ')
    else:
        file = open('./train_valid_indexes.txt', 'r')
        contents = file.readlines()
        train_content = contents[0].strip().split(' ')
        valid_content = contents[1].strip().split(' ')
        train_indexes = [int(i) for i in train_content]
        valid_indexes = [int(i) for i in valid_content]

    # 训练
    history = train(train_dataset_info, args, train_indexes, valid_indexes)

    # 可视化训练曲线图
    #plot_history(history, args)

    #predict(args)
    if not args.stage == 1:
        # 预测
        predict(args)
