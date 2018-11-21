# -*- coding: utf-8 -*-

import tensorflow as tf
import keras.backend as K

def acc(threshold=0.5):
    def acc_fix(y_true, y_pred):
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
        y_true = K.cast(K.greater(K.clip(y_true, 0, 1), threshold), K.floatx())
        all_ones = tf.ones_like(y_true)
        all_zeors = tf.zeros_like(y_true)
        f = tf.where(tf.equal(y_pred, y_true), all_ones, all_zeors)
        return K.mean(f)
    return acc_fix

def precision(threshold=0.5):
    def precision_fix(y_true, y_pred):
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
        y_true = K.cast(K.greater(K.clip(y_true, 0, 1), threshold), K.floatx())
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        precision_ = tp / (tp+fp+K.epsilon())
        precision_ = tf.where(tf.is_nan(precision_), tf.zeros_like(precision_), precision_)
        return K.mean(precision_)
    return precision_fix

def precision_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    precision_ = tp / (tp+fp+K.epsilon())
    precision_ = tf.where(tf.is_nan(precision_), tf.zeros_like(precision_), precision_)
    return 1 - K.mean(precision_)


def recall(threshold=0.5):
    def recall_fix(y_true, y_pred):
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
        y_true = K.cast(K.greater(K.clip(y_true, 0, 1), threshold), K.floatx())
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
        recall_ = tp / (tp+fn+K.epsilon())
        recall_ = tf.where(tf.is_nan(recall_), tf.zeros_like(recall_), recall_)
        return K.mean(recall_)
    return recall_fix


def f1(threshold=0.5):
    def f1_fix(y_true, y_pred):
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
        #y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

        return K.mean(f1)
    return f1_fix

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)

# focal loss单标签多分类
# 与https://www.kaggle.com/rejpalcz/focalloss-for-keras中的不太一样，因为那里面的输出的激活函数是线性的
# another reference : https://blog.csdn.net/legalhighhigh/article/details/81409551
def focal_loss_categorical(target_tensor, prediction_tensor):
    from tensorflow.python.ops import array_ops
    gama = 1.0
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
    one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor-prediction_tensor, zeros)
    FT = -1 * (one_minus_p ** gama) * tf.log(tf.clip_by_value(prediction_tensor, 1e-08, 1.0))
    FT = tf.reduce_sum(FT)
    return FT

# focal loss多标签多分类
# reference : https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
def focal_loss_binary(y_true, y_pred):
    gamma = 2. ; alpha=.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    #FT = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    FT = -tf.reduce_mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-tf.reduce_mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return FT
