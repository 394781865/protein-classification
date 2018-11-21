# coding=utf-8
# 帮助函数

from skimage.transform import resize
from tqdm import tqdm
import skimage.io
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import cv2

from DataAugmentForClassifer import *

# 得到最优模型路径
def get_check_path(root):
    first = True
    check_name = ''
    for parent, _, files in os.walk(root):
        for file in files:
            if first:
                check_name = file
                first = False
            else:
                cur = file.split('.')[1]
                cur = cur.split('-')[0]
                pre = check_name.split('.')[1]
                pre = pre.split('-')[0]
                if int(cur) > int(pre):
                    check_name = file
    check_path = os.path.join(root, check_name)
    return check_path

# show pic
def show_pic(img):
    cv2.imwrite('./1.jpg', img)
    cv2.namedWindow('pic', 0)
    cv2.moveWindow('pic', 0, 0)
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')

# transform skimage to opencv image
def skimage2opencv(img):
    from skimage import img_as_ubyte
    #img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# transform opencv image to skimage
def opencv2skimage(img):
    from skimage import img_as_float
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img_as_float(img)
    return img

# 数据迭代器
class data_generator():

    def init(self):
        pass

    def create_train(self, dataset_info, batch_size, shape, augument=False, preprocessing_function=None):
        '''
        输入：
            dataset_info : 输入数据信息，包括图片路径以及对应的label
            batch_size : 一个batch的大小
            shape : 图片的shape
            augument : 是否使用图像增强，默认否
            preprocessing_function : 读入图片的时候，对图片预处理的方法，默认为None
        输出：
            batch_images : 一个batch的图像array
            batch_labesl : 一个batch的标签one-hot向量
        '''
        assert shape[2] in [3,4]
        cnt = 0
        indexes = np.arange(dataset_info.shape[0])
        np.random.shuffle(indexes)
        while True:
            #random_indexes = np.random.choice(len(dataset_info), batch_size)
            if cnt >= len(indexes):
                cnt = 0
                np.random.shuffle(indexes)
            #print(min(cnt+batch_size, len(indexes)))
            batch_indexes = indexes[range(cnt, min(cnt+batch_size,len(indexes)))]
            #print(batch_indexes)
            cnt += batch_size
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(batch_indexes):
                image = self.load_image(
                        dataset_info[idx]['path'],
                        shape, augument, preprocessing_function)

                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1

            yield batch_images, batch_labels

    def create_train_ohem(self, dataset_info, batch_size, shape, augument=False, preprocessing_function=None, model=None, graph=None):
        '''
        输入：
            dataset_info : 输入数据信息，包括图片路径以及对应的label
            batch_size : 一个batch的大小
            shape : 图片的shape
            augument : 是否使用图像增强，默认否
            preprocessing_function : 读入图片的时候，对图片预处理的方法，默认为None
        model : 当前模型
        输出：
            batch_images : 一个batch的图像array
            batch_labesl : 一个batch的标签one-hot向量
        '''
        assert shape[2] in [3,4]
        cnt = 0
        indexes = np.arange(dataset_info.shape[0])
        np.random.shuffle(indexes)
        while True:
            #random_indexes = np.random.choice(len(dataset_info), batch_size)
            if cnt >= len(indexes):
                cnt = 0
                np.random.shuffle(indexes)
            #print(min(cnt+batch_size, len(indexes)))
            batch_indexes = indexes[range(cnt, min(cnt+batch_size,len(indexes)))]
            cnt += batch_size
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(batch_indexes):
                image = self.load_image(
                        dataset_info[idx]['path'],
                        shape, augument, preprocessing_function)

                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1

            if model:
                batch_images, batch_labels = self.generate_ohem_batch(model, batch_images, batch_labels, batch_size // 3 * 2, graph)
            yield batch_images, batch_labels

    def generate_ohem_batch(self, model, batch_images, batch_labels, k, graph=None):
        '''
            get top k loss samples
        '''

        assert batch_images.shape[0]==batch_labels.shape[0]
        assert batch_images.shape[0]>k

        import keras.backend as K
        K.set_learning_phase(0)
        # calculate loss
        losses_dict = {}
        for i in range(batch_images.shape[0]):
            img = batch_images[i]
            y_true = batch_labels[i]
            #model._make_predict_function()
            with graph.as_default():
                #print(model.get_weights()[0][0][0][0][:4])
                y_predict = model.predict(img[np.newaxis])[0]

            loss = self.cal_loss(y_true, y_predict)
            losses_dict[loss] = i

        K.set_learning_phase(1)
        # get top k loss sample
        keys = losses_dict.keys()
        sorted(keys)
        indexes = [losses_dict[key] for key in keys]
        indexes = indexes[len(indexes)-k:]

        return batch_images[indexes], batch_labels[indexes]

    def cal_loss(self, y_true, y_predict, type='f1'):
        '''
        输入:
            y_true : 真实标签
            y_predict : 预测值
            type : loss类型,默认为F1损失
        输出：
            loss : 损失值
        '''

        # need rewrite!!!!!!!
        assert len(y_true)==len(y_predict)

        if type=='f1':
            tp = 0;tn = 0;fp = 0;fn = 0
            for t, f in zip(y_true, y_predict):
                if t==1:
                    tp += f
                    fn += (1-f)
                else:
                    tn += (1-f)
                    fp += f
            f1 = 2*tp / 2*tp+fp+fn
            loss = -f1
        elif type=='category_crossentropy':
            loss = 0
            for t, f in zip(y_true, y_predict):
                if t==1:
                    loss -= np.log(f)
        elif type=='binary_crossentropy':
            loss = 0
            for t, f in zip(y_true, y_predict):
                if t==1:
                    loss += -np.log(f)
                else:
                    loss += -np.log(1-f)
        else:
            raise('no such loss yet!')

        return loss

    def load_image(self, path, shape, augument=False, preprocessing_function=None):
        '''
        输入：
            path : 图片的伪根路径
            shape : 图片的shape
            augument : 是否进行图像增强
            preprocessing_function : 对图片进行预处理的方法
        输出:
            image : 堆叠各个通道后的图像
        '''

        flags = cv2.IMREAD_GRAYSCALE
        image_red_ch = cv2.imread(path+'_red.png', flags)
        image_green_ch = cv2.imread(path+'_green.png', flags)
        image_blue_ch = cv2.imread(path+'_blue.png', flags)
        image_yellow_ch = cv2.imread(path+'_yellow.png', flags)

        #image_red_ch = skimage.io.imread(path+'_red.png')
        #image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        #image_green_ch = skimage.io.imread(path+'_green.png')
        #image_blue_ch = skimage.io.imread(path+'_blue.png')

        #print(image_red_ch.shape)
        #print(image_yellow_ch.shape)
        #print(image_green_ch.shape)
        #print(image_blue_ch.shape)

        #image_red_ch += (image_yellow_ch/2).astype(np.uint8)
        #image_green_ch += (image_yellow_ch/2).astype(np.uint8)

        image = np.stack((
                image_red_ch,
                image_green_ch,
                image_blue_ch,
                image_yellow_ch
            ), -1)

        if augument:
            image = self.augument(image)

        #image = resize(image, (shape[0], shape[1]), mode='reflect')
        image = cv2.resize(image, (shape[0], shape[1]), cv2.INTER_AREA)

        image = image.astype(np.float32)/255
        if preprocessing_function:
            image = preprocessing_function(image)

        return image

    def augument(self, image):
        #image_opencv = skimage2opencv(image)
        #show_pic(image)
        DataAug = DataAugmentForClassifer(
            rotation_rate=0.5,
            crop_rate=0.5,
            shift_rate=0,
            change_light_rate=0,
            add_noise_rate=0,
            flip_rate=0.5,
            cutout_rate=0,
            tongtai_filter_rate=0
        )
        aug_image = DataAug.dataAugment(image)
        #show_pic(aug_image)
        #aug_image = aug_image.astype(np.uint8)
        #image_skimage = opencv2skimage(aug_image)
        return aug_image

# 得到训练数据信息
def load_csv(pic_root_path, csv_root_path):
    '''
    输入：
        csv_root_path : csv文件的根目录
        pic_root_path : pic文件的根目录
    输出:
        train_data_info : 训练数据信息
    '''
    csv_path = csv_root_path+'/train.csv'
    data = pd.read_csv(csv_path)
    train_data_info = list()
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_data_info.append({'path':os.path.join(pic_root_path,name),
                                'labels':np.array([int(label) for label in labels])})
    return np.array(train_data_info)

# 画出训练的结果曲线
def plot_history(history, args, show=False, save=True):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc_fix"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc_fix"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    if show:
        plt.show()
    if save:
        save_root = args.save_pic_root
        save_root = os.path.join(save_root, args.check_root.split('/')[-1])
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        save_path1 = os.path.join(save_root, str(args.stage)+'.jpg')
        savefig(save_path1)

if __name__ == '__main__':

    csv_root_path = '../AllData'
    pic_root_path = '../AllData/train'
    train_dataset_info = load_csv(pic_root_path, csv_root_path)

    batch_size = 1
    dataGen = data_generator()
    ii = dataGen.create_train(
        train_dataset_info,
        batch_size,
        (256, 256, 4),
        augument=True
    )

    for _ in range(train_dataset_info.shape[0]//batch_size):
        next(ii)

    # cal train data mean and std
    x_tot = np.zeros(4)
    x2_tot = np.zeros(4)
    for _ in tqdm(range(train_dataset_info.shape[0] // batch_size)):
        x, _ = next(ii)
        x = x.reshape(-1, 4)
        x_tot += np.mean(x, axis=0)
        x2_tot += np.mean(x**2, axis=0)

    channel_avr = x_tot / train_dataset_info.shape[0]
    channel_std = np.sqrt(x2_tot/train_dataset_info.shape[0] - channel_avr**2)
    print(channel_avr)
    print(channel_std)

