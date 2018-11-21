# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for classifer
# author:
#     maozezhong 2018-11-3
##############################################################

# 包括:
#     1. 裁剪
#     2. 平移
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度
#     6. 镜像
#     7. cutout
# 注意:
#     random.seed(),相同的seed,产生的随机数是一样的!!

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

# 图像均为cv2读取
class DataAugmentForClassifer():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5,
                cutout_rate=0.5, cut_out_length=50, cut_out_holes=2, cut_out_threshold=0.5,
                tongtai_filter_rate=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate
        self.tongtai_filter_rate = tongtai_filter_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    # 同态滤波
    def _tongtai_filter(self, img):
        '''
        https://blog.csdn.net/cjsh_123456/article/details/79351654
        '''
        img = np.float32(img)
        img = img/255

        rows,cols,dim=img.shape
        rh, rl, cutoff = 2.5, 0.5, 64

        imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y,cr,cb = cv2.split(imgYCrCb)

        y_log = np.log(y+0.01)

        # 傅里叶变换
        y_fft = np.fft.fft2(y_log)

        # 中心化?
        y_fft_shift = np.fft.fftshift(y_fft)

        DX = cols/cutoff
        G = np.ones((rows,cols))
        for i in range(rows):
            for j in range(cols):
                G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl

        result_filter = G * y_fft_shift
        result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
        result = np.exp(result_interm)-0.01

        target_pic_path = './tongtai.jpg'
        cv2.imwrite(target_pic_path, result*255)
        img = cv2.imread(target_pic_path)
        os.remove(target_pic_path)

        return img

    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time()))
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True)*255


    # 调整亮度
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5) #flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)

    # cutout
    def _cutout(self, img, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        # 得到h和w
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape

        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)    #numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        target_pic_path = './cut.jpg'
        cv2.imwrite(target_pic_path, img)
        img = cv2.imread(target_pic_path)
        os.remove(target_pic_path)

        return img

    # 旋转
    def _rotate_img(self, img, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
        '''
        #---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        return rot_img

    # 裁剪
    def _crop_img(self, img):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
        输出:
            crop_img:裁剪后的图像array
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]

        lx = int(w*0.1)
        ly = int(h*0.1)

        #随机扩展这个最小框
        crop_x_min = int(random.uniform(0, lx))
        crop_y_min = int(random.uniform(0, ly))
        crop_x_max = int(w - random.uniform(0, lx))
        crop_y_max = int(h - random.uniform(0, ly))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        return crop_img

    # 平移
    def _shift_pic(self, img):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
        输出:
            shift_img:平移后的图像array
        '''
        #---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]

        d_to_left = int(w*0.08)           #包含所有目标框的最大左移动距离
        d_to_right = int(w*0.08)      #包含所有目标框的最大右移动距离
        d_to_top = int(h*0.08)            #包含所有目标框的最大上移动距离
        d_to_bottom = int(h*0.08)     #包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left-1), (d_to_right-1))
        y = random.uniform(-(d_to_top-1), (d_to_bottom-1))

        M = np.float32([[1, 0, x], [0, 1, y]])  #x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return shift_img

    # 镜像
    def _filp_pic(self, img):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
            输出:
                flip_img:平移后的图像array
        '''
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon: #水平翻转
            flip_img =  cv2.flip(flip_img, -1)
        else:
            flip_img = cv2.flip(flip_img, 0)

        return flip_img

    def dataAugment(self, img):
        '''
        图像增强
        输入:
            img:图像array
        输出:
            img:增强后的图像
        '''
        change_num = 0  #改变的次数
        #print('------')
        while change_num < 1:   #默认至少有一种数据增强生效
            if random.random() < self.crop_rate:        #裁剪
                #print('裁剪')
                change_num += 1
                img = self._crop_img(img)

            if random.random() < self.rotation_rate:    #旋转
                #print('旋转')
                change_num += 1
                # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.7, 0.8)
                img = self._rotate_img(img, angle, scale)

            if random.random() < self.shift_rate:        #平移
                #print('平移')
                change_num += 1
                img = self._shift_pic(img)

            if random.random() < self.change_light_rate: #改变亮度
                #print('亮度')
                change_num += 1
                img = self._changeLight(img)

            if random.random() < self.add_noise_rate:    #加噪声
                #print('加噪声')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  #cutout
                #print('cutout')
                change_num += 1
                img = self._cutout(img, length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:    #翻转
                #print('翻转')
                change_num += 1
                img = self._filp_pic(img)

            if random.random() < self.tongtai_filter_rate:  #同态滤波
                change_num += 1
                img = self._tongtai_filter(img)
            #print('\n')
        # print('------')
        return img


if __name__ == '__main__':

    ### test ###

    import shutil
    need_aug_num = 1

    dataAug = DataAugmentForObjectDetection()

    source_pic_root_path = './data_split'
    source_xml_root_path = './data_voc/VOC2007/Annotations'


    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:
            cnt = 0
            while cnt < need_aug_num:
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
                coords = parse_xml(xml_path)        #解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
                coords = [coord[:4] for coord in coords]

                img = cv2.imread(pic_path)
                show_pic(img, coords)    # 原图

                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                cnt += 1

                show_pic(auged_img, auged_bboxes)  # 强化后的图


