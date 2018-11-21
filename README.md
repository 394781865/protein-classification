## 想法
- [x] 废除，keras不合适(数据产生器那边可以再加个k折,https://www.kaggle.com/allunia/protein-atlas-exploration-and-baseline)
- [ ] 在预测的时候，不同的类别给与不同的判断阈值，https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb的评论里面也有
- [ ] 增强方法加上同态滤波和cutout试一试
- [x] focal loss
- [x] OHEM : reference (https://stackoverflow.com/questions/43883271/how-to-do-ohemonline-hard-example-mining-on-keras)
- [ ] 4通道输入，keras load weight的时候按名称：https://blog.csdn.net/dongapple/article/details/77530212
- [ ] 两个分支，https://www.kaggle.com/kwentar/two-branches-xception-lb-0-3
- [ ] 转化成多个二分类，训练28个小模型

## 注意
- skimage读入的图片是：(h,w,c), RGB, uint8(255)
- opencv读入的图片是：(h,w,c), BGR, uint8(255)
- pil读入的图片是：PIL类型，RGB，(w,h)

