## 想法
- [x] 在预测的时候，不同的类别给与不同的判断阈值，https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb的评论里面也有
- [x] focal loss
- [x] OHEM : reference (https://stackoverflow.com/questions/43883271/how-to-do-ohemonline-hard-example-mining-on-keras)
- [x] 4通道输入，keras load weight的时候按名称：https://blog.csdn.net/dongapple/article/details/77530212

## 注意
- skimage读入的图片是：(h,w,c), RGB, uint8(255)
- opencv读入的图片是：(h,w,c), BGR, uint8(255)
- pil读入的图片是：PIL类型，RGB，(w,h)

## time slot
- [ ] 尝试512+256, 阈值用动态调整的阈值(2018.12.2)
