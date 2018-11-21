- 复现：[link](https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb)

## train
- 先固定前面所有层，训练10个epoch， lr=0.01
- keras貌似不行。设定lr，base用lr/10的学习率，middle用lr/3的学习率，head用lr的学习率, lr=0.001, 训练50个回合
