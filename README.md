# 粗糙度检测

一、使用预训练的 vgg 模型
    
    1、直接运行main.py即可。其中需要设置的超参数在上面的def_args中
    2、报告显存不够时，降低batchsize
    3、跑完一百代后，会画出训练和测试曲线。
        当训练曲线不稳定时，降低学习率，
        当训练损失降不下去时，尝试提高学习率；
        当过拟合时，增加batchsize和weight_decay
    4、目前使用的损失函数为l1损失，可以尝试mse损失
    5、如果训练损失无论如何都降不下去时，尝试使用ResNet；
        如果还不行，考虑数据自身的问题；
        由于数据量较少，vit或者swin vit可能效果不一定好于resnet。
        等后面有时间帮你把这些涉及到的网络写一下