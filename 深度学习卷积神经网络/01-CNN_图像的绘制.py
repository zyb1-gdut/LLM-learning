import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 绘制纯黑图片
    # 创建一个200x200像素的黑色RGB图像数组，形状为[高度, 宽度, 通道数]
    img_np = np.zeros(shape=[200,200,3])
    # print(img_np)
    # img_np = torch.zeros(size=[200,200,3])
    # 显示图像
    plt.imshow(img_np)
    plt.show()


    # 绘制纯白图片
    img_np = np.full(shape=[200, 200, 3],fill_value=255)
    # plt.axis("off")
    plt.imshow(img_np)
    plt.show()

