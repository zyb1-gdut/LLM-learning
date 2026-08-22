import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import matplotlib.pyplot as plt  # 导入matplotlib绘图库

"""
CNN卷积神经网络的组成
    卷积层：提取特征
    池化层：降维
    全连接层：输出结果

处理流程
    1- 读取图片
    2- 创建卷积层
    3- 将图片输入到卷积层中处理，经过Filter卷积核进行特征提取，得到特征图
        3.1- 图片的原始形状[H高度,W宽度,C通道]
        3.2- 输入进卷积层中的图片形状需要调整。首先是调整为[C,H,W]将通道对应的维度提到最前面
        3.3- 然后需要指定图片的张数，也就是需要继续将形状改为[N图片的张数,C,H,W]
    4- 将经过卷积层处理后的图片数据，再以图片的形式展示出来，那么又需要进行相反的形状操作
        [N图片的张数,C,H,W] -> [C,H,W] -> [H,W,C]

遇到如下异常，如何处理呢？
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
原因：matplotlib和torch框架中都有libiomp5md.dll组件，但是该组件只允许初始化一次
解决：删除torch框架下的libiomp5md.dll
"""

if __name__ == '__main__':  # 程序入口点，确保只有直接运行此脚本时才执行以下代码
    # 1 - 读取图片
    img_data = plt.imread("data/img.jpg")  # 使用matplotlib读取图片文件，返回numpy数组
    print("原始图片形状：", img_data.shape)  # 打印原始图片的形状，通常是(H, W, C)格式

    # 2 - 创建卷积层【掌握】
    """
        in_channels：输入的图片通道数，RGB的取值是3
        out_channels：输出的图片通道数，也就是卷积核个数，可以根据业务调整
        kernel_size：卷积核的高度宽度大小。一般是3*3、5*5、7*7
        stride：移动的步长。有两类值
            如果是整数，表示向右、向下移动的距离是相同
            如果是元组，参数含义是(向右的步长, 向下的步长)
        padding：图片四周填充的圈数
            padding=数字：最常用。表示填充多少圈。一般工作中填充2圈以内
            padding="same"：CNN网络自动进行填充，实现特征图的形状与原始图片的形状相同
    """
    # 创建一个2D卷积层对象
    conv2d = nn.Conv2d(
        in_channels=3,     # 输入通道数为3（RGB图像）
        out_channels=4,    # 输出通道数为4（使用4个卷积核）
        kernel_size=3,     # 卷积核大小为3x3
        stride=1,          # 步长为1
        padding=0         # 不进行边缘填充
    )

    # 3 - 将图片输入到卷积层中处理，经过Filter卷积核进行特征提取，得到特征图
    # 3.1 - 图片的原始形状[H高度, W宽度, C通道]
    # 3.2 - 输入进卷积层中的图片形状需要调整。首先是调整为[C, H, W]
    # 将numpy数组转换为torch张量，并调整维度顺序从(H, W, C)变为(C, H, W)
    permute_img_data = torch.tensor(img_data).permute(dims=[2, 0, 1])
    print("维度交换后的形状：", permute_img_data.shape)  # 打印调整后的形状

    # 3.3 - 然后需要指定图片的张数，也就是需要继续将形状改为[N图片的张数, C, H, W]
    # 在第0维度添加一个维度，表示批次大小为1，形状变为(1, C, H, W)
    # 将permute_img_data张量在维度0上增加一个维度
    # 该操作会修改原张量，不返回新张量
    # 参数dim=0指定在第0维添加新维度
    permute_img_data.unsqueeze_(dim=0)
    print("升维后的形状：", permute_img_data.shape)  # 打印升维后的形状

    # 3.4- 卷积层对形状修改后的图片数据进行实际的处理
    """
        这里为什么要将数据类型变成float32小数？
        因为带入到特征图计算公式中，过程中可能会产生小数的情况。如果不调整类型，会报如下的错误：
        RuntimeError: Input type (unsigned char) and bias type (float) should be the same
    """
    # 将数据类型转换为float32并传入卷积层进行前向传播计算
    feature_map = conv2d(permute_img_data.type(torch.float32))
    print(feature_map, type(feature_map))  # 打印特征图和其类型

    # 4 - 将经过卷积层处理后的图片数据，再以图片的形式展示出来，那么又需要进行相反的形状操作
    # [N图片的张数, C, H, W] -> [C, H, W] -> [H, W, C]
    # 移除批次维度，从(1, C, H, W)变为(C, H, W)
    squeeze_img_data = feature_map.squeeze(dim=0)
    # 调整维度顺序从(C, H, W)变为(H, W, C)，以便于图像显示
    new_img_data = squeeze_img_data.permute(dims=[1, 2, 0])
    print(new_img_data.shape)  # 打印最终形状

    # 分别展示4个通道的图片
    # 提取第0个通道的特征图并转换为numpy数组用于显示
    img_channel_0 = new_img_data[:, :, 0].detach().numpy()
    plt.imshow(img_channel_0)  # 显示图像
    plt.show()  # 展示图像窗口

    # 提取第1个通道的特征图并显示
    img_channel_1 = new_img_data[:, :, 1].detach().numpy()
    plt.imshow(img_channel_1)
    plt.show()

    # 提取第2个通道的特征图并显示
    img_channel_2 = new_img_data[:, :, 2].detach().numpy()
    plt.imshow(img_channel_2)
    plt.show()

    # 提取第3个通道的特征图并显示
    img_channel_3 = new_img_data[:, :, 3].detach().numpy()
    plt.imshow(img_channel_3)
    plt.show()
