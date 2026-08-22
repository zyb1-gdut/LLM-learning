import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块

"""
    池化层总结：
        1- 作用：对卷积层输出的特征图激活以后的数据进行降维（下采样）操作
        2- 【非常重要】特点：池化层是一个无参数的操作，也就是不需要根据输入数据进行训练学习得到权重、偏置
"""

def single_channel():
    # 准备数据
    torch.manual_seed(817)  # 设置随机种子以确保结果可重现
    # 生成一个1×4×4的随机整数张量作为单通道特征图数据，数值范围在1-49之间
    img_data = torch.randint(low=1, high=50, size=[1, 4, 4], dtype=torch.float32)
    print("特征图数据：", img_data)  # 打印生成的特征图数据

    # 最大池化
    """
        kernel_size：池化层核的形状大小，一般是3*3、5*5、7*7
        stride：向下、向右移动的步长
        padding：在卷积层输出的特征图四周进行填充，一般是0层、1层、2层
    """
    # 创建最大池化层实例，使用2×2的池化窗口，步长为1，无填充
    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    # 对特征图数据执行最大池化操作
    max_result = max_pool(img_data)
    print("最大池化输出结果", max_result)  # 打印最大池化结果

    # 平均池化
    # 创建平均池化层实例，使用2×2的池化窗口，步长为1，无填充
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    # 对特征图数据执行平均池化操作
    avg_result = avg_pool(img_data)
    print("平均池化输出结果", avg_result)  # 打印平均池化结果

def multi_channel():
    # 准备数据
    torch.manual_seed(817)  # 设置随机种子以确保结果可重现
    # 生成一个3×4×4的随机整数张量作为多通道特征图数据，数值范围在1-49之间
    img_data = torch.randint(low=1, high=50, size=[3, 4, 4], dtype=torch.float32)
    print("特征图数据：", img_data)  # 打印生成的特征图数据

    # 最大池化
    # 创建最大池化层实例，使用2×2的池化窗口，步长为1，无填充
    max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    # 对多通道特征图数据执行最大池化操作
    max_result = max_pool(img_data)
    print("最大池化输出结果", max_result)  # 打印最大池化结果

# 程序主入口点
if __name__ == '__main__':
    # 单通道池化示例
    # single_channel()

    # 多通道池化示例（当前被注释掉）
    multi_channel()
