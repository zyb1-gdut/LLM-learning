
import torch

"""
    批量归一化分为如下3类：
        BatchNorm1d：主要用来处理一维数据，例如文本。输入数据的形状要求是(N条样本,num_features多个特征)
        BatchNorm2d：主要用来处理二维数据，例如图片。输入数据的形状要求是(N, C, H, W)。
                     N表示N张图片，C表示颜色通道，H图片的高度，W图片的宽度
        BatchNorm3d：主要处理三维数据，例如：视频、医学影像等。输入数据的形状要求是(N, C, D, H, W)
"""
def demo1():
    # 准备输入的特征数据
    # 1张图片，2个颜色通道，图片高度是3，宽度是4
    input_2d = torch.randn(size=(1,2,3,4))
    print("批量归一化前的数据：",input_2d)

    # 批量归一化处理
    """
        num_features：在处理图片的时候，该值需要与图片的颜色通道保持一致
        eps：小常数，为了避免分母为0
        affine：通常都设置为True，表示神经网络自动去调整γ和β
    """
    # 创建一个二维批归一化层(torch.nn.BatchNorm2d)实例
    # 参数说明:
    #   num_features: 输入特征图的通道数，设置为2表示处理2通道的输入
    #   eps: 用于数值稳定性的小常数，防止除零错误，设置为1e-5
    #   momentum: 动量参数，用于计算运行统计量的动量值，设置为0.1
    #   affine: 布尔值，控制是否使用可学习的仿射变换参数，设置为True表示启用
    # 返回值:
    #   torch.nn.BatchNorm2d对象，用于对2D输入进行批归一化操作
    bn2d = torch.nn.BatchNorm2d(num_features=2,eps=1e-5,momentum=0.1,affine=True)

    output = bn2d(input_2d)

    print(output)
    print(bn2d.weight) # 也就是γ
    print(bn2d.bias) # 也就是β

def demo2():
    # 创建测试样本
    # 2个样本, 1个特征
    # 不能创建1个样本, 无法统计均值和方差
    input_1d = torch.randn(size=(2, 2))
    # 创建线性层对象
    linear1 = torch.nn.Linear(in_features=2, out_features=3)
    # 创建BN层对象
    # num_features：输入特征数
    bn1d = torch.nn.BatchNorm1d(num_features=3)  # 20 output features
    output_1d = linear1(input_1d)
    # 进行批量归一化
    output = bn1d(output_1d)
    print("output-->", output)
    print(output.size()) # (32, 20)

if __name__ == '__main__':
    # demo1()
    demo2()