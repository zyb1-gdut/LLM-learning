"""
搭建神经网络的流程：
    1- 定义一个类，继承自nn.Module
    2- 类中实现两个方法：__init__、forward
        2.1- __init__：定义神经网络结构和参数初始化，也就是定义除了输入层以外的层有哪些；每层的神经元个数有多少个
        2.2- forward：激活函数、传播过程

运行报错如下：
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cpu and cuda:0! (when checking argument for
argument mat1 in method wrapper_CUDA_addmm)
该错误表示在执行矩阵运算时，参与计算的张量分布在不同的设备上（CPU和GPU），而PyTorch要求所有参与同一运算的张量必须在同一设备上。


"""
import torch
import torch.nn as nn
from torchsummary import summary


class MyANN(nn.Module):
    """
    自定义全连接神经网络类，继承自torch.nn.Module。

    网络结构包括两个隐藏层和一个输出层：
        - 隐藏层1: 输入维度3 -> 输出维度3，使用Sigmoid激活函数
        - 隐藏层2: 输入维度3 -> 输出维度2，使用ReLU激活函数
        - 输出层: 输入维度2 -> 输出维度2，使用Softmax激活函数

    权重初始化采用Xavier和Kaiming方法，偏置初始化为0。
    """

    def __init__(self):
        """
        初始化神经网络结构及参数。

        定义了以下线性变换层：
            linear1: 隐藏层1 (3 -> 3)
            linear2: 隐藏层2 (3 -> 2)
            output: 输出层 (2 -> 2)

        对各层权重和偏置进行了初始化：
            linear1 使用 Xavier 正态分布初始化
            linear2 使用 Kaiming 正态分布初始化
            所有偏置初始化为零
        """
        # 调用父类的构造函数
        super().__init__()

        # 定义神经网络各层结构
        self.linear1 = nn.Linear(3, 3)   # 隐藏层1: 3个输入特征，3个输出特征
        self.linear2 = nn.Linear(3, 2)   # 隐藏层2: 3个输入特征，2个输出特征
        self.output = nn.Linear(2, 2)    # 输出层: 2个输入特征，2个输出特征

        # 参数初始化
        nn.init.xavier_normal_(self.linear1.weight)  # 隐藏层1权重使用Xavier初始化
        nn.init.zeros_(self.linear1.bias)            # 隐藏层1偏置初始化为0

        nn.init.kaiming_normal_(self.linear2.weight) # 隐藏层2权重使用Kaiming初始化
        nn.init.zeros_(self.linear2.bias)            # 隐藏层2偏置初始化为0

    def forward(self, x):
        """
        前向传播函数，定义数据在网络中的流动过程和激活函数应用。

        参数:
            x (Tensor): 输入张量，形状为(batch_size, 3)

        返回:
            Tensor: 经过神经网络处理后的输出张量，形状为(batch_size, 2)，表示分类概率分布
        """
        # 隐藏层1：线性变换 + Sigmoid激活函数
        x = torch.sigmoid(self.linear1(x))

        # 隐藏层2：线性变换 + ReLU激活函数
        x = torch.relu(self.linear2(x))

        # 输出层：线性变换 + Softmax激活函数（按最后一个维度计算概率）
        x = torch.softmax(self.output(x), dim=-1)

        return x


def train_ann():
    """
    训练并测试自定义神经网络模型。

    主要功能包括：
        1. 构造随机输入数据用于模拟训练
        2. 实例化MyANN模型
        3. 进行前向传播获取预测结果
        4. 使用torchsummary打印模型结构信息
        5. 打印模型所有可学习参数的名称与数值
    """
    # 构造模拟输入数据：10条样本，每条样本具有3个特征
    # 检查当前电脑是否有可用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备是：", device)

    my_data = torch.randn(10, 3).to(device)

    # 创建神经网络模型实例
    model = MyANN().to(device)

    # 将数据传入模型进行前向传播，获得输出结果
    output = model(my_data)#传入数据，并进行了前向传播，得到预测值
    print("神经网络预测结果是：", output)

    print("-" * 30)

    # 打印模型结构摘要信息
    # input_size=(3,) 表示每个样本有3个特征
    # batch_size=1 表示一次处理一条数据
    summary(model, input_size=(3,), batch_size=1)

    print("-" * 30)

    # 遍历并打印模型的所有参数名称及其对应的值
    for name, param in model.named_parameters():
        print(f"参数名称{name}，参数值{param}")


if __name__ == '__main__':
    train_ann()
