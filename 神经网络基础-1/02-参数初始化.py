import torch.nn as nn
import torch
import math

"""
    参数（w权重和b偏置）初始化如何选择？
        1- 激活函数的角度
            1.1- ReLU：使用kaiming初始化
            1.2- 其他激活函数：例如Tanh、Sigmoid推荐使用xavier初始化
            
        2- 神经网络的层数/深度
            2.1- 浅层神经网络：可以使用kaiming、xavier、均匀分布初始化、正态分布初始化
            2.2- 深层神经网络：一般是层数>=10，推荐使用kaiming、xavier
"""


def uniform_demo():
    """
    使用均匀分布初始化线性层的权重和偏置参数。
    说明：
        - 权重和偏置都被初始化为U(-1, 1)范围内的随机值
        - 适用于浅层网络或特定实验场景，但可能不是最优选择
    """
    # 创建神经网络层：输入特征5维，输出特征3维
    linear = nn.Linear(5, 3)

    # 参数初始化：权重和偏置都使用均匀分布U(0,1)
    nn.init.uniform_(linear.weight)  # 默认范围[0,1]
    nn.init.uniform_(linear.bias)  # 默认范围[0,1]

    print("均匀分布初始化 - 权重形状:", linear.weight.shape)
    print("权重前3个值:", linear.weight.data[0, :3])
    print("偏置值:", linear.bias.data)

# 预期输出分析：
# 权重形状: torch.Size([3, 5]) - 符合out_features×in_features
# 权重值: 在[0,1]范围内均匀分布的随机数
# 优点: 实现简单，打破对称性
# 缺点: 范围选择不当可能导致梯度问题

def normal_demo():
    """
    使用正态分布初始化线性层的权重参数。
    说明：
        - 权重被初始化为均值为0、标准差为1的正态分布N(0,1)
        - 偏置未显式初始化，将使用PyTorch默认初始化（通常是均匀分布）
    """
    # 创建神经网络层
    linear = nn.Linear(5, 3)

    # 参数初始化：权重使用正态分布，偏置使用默认初始化
    # 使用正态分布初始化线性层权重参数
    # 参数说明:
    #   linear.weight - 线性层的权重张量
    #   mean=0 - 正态分布的均值为0
    #   std=1 - 正态分布的标准差为1
    # 该操作将线性层权重按照标准正态分布N(0,1)进行随机初始化
    nn.init.normal_(linear.weight, mean=0, std=1)  # N(0,1)分布

    # 偏置bias保持PyTorch默认初始化（通常为均匀分布）

    print("正态分布初始化 - 权重形状:", linear.weight.shape)
    print("权重统计 - 均值: {:.4f}, 标准差: {:.4f}".format(
        linear.weight.mean().item(), linear.weight.std().item()))
    print("偏置形状:", linear.bias.shape)

# 预期输出分析：
# 权重近似服从N(0,1)，实际均值和标准差接近0和1
# 优点: 有效打破对称性，适合浅层网络
# 缺点: 标准差设置不当可能导致梯度爆炸或消失

def zeros_demo():
    """
    将线性层的权重参数初始化为全零。
    说明：
        - 所有权重设置为0
        - 严重不推荐用于实际训练，因为会导致对称性问题
        - 所有神经元会有相同的输出和梯度，无法有效学习不同特征
    """
    # 创建神经网络层
    linear = nn.Linear(5, 3)

    # 参数初始化：权重全零，偏置也全零（演示用途）
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)  # 偏置通常可以初始化为0

    print("全零初始化 - 权重:")
    print(linear.weight.data)
    print("所有权重是否为0:", torch.all(linear.weight == 0).item())


# 预期输出分析：
# 权重矩阵全部为0，偏置向量也全部为0
# 问题: 前向传播时所有神经元输出相同，反向传播时梯度相同
# 适用场景: 仅用于调试或测试，实际训练中应避免

def ones_demo():
    """
    将线性层的权重参数初始化为全1。
    说明：
        - 所有权重设置为1
        - 可能导致激活值过大，引发梯度爆炸问题
        - 实际应用中极少使用
    """
    # 创建神经网络层
    linear = nn.Linear(5, 3)

    # 参数初始化：权重全1
    nn.init.ones_(linear.weight)
    # 偏置使用默认初始化

    print("全一初始化 - 权重:")
    print(linear.weight.data)
    print("权重总和:", linear.weight.sum().item())


# 预期输出分析：
# 权重矩阵所有元素均为1
# 风险: 网络层输出会很大，容易导致梯度爆炸
# 适用场景: 基本不用于实际训练，仅用于特定测试

def constant_demo():
    """
    将线性层的权重参数初始化为指定常数值。
    说明：
        - 所有权重设置为val参数指定的固定值
        - 主要用于实验调试或特殊网络结构
    """
    # 创建神经网络层
    linear = nn.Linear(5, 3)

    # 参数初始化：权重设为常数10
    nn.init.constant_(linear.weight, val=10)
    # 偏置可以设为其他常数，如0.1
    nn.init.constant_(linear.bias, val=0.1)

    print("常数初始化 - 权重前几个值:", linear.weight.data[0, :3])
    print("偏置值:", linear.bias.data)
    print("所有权重是否等于10:", torch.all(linear.weight == 10).item())


# 预期输出分析：
# 权重矩阵所有元素均为10，偏置均为0.1
# 问题: 无法打破对称性，所有神经元学习相同特征
# 用途: 调试网络结构，测试特定数值行为

def kaiming_demo():
    """
    使用Kaiming方法初始化线性层的权重参数。
    说明：
        - 专为ReLU及其变体设计，考虑激活函数的特性
        - 包括均匀分布和正态分布两种方式
        - 推荐用于ReLU、Leaky ReLU等激活函数的网络
    """
    # 创建神经网络层
    linear = nn.Linear(5, 3)

    # Kaiming均匀初始化：适合ReLU激活函数
    nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    print("Kaiming均匀初始化 - 权重范围: [{:.4f}, {:.4f}]".format(
        linear.weight.min().item(), linear.weight.max().item()))

    # 重新初始化：Kaiming正态分布初始化
    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
    mean_val = linear.weight.mean().item()
    std_val = linear.weight.std().item()
    print("Kaiming正态初始化 - 均值: {:.4f}, 标准差: {:.4f}".format(mean_val, std_val))

    # 验证Kaiming初始化的数学原理
    fan_in = linear.weight.size(1)  # 输入特征数5
    expected_std = math.sqrt(2.0 / fan_in)  # ReLU的增益为√2
    print("理论标准差: {:.4f}, 实际标准差: {:.4f}".format(expected_std, std_val))


# 预期输出分析：
# 均匀分布：权重在[-bound, bound]范围内，bound = √(6/fan_in)
# 正态分布：权重服从N(0, √(2/fan_in))
# 优点: 保持每层输出的方差稳定，避免梯度消失/爆炸
# 适用: 深层网络，ReLU家族激活函数

def xavier_demo():
    """
    使用Xavier方法初始化线性层的权重参数。
    说明：
        - 适用于Sigmoid、Tanh等S型激活函数
        - 目标是使输入和输出的方差相同
        - 包括均匀分布和正态分布两种实现
    """
    # 创建神经网络层
    linear = nn.Linear(5, 3)

    # Xavier均匀初始化
    nn.init.xavier_uniform_(linear.weight)
    uniform_range = math.sqrt(6.0 / (linear.weight.size(1) + linear.weight.size(0)))
    print("Xavier均匀初始化 - 理论范围: ±{:.4f}".format(uniform_range))
    print("实际范围: [{:.4f}, {:.4f}]".format(
        linear.weight.min().item(), linear.weight.max().item()))

    # Xavier正态初始化
    nn.init.xavier_normal_(linear.weight)
    mean_val = linear.weight.mean().item()
    std_val = linear.weight.std().item()
    expected_std = math.sqrt(2.0 / (linear.weight.size(1) + linear.weight.size(0)))
    print("Xavier正态初始化 - 均值: {:.4f}, 标准差: {:.4f}".format(mean_val, std_val))
    print("理论标准差: {:.4f}".format(expected_std))


# 预期输出分析：
# 均匀分布：U(-a, a)，其中a = √(6/(fan_in + fan_out))
# 正态分布：N(0, √(2/(fan_in + fan_out)))
# 优点: 适合S型函数，保持输入输出方差一致
# 适用: Tanh、Sigmoid激活函数，中等深度网络

if __name__ == '__main__':
    """
    主函数：测试所有初始化方法并比较效果
    """
    print("=" * 60)
    print("神经网络参数初始化方法综合演示")
    print("=" * 60)

    # 测试所有初始化方法
    initialization_methods = [
        ("均匀分布", uniform_demo),
        ("正态分布", normal_demo),
        ("全零初始化", zeros_demo),
        ("全一初始化", ones_demo),
        ("常数初始化", constant_demo),
        ("Kaiming初始化", kaiming_demo),
        ("Xavier初始化", xavier_demo)
    ]

    for name, method in initialization_methods:
        print(f"\n>>> 测试{name}:")
        method()
        print("-" * 40)

# 预期输出总结：
# 均匀分布：简单实现，适用于实验调试
# 正态分布：打破对称性，适用于深层网络
# 全零/全一：调试用，不建议用于训练
# 常数初始化：特殊场景，需谨慎选择常数
# Kaiming：针对ReLU，避免梯度消失/爆炸
# Xavier：针对S型函数，保持输入输出方差一致
"""
    各初始化方法比较表：

    方法            | 适用场景                  | 优点                  | 缺点
    ---------------+-------------------------+-----------------------+---------------------------
    均匀分布         | 浅层网络，实验调试         | 实现简单              | 可能不是最优选择
    正态分布         | 浅层网络                 | 打破对称性            | 需要谨慎选择标准差
    全零/全一        | 仅用于调试               | 实现简单              | 导致对称性问题，不用于训练
    常数初始化       | 特殊实验                 | 可控性强              | 无法打破对称性
    Kaiming        | ReLU网络，深层网络       | 避免梯度消失/爆炸        | 对S型函数效果一般  
    Xavier         | Tanh/Sigmoid，中等深度  | 保持输入输出方差一致      | 对ReLU效果欠佳
"""
