import torch
import torch.nn as nn

def multi():
    """
    演示多分类交叉熵损失函数的使用方法

    该函数展示了如何使用PyTorch的CrossEntropyLoss计算多分类问题的损失值。
    支持两种真实值格式：类别索引和one-hot编码格式。

    参数:
        无

    返回值:
        无

    注意事项:
        - 当使用类别索引时，数据类型必须是torch.int64 (Long类型)
        - 当使用one-hot编码时，数据类型必须是浮点数类型
        - 预测值通常需要是浮点数类型并设置requires_grad=True以支持梯度计算
    """
    # 真实值
    # 可以是类别。注意：类型必须是长整数Long，否则会报错：expected scalar type Long but found Int
    # y_true = torch.tensor([1,2],dtype=torch.int64)

    # 也可以是热编码之后的数据。注意：类型必须是小数，只要是小数就行
    y_true = torch.tensor([[0,1,0], [0,0,1]],dtype=torch.float32)

    # 预测值
    y_pred = torch.tensor([[0.1,0.6,0.3], [0.7,0.1,0.2]],requires_grad=True,dtype=torch.float32)

    # 创建损失函数实例对象
    # CrossEntropyLoss = 线性求和结果 + softmax激活函数
    loss_obj = nn.CrossEntropyLoss()

    # 计算损失值
    loss_value = loss_obj(y_pred,y_true)
    print(loss_value)

def binary():
    """
    演示二分类交叉熵损失函数的使用方法

    该函数展示了如何使用PyTorch的BCELoss计算二分类问题的损失值。
    真实值和预测值都应该是概率形式，取值范围在[0,1]之间。

    参数:
        无

    返回值:
        无

    注意事项:
        - 真实值应该是0或1的标签
        - 预测值应该是模型输出的概率值
        - 通常需要在模型输出后添加sigmoid激活函数
    """
    # 真实概率值，要么是0，要么是1
    y_true = torch.tensor([0,1,0],dtype=torch.float32)

    # 预测概率值
    y_pred = torch.tensor([0.6901,0.5459,0.2469],requires_grad=True,dtype=torch.float32)

    # 创建损失函数实例对象
    loss = nn.BCELoss()

    # 计算损失值
    loss = loss(y_pred,y_true)
    print("二分类的损失值",loss)

if __name__ == '__main__':
    # 多分类
    multi()

    # 二分类
    binary()
