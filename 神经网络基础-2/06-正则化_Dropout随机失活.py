import torch

if __name__ == '__main__':
    # 神经网络层的数据输入数据
    input_data = torch.randint(1,10,size=(1,4),dtype=torch.float32)

    # 构建一层网络
    linear1 = torch.nn.Linear(4,5)

    # 创建Dropout随机失活层（主要是为了防止过拟合）
    # 创建一个Dropout层，用于在训练过程中随机将输入张量中的部分元素置零
    # 参数p: 置零的概率，默认为0.2，即20%的元素会被随机置零
    # 返回值: Dropout模块实例，可以在前向传播中使用
    dropout = torch.nn.Dropout(p=0.2)

    # 线性回归加权求和计算
    x = linear1(input_data)

    # 激活函数计算
    x = torch.tanh(x)

    # Dropout随机失活：经过Dropout随机失活层以后，输出结果值变成0
    print("随机失活前的结果",x)
    x = dropout(x)
    print("随机失活后的结果",x)
