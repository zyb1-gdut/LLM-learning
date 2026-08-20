import torch
import torch.nn as nn

"""
    梯度下滑优化方式总结：
        1- 优化分类有：momentum、adagrad、rmsprop、adam
        2- momentum动量法:
            特点:只对梯度进行优化
            作用:平缓、鞍点
        
        3- adagrad
            特点：只对学习率进行优化
            作用：因为可能会导致学习率更新速度过快，导致学习率国小，出现鞍点或缓慢的情况
        
        4- rmsprop
            特点:只对学习率进行优化。相对adagrad增加了β的系数
            
        5- adam
            特点：对学习率、梯度都进行了优化
           
"""
def momentum_demo():
    #初始化权重值
    w = torch.tensor([1.0],requires_grad = True, dtype = torch.float32)

    #自定义损失函数
    criterion = (w**2)/2

    #创建优化器对象，作用：1- 计算梯度；2- 自动更新权重
    #创建随机梯度下降优化器，用于更新模型参数
    #参数说明：
    #   [w] - 需要优化的参数列表，这里是对变量w进行优化
    #   lr = 0.1 - 学习率，控制参数更新的步长大熊
    #   momentum = 0.9 - 动量参数，用于加速收敛并减小震荡。
    optimizer = torch.optim.SGD([w], lr = 0.1,momentum=0.9)

    #反向传播计算梯度
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    criterion.sum().backward()
    #更新权重
    optimizer.step()
    print(f"第一次，梯度的结果：{w.grad},更新后的w权重值：{w.detach()}")
    print(f"第一次criterion值为{criterion.sum().detach()}")

    #第二次
    criterion = (w**2)/2
    #反向传播计算梯度
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    criterion.sum().backward()
    #更新权重
    optimizer.step()
    print(f"第二次，梯度的结果：{w.grad},更新后的w权重值：{w.detach()}")
    print(f"第二次criterion值为{criterion.sum().detach()}")

    criterion = (w**2)/2
    #反向传播计算梯度
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    criterion.sum().backward()
    #更新权重
    optimizer.step()
    print(f"第3次，梯度的结果：{w.grad},更新后的w权重值：{w.detach()}")
    print(f"第3次criterion值为{criterion.sum().detach()}")
def adagrad_demo():
    #初始化权重值
    w = torch.tensor([1.0],requires_grad = True, dtype = torch.float32)

    #自定义损失函数
    criterion = (w**2)/2

    #创建优化器对象，作用：1-读取梯度：2- 自动化更新权重
    optimizer = torch.optim.Adagrad([w], lr = 0.1)

    #反向传播 计算梯度
    #梯度清零
    optimizer.zero_grad
    #反向传播
    criterion.sum().backward()
    #更新权重
    optimizer.step()
    print(f"第1次，梯度的结果{w.grad},更新后的w权重值{w.detach()}")

    #第二次
    criterion = (w**2)/2
    #反向传播 梯度计算
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    criterion.sum().backward()
    #更新权重
    optimizer.step()
    print(f"第2次，梯度的结果{w.grad},更新后的w权重值{w.detach()}")

    # 第二次
    criterion = (w ** 2) / 2
    # 反向传播 梯度计算
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    criterion.sum().backward()
    # 更新权重
    optimizer.step()
    print(f"第3次，梯度的结果{w.grad},更新后的w权重值{w.detach()}")
    print(f"第3次criterion值为{criterion.sum().detach()}")
if __name__ == "__main__":
    momentum_demo()
    adagrad_demo()