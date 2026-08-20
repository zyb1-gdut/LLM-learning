#导入相关包
import torch
import torch.nn as nn

"""
    回归的损失函数总结：
        1- L1 损失
            特点： 也称之为MAE损失，曲线不光滑，容易错误极值点。
            使用：很少使用，一般作为正则化下添加到其他损失函数中。
        2- L2损失
            特点：也称之为MSE损失，因为对损失值求和平方，容易出现梯度爆炸的情况
            使用：较少使用，相对比MAE用的多一些。一般作为正则化下添加到其他损失函数中
        3- SmoothL1损失
            特点：集成了L1损失和L2损失，而且避免了它们的确定
            使用：经常使用
"""

def mae_demo():
    """
    演示L1损失（Mean Absolute Error）的计算过程：

    该函数通过定义真实值和预测值，使用Pytorch的L1Loss函数计算MAE损失值，
    并打印结果，L1损失是预测值与真实值之间绝对差值的平均值。

    参数：
        无

    返回值：
        无

    """
    #真实值
    y_true = torch.tensor([2.0,2.0,2.0],dtype=torch.float32)

    #预测值
    y_pred = torch.tensor([1.0,1.0,1.9],dtype=torch.float32)

    #创建损失函数实例对象
    loss = nn.L1Loss()

    #计算损失值
    loss = loss(y_pred, y_true)
    print("MAE损失值",loss)

def mse_demo():
    """
    演示L2损失（Mean Squared Error）的计算过程

    该函数通过定义真实值和预测值，使用Pytorch的MSELoss函数计算MSE损失值。
    并打印结果，L2损失是预测值与真实值之间平方差的平均值

    参数：
        无

    返回值：
        无

    :return:
    """
    #真实值
    y_true = torch.tensor([2.0,2.0,2.0],dtype=torch.float32)

    #预测值
    y_pred = torch.tensor([1.0,1.0,1.9],dtype=torch.float32)

    #创建损失函数实例对象
    loss = nn.MSELoss()

    #计算损失值
    loss= loss(y_pred, y_true)
    print("MSE损失值",loss)

def smoothl1_demo():
    """
    演示Smooth L1孙hi的计算过程：

    该函数通过定义真实值和预测值，使用Pytorch的SmoothL1Loss函数计算Smooth L1损失值
    并打印结果。Smooth L1 损失结合了L1和L2损失的优点。在误差较小时，使用平方损失，
    在误差较大时使用线性损失，具有更好的鲁棒性。

    参数：
        无

    返回值：
        无

    """
    #真实值
    y_true = torch.tensor([0,3],dtype=torch.float32)

    #预测值
    y_pred = torch.tensor([0.6,0.4],dtype=torch.float32)

    #创建损失函数是咧对象
    loss = nn.SmoothL1Loss()

    #计算损失值
    loss = loss(y_pred, y_true)
    print("SmoothL1损失值：",loss)

if __name__ == "__main__":
    #L1 Loss. MAE
    mae_demo()

    #L2 Loss. MSE
    mse_demo()

    #SmoothL1 loss
    smoothl1_demo()
