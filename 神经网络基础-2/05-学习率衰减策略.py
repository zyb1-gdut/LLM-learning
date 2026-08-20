import matplotlib.pyplot as plt
import torch
import torch.nn

def demo():
    # 定义真实值、特征值、权重值
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype=torch.float32)
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float32)

    # 梯度下降优化器
    optimizer = torch.optim.SGD([w],lr=0.1,momentum=0.9)

    # 学习率_等间隔下降策略
    """
        step_size：每隔多少个轮次更新一次学习率
        gamma：学习率的更新系数
    """

    # 创建学习率调度器，用于在训练过程中动态调整优化器的学习率
    # 该调度器每隔step_size个epoch将学习率乘以gamma因子
    # optimizer: 优化器对象，用于更新模型参数
    # step_size: 学习率衰减的间隔步数，每经过step_size个epoch执行一次衰减
    # gamma: 学习率衰减因子，新的学习率 = 当前学习率 * gamma
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    # 学习率_指定间隔下降策略
    # milestones用来指定在什么地方进行学习率的衰减
    # milestones = [50,125,160]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.5)

    # 学习率_指数学习率策略
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)

    # 训练的总轮次
    epochs = 200

    iteration = 10

    epoch_list = [] # 记录循环到哪个轮次了
    lr_list = [] # 记录对应轮次的学习率
    for epoch in range(epochs):
        for i in range(iteration):
            # 获得预测值
            y_pred = w*x

            # 计算损失值
            loss_value = (y_pred-y_true)**2

            # 梯度清零、反向传播、更新权重
            optimizer.zero_grad()
            loss_value.sum().backward()
            optimizer.step()

        # 记录对应轮次和对应的学习率
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        print(f"第{epoch}轮次，学习率是多少{scheduler.get_last_lr()}")
        # 按照等间隔（epoch的等间隔）更新学习率
        scheduler.step()

    # 绘制折线图
    plt.plot(epoch_list,lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("lr")
    plt.show()


if __name__ == '__main__':
    demo()