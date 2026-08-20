import torch
import matplotlib.pyplot as plt

# 设置中文显示（关键步骤）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 支持中文的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 1. 准备数据和模型（简单线性回归为例）
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.1, 5.9, 8.1])
w = torch.tensor(0.0, requires_grad=True)  # 权重
b = torch.tensor(0.0, requires_grad=True)  # 偏置

# 2. 设置超参数和记录列表
learning_rate = 0.01
max_epochs = 2000
convergence_threshold_loss = 1e-6  # 损失变化阈值
convergence_threshold_grad = 1e-4  # 梯度范数阈值

loss_history = []
grad_norm_history = []
param_change_history = []

# 3. 梯度下降循环
prev_w = w.clone().detach()
prev_b = b.clone().detach()

for epoch in range(max_epochs):
    # 前向传播，计算损失 (MSE)
    y_pred = w * x + b
    loss = torch.mean((y_pred - y) ** 2)
    loss_history.append(loss.item())

    # 反向传播，计算梯度
    loss.backward()

    # 计算当前梯度范数 (判断收敛性1)
    # 计算当前梯度的范数并记录到历史列表中
    # 该代码块用于监控模型训练过程中的梯度变化情况
    current_grad_norm = torch.norm(torch.tensor([w.grad, b.grad])).item()
    grad_norm_history.append(current_grad_norm)


    # 计算参数变化量 (判断收敛性2)
    with torch.no_grad():
        current_param_change = torch.norm(torch.tensor([w - prev_w, b - prev_b])).item()
        param_change_history.append(current_param_change)
        # 更新前一次参数记录
        prev_w = w.clone().detach()
        prev_b = b.clone().detach()

    # 判断收敛 (添加epoch>0防止首次迭代就判断)
    # 检查模型是否收敛的条件判断
    # 通过比较相邻轮次的损失变化和梯度范数来判断训练是否收敛
    # 收敛条件：损失变化小于阈值且梯度范数小于阈值
    if epoch > 0:
        loss_change = abs(loss_history[-1] - loss_history[-2])
        # 同时满足损失变化小且梯度很小，才认为收敛
        if loss_change < convergence_threshold_loss and current_grad_norm < convergence_threshold_grad:
            print(f'在第 {epoch + 1} 轮迭代收敛！')
            print(f'最终损失: {loss.item():.8f}')
            break


    # 梯度下降更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        # 清零梯度，防止累积
        w.grad.zero_()
        b.grad.zero_()
else:
    # 若循环正常结束（非break），说明达到最大迭代次数
    print(f'达到最大迭代次数 {max_epochs}，停止迭代。')

# 4. 可视化收敛过程
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 损失变化曲线
axes[0].plot(loss_history)
axes[0].set_title('损失函数下降曲线')
axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

# 梯度范数曲线
axes[1].plot(grad_norm_history)
axes[1].set_title('梯度范数变化')
axes[1].set_xlabel('迭代次数')
axes[1].set_ylabel('Gradient Norm')
axes[1].grid(True)
axes[1].set_yscale('log')  # 用对数坐标更易观察

# 参数变化量曲线
axes[2].plot(param_change_history)
axes[2].set_title('参数变化量')
axes[2].set_xlabel('迭代次数')
axes[2].set_ylabel('Parameter Change Norm')
axes[2].grid(True)
axes[2].set_yscale('log')

plt.tight_layout()
plt.show()

# 打印最终参数
print(f'\n最终模型参数: w = {w.item():.4f}, b = {b.item():.4f}')