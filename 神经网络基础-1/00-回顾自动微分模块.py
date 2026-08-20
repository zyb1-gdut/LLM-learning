import torch

# 1. 定义数据（对应图中的各个节点）
x = torch.tensor([2.0])      # 输入特征
y_true = torch.tensor([3.0]) # 真实标签

# 2. 初始化参数（需要梯度计算）
w = torch.tensor([1.0], requires_grad=True)  # 权重
b = torch.tensor([0.5], requires_grad=True)  # 偏置

print("=== 前向传播过程 ===")
print(f"输入 x = {x.item()}, 真实值 y = {y_true.item()}")
print(f"初始参数: w = {w.item()}, b = {b.item()}")

# 3. 前向传播（对应图中的计算路径）
z1 = x * w          # 乘法运算
z = z1 + b          # 加法运算 → 预测值
loss = (z - y_true) ** 2  # MSE损失

print(f"\n计算过程:")
print(f"z₁ = x * w = {x.item()} × {w.item()} = {z1.item()}")
print(f"z = z₁ + b = {z1.item()} + {b.item()} = {z.item()}")
print(f"loss = (z - y)² = ({z.item()} - {y_true.item()})² = {loss.item()}")

# 4. 反向传播（自动微分！）
loss.backward()  # 自动计算所有requires_grad=True的张量的梯度

print("\n=== 反向传播结果 ===")
print(f"梯度计算完成!")
print(f"∂loss/∂w = {w.grad.item()}")  # w的梯度
print(f"∂loss/∂b = {b.grad.item()}")  # b的梯度

# 5. 梯度下降更新参数
learning_rate = 0.1
with torch.no_grad():  # 更新时不跟踪梯度
    w_new = w - learning_rate * w.grad
    b_new = b - learning_rate * b.grad

print(f"\n=== 参数更新 ===")
print(f"学习率 = {learning_rate}")
print(f"新权重 w = {w.item()} - {learning_rate} × {w.grad.item()} = {w_new.item()}")
print(f"新偏置 b = {b.item()} - {learning_rate} × {b.grad.item()} = {b_new.item()}")

# 验证效果
new_z = x * w_new + b_new
new_loss = (new_z - y_true) ** 2
print(f"更新后损失: {new_loss.item():.4f} (比原来的 {loss.item():.4f} 减小了!)")