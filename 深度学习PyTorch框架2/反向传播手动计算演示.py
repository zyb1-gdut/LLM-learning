import torch
import torch.nn as nn
import numpy as np


def manual_backpropagation_demo():
    """
    手动实现反向传播的详细演示
    """
    print("=" * 60)
    print("反向传播手动计算演示")
    print("=" * 60)

    # 1. 定义数据和参数
    x = torch.tensor([2.0], requires_grad=False)  # 输入数据
    w = torch.tensor([1.0], requires_grad=True)  # 权重参数
    b = torch.tensor([0.5], requires_grad=True)  # 偏置参数
    y_true = torch.tensor([3.0])  # 真实值

    print(f"\n1. 初始参数:")
    print(f"   输入 x = {x.item()}")
    print(f"   权重 w = {w.item()}")
    print(f"   偏置 b = {b.item()}")
    print(f"   真实值 y_true = {y_true.item()}")

    # 2. 前向传播（手动计算每一步）
    print(f"\n2. 前向传播计算:")

    # 第一步：线性变换 z1 = x * w
    z1 = x * w
    print(f"   z1 = x * w = {x.item()} × {w.item()} = {z1.item()}")

    # 第二步：加上偏置 z = z1 + b
    z = z1 + b
    print(f"   z = z1 + b = {z1.item()} + {b.item()} = {z.item()}")

    # 第三步：计算损失 loss = (z - y_true)^2
    loss = (z - y_true) ** 2
    print(f"   loss = (z - y_true)² = ({z.item()} - {y_true.item()})² = {loss.item()}")

    # 3. 反向传播（手动计算梯度）
    print(f"\n3. 反向传播梯度计算:")

    # 第一步：计算 ∂loss/∂loss = 1
    dloss_dloss = 1.0
    print(f"   ∂loss/∂loss = 1")

    # 第二步：计算 ∂loss/∂z
    # loss = (z - y_true)^2，所以 ∂loss/∂z = 2(z - y_true)
    dloss_dz = 2 * (z - y_true)
    print(f"   ∂loss/∂z = 2(z - y_true) = 2×({z.item()} - {y_true.item()}) = {dloss_dz.item()}")

    # 第三步：计算 ∂loss/∂b
    # z = z1 + b，所以 ∂z/∂b = 1
    # ∂loss/∂b = (∂loss/∂z) × (∂z/∂b)
    dloss_db = dloss_dz * 1
    print(f"   ∂loss/∂b = (∂loss/∂z) × (∂z/∂b) = {dloss_dz.item()} × 1 = {dloss_db.item()}")

    # 第四步：计算 ∂loss/∂w
    # z = z1 + b，所以 ∂z/∂z1 = 1
    # z1 = x * w，所以 ∂z1/∂w = x
    # ∂loss/∂w = (∂loss/∂z) × (∂z/∂z1) × (∂z1/∂w)
    dloss_dw = dloss_dz * 1 * x
    print(f"   ∂loss/∂w = (∂loss/∂z) × (∂z/∂z1) × (∂z1/∂w)")
    print(f"            = {dloss_dz.item()} × 1 × {x.item()} = {dloss_dw.item()}")

    # 4. 使用PyTorch自动微分验证
    print(f"\n4. PyTorch自动微分验证:")

    # 清空之前的梯度
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # 重新进行前向传播（这次用PyTorch记录计算图）
    z1 = x * w
    z = z1 + b
    loss = (z - y_true) ** 2

    # 反向传播
    loss.backward()

    print(f"   PyTorch计算的 ∂loss/∂w: {w.grad.item()}")
    print(f"   PyTorch计算的 ∂loss/∂b: {b.grad.item()}")
    print(f"   手动计算验证: {'✓ 一致' if abs(w.grad.item() - dloss_dw.item()) < 1e-6 else '✗ 不一致'}")

    return dloss_dw.item(), dloss_db.item(), w.grad.item(), b.grad.item()


def neural_network_example():
    """
    简单的神经网络反向传播示例
    """
    print(f"\n" + "=" * 60)
    print("神经网络反向传播示例")
    print("=" * 60)

    # 定义一个简单的神经网络：输入层(2) -> 隐藏层(3) -> 输出层(1)
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.hidden = nn.Linear(2, 3)  # 2输入, 3隐藏神经元
            self.output = nn.Linear(3, 1)  # 3隐藏, 1输出

        def forward(self, x):
            x = torch.sigmoid(self.hidden(x))  # 隐藏层使用sigmoid激活
            x = self.output(x)  # 输出层线性变换
            return x

    # 创建网络和样本数据
    model = SimpleNN()
    inputs = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
    targets = torch.tensor([[1.0]], dtype=torch.float32)

    print(f"\n网络结构: 2输入 → 3隐藏神经元(sigmoid) → 1输出")
    print(f"输入数据: {inputs.tolist()}")
    print(f"目标值: {targets.tolist()}")

    # 前向传播
    outputs = model(inputs)
    criterion = nn.MSELoss()
    loss = criterion(outputs, targets)

    print(f"\n前向传播结果:")
    print(f"网络输出: {outputs.item():.4f}")
    print(f"损失值: {loss.item():.4f}")

    # 反向传播前查看参数梯度
    print(f"\n反向传播前参数梯度:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.grad}")

    # 执行反向传播
    model.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播

    print(f"\n反向传播后参数梯度:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.norm().item():.6f} (梯度范数)")

    return model, loss.item()


def gradient_descent_update():
    """
    梯度下降参数更新演示
    """
    print(f"\n" + "=" * 60)
    print("梯度下降参数更新演示")
    print("=" * 60)

    # 初始参数
    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.5], requires_grad=True)
    x = torch.tensor([2.0])
    y_true = torch.tensor([3.0])
    learning_rate = 0.1

    print(f"初始状态:")
    print(f"  w = {w.item():.3f}, b = {b.item():.3f}")
    print(f"  学习率 = {learning_rate}")

    # 多次迭代演示
    for epoch in range(3):
        # 前向传播
        z = x * w + b
        loss = (z - y_true) ** 2

        # 反向传播
        loss.backward()

        # 梯度下降更新（不跟踪梯度）
        with torch.no_grad():
            w_old, b_old = w.item(), b.item()
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        print(f"\n迭代 {epoch + 1}:")
        print(f"  损失: {loss.item():.4f}")
        print(f"  w梯度: {w.grad.item():.3f}, b梯度: {b.grad.item():.3f}")
        print(f"  参数更新: w {w_old:.3f} → {w.item():.3f}, b {b_old:.3f} → {b.item():.3f}")

        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()

    return w.item(), b.item()


if __name__ == "__main__":
    # 运行手动反向传播演示
    manual_dw, manual_db, auto_dw, auto_db = manual_backpropagation_demo()

    # 运行神经网络示例
    model, final_loss = neural_network_example()

    # 运行梯度下降演示
    final_w, final_b = gradient_descent_update()

    print(f"\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. 反向传播通过链式法则计算每个参数的梯度")
    print("2. 梯度方向指示参数应该如何调整来减小损失")
    print("3. 梯度大小表示调整的幅度")
    print("4. PyTorch的autograd自动处理这些复杂计算")
