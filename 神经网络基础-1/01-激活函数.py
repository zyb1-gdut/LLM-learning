import torch
import matplotlib.pyplot as plt

# 设置中文字体显示 - 确保图表能正确显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    原因：torch和matplotlib工具中都有libiomp5md.dll，只能保留一个
    解决：删除torch下的libiomp5md.dll
    注意：这是Windows系统下的常见问题，不影响代码逻辑功能[1](@ref)
"""


def sigmoid_demo():
    """
    Sigmoid激活函数演示
    函数定义: σ(x) = 1 / (1 + exp(-x))
    输出范围: (0, 1)
    特点：将输入映射到0-1之间，适合二分类问题，但存在梯度消失问题[1,5](@ref)
    """
    # 创建1行2列的子图，用于并排显示函数图像和导数图像
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)  # 设置图形大小

    # 创建x轴的刻度：从-20到20生成1000个等间距点
    x = torch.linspace(-20, 20, 1000)

    # sigmoid函数_原始图像计算
    # Sigmoid将任何实数映射到(0,1)区间，适合表示概率[5](@ref)
    y = torch.sigmoid(x)

    # 绘制原始函数图像
    axes[0].plot(x.numpy(), y.numpy())
    axes[0].grid()
    axes[0].set_title("Sigmoid函数原始图像")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("sigmoid(x)")

    # sigmoid函数_导数图像计算
    # 重新创建x张量，并设置requires_grad=True以计算梯度
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    # 计算sigmoid并求和，然后反向传播计算梯度
    torch.sigmoid(x).sum().backward()

    # 绘制导数图像
    # sigmoid的导数公式：f'(x) = f(x)(1-f(x))，值域(0, 0.25)[5](@ref)
    axes[1].plot(x.detach().numpy(), x.grad.numpy())
    axes[1].grid()
    axes[1].set_title("Sigmoid函数导数图像")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("sigmoid'(x)")

    plt.tight_layout()
    plt.show()


def tanh_demo():
    """
    Tanh激活函数演示
    函数定义: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    输出范围: (-1, 1)
    特点：零中心化，比Sigmoid梯度更强，但仍有梯度消失问题[2,5](@ref)
    """
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    # 创建x轴的刻度
    x = torch.linspace(-20, 20, 1000)

    # tanh函数_原始图像
    y = torch.tanh(x)

    # 绘制图形
    axes[0].plot(x.numpy(), y.numpy())
    axes[0].grid()
    axes[0].set_title("Tanh函数原始图像")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("tanh(x)")

    # tanh函数_导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.tanh(x).sum().backward()

    # 绘制图形
    # tanh的导数：f'(x) = 1 - tanh²(x)，值域(0, 1)[2](@ref)
    axes[1].plot(x.detach().numpy(), x.grad.numpy())
    axes[1].grid()
    axes[1].set_title("Tanh函数导数图像")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("tanh'(x)")

    plt.tight_layout()
    plt.show()


def relu_demo():
    """
    ReLU激活函数演示
    函数定义: ReLU(x) = max(0, x)
    特点：计算简单，解决梯度消失问题，但存在神经元死亡问题[1,3](@ref)
    """
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    # 创建x轴的刻度
    x = torch.linspace(-20, 20, 1000)

    # relu函数_原始图像
    y = torch.relu(x)  # 负数归零，正数保持不变

    # 绘制图形
    axes[0].plot(x.numpy(), y.numpy())
    axes[0].grid()
    axes[0].set_title("ReLU函数原始图像")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("ReLU(x)")

    # relu函数_导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.relu(x).sum().backward()

    # 绘制图形
    # ReLU的导数：x>0时为1，x<0时为0[3](@ref)
    axes[1].plot(x.detach().numpy(), x.grad.numpy())
    axes[1].grid()
    axes[1].set_title("ReLU函数导数图像")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("ReLU'(x)")

    plt.tight_layout()
    plt.show()


def softmax_demo():
    """
    Softmax激活函数演示
    函数定义: softmax(x_i) = exp(x_i) / Σexp(x_j)
    用途：多分类问题，将输出转换为概率分布[5,8](@ref)
    """
    # 示例数据：2个样本，每个样本4个类别的得分
    scores = torch.tensor([[0.3, 0.24, 5, -3.1],
                           [-0.1, 0.56, 2.4, 0.35]])

    print("原始分数张量:")
    print(scores)
    print("张量形状:", scores.shape)
    print()

    # 计算得到概率值
    """
    dim参数说明:
    - dim=0: 按列计算softmax（每列元素进行softmax）
    - dim=1: 按行计算softmax（每行元素进行softmax，常用方式）
    - dim=-1: 最后一个维度[8](@ref)
    """

    # 按行计算softmax（常用方式）- 每个样本各类别概率和为1
    prob_dim1 = torch.softmax(scores, dim=1)
    print("按行softmax (dim=1) - 每个样本各类别概率:")
    print(prob_dim1)
    print("每行概率和:", torch.sum(prob_dim1, dim=1))  # 验证每行和为1
    print()

    # 按列计算softmax - 每个类别在各样本间的概率分布
    prob_dim0 = torch.softmax(scores, dim=0)
    print("按列softmax (dim=0) - 每个类别在各样本间的概率:")
    print(prob_dim0)
    print("每列概率和:", torch.sum(prob_dim0, dim=0))  # 验证每列和为1
    print()

    # 分析softmax特性：放大最大值效应
    print("Softmax特性分析:")
    print("第一行原始分数:", scores[0])
    print("第一行softmax结果:", prob_dim1[0])
    print("最大值5.0对应的概率:", prob_dim1[0][2].item())
    print("概率分布情况: 最大值被显著放大")


if __name__ == '__main__':
    """
    主函数：依次演示四种激活函数
    激活函数的作用：为神经网络引入非线性因素，使其能够学习复杂模式
    """

    print("=== Sigmoid函数演示 ===")
    print("Sigmoid特点: 输出范围(0,1), 适合二分类, 但存在梯度消失问题")
    print("梯度消失: 当|x|较大时梯度接近0, 参数更新困难")
    """
    1. Sigmoid函数分析结果
        预期输出特征：
        原始函数：S形曲线，值域(0,1)，在x=0时y=0.5
        导数函数：钟形曲线，最大值0.25（在x=0处）
        数学特性验证：
            # 示例计算
            x_test = torch.tensor([-10.0, 0.0, 10.0])
            y_test = torch.sigmoid(x_test)
            print(f"Sigmoid测试: {x_test} -> {y_test}")
            # 输出: tensor([0.0000, 0.5000, 1.0000])
        实际问题：当x=-10或x=10时，梯度接近0，导致梯度消失
    """
    sigmoid_demo()

    print("\n" + "=" * 50)
    print("=== Tanh函数演示 ===")
    print("Tanh特点: 输出范围(-1,1), 零中心化, 比Sigmoid收敛快")
    print("但仍存在梯度消失问题, 常用于隐藏层")
    """
    2. Tanh函数分析结果
        预期输出特征：        
        原始函数：S形曲线，值域(-1,1)，通过原点(0,0)        
        导数函数：类似sigmoid但值域更大，最大值1（在x=0处）        
        优势分析：
            # 零中心化优势示例
            x_test = torch.tensor([-1.0, 0.0, 1.0])
            y_test = torch.tanh(x_test)
            print(f"Tanh测试: {x_test} -> {y_test}")
            # 输出: tensor([-0.7616, 0.0000, 0.7616])
    """
    tanh_demo()


    print("\n" + "=" * 50)
    print("=== ReLU函数演示 ===")
    print("ReLU特点: 计算简单, 缓解梯度消失, 但存在神经元死亡问题")
    print("广泛应用在隐藏层, 是现代深度学习最常用的激活函数")
    """
    3. ReLU函数分析结果
    预期输出特征：
    原始函数：x<0时为0，x≥0时为线性函数
    导数函数：x<0时为0，x>0时为1的阶跃函数
    实际应用示例：
        # ReLU在前向传播中的应用
        x_test = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_test = torch.relu(x_test)
        print(f"ReLU测试: {x_test} -> {y_test}")
        # 输出: tensor([0., 0., 0., 1., 2.])[3]
    """
    relu_demo()

    #exit()
    print("\n" + "=" * 50)
    print("=== Softmax函数演示 ===")
    print("Softmax特点: 将输出转换为概率分布, 所有输出之和为1")
    print("主要用于多分类问题的输出层")
    """
    === Softmax函数演示 ===
    Softmax特点: 将输出转换为概率分布, 所有输出之和为1
    主要用于多分类问题的输出层
    
    原始分数张量:
    tensor([[ 0.3000,  0.2400,  5.0000, -3.1000],
            [-0.1000,  0.5600,  2.4000,  0.3500]])
    张量形状: torch.Size([2, 4])
    
    按行softmax (dim=1) - 每个样本各类别概率:
    tensor([[6.2339e-03, 5.8483e-03, 9.8677e-01, 1.1358e-04],
            [4.6618e-02, 1.2703e-01, 7.0183e-01, 1.0452e-01]])
    每行概率和: tensor([1.0000, 1.0000])
    
    按列softmax (dim=0) - 每个类别在各样本间的概率:
    tensor([[0.5987, 0.4219, 0.9241, 0.0341],
            [0.4013, 0.5781, 0.0759, 0.9659]])
    每列概率和: tensor([1.0000, 1.0000, 1.0000, 1.0000])
    
    Softmax特性分析:
    第一行原始分数: tensor([ 0.3000,  0.2400,  5.0000, -3.1000])
    第一行softmax结果: tensor([6.2339e-03, 5.8483e-03, 9.8677e-01, 1.1358e-04])
    最大值5.0对应的概率: 0.98676997423172
    概率分布情况: 最大值被显著放大
    """
    softmax_demo()