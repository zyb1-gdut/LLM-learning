import time

import torch
import torch.nn  as nn
import torch.optim as optim
from dask.array import true_divide
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchsummary import summary

#假设你已经定义好模型model 和输入数据 input
#统一运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data():
    #直接加载数据：如果本地有离线的图片数据，那么不用联网下载，否则需要在线下载
    """
        root:离线图片所在的位置
        train：True表示加载的训练集的5w张图片，False表示加载测试的1w张图片
        transforms：是否要对图片转化为张量
        download：如果本地没有，允许在线下载
    :return:
    """
    train = CIFAR10(root="data",train=True,download=True,transform=ToTensor())
    test  = CIFAR10(root="data",train=False,download=True,transform=ToTensor())

    return train,test

class ImgCNNModel(nn.Module):
    def __init__(self):
        #1- 初始化父类
        super().__init__()

        #2- 搭建网络结构
        #2.1- 隐藏层中的第一套卷积和池化
        #out_channels:即表示输出图片的通道数，也表示卷积核的个数
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels=6,kernel_size = 3,stride =1,padding = 0 )
        #尺寸：通道数：3-->6,高度=（32-3+×0）/1+1=30，输出尺寸：6*30*30
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0 )
        #尺寸：通道数：6 ，高度=(30-2+2×0)/2+1=15，输出尺寸：6*15*15
        #2.2- 隐藏层中的第二套卷积和池化
        self.conv2 = nn.Conv2d(in_channels = 6,out_channels=16,kernel_size = 3,stride = 1,padding = 0)
        #尺寸：通道数：6->16,高度=（15-3+2*0）/1+1=12，输出尺寸：16*12*12
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2 ,padding = 0 )
        #尺寸：通道数：16 ，高度=（12-2+2*0）/2+1=6，输出尺寸：16*6*6 =576
        #2.3- 全连接层
        """
        输出尺寸 = (输入尺寸 - 核大小 + 2×填充) / 步长 + 1
            全连接层的第一层中输入的特征个数不能随意的指定，需要与最后一层池化层的特征个数保持一致
            在隐藏层中，卷积/池化后的尺寸变形： 输出尺寸 = （输入尺寸 - 核大小 + 2×填充）/步长 + 1 
            特征个数576 = 16个特征图*6高度*6宽度
        """
        self.linear1 = nn.Linear(in_features= 576,out_features=120)
        self.linear2 = nn.Linear(in_features=120,out_features=84)

        #优化代码：随机失活
        #self.dropout = nn.Dropout(p=0.5)

        #2.4- 输出层
        self.output = nn.Linear(in_features = 84,out_features = 10)


    #数据传播过程+激活函数
    def forward(self, x):
        #隐藏层中的第一套卷积层、激活层、池化层
        # 分开写
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.pool1(x)

        #合并写
        x = self.pool1(torch.relu(self.conv1(x)))
        #隐藏层中的第二套卷积层、激活层、池化层
        x = self.pool2(torch.relu(self.conv2(x)))

        """
            池化层中输出的结果不是二维的，但是全连接层只能处理二维数据[样本数据条数，一条样本数据特征个数]
            x.size(0),取的是传递进来的这一批次的图片张数。注意：值最小为1
        """

        print("特征图修改前的形状，",x.shape)
        #将输入张量x重塑为二维矩阵形式
        #第一维保持不变，第二维度展开所有剩余维度
        #例如：形状为（batch_size, channels , height , width）的张量
        #会被重塑为（batch_size, channels*height*width）的二维张量
        x = x.reshape(x.size(0),-1)
        print("特征图修改后的结果：",x.shape)

        #全连接层
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))

        #优化
        #x = self.dropout(x)

        #输出层
        #注意：不能在外面调用softmax激活。因此是多分类问题，后续会调用CrossEntropyLoss，自带softmax功能
        x = self.output(x)

        return x

def train_model(train_data):
        #1- 数据封装：TensorDataset -> DataLoader, 为了防止内存溢出
        torch.manual_seed(817)
        #创建数据加载器，用于批量加载训练数据
        #参数说明：
        #train_data:训练数据集对象
        #batch_size:每个批次的数据量大小，设置为8
        #shuffle:是否每个epoch前打乱数据顺序，设置为True表示打乱
        dataloader = DataLoader(train_data,batch_size=8,shuffle=True,drop_last=True)

        #2- 创建神经网络实例对象
        model = ImgCNNModel()

        #3- 创建损失函数
        criterion = nn.CrossEntropyLoss()

        #4- 创建优化器
        optimizer = optim.Adam(params = model.parameters(),lr=1e-3,betas=(0.9,0.99))

        #5- 训练数据
        epochs =1

        for epoch in range(epochs):
            start_time = time.time()#用于统计训练耗时
            count = 0 #用于查看进度

            total_loss_value = 0.0 #每轮次的总损失
            total_sample_count = 0 #每轮次的总训练样本条数

            for x_train,y_train in dataloader:
                #模式切换
                model.train()

                #预测
                y_pred = model(x_train)
                #计算损失值
                loss = criterion(y_pred,y_train)
                print(f"当前{count}批次的平均损失值{loss}")
                count +=1

                #累计总损失和总样本条数
                total_loss_value += loss.item()*len(x_train)
                #loss.item 要注意损失函数算的损失值默认是平均值
                total_sample_count += len(x_train)

                #梯度清零、反向传播、更新参数
                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()

            #计算每轮的总平均损失
            print(f"第{epoch+1}轮次，总的平均损失{total_loss_value/total_sample_count},耗时{time.time()-start_time:.4f}")

        torch.save(model.state_dict(),"./model/img_1.pt")

def predict_model(test_data):
        #1-加载训练好的模型
        model = ImgCNNModel()
        model.load_state_dict(torch.load("./model/img_1.pt"))

        #2- 将数据封装为DataLoader
        dataloader = DataLoader(test_data,batch_size=8)

        #3- 进行预测
        correct = 0 #预测正确的样本条数
        for x_test,y_test in dataloader:
            #切换模式
            model.eval()

            #预测
            y_pred = model(x_test)
            # print("原始的线性求和结果：",y_pred)
            # print("10种类别的概率",torch.softmx(y_pred,dim=1))
            # print("预测的分类",torch.argmax(y_pred,dim=1))
            # print("预测是否准确",torch.argmax(y_pred)==y_test)
            # #注意：这里的dim要设置为1，表示以行为单位取概率值的最大的作为最终的预测分类结果
            # #torch.argmax 是 PyTorch 里最常用的“裁判员”函数。
            # 它的工作极其简单粗暴：在一堆数字里，找出最大那个数字所在的位置（索引），然后把位置号报给你。

            #计算预测正确的样本数量
            #通过比较预测结果和真实标签，统计当前批次种预测正确的样本数并累加到correct变量中
            correct = correct + (torch.argmax(y_pred,dim=1)==y_test).sum()

            #break

        print("最终的预测准确度：",correct/len(test_data))

if __name__ == "__main__":
    #1- 准备数据
    train,test = create_data()
    #
    # #2- 搭建CNN卷积神经网络
    # """
    #     卷积层的总参数个数=输入通道数*卷积核形状（H*W）*卷积核个数 + 卷积核个数
    #     池化层也是一个无参数的操作，也就是不需要根据输入数据进行训练学习得到权重、偏置。因此池化层的参数个数=0
    #
    # """
    # model = ImgCNNModel()
    # model = model.to(device)
    # summary(model,input_size=(3,32,32),batch_size=1)
    #
    # #3- 训练模型
    # train_model(train)

    #4- 模型预测和评估
    predict_model(test)
