import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 标准化处理
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchsummary import summary

# 文件->DataFrame->Tensor->TensorDataset->DataLoader
def create_data():
    # 1- 读取文件
    data_df = pd.read_csv("data/手机价格预测.csv",encoding="UTF-8")

    # 2- 得到特征数据和目标值
    x = data_df.iloc[:,:-1]
    y = data_df.iloc[:,-1]

    # 查看4种手机价格对应的数据条数。目的：是检查数据是否均衡，如果不均衡，推荐下面的stratify设置为True
    # print(y.value_counts())
    # 3- 划分训练集和测试集
    # 将数据集划分为训练集和测试集
    # x: 特征数据，用于训练和测试
    # y: 标签数据，与特征数据对应
    # test_size: 测试集占比，设置为0.2表示20%的数据作为测试集
    # random_state: 随机种子，设置为10以确保结果可重现
    # shuffle: 是否打乱数据，设置为True表示在划分前打乱数据顺序
    # 返回值: 包含训练集特征、测试集特征、训练集标签、测试集标签的元组
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10,shuffle=True)

    # 4-特征预处理
    # 对数据进行标准化处理：目的为了消除量纲（单位）的影响
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    # 5- 数据类型转换：DataFrame->Tensor->TensorDataset
    """
        下面为什么需要设置数据的类型，也就是dtype参数呢？
            1- 因为需要保证特征数据类型一致，都是相同长度的小数。
            2- 分类问题中，目标值类型必须是int64
        否则会报如下的异常：
        RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
    """
    # 下面两行给没有标准化处理使用
    # train_dataset = TensorDataset(torch.tensor(x_train.values,dtype=torch.float32),torch.tensor(y_train.values,dtype=torch.int64))
    # test_dataset = TensorDataset(torch.tensor(x_test.values,dtype=torch.float32),torch.tensor(y_test.values,dtype=torch.int64))

    # 下面两行给标准化处理后使用
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.int64))

    # 6- 返回结果
    # 特征个数
    features = x_train.shape[1]

    # 目标值类别个数
    targets = len(np.unique(y_train))
    return train_dataset,test_dataset,features,targets

# 构建人工神经网络
class PhoneANNModel(nn.Module):
    def __init__(self,features,targets):
        # 1- 首先初始化父类
        super().__init__()

        # 2- 搭建神经网络各层
        # 2.1- 隐藏层1
        self.linear1 = nn.Linear(in_features=features,out_features=128)
        # 2.2- 隐藏层2
        self.linear2 = nn.Linear(in_features=128, out_features=256)
        # 2.3- 输出层
        self.output = nn.Linear(in_features=256, out_features=targets)

    def forward(self,x):
        # 前向传播的过程
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))

        """
            为什么这个地方没有写softmax？
            因为该业务问题是多分类的问题，因此损失函数需要使用CrossEntropyLoss损失函数
            由于CrossEntropyLoss自带softmax激活函数，因此这里不能使用softmax
        """
        x = self.output(x)

        return x

def train_model(train_dataset,features, targets):
    # 1- 数据封装：TensorDataset->DataLoader，为了防止内存溢出
    # 设置随机数种子是为了与shuffle=True配合使用，让每次的数据固定
    torch.manual_seed(815)
    dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)

    # 2- 创建神经网络实例对象
    model = PhoneANNModel(features, targets)

    # 3- 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 4- 创建优化器
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9, 0.99))

    # 5- 训练数据
    epochs = 50

    for epoch in range(epochs):

        total_loss = 0.0        # 每轮次的总损失值
        total_sample_num = 0    # 每轮次训练的总样本条数

        for x_train_tmp,y_train_tmp in dataloader:
            # 设置模式：训练模式，也就是允许神经元随机失活
            model.train()

            # 数据预测
            y_pred = model(x_train_tmp)
            # 计算损失值
            loss = criterion(y_pred,y_train_tmp)

            # 累计损失值【了解】
            total_loss += loss.item() * len(x_train_tmp)
            total_sample_num += len(x_train_tmp)

            # 梯度清零、反向传播、更新参数【固定写法】
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 计算每轮次总的平均损失值【了解】
        total_avg_loss = total_loss/total_sample_num
        print(f"第{epoch+1}轮次，总的平均交叉熵损失值{total_avg_loss}")

    # 6- 保存训练好的模型
    torch.save(model.state_dict(),"model/phone_ann.pt")

def predict_model(test_dataset,features,targets):
    # 1- 加载训练好的模型
    model = PhoneANNModel(features, targets)
    model.load_state_dict(torch.load("model/phone_ann.pt"))

    # 2- 将数据封装为DataLoader
    dataloader = DataLoader(test_dataset,batch_size=3,shuffle=False)

    # 3- 进行预测
    correct_count = 0 # 预测正确的样本条数
    for x_test,y_test in dataloader:
        # 设置模式：测试模式，也就是不允许神经元随机失活
        model.eval()

        # 预测数据
        y_pred = model(x_test)

        """
            预测结果： tensor([[ 0.2841,  0.8648,  0.2713, -1.5496],
            [-2.3711,  0.1373,  1.0892,  0.6970],
            [ 2.8117,  1.5394, -0.4526, -3.8784]], grad_fn=<AddmmBackward0>)
            
            为什么预测结果中一条数据的各个值累加求和不是1，而且还有负数、大于1的情况？
                原因：目前神经网络的输出层只是进行了线性加权求和，没有应用激活函数
        """
        # print("预测结果_原始线性加权求和结果：",y_pred)
        # print("预测结果_softmax计算后的概率值：",torch.softmax(y_pred,dim=1))
        # print("预测结果：",torch.argmax(y_pred,dim=1))
        # print("真实结果：",y_test)

        y_pred_target = torch.argmax(y_pred,dim=1)
        correct_count =  correct_count + (y_pred_target==y_test).sum()

    # 最终的准确率
    acc_rate = correct_count.item()/len(test_dataset)
    print("预测准确率：",acc_rate)

if __name__ == '__main__':
    # 1- 准备数据
    train_dataset,test_dataset,features,targets = create_data()

    # 2- 构建人工神经网络（演示，不是正式代码，只是为了计算参数个数）
    # model = PhoneANNModel(features,targets)
    # summary(model,input_size=(features,),batch_size=1)

    # 3- 模型训练
    train_model(train_dataset,features,targets)

    # 4- 预测和评估
    predict_model(test_dataset,features,targets)
