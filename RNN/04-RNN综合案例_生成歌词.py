import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import jieba

def create_data():
    # 1- 读取文本内容
    original_words = [] # 每句歌词分词以后，在没有去重之前的内容
    unique_words = [] # 用来存储去重后的单词
    # 打开周杰伦歌词数据文件，以只读模式打开并指定UTF-8编码格式
    """
    with 关键字的主要作用如下：
        上下文管理：with 用于实现上下文管理协议，自动处理资源的获取和释放
        自动关闭文件：在本例中，with open("data/jaychou_lyrics.txt", mode="r", encoding="UTF-8") as file_obj: 确保文件在使用完毕后自动关闭，即使发生异常也会正确关闭
        异常安全：如果在 with 代码块中发生异常，文件也会被正确关闭，避免资源泄露
        简化代码：无需手动调用 file_obj.close() 方法，减少代码冗余并提高代码安全性
    """
    with open("data/jaychou_lyrics.txt",mode="r",encoding="UTF-8") as file_obj:

        while True:
            line = file_obj.readline()
            if line=="":
                break

            # 2- 对每行内容进行分词处理
            words = jieba.lcut(line)
            original_words.append(words)
            # print(line)
            # print(words)

            # 3- 去重
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)

    # print("原始歌词的行数：",len(original_words),original_words)
    # print("去重以后的词的个数：",len(unique_words),unique_words)

    # 4- 数据结构转成字典的形式 {词:索引}
    word_dict = {word:index for index,word in enumerate(unique_words)}
    print("词汇表的大小：",len(word_dict),word_dict)

    # 5- 构建词汇表
    """
        步骤：
            1- 将原始的歌词中的单词替换成单词对应的索引，具体需求如下：
                line_words ->   ['想要', '有', '直升机', '\n']
                使用索引替代单词-> [0,1,2,3]
            2- 将所有行的歌词拼接到一起
            3- 以空格对应的索引分隔拼接
    """
    corpus_idx = []

    for line_words in original_words:
        # 1 - 将原始的歌词中的单词替换成单词对应的索引
        for word in line_words:
            # 2- 将所有行的歌词拼接到一起
            index = word_dict.get(word)
            corpus_idx.append(index)
        # 3 - 以空格对应的索引拼接
        corpus_idx.append(word_dict.get(" "))

    print("原始歌词的：", original_words)
    print("索引替换以后的歌词：", corpus_idx)
    print("索引替换以后的歌词长度：", len(corpus_idx))

    """
        unique_words：去重以后的词列表
        word_dict：去重以后的词和索引的字典形式
        corpus_idx：将所有歌词以索引替换并且以40代表的空格分隔以后的完整内容
        len(unique_words)：去重以后词的个数
    """
    return unique_words,word_dict,corpus_idx,len(unique_words)

class LyricsDataset(Dataset):
    """
        为什么需要自己实现Dataset？
        因为普通的Dataset不支持数据往前每次移动1位的过程
        自己实现Dataset，需要实现如下3个方法：
            __init__：初始化参数
            __getitem__：获取x_train、y_train，y_train相对x_train是将位置往后整体移动了1位
            __len__：获取分段的个数
    """
    def __init__(self,corpus_idx,num_chars):
        # 歌词文件经过索引替换、增加空格对应的索引进行分隔处理后的内容
        self.corpus_idx = corpus_idx

        # 从corpus_idx每次取连续的多个少词
        self.num_chars = num_chars

        # corpus_idx中词的总数量
        self.word_count = len(corpus_idx)

        # 计算分段的个数，向下取整 6.99999 -> 6
        """
            corpus_idx = [0, 1, 2, 3, 40, 0, 4, 5, 6, 7, 8]
            word_count = 11
            num_chars = 5
            number = 2
        """
        self.number = self.word_count//self.num_chars
        print("分段的个数：",self.number)

    # 对象[索引]的时候会自动调用该方法用来获取对应x_train/y_train、x_test/y_test
    def __getitem__(self, index):
        """
            corpus_idx = [0, 1, 2, 3, 40, 0, 4, 5, 6, 7, 8]
            word_count = 11
            num_chars = 5
            self.word_count - self.num_chars - 1 = 11 -5 -1=5

            index=5，start=5，end=10
            x_train [0, 4, 5, 6, 7]
            y_train [4, 5, 6, 7, 8]

            index=6，start=5，end=10
            x_train [0, 4, 5, 6, 7]
            y_train [4, 5, 6, 7, 8]
        """
        # 获得取数范围的start和end索引
        # 计算截取文本的起始和结束位置，确保索引不越界
        # start位置在[0, word_count-num_chars-1]范围内
        # end位置为start加上要截取的字符数量
        start = min(max(0,index), self.word_count - self.num_chars - 1)
        end = start + self.num_chars

        # 根据start和end索引取出对应的x和y
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start+1:end+1]

        return torch.tensor(x),torch.tensor(y)


    def __len__(self):
        return self.number

class LyricsRNNModel(nn.Module):
    def __init__(self, unique_word_count):
        # 初始化父类
        super().__init__()
        # 定义网络结果
        # 词嵌入层
        self.ebd = nn.Embedding(num_embeddings=unique_word_count,embedding_dim=128)
        # 循环层
        self.rnn = nn.RNN(input_size=128,hidden_size=256,num_layers=1)
        # 全连接的输出层
        self.output = nn.Linear(in_features=256,out_features=unique_word_count)

    def forward(self,inputs,hidden):
        # 词嵌入层：输入词的索引，得到词向量
        embed = self.ebd(inputs)

        # 循环层
        """
            输入 词向量 和 上一次的隐藏状态
            输出 本次的预测结果 和 更新以后的隐藏状态

            为什么transpose(0,1)？
            因为rnn中对输入数据的格式要求是[分段的长度num_chars, 批次的大小, 一个词用多少个数字表示]
        """
        print("维度交换前",embed.shape)
        out,hidden = self.rnn(embed.transpose(0,1), hidden)
        print("维度交换后",embed.transpose(0,1).shape)

        # 输出层：因为输出层是全连接层，只能处理二维数据，而out的维度是三维
        output = self.output(out.reshape(-1,out.shape[-1]))

        return output,hidden

    # 初始化隐藏层
    def init_hidden(self,batch_size):
        return torch.zeros(1, batch_size,256)

def train():
    # 构建词典
    unique_words, word_to_index, corpus_idx, unique_word_count = create_data()

    # 数据集 LyricsDataset对象，并实现了 __getitem__ 方法
    lyrics = LyricsDataset(corpus_idx=corpus_idx, num_chars=32)
    # 查看句子数量
    print(lyrics.number) # 输出：1535

    # 初始化模型
    model = LyricsRNNModel(unique_word_count)
    # 数据加载器 DataLoader对象，并将lyrics dataset对象传递给它
    lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=5)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法：alt+回车
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练轮数
    epoch = 2
    for epoch_idx in range(epoch):
        # 训练时间
        start = time.time()
        iter_num = 0  # 迭代次数
        # 训练损失
        total_loss = 0.0
        # 遍历数据集 DataLoader 会在后台调用 dataset.__getitem__(index) 来获取每个样本的数据和标签，并将它们组合成一个 batch
        for x, y in lyrics_dataloader:
            # 隐藏状态的初始化
            hidden = model.init_hidden(batch_size=5)
            # 模型计算
            output, hidden = model(x, hidden)
            # 计算损失
            # y形状为(batch, seq_len), 需要转换成一维向量->160个词的下标索引
            # output形状为(seq_len, batch, 词向量维度)
            # 需要先将y进行维度交换(和output保持一致)再改变形状
            # 转置张量y并在第一个维度上展平，用于后续计算
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            # 计算模型输出与真实标签之间的损失值
            loss = criterion(output, y)
            # 清零优化器中的梯度缓存
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            iter_num += 1  # 迭代次数加1
            # 累加当前批次的损失值
            total_loss += loss.item()

        # 打印训练信息
        print('epoch %3s loss: %.5f time %.2f' % (epoch_idx + 1, total_loss / iter_num, time.time() - start))
    # 模型存储
    torch.save(model.state_dict(), 'model/lyrics_model.pth')

def predict(start_word, sentence_length):
    # 构建词典
    unique_words, word_to_index, _, unique_word_count = create_data()

    # 构建模型
    model = LyricsRNNModel(unique_word_count)
    # 加载参数
    model.load_state_dict(torch.load('model/lyrics_model.pth'))
    # 隐藏状态
    hidden = model.init_hidden(batch_size=1)
    # 将起始词转换为索引
    word_idx = word_to_index[start_word]
    # 产生的词的索引存放位置
    generate_sentence = [word_idx]

    # 遍历到句子长度，获取每一个词
    for _ in range(sentence_length):
        # 模型预测
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 获取预测结果
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)

    # generate_sentence -> [0, 1, 2, 3, 40, .......]
    # 根据产生的索引获取对应的词，并进行打印
    for idx in generate_sentence:
        print(unique_words[idx], end='')

if __name__ == '__main__':
    # 1- 准备数据
    unique_words,word_dict,corpus_idx,word_count = create_data()

    # 2- 自定义数据集Dataset
    # 3- 创建RNN循环神经网络结构
    model = LyricsRNNModel(word_count)

    # 4- 模型训练
    # train()

    # 5- 生成歌词，模型测试
    predict("东风破",100)

