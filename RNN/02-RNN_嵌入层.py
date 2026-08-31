import jieba  # 导入jieba中文分词库
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块

if __name__ == '__main__':  # 程序入口判断，确保只在直接运行脚本时执行
    # 语料库：文字、文章、小说、新闻。。。
    # 定义待处理的中文文本内容
    content = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"

    # 分词器分词处理
    # 使用jieba对文本进行分词，得到词汇列表
    words = jieba.lcut(content)
    print(words)  # 打印分词结果

    # 将词表中单词去重处理，同时保留原有的顺序
    # 不能使用下面的代码实现，因为set无序
    # print(list(set(words)))  # 注释说明不能使用set去重的原因是会丢失顺序
    # 手动实现去重并保持原有顺序
    unique_words = []
    for word in words:
        if word not in unique_words:  # 如果单词不在唯一词列表中
            unique_words.append(word)  # 添加到唯一词列表
    print(unique_words)  # 打印去重后的词汇列表

    # 使用词嵌入层将单词对应的索引变成词向量
    # 创建词嵌入层对象
    # num_embeddings: 词汇表大小，embedding_dim: 词向量维度
    ebd = nn.Embedding(num_embeddings=len(unique_words), embedding_dim=4) # 这个词嵌入层的参数是词汇表大小和词向量的维度
    print(ebd, type(ebd))  # 打印嵌入层对象和其类型

    # 单词对应的索引变成词向量
    # 遍历唯一词汇列表，为每个单词生成对应的词向量
    """
    在代码中使用 enumerate 的原因如下：
        1）同时获取索引和值：enumerate 函数能够同时提供列表中元素的索引(index)和值(word)，避免了手动维护计数器的需要
        2）满足嵌入层输入要求：nn.Embedding 层需要使用整数索引来获取对应的词向量，index 变量正好提供了这个功能
        3）简化代码逻辑：相比手动使用 while 循环或单独维护索引变量，enumerate 让代码更加简洁易读
        4）保证一一对应关系：确保每个词汇与其在嵌入层中的索引位置正确匹配，这对于后续的词向量查找至关重要
    """
    for index, word in enumerate(unique_words):
        # 得到词向量
        # 将索引转换为tensor并输入到嵌入层中获取词向量
        word_vec = ebd(torch.tensor(index))
        # 打印单词、索引和对应的词向量
        print(f"{word} \t {index} \t {word_vec}")

    # print("-"*30)
    # 普通的while循环也能实现相同的效果
    # index = 0
    # while index<len(unique_words):
    #     print(f"{unique_words[index]} \t {index}")
    #     index+=1
