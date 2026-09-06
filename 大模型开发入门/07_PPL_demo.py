import math
# 定义语料库
sentences = [
    ['I', 'have', 'a', 'pen'],
    ['He', 'has', 'a', 'book'],
    ['She', 'has', 'a', 'cat']
]
# 定义语言模型
# 定义一元语言模型，每个单词出现的概率
unigram = {
    'I': 1/12,  # I 出现的概率
    'have': 1/12, # have 出现的概率
    'a': 3/12,
    'pen': 1/12,
    'He': 1/12,
    'has': 2/12,
    'book': 1/12,
    'She': 1/12,
    'cat': 1/12
}

# 初始化困惑度累加器为0
all_sentence_perplexity = 0

# 遍历语料库的每个句子，计算每个句子的困惑度
for sentence in sentences:
    # 训练内，初始话当前句子的概率为1
    sentence_prob = 1
    # 遍历句子中的每个单词，累乘每个单词的概率
    for word in sentence:
        # 取出来每个单词的概率并累乘
        sentence_prob *= unigram[word]
    # 计算当前句子的平均负对数概率（以2为底），即-1/N * log2（P（W1W2.。。））
    temp = -math.log(sentence_prob, 2) / len(sentence)
    print("当前句子的平均负对数概率为：", temp)
    # 计算当前句子的困惑度（2的上述值的幂），并累加到总困惑度
    sentence_prob = 2 ** temp
    print("当前句子的困惑度为：", sentence_prob)
    all_sentence_perplexity += sentence_prob

print("所有句子的总困惑度为：", all_sentence_perplexity)
#计算所有句子的平均困惑度
perplexity = all_sentence_perplexity / len(sentences)
print("所有句子的平均困惑度为：", perplexity)
