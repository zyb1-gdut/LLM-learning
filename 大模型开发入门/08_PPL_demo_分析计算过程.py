import math
def calculate_perplexity(sentences, unigram):
    """
    计算语言模型的困惑度并生成评估报告
    参数:
        sentences: 句子列表，每个句子是单词列表
        unigram: 一元语言模型概率字典
    返回:
        包含详细评估结果的字典
    """
    # 初始化结果存储
    # 初始化一个名为results的字典，用于存储语言模型困惑度评估的结果。该字典包含以下字段：
    #   'sentences': 存储句子评估结果的列表
    #   'total_perplexity': 总困惑度值
    #   'average_perplexity': 平均困惑度值
    #   'vocabulary_size': 词汇表大小，通过unigram字典长度获得
    #   'evaluation': 模型性能评估文本
    # 这是评估函数的初始结果容器。
    results = {
        'sentences': [],
        'total_perplexity': 0,
        'average_perplexity': 0,
        'vocabulary_size': len(unigram),
        'evaluation': ""
    }

    # 初始化困惑度累加器
    all_sentence_perplexity = 0

    # 遍历每个句子，计算困惑度
    for sentence in sentences:
        sentence_prob = 1
        sentence_log_prob = 0

        # 遍历句子中的每个词，计算句子的概率和对数概率
        # 通过累乘词的概率得到句子概率，同时累加对数概率避免数值下溢
        for word in sentence:
            word_prob = unigram.get(word, 1e-10)  # 处理未登录词：当单词不在词汇表中时，返回默认的极小概率值 1e-10
            sentence_prob *= word_prob
            sentence_log_prob += math.log(word_prob, 2)


        # 计算平均负对数概率
        avg_neg_log_prob = -sentence_log_prob / len(sentence)
        # 计算当前句子的困惑度
        sentence_ppl = 2 ** avg_neg_log_prob
        # 构建句子分析结果字典
        # 包含句子文本、概率、平均负对数概率、困惑度和词数等统计信息
        sentence_result = {
            'text': " ".join(sentence),
            'probability': sentence_prob,
            'avg_neg_log_prob': avg_neg_log_prob,
            'perplexity': sentence_ppl,
            'word_count': len(sentence)
        }

        results['sentences'].append(sentence_result)

        # 累加所有句子的困惑度
        all_sentence_perplexity += sentence_ppl

    # 计算总困惑度和平均困惑度
    results['total_perplexity'] = all_sentence_perplexity
    results['average_perplexity'] = all_sentence_perplexity / len(sentences)
    return results

def generate_perplexity_report(results):
    """
    生成困惑度评估报告
    参数:
        results: 包含评估结果的字典
    返回:
        格式化的评估报告字符串
    """
    report = "===== 语言模型困惑度评估报告 =====\n\n"

    # 报告标题部分：展示模型基本信息
    report += f"评估模型: 一元语言模型 (Unigram)\n"
    report += f"词汇表大小: {results['vocabulary_size']} 个单词\n"
    report += f"测试句子数量: {len(results['sentences'])}\n"
    report += "\n"

    # 详细句子分析部分：逐句输出困惑度相关指标
    report += "=== 句子级困惑度分析 ===\n"
    for i, sent in enumerate(results['sentences'], 1):
        report += f"\n句子 {i}: \"{sent['text']}\"\n"
        report += f"  单词数: {sent['word_count']}\n"
        report += f"  句子概率: {sent['probability']:.6f}\n"
        report += f"  平均负对数概率: {sent['avg_neg_log_prob']:.4f}\n"
        report += f"  困惑度: {sent['perplexity']:.2f}\n"

    # 总体评估部分：输出整体困惑度统计信息
    report += "\n=== 总体评估 ===\n"
    report += f"总困惑度: {results['total_perplexity']:.2f}\n"
    report += f"平均困惑度: {results['average_perplexity']:.2f}\n"

    return report

if __name__ == '__main__':
    # 测试数据
    sentences = [
        ['I', 'have', 'a', 'pen'],
        ['He', 'has', 'a', 'book'],
        ['She', 'has', 'a', 'cat']
    ]

    unigram = {
        'I': 1 / 12,  # 'I' 出现的概率
        'have': 1 / 12,  # 'have' 出现的概率
        'a': 3 / 12,  # 'a' 出现的概率
        'pen': 1 / 12,  # 'pen' 出现的概率
        'He': 1 / 12,  # 'He' 出现的概率
        'has': 2 / 12,  # 'has' 出现的概率
        'book': 1 / 12,  # 'book' 出现的概率
        'She': 1 / 12,  # 'She' 出现的概率
        'cat': 1 / 12  # 'cat' 出现的概率
    }

    # 计算困惑度并生成报告
    results = calculate_perplexity(sentences, unigram)
    report = generate_perplexity_report(results)

    # 打印报告
    print(report)


"""
语言模型困惑度计算代码
一、计算过程详解
1. 第一个句子：['I', 'have', 'a', 'pen']
    • 句子概率：    
      • P(I) = 1/12    
      • P(have) = 1/12    
      • P(a) = 3/12    
      • P(pen) = 1/12
      • 乘积 = (1/12) × (1/12) × (3/12) × (1/12) = 3/20736 ≈ 0.0001447  
    • 平均负对数概率：
      • -log₂(0.0001447) ≈ -(-12.754) = 12.754
      • 除以句子长度4：12.754 / 4 = 3.1885
    • 困惑度：
      • 2^{3.1885} ≈ 9.118
2. 第二个句子：['He', 'has', 'a', 'book']
    • 句子概率：
      • P(He) = 1/12
      • P(has) = 2/12
      • P(a) = 3/12
      • P(book) = 1/12
      • 乘积 = (1/12) × (2/12) × (3/12) × (1/12) = 6/20736 ≈ 0.0002894
    • 平均负对数概率：
      • -log₂(0.0002894) ≈ -(-11.754) = 11.754
      • 除以句子长度4：11.754 / 4 = 2.9385  
    • 困惑度：
      • 2^{2.9385} ≈ 7.667
3. 第三个句子：['She', 'has', 'a', 'cat']
    • 句子概率：
      • P(She) = 1/12
      • P(has) = 2/12
      • P(a) = 3/12
      • P(cat) = 1/12
      • 乘积 = (1/12) × (2/12) × (3/12) × (1/12) = 6/20736 ≈ 0.0002894
    • 平均负对数概率：
      • -log₂(0.0002894) ≈ -(-11.754) = 11.754
      • 除以句子长度4：11.754 / 4 = 2.9385
    • 困惑度：
      • 2^{2.9385} ≈ 7.667
4. 平均困惑度
    • 总困惑度 = 9.118 + 7.667 + 7.667 = 24.452
    • 平均困惑度 = 24.452 / 3 ≈ 8.151

"""
