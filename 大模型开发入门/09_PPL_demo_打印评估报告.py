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

        # 计算句子概率
        for word in sentence:
            word_prob = unigram.get(word, 1e-10)  # 防止未登录词导致概率为0
            sentence_prob *= word_prob
            sentence_log_prob += math.log(word_prob, 2)

        # 计算平均负对数概率
        avg_neg_log_prob = -sentence_log_prob / len(sentence)

        # 计算困惑度
        sentence_ppl = 2 ** avg_neg_log_prob

        # 存储句子结果
        sentence_result = {
            'text': " ".join(sentence),
            'probability': sentence_prob,
            'avg_neg_log_prob': avg_neg_log_prob,
            'perplexity': sentence_ppl,
            'word_count': len(sentence)
        }
        results['sentences'].append(sentence_result)

        # 累加困惑度
        all_sentence_perplexity += sentence_ppl

    # 计算平均困惑度
    results['total_perplexity'] = all_sentence_perplexity
    results['average_perplexity'] = all_sentence_perplexity / len(sentences)

    # 评估模型性能
    avg_ppl = results['average_perplexity']
    if avg_ppl < 10:
        results['evaluation'] = "模型性能优秀：困惑度极低，预测能力接近完美"
    elif avg_ppl < 30:
        results['evaluation'] = "模型性能良好：困惑度较低，预测能力较强"
    elif avg_ppl < 50:
        results['evaluation'] = "模型性能一般：困惑度中等，预测能力尚可"
    elif avg_ppl < 100:
        results['evaluation'] = "模型性能较差：困惑度较高，预测能力有限"
    else:
        results['evaluation'] = "模型性能差：困惑度很高，预测能力弱"

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

    # 报告标题
    report += f"评估模型: 一元语言模型 (Unigram)\n"
    report += f"词汇表大小: {results['vocabulary_size']} 个单词\n"
    report += f"测试句子数量: {len(results['sentences'])}\n"
    report += "\n"

    # 详细句子分析
    report += "=== 句子级困惑度分析 ===\n"
    for i, sent in enumerate(results['sentences'], 1):
        report += f"\n句子 {i}: \"{sent['text']}\"\n"
        report += f"  单词数: {sent['word_count']}\n"
        report += f"  句子概率: {sent['probability']:.6f}\n"
        report += f"  平均负对数概率: {sent['avg_neg_log_prob']:.4f}\n"
        report += f"  困惑度: {sent['perplexity']:.2f}\n"

    # 总体评估
    report += "\n=== 总体评估 ===\n"
    report += f"总困惑度: {results['total_perplexity']:.2f}\n"
    report += f"平均困惑度: {results['average_perplexity']:.2f}\n"
    report += f"\n性能评估: {results['evaluation']}\n"

    # 解释困惑度
    report += "\n=== 困惑度解释 ===\n"
    report += "困惑度(Perplexity)是语言模型评估的核心指标，表示模型预测下一个词的不确定性。\n"
    report += "数值越小表示模型越好，理想值为1（完美预测）。实际应用中：\n"
    report += "  < 50: 优秀\n  50-100: 良好\n  100-200: 一般\n  > 200: 较差\n"

    # 改进建议
    report += "\n=== 改进建议 ===\n"
    if results['average_perplexity'] > 100:
        report += "1. 增加训练数据量，提高模型泛化能力\n"
        report += "2. 使用高阶语言模型（如bigram, trigram）\n"
        report += "3. 应用平滑技术处理未登录词\n"
        report += "4. 优化词汇表，减少低频词\n"
    elif results['average_perplexity'] > 50:
        report += "1. 优化模型参数，提高预测准确性\n"
        report += "2. 增加训练数据多样性\n"
        report += "3. 应用回退或插值平滑技术\n"
    else:
        report += "模型性能良好，建议：\n"
        report += "1. 继续优化高阶语言模型\n"
        report += "2. 在更大数据集上验证模型性能\n"

    report += "\n" + "=" * 40
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
一、指标质量评估
    1. 困惑度指标解读
        • 困惑度定义：困惑度是语言模型预测能力的度量，表示模型预测下一个词的不确定性
        • 数值范围：理论上≥1，值越小表示模型越好
        • 实际意义：困惑度8.15表示模型平均每次预测有8.15个等可能的候选词
    2. 评估结果分析
        • 平均困惑度8.15：对于小型语料库和简单模型，这个值在合理范围内
        • 句子间差异：
          • 第一个句子困惑度9.12（较高）
          • 第二、三个句子困惑度7.67（较低）
        • 差异原因：
          • 第一个句子包含"I"和"have"，这些词在语料库中出现频率较低
          • 第二、三个句子包含"has"，这个词出现频率较高（2/12）
    3. 模型质量评估
        • 优点：
          • 简单直观，易于计算
          • 直接反映模型预测能力
        • 局限性：
          • 未考虑词序和上下文（一元模型）
          • 对罕见词敏感（如"pen"）
          • 受语料库大小影响大
    4. 改进建议
        1. 使用高阶模型：考虑bigram或trigram模型，捕捉词序信息
        2. 平滑技术：应用加一平滑或Good-Turing平滑处理罕见词
        3. 更大语料库：增加训练数据提高模型泛化能力
        4. 交叉验证：使用训练集和测试集评估模型泛化能力

二、困惑度指标的实际应用
    1. 困惑度阈值参考
        # 不同任务的困惑度参考值
        机器翻译: < 10 (优秀)
        文本摘要: < 15 (良好)
        对话系统: < 20 (可用)
        语言生成: < 30 (基本)

三、总结
    1. 计算过程正确：代码正确实现了困惑度的计算逻辑
    2. 结果合理：困惑度值在预期范围内（8.15）
    3. 模型局限性：一元模型未考虑上下文信息
    4. 评估价值：困惑度是语言模型的重要评估指标
    5. 改进方向：使用高阶模型和更大语料库提高模型质量

困惑度作为语言模型的核心评估指标，能够有效衡量模型预测能力。在实际应用中，应结合具体任务需求和数据集特性，选择合适的模型和评估方法。
"""