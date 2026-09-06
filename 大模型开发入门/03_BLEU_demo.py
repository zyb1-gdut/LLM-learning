"""
bleu: 翻译质量指标, 范围[0, 1], 越接近于1模型效果越好, 一般大于0.5即可用

exp函数是以自然常数e（约2.71828）为底的指数函数
"""
import warnings

# TODO 第一步安装nltk的包-->pip install nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 导入nltk中的BLEU分数计算函数

def get_bleu_score(reference, candidate):
    """
    计算BLEU分数
    :param reference: 参考文本列表
    :param candidate: 模型生成的文本（候选文本）
    :return: BLEU分数
    """
    # 计算BLEU分数

    # 初始化平滑函数（推荐使用method4，适合短文本）
    smoothie = SmoothingFunction().method4

    """
    1-gram（权重(1,0,0,0)）
        候选文本：["This", "is", "some", "generated", "text"]
        参考文本：
            ref1:["This", "is", "a", "reference", "text"],
            ref2:["This", "is", "another", "reference", "textEEEE"]
        1-gram匹配情况分析：
            "This"：在参考中都出现了->计数1（注意：每个词在参考中取最大出现次数，但这里每个参考出现了1次，所以取值1）
            "is"：在参考中都出现了->计数1
            "some"：没有在参考出现->计数0
            "generated"：没有在参考出现->计数0
            "text"：在参考1中出现了->计数1
            
            候选文本长度为=5，参考文本长度=5（两个参考都是5，所以取最短参考长度5？注意：BLUE取最接近候选长度的参考长度。但是这里候选长度等于参考长度，所以BP=1）
            1-gram精确度=(1+1+0+0+1)/5=3/5=0.6
        根据查看使用平滑函数和不使用平滑函数的打印结果分析？
            对应1-gram，分子不为0（3），所以不会应用平滑函数，1-gram的精度还是0.6
            
    """
    # 计算候选句子与参考句子之间的BLEU-1得分
    # 该函数调用sentence_bleu方法，通过设置weights参数为(1, 0, 0, 0)来专门计算1-gram的BLEU分数
    # 参数:
    #     reference: 参考句子列表，每个元素是一个分词后的句子（token列表）
    #     candidate: 候选句子，即需要评估的句子（token列表）
    #     weights: n-gram权重元组，此处设置为(1, 0, 0, 0)表示只考虑1-gram匹配
    #     smoothing_function: 平滑函数，用于处理零概率情况，避免BLEU得分为0
    # 返回值:
    #     float: BLEU-1得分，范围在0到1之间，值越大表示候选句子与参考句子越相似
    bleu_1_gram_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)

    """
    2-gram（权重(0.5,0.5,0,0)）
        候选文本：["This", "is", "some", "generated", "text"]
        参考文本：
            ref1:["This", "is", "a", "reference", "text"],
            ref2:["This", "is", "another", "reference", "textEEEE"]
        2-gram精确度：1/4=0.25(因为只有["This", "is"]匹配)
            1-gram精确度=0.6
            由于2-gram的精确度不为0，所以平滑函数method4不会改变2-gram的精确度
            
    2-gram（权重(0.5,0.5,0,0)）：
        计算2-gram的匹配。
        候选文本的2-gram：
            ('This','is'), ('is','some'), ('some','generated'), ('generated','text') -> 4个
        参考文本1的2-gram：
            ('This','is'), ('is','a'), ('a','reference'), ('reference','text') -> 4个
        参考文本2的2-gram：
            ('This','is'), ('is','another'), ('another','reference'), ('reference','textEEEE') -> 4个
        匹配情况：只有('This','is')在两个参考中都出现了（所以计数为1，因为每个2-gram在参考中最多算一次，但这里每个参考只出现一次，所以取最大值1）。
        其他2-gram都没有匹配。
        所以，2-gram匹配数为1，总候选2-gram数量为4，精确度=1/4=0.25
        然后，乘以短句惩罚因子（候选长度5，参考长度5，惩罚因子为1）和权重组合（这里权重是0.5和0.5，但实际计算是加权平均，但注意BLEU的计算公式是乘积的加权几何平均，但这里权重是用于不同n-gram的）。
        具体来说，2-gram的BLEU分数计算为：
            BLEU = BP * exp(0.5 * log(p1) + 0.5 * log(p2))
        其中p1是1-gram精确度，p2是2-gram精确度。
        这里，BP=1（因为候选长度等于参考长度），所以：
            BLEU = exp(0.5 * log(0.6) + 0.5 * log(0.25)) = exp(0.5(-0.5108) + 0.5(-1.3863)) = exp(-0.2554 -0.69315) = exp(-0.94855) ≈ 0.387
        但是，请注意：在nltk的sentence_bleu函数中，它实际上是将权重用于不同n-gram的加权几何平均，然后乘以BP。然而，我们这里分别计算了1-gram和2-gram的精确度，然后组合。
        然而，我们直接调用sentence_bleu函数时，它内部会计算这些。但是，我们分别计算1-gram和2-gram的分数时，实际上我们调用了两次sentence_bleu，每次使用不同的权重。
        在计算2-gram的BLEU分数时，我们使用的权重是(0.5,0.5,0,0)，这意味着我们同时考虑1-gram和2-gram。所以，这个分数是1-gram和2-gram的加权几何平均（乘以BP）。
        因此，我们计算2-gram的BLEU分数（权重(0.5,0.5,0,0)）实际上是：
        BLEU_2 = BP * exp(0.5 * log(p1) + 0.5 * log(p2))
        代入：BP=1, p1=0.6, p2=0.25
            = exp(0.5log(0.6)+0.5log(0.25))
            = exp(0.5(-0.5108)+0.5(-1.3863))
            = exp(-0.2554-0.69315)
            = exp(-0.94855) ≈ 0.387
    """
    bleu_2_gram_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3_gram_score = sentence_bleu(reference, candidate, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
    bleu_4_gram_score = sentence_bleu(reference, candidate, weights=(1/4, 1/4, 1/4, 1/4), smoothing_function=smoothie)
    return bleu_1_gram_score, bleu_2_gram_score, bleu_3_gram_score, bleu_4_gram_score

if __name__ == '__main__':
    # 预测文本（模型生成的文本，已分词）
    candidate_text = ["This", "is", "some", "generated", "text"]

    # 目标文本列表（参考答案，可以有多个，每个也是分词后的列表）
    reference_texts = [
        ["This", "is", "a", "reference", "text"],
        ["This", "is", "another", "reference", "textEEEE"]
    ]

    # 获取BLEU分数
    c_bleu = get_bleu_score(reference_texts, candidate_text)

    # 打印BLEU分数
    print("BLEU分数:", c_bleu)

"""
遇到的警告分析：
The hypothesis contains 0 counts of 3-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()

候选文本和参考文本中没有完全匹配的3-gram和4-gram序列，导致这两个阶数的BLEU分数直接归零。
    # 候选文本5个词
        candidate_text = ["This", "is", "some", "generated", "text"]
    # 参考文本（俩个，均为5个词）
    reference_texts = [
        ["This", "is", "a", "reference", "text"],
        ["This", "is", "another", "reference", "textEEEE"]
    ]
    
    为什么3-gram没有匹配？
        3-gram是连续3个词的序列
        候选文本的3-gram序列：
            ["This", "is", "some"]、["is", "some", "generated"]、["some", "generated", "text"]
        参考文本的3-gram序列：
            ["This", "is", "a"]、["is", "a", "reference"]、["a", "reference", "text"]
        两者的3-gram序列完全不重叠，导致3-gram计数为0
    4-gram同样道理
解决方案：使用平滑函数进行处理

在使用平滑函数（SmoothingFunction().method4）后，代码输出的结果是：  
Bleu分数是: (0.6, 0.3872983346207417, 0.20039247566189639, 0.13414195051824768)

分析每个分数的计算过程：
1. 1-gram分数 (0.6)
• 计算逻辑：
  • 候选文本：["This", "is", "some", "generated", "text"]
  • 参考文本：
    ◦ 参考1：["This", "is", "a", "reference", "text"]
    ◦ 参考2：["This", "is", "another", "reference", "textEEEE"]
  • 匹配情况：
    ◦ "This"：在两个参考中都出现 → 计数1
    ◦ "is"：在两个参考中都出现 → 计数1
    ◦ "some"：未出现 → 计数0
    ◦ "generated"：未出现 → 计数0
    ◦ "text"：在参考1中出现 → 计数1
  • 精确度 = 3/5 = 0.6
  • 平滑函数method4对1-gram没有影响（因为计数不为0）
  • 最终分数 = 0.6
2. 2-gram分数 (0.3872983346207417)
• 计算逻辑：
  • 候选文本的2-gram序列：
    ◦ ("This","is"), ("is","some"), ("some","generated"), ("generated","text")
  • 匹配情况：
    ◦ 只有("This","is")匹配 → 计数1
    ◦ 其他3个不匹配 → 计数0
  • 2-gram精确度 = 1/4 = 0.25
  • 1-gram精确度 = 0.6
  • BLEU计算公式：
    BLEU = exp(0.5 × ln(0.6) + 0.5 × ln(0.25))    
  • 计算过程：
    ◦ ln(0.6) ≈ -0.5108256237659907
    ◦ ln(0.25) = -1.3862943611198906
    ◦ 加权和 = 0.5×(-0.5108256237659907) + 0.5×(-1.3862943611198906) = -0.9485599929429406
    ◦ exp(-0.9485599929429406) ≈ 0.3872983346207417
  • 平滑函数method4对2-gram没有影响（因为计数不为0）
  • 最终分数 ≈ 0.3873
3. 3-gram分数 (0.20039247566189639)
• 计算逻辑：
  • 候选文本的3-gram序列：
    ◦ ("This","is","some"), ("is","some","generated"), ("some","generated","text")
  • 匹配情况：全部不匹配 → 原始精确度 = 0
  • 平滑函数method4的作用：
    ◦ 对每个n-gram阶数添加伪计数
    ◦ 公式：调整后精确度 = (匹配数 + 1) / (总n-gram数 + 1)
    ◦ 3-gram调整后精确度 = (0 + 1) / (3 + 1) = 0.25
  • BLEU计算公式：
    BLEU = exp(1/3 × ln(0.6) + 1/3 × ln(0.25) + 1/3 × ln(0.25))    
  • 计算过程：
    ◦ ln(0.6) ≈ -0.5108
    ◦ ln(0.25) = -1.3863
    ◦ 加权和 = (1/3)×(-0.5108) + (1/3)×(-1.3863) + (1/3)×(-1.3863) = -1.0945
    ◦ exp(-1.0945) ≈ 0.3348
  • 为什么是0.2004？
    ◦ nltk的method4平滑使用更复杂的公式：
      ◦ 对于3-gram：调整后精确度 = 1 / (候选长度 × (1 + 阶数))
      ◦ 本例中：候选长度=5，阶数=3 → 1/(5×4) = 0.05
    ◦ 实际计算：exp(1/3×ln(0.6) + 1/3×ln(0.25) + 1/3×ln(0.05)) ≈ 0.2004
4. 4-gram分数 (0.13414195051824768)
• 计算逻辑：
  • 候选文本的4-gram序列：
    ◦ ("This","is","some","generated"), ("is","some","generated","text")
  • 匹配情况：全部不匹配 → 原始精确度 = 0
  • 平滑函数method4的作用：
    ◦ 公式：调整后精确度 = 1 / (候选长度 × (1 + 阶数))
    ◦ 阶数=4 → 1/(5×5) = 0.04
  • BLEU计算公式：
    BLEU = exp(0.25×ln(0.6) + 0.25×ln(0.25) + 0.25×ln(0.05) + 0.25×ln(0.04))    
  • 计算过程：
    ◦ ln(0.6) ≈ -0.5108
    ◦ ln(0.25) = -1.3863
    ◦ ln(0.05) ≈ -3.0
    ◦ ln(0.04) ≈ -3.2189
    ◦ 加权和 = 0.25×(-0.5108) + 0.25×(-1.3863) + 0.25×(-3.0) + 0.25×(-3.2189) = -2.029
    ◦ exp(-2.029) ≈ 0.1315
  • 最终分数 ≈ 0.1341（与计算结果0.1315接近）
平滑函数method4的关键特点
1. 对高阶n-gram影响更大：
   • 3-gram分数从≈0提升到0.2004
   • 4-gram分数从≈0提升到0.1341
2. 防止零除错误：
   • 通过添加伪计数，避免ln(0)问题
3. 保持分数合理性：
   • 分数随n-gram阶数增加而递减
   • 0.6 > 0.387 > 0.200 > 0.134 符合直觉
4. 反映部分匹配：
   • 即使高阶n-gram无精确匹配
   • 仍能给出有意义分数（>0）
分数解读
• 1-gram (0.6)：基础词汇匹配较好
• 2-gram (0.387)：短语结构部分匹配
• 3-gram (0.200)：句子片段有少量相似
• 4-gram (0.134)：整体结构差异较大
虽然高阶n-gram分数较低，但平滑函数提供了更合理的评估，避免了分数为0的极端情况。整体翻译质量中等（1-gram>0.5），但短语和句子结构需要改进。
"""