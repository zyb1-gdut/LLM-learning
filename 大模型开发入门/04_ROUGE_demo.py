"""
ROUGE的评估指标

rouge-1（词汇覆盖评估）
rouge-2（短语结构评估）
rouge-l（句子连贯性评估）

"""
# 安装依赖： pip install rouge
from rouge import Rouge

# 创建Rouge对象，用来统计Rouge分数
rouge = Rouge()

# 定义参考文本和候选文本
generated_text = "这 是 一些 生成 文本"
reference = [
    "这 是 一个 参考 文本",
    "这 是 另 一个 参考 文本"
]

scores = rouge.get_scores(generated_text, reference[1])

"""
[{
'rouge-1': {'r': 0.5, 'p': 0.6, 'f': 0.5454545404958678}, 
'rouge-2': {'r': 0.2, 'p': 0.25, 'f': 0.22222221728395072}, 
'rouge-l': {'r': 0.5, 'p': 0.6, 'f': 0.5454545404958678}
}]
"""
# 打印rouge-1的数据
print("ROUGE-1 precision（精确率）：", scores[0]["rouge-1"]["p"])
print("ROUGE-1 recall（召回率）：", scores[0]["rouge-1"]["r"])
print("ROUGE-1 f1分数：", scores[0]["rouge-1"]["f"])

# 打印rouge-2的数据
print("ROUGE-2 precision（精确率）：", scores[0]["rouge-2"]["p"])
print("ROUGE-2 recall（召回率）：", scores[0]["rouge-2"]["r"])
print("ROUGE-2 f1分数：", scores[0]["rouge-2"]["f"])

# 打印rouge-l的数据
print("ROUGE-l precision（精确率）：", scores[0]["rouge-l"]["p"])
print("ROUGE-l recall（召回率）：", scores[0]["rouge-l"]["r"])
print("ROUGE-l f1分数：", scores[0]["rouge-l"]["f"])