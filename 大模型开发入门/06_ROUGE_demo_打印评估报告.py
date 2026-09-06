"""
ROUGE评估器实现
使用rouge库计算文本生成质量指标
包含ROUGE-1、ROUGE-2和ROUGE-L的计算
"""

# 安装依赖: pip install rouge
from rouge import Rouge
from typing import List, Dict, Union

def calculate_rouge(
        generated_text: str,
        reference_texts: Union[str, List[str]],
        metrics: List[str] = ["rouge-1", "rouge-2", "rouge-l"]
) -> List[Dict]:
    """
    计算生成文本与参考文本之间的ROUGE分数

    参数:
        generated_text: 模型生成的文本
        reference_texts: 参考文本，可以是单个字符串或字符串列表
        metrics: 需要计算的ROUGE指标列表

    返回:
        包含各ROUGE指标分数的字典列表
    """
    # 确保参考文本是列表格式
    if isinstance(reference_texts, str):
        reference_texts = [reference_texts]

    # 初始化Rouge评估器
    rouge = Rouge()

    # 存储所有参考文本的评分结果
    all_scores = []

    # 计算每个参考文本的ROUGE分数
    for ref_text in reference_texts:
        try:
            # 计算ROUGE分数
            scores = rouge.get_scores(generated_text, ref_text)[0]

            # 只保留指定的指标
            filtered_scores = {metric: scores[metric] for metric in metrics if metric in scores}
            all_scores.append(filtered_scores)
        except Exception as e:
            print(f"计算ROUGE分数时出错: {e}")
            # 返回空分数表示计算失败
            all_scores.append({metric: {"p": 0.0, "r": 0.0, "f": 0.0} for metric in metrics})

    return all_scores


def print_rouge_scores(scores_list: List[Dict]):
    """
    格式化并打印ROUGE分数

    参数:
        scores_list: ROUGE分数列表
    """
    if not scores_list:
        print("没有可用的ROUGE分数")
        return

    # 打印每个参考文本的评分
    for i, scores in enumerate(scores_list, 1):
        print(f"\n===== 参考文本 {i} 的ROUGE分数 =====")
        for metric, values in scores.items():
            print(f"\n{metric.upper()} 指标:")
            print(f"  精确率 (Precision): {values['p']:.4f}")
            print(f"  召回率 (Recall):    {values['r']:.4f}")
            print(f"  F1分数:             {values['f']:.4f}")
        print("-" * 50)


def evaluate_quality(scores_list: List[Dict]) -> Dict:
    """
    基于ROUGE分数评估文本生成质量

    参数:
        scores_list: ROUGE分数列表

    返回:
        包含质量评估结果的字典
    """
    if not scores_list:
        return {"quality_level": "无法评估", "analysis": "没有可用的ROUGE分数"}

    # 初始化最佳分数
    best_scores = {}

    # 找出每个指标的最佳分数（取最大值）
    for scores in scores_list:
        for metric, values in scores.items():
            if metric not in best_scores or values["f"] > best_scores[metric]["f"]:
                best_scores[metric] = values

    # 质量等级评估
    quality_level = "优秀"
    if best_scores.get("rouge-1", {}).get("f", 0) < 0.7:
        quality_level = "良好"
    if best_scores.get("rouge-1", {}).get("f", 0) < 0.5:
        quality_level = "一般"
    if best_scores.get("rouge-1", {}).get("f", 0) < 0.3:
        quality_level = "较差"

    # 详细分析
    analysis = {
        "vocabulary_coverage": "词汇覆盖良好" if best_scores.get("rouge-1", {}).get("r", 0) >= 0.5 else "词汇覆盖不足",
        "phrase_structure": "短语结构良好" if best_scores.get("rouge-2", {}).get("f", 0) >= 0.3 else "短语结构需改进",
        "sentence_coherence": "句子连贯性好" if best_scores.get("rouge-l", {}).get("f", 0) >= 0.5 else "句子连贯性需提高"
    }

    return {
        "quality_level": quality_level,
        "best_scores": best_scores,
        "analysis": analysis
    }


if __name__ == "__main__":
    # 示例用法
    generated_text = "这 是 一些 生成 文本"
    reference_texts = [
        "这 是 一个 参考 文本",
        "这 是 另外 一个 参考 文本",
    ]

    print("生成文本:", generated_text)
    print("参考文本:")
    for i, ref in enumerate(reference_texts, 1):
        print(f"  参考文本 {i}: {ref}")

    # 计算ROUGE分数
    rouge_scores = calculate_rouge(
        generated_text=generated_text,
        reference_texts=reference_texts,
        metrics=["rouge-1", "rouge-2", "rouge-l"]
    )

    # 打印结果
    print("\n===== ROUGE分数详情 =====")
    print_rouge_scores(rouge_scores)

    # 评估质量
    quality_result = evaluate_quality(rouge_scores)
    print("\n===== 质量评估 =====")
    print(f"总体质量: {quality_result['quality_level']}")
    print("详细分析:")
    for aspect, comment in quality_result["analysis"].items():
        print(f"  - {aspect.replace('_', ' ')}: {comment}")

    # 打印最佳分数
    print("\n最佳ROUGE分数:")
    for metric, values in quality_result["best_scores"].items():
        print(f"  {metric.upper()}: F1={values['f']:.4f}")
