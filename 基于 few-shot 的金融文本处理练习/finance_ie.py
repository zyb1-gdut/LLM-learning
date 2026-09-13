# coding:utf-8
import json
import ollama
import re
# from rich import print

# 定义不同类型下的实体类型
schema = {
    '金融': ['日期', '股票名称', '开盘价', '收盘价', '成交量']
}

IE_PATTERN = "{}\n\n提取上述句子中{}的实体，并按照JSON格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','分隔。"



# 提供一些例子供模型参考
ie_examples = {
    '金融': [
        {
            'content': '2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，一度飙升至105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。',
            'answers': {
                '日期': ['2023-01-10'],
                '股票名称': ['古哥-D[EOOE]美股'],
                '开盘价': ['100美元'],
                '收盘价': ['102美元'],
                '成交量': ['520000'],
            }
        }
    ]
}


def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """

def clean_response(response: str):
    """
    后处理模型输出。

    Args:
        response (str): _description_
    """
    # response1='```json["name":lucy]```abc```json["name":lucy]```'


def inference(
        sentences: list,
        custom_settings: dict
    ):
    """
    推理函数。
    Args:
        sentences (List[str]): 待抽取的句子。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """



if __name__ == '__main__':

    # 初始化句子和自定义设置
    sentences = [
        '2023-02-15，寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。',
        '2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，投资者保持观望态度。',
    ]

    # 初始化自定义设置

    # print(custom_settings)

    # # 开始推理
