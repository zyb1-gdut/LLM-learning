import os
from openai import OpenAI


# 配置 API 客户端
client = OpenAI(
    api_key = os.environ['DASHSCOPE_API_KEY'],
    base_url= os.environ['DASHSCOPE_BASE_URL'],
)

# 调用模型，创建一次响应
response = client.responses.create(
    model="qwen3.7-plus",
    instructions="你是一位python助教，用简短的中文回答。",
    input="用一个简单的例子解释Python字典。",
)
#打印输出结果
print(response.output_text)
print("响应ID：",response.id)

#开启第二轮对话
second_response = client.responses.create(
    model="qwen3.7-plus",
    instructions="你是一位Python助教，用简短的中文回答。",
    previous_response_id = response.id,
    input="针对你刚才解释的知识，给我一道练习题，先不要给答案。"
)

print(second_response.output_text)
print("响应ID：",second_response.id)