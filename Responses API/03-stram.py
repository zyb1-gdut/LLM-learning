import os
from openai import OpenAI

#配置API客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

#设置流式输出
stream = client.responses.create(
    model = 'qwen3.7-plus',
    instructions = "你是一位Python助教，用简短中文回答。",
    input = "用一个简单的例子解释Python列表",
    stream = True,
)

#打印事件类型名称，
# for event in stream:
#     print(event.type)

#同时打印正文
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta,end="",flush=True)

    elif event.type == "response.completed":
        print("\n\n本轮回答完成。")
        print("响应ID：",event.response.id)
