import os
from openai import OpenAI , APIError


#配置API客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

#调用模型
model = "qwen3.7-plus"
instructions="你是一位Python助教，用简短的中文回答。"

#开始时，没有上一轮的相应
previous_id = None

while True:
    user_input = input("\n你：").strip()

    if user_input.lower() == "exit":
        print("对话结束。")
        break

    if user_input == "/reset":
        previous_id = None
        print("已开启新对话。")
        continue

    if not user_input:
        continue

    #构造本轮请求参数
    request_params = {
        'model': model,
        'instructions': instructions,
        'input': user_input,
    }

    #第一轮不传历史ID，后续轮次再传
    if previous_id is not None:
        request_params["previous_response_id"] = previous_id

    # response = client.responses.create(**request_params)
    #
    # print("\n助手：",response.output_text)
    #
    # #更新为本轮响应ID，下一轮从这里继续
    # previous_id = response.id

    # #一次性全部回复
    # try:
    #     response = client.responses.create(**request_params)
    # except APIError as e:
    #     print("本轮请求失败：", e)
    #     continue
    #
    # print("\n助手：",response.output_text)
    # previous_id = response.id

    #流式输出
    #给completed 初始化False，用来区分，“收到完成事件”和“接收结束但没有确认完成”
    completed = False

    try :
        stream = client.responses.create(
            **request_params,
            stream = True,
        )

        print("\n助手：",end="",flush=True)

        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta,end="",flush=True)

            elif event.type == "response.completed":
                previous_id = event.response.id
                completed = True
                print()

        if not completed :
            print("\n本轮未确认完整完成，保留上一轮的历史ID。")

    except APIError as e:
        print("\n本轮请求失败",e)
        continue
