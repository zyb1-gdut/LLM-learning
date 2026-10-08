import os
from openai import OpenAI,APIError

#配置API客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

#调用模型
try :
    response = client.responses.create(
        model = "qwen3.7-plus",
        instructions = "请搜索网络后回答，优先使用官方网站，并附来源链接。",
        input = "请查询Python官网目前最新的稳定版本及其分布日期，不含预发布版本。",
        tools = [
            {"type":"web_search"}
        ],
    )

except APIError as e :
    print("请求失败",e)

else :
    print("助手：",response.output_text)

    print("\n本次响应包含的输出项：")
    for item in response.output :
        print("类型：",item.type)

        if item.type == "web_search_call":
            print(item.model_dump_json(indent=2))

