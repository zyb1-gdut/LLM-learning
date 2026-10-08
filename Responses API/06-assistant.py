import os
import json
from openai import OpenAI, APIError


# 1. 真正负责计算的本地函数
def calculate_sales(name, quantity, cost_price, sale_price):
    total_cost = quantity * cost_price
    total_income = quantity * sale_price
    gross_profit = total_income - total_cost

    return {
        "name": name,
        "total_cost": total_cost,
        "total_income": total_income,
        "gross_profit": gross_profit,
    }


# 2. 配置模型客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

MODEL = "qwen3.7-plus"

INSTRUCTIONS = (
    "你是水果销售计算助手，用简短中文回答。"
    "需要计算时使用工具，缺少必要信息时先询问用户。"
    "不要编造价格、销量等参数。"
    "根据工具结果回答，区分毛利与净利润。"
    "工具返回错误时，检查原因，不要原样重复错误调用。"
)


# 3. 给模型看的工具说明，不是函数实现代码
tools = [
    {
        "type": "function",
        "name": "calculate_sales",
        "description": "计算一种水果的总进货成本、总收入和毛利。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "水果名称",
                },
                "quantity": {
                    "type": "number",
                    "description": "售出数量，单位为斤",
                },
                "cost_price": {
                    "type": "number",
                    "description": "每斤进货价，单位为元",
                },
                "sale_price": {
                    "type": "number",
                    "description": "每斤售价，单位为元",
                },
            },
            "required": [
                "name",
                "quantity",
                "cost_price",
                "sale_price",
            ],
        },
    }
]


# 保存对话衔接位置，开始时没有历史
previous_id = None

# 每条用户消息最多执行四轮工具
max_tool_rounds = 4

print("水果算账助手已启动。输入 /reset 重置，输入 exit 退出。")


# 4. 外层循环：持续接收用户输入
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

    request_params = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": user_input,
        "tools": tools,
    }

    if previous_id is not None:
        request_params["previous_response_id"] = previous_id

    try:
        response = client.responses.create(**request_params)
    except APIError as error:
        print("请求失败：", error)
        continue

    # 每次收到新的用户输入，工具轮数重新计数
    tool_rounds = 0


    # 5. 内层循环：处理当前任务的工具调用
    while True:
        function_calls = []

        for item in response.output:
            if item.type == "function_call":
                function_calls.append(item)

        # 没有工具调用：显示回答或追问，回去等待用户输入
        if not function_calls:
            print("\n助手：", response.output_text)
            previous_id = response.id
            break

        if tool_rounds >= max_tool_rounds:
            print("达到工具轮数上限，本次停止，保留此前的对话位置。")
            break

        tool_rounds += 1
        print(f"\n第 {tool_rounds} 轮工具执行")

        # 只保存这一轮工具执行的结果
        tool_outputs = []

        for item in function_calls:
            print("函数名称：", item.name)
            print("调用编号：", item.call_id)
            print("原始参数：", item.arguments)

            try:
                # 检查请求的函数是否在允许范围内
                if item.name != "calculate_sales":
                    raise ValueError(f"不支持的函数：{item.name}")

                # JSON 字符串转成 Python 对象
                arguments = json.loads(item.arguments)

                if not isinstance(arguments, dict):
                    raise ValueError("参数必须是一组键值字段")

                required_fields = {
                    "name", "quantity", "cost_price", "sale_price"
                }

                # 检查参数名是否齐全、有无多余字段
                if set(arguments) != required_fields:
                    raise ValueError("参数有缺失或包含未知字段")

                if not isinstance(arguments["name"], str):
                    raise ValueError("水果名称必须是字符串")

                # 检查数字参数
                for field in ("quantity", "cost_price", "sale_price"):
                    value = arguments[field]

                    if type(value) not in (int, float) or value < 0:
                        raise ValueError(f"{field} 必须是非负数字")

                # 真正执行本地函数
                result = calculate_sales(**arguments)
                print("本地计算结果：", result)

                tool_result = {
                    "ok": True,
                    "data": result,
                }

            except (ValueError, TypeError) as error:
                print("工具执行失败：", error)

                tool_result = {
                    "ok": False,
                    "error": str(error),
                }

            # 包装成接口要求的工具结果格式
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(tool_result, ensure_ascii=False),
            })

        # 将本轮全部工具结果回传给模型
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools,
                tool_choice="auto",
            )
        except APIError as error:
            print("结果回传失败，本次停止：", error)
            break
