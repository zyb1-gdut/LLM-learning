import os
from openai import OpenAI,APIError
import json
#可连续使用工具助手
#自定义函数
def calculate_sales(name,quantity,cost_price,sale_price):
    total_cost = quantity * cost_price
    total_income = quantity * sale_price
    gross_profit = total_income - total_cost

    return {
        "name": name,
        "total_cost": total_cost,
        "total_income": total_income,
        "gross_profit":gross_profit,
    }

# #采用字典解包
# argument =  {"name":"苹果","quantity":2,"cost_price":3,"sale_price":3.5,}
# result = calculate_sales(**argument)


# result = calculate_sales(
#     name="苹果",
#     quantity=5,
#     cost_price=3,
#     sale_price=3.5,
# )
# print(result)

#配置API客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

#工具说明,描述函数名和参数
tools = [
    {
        "type":"function",
        "name":"calculate_sales",
        "description":"计算一种水果售出后的总进货成本、总收入和毛利",
        "parameters":{
            "type":"object",#object 表示一组键值字段，string 表示字符串，number 表示数字。
            "properties":{
                "name":{
                    "type":"string",
                    "description":"水果名称",
                },
                "quantity":{
                    "type":"number",
                    "description":"售出数量，单位为斤",
                },
                "cost_price":{
                    "type":"number",
                    "description":"每斤进货价，单位为元",
                },
                "sale_price":{
                    "type":"number",
                    "description":"每斤售价，单位为元"
                },
            },
            "required":[
            "name",
            "quantity",
            "cost_price",
            "sale_price",
            ],
        },
    }
]

#调用模型
response = client.responses.create(
    model = "qwen3.7-plus",
    instructions= "水果销售计算请使用工具，缺少必要信息时先询问用户。",
    input=(
        "苹果进货价每斤3元，售价每斤3.5元，卖了8斤；"
        "香蕉进货价每斤2元，售价每斤3元，卖了5斤。"
        "请分别计算两种水果的成本、收入和毛利。"
    ),
    tools = tools,
)
#若没有调用函数，可能是缺少信息，补充信息
#最多接收三次补充信息
followup_count = 0
max_followups = 3
while True:
    #每次检查当前response是否包含函数调用
    has_function_call = False

    for item in response.output:
        if item.type == "function_call":
            has_function_call = True

    #已经拿到调用请求，离开追问循环
    if has_function_call:
        break

    print("助手：",response.output_text)

    if followup_count >= max_followups:
        print("已达到补充次数上限，本次结束。")
        raise SystemExit

    extra_input = input("你(输入exit退出):").strip()

    if extra_input.lower() == "exit":
        print("已退出。")
        raise SystemExit

    if not extra_input:
        print("输入不能为空。")
        continue

    response = client.responses.create(
        model = "qwen3.7-plus",
        instructions= "水果销售计算请使用工具，缺少必要信息时先询问用户。",
        previous_response_id = response.id,
        input = extra_input,
        tools = tools,
    )
    followup_count += 1

max_tool_rounds = 4
tool_rounds = 0

while True:
    # 收集当前响应中的全部函数调用请求
    function_calls = []#模型交给程序的任务

    for item in response.output:
        if item.type == "function_call":
            function_calls.append(item)

    # 没有更多工具请求，就展示本轮文本并结束
    if not function_calls:
        print("\n助手：", response.output_text)
        break

    # 防止模型一直请求工具
    if tool_rounds >= max_tool_rounds:
        print("已达到工具执行轮数上限，停止本次任务。")
        break

    tool_rounds += 1
    print(f"\n第 {tool_rounds} 轮工具执行")

    # 每轮重新收集结果，不能混入上一轮的结果
    tool_outputs = []#程序准备交回模型的结果

    for item in function_calls:
        print("函数名称：", item.name)
        print("调用编号：", item.call_id)
        print("原始参数：", item.arguments)

        try:
            if item.name != "calculate_sales":
                raise ValueError(f"不支持的函数：{item.name}")

            arguments = json.loads(item.arguments)

            if not isinstance(arguments, dict):
                raise ValueError("参数必须是一组键值字段")

            required_fields = {
                "name", "quantity", "cost_price", "sale_price"
            }

            if set(arguments) != required_fields:
                raise ValueError("参数有缺失或包含未知字段")

            if not isinstance(arguments["name"], str):
                raise ValueError("水果名称必须是字符串")

            for field in ("quantity", "cost_price", "sale_price"):
                value = arguments[field]

                if type(value) not in (int, float) or value < 0:
                    raise ValueError(f"{field} 必须是非负数字")

            result = calculate_sales(**arguments)
            print("本地计算结果：", result)
            #包装结果
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

        # 成功或失败，都给这次调用返回明确结果.type、call_id、output 是接口约定的外层格式；里面的 ok、data、error 是我们自己设计的结果内容。
        tool_outputs.append({
            "type": "function_call_output",#我是一份函数执行结果。
            "call_id": item.call_id,#我回答的是编号为 XXX 的调用
            "output": json.dumps(tool_result, ensure_ascii=False),#这是执行得到的数据。
        })

    try:
        response = client.responses.create(
            model="qwen3.7-plus",
            instructions=(
                "你是水果销售计算助手。"
                "需要计算时使用工具，缺少信息时询问用户。"
                "根据工具结果回答，不要编造计算结果。"
                "如果工具返回错误，检查原因，不要原样重复错误调用。"
                "所有所需计算完成后给出最终答案，区分毛利与净利润。"
            ),
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
            tool_choice="auto",
        )

    except APIError as error:
        print("模型请求失败：", error)
        break