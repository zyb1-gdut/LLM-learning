import os
from openai import OpenAI
import json

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

#工具说明
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


tool_outputs = [] #保存准备回传的结果
tool_call_count = 0 #记录模型提出了几次函数调用

for item in response.output:
    print("输出类型：",item.type)

    if item.type == "function_call":
        tool_call_count += 1
        print("函数名称：",item.name)
        print("原始参数：",item.arguments)

        #只允许执行明确支持的函数
        if item.name != "calculate_sales":
            print("不支持的函数：",item.name)
            continue

        try:
            #把JSON字符串转化为Python对象
            arguments= json.loads(item.arguments)#json.loads：JSON 字符串 → Python 字典

            #检查参数结构：
            if not isinstance(arguments,dict):
                raise ValueError("参数必须时一组键值字段")

            required_fields = {"name","quantity","cost_price","sale_price"}

            if set(arguments) != required_fields:
                raise ValueError("参数有缺失或包含未知字段")
            if not isinstance(arguments["name"],str):
                raise ValueError("水果名称必须是字符串")
            for field in ("quantity","cost_price","sale_price"):
                value = arguments[field]

                if type(value) not in (int,float) or value < 0:
                    raise ValueError(f"{field}必须是非负数字")

            #执行本地函数
            result = calculate_sales(**arguments)

        except (ValueError, TypeError) as e:
            print("工具参数或计算出错",e)
            continue

        print("解析后的参数：",arguments)
        print("本地计算结果：",result)

        tool_outputs.append({
            "type":"function_call_output",
            "call_id":item.call_id ,#函数调用请求”和“函数执行结果”配对
            "output":json.dumps(result,ensure_ascii=False),#json.dumps：Python 字典 → JSON 字符串

        })

#第二次模型请求
if tool_call_count == 0:
    print("模型没有请求工具:",response.output_text)

elif len(tool_outputs) != tool_call_count:
    print("部分工具没有成功执行，本次先不生成最终回答。")

else:
    final_response = client.responses.create(
        model = "qwen3.7-plus",
        instructions = (
            "根据工具返回的计算结果，用简短的中文回答用户。"
            "说明总成本、总收入和毛利，不要将毛利称为净利润。"
        ),
        previous_response_id=response.id,
        input=tool_outputs,
        tools=tools,
        tool_choice="none",
    )
    print("模型请求的调用次数：", tool_call_count)
    print("成功准备的结果数量：", len(tool_outputs))
    print("\n最终回答：",final_response.output_text)