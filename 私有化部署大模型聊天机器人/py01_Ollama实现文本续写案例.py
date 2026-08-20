"""
需求：构建一个简单的命令行交互程序，用户可以通过输入问题与AI模型进行对话。该程序将使用qwen2:0.5b语言模型。用户可以通过命令行输入问题，程序将调用AI模型生成回答，并将结果输出到终端。
"""
#需要先安装模块包：pip install ollama
import ollama

#使用ollama模块调用chat聊天函数，传入api接口对应的参数
#model：指定使用的模型名称，这里是qwen2：1.5b
#message：一个列表，包含了对话的历史记录，每个元素都是一个字典，包含role 和 content 两个键值对，role 表示角色，user表示用户，
response = ollama.chat(model="qwen2:1.5b",messages=[{'role':'user','content':'从前有座山，山里有座庙，庙里有一个和尚，续写故事'}])

#打印输出结果（模型返回的内容）
print(response["message"]["content"])

