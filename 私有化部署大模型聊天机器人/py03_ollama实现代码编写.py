#需求：构建一个基于AI模型的代码生成工具，用户可以通过输入自然语言描述所需功能，AI模型将自动生成相应的Python代码。该工具旨在帮助开发者快速获取代码片段，减少手动编写代码的时间。
import ollama

#指令
prompt = """
请为以下功能生成一段Python代码：
求两个数的最大公约数
"""
#调用模型
response = ollama.chat(model="qwen2:1.5b",messages=[{'role':'user','content':prompt}])

result = response['message']['content']
print(result)

def gcd(a, b):
    if a == 0:
        return b
    return gcd(b % a, a)

num1 = int(input("请输入第一个数: "))
num2 = int(input("请输入第二个数: "))

print("最大公约数为:", gcd(num1, num2))