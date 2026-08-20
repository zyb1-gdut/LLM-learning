# 1. 导入相关包
import ollama

# 2. 定义一个函数，用于发起请求，返回结果
# def get_response(prompt):
#     messages = []
#     messages.append({'role': 'user', 'content': prompt, })
#     response = ollama.chat(model='qwen2.5:7b', messages=messages)
#     return response['message']['content']

# 3. 定义一个函数，用于发起请求，返回结果
def get_response(prompt):
    # response = ollama.chat(model='qwen2:0.5b', messages=prompt[-50:])
    if isinstance(prompt,str):
        response = ollama.chat(model='deepseek-r1:8b', messages=[{'role':'user','content':prompt}])
    else:
        response = ollama.chat(model='deepseek-r1:8b', messages=prompt[-50:])
    return response['message']['content']


# 7. 测试，测试结束后，终止
if __name__ == '__main__':
    s=input("请输入你要表达的内容")
    prompt = s
    result = get_response(prompt)
    print(result)
