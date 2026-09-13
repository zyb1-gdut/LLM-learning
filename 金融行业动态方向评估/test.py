import ollama
# 聊天式
response = ollama.chat(model='qwen3:8b',messages=[{'role': 'user', 'content': '为什么天空是蓝⾊的？', }])
print(response)
print('*' * 80)
print(response['message']['content'])
# ⽣成式
## response = ollama.generate(model='qwen2:1.5b', prompt='为什么天空是蓝⾊的？')