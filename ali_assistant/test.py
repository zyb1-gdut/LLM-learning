import os
import requests
import yaml
def read_conf(path='conf_ali.yml'):
    with open(path,'r',encoding='utf-8') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    return conf

conf = read_conf()

API_KEY = conf['API_KEY']

url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

request_body = {
    "model": "MiniMax-M1",
    "tokens_to_generate": 8192,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
}

while True:
    line = input("User input: ")
    if line.lower() in ['quit', 'exit']:
        break

    request_body["messages"].append({"role": "user", "content": line})

    response = requests.post(url, headers=headers, json=request_body)
    print(f'response:{response}')
    response_data = response.json()
    reply = response_data["choices"][0]["message"]["content"]
    reasoning_content = response_data["choices"][0]["message"]["reasoning_content"]
    print(f"Reasoning: {reasoning_content}")
    print(f"Reply: {reply}")

    request_body["messages"].append({"role": "assistant", "content": reply})

