#需求：构建一个简单的命令行交互程序，用户可以通过输入问题与AI模型进行对话。该程序将使用qwen2:0.5b语言模型。用户可以通过命令行输入问题，程序将调用AI模型生成回答，并将结果输出到终端。
import ollama

while True:
        #输入问题
        prompt = input("请输入您的问题：")
        response = ollama.chat(model="qwen2:1.5b",messages=[{'role':'user','content':prompt}])
        #调用模型回答
        result = response["message"]["content"]
        #打印结果
        print(result)
