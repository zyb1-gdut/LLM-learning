#导入相关的包
import ollama
import re #正则表达式
import json  #美化输出

from tblib.decorators import return_error

#定义实体类型
schema = {
    '金融':['日期','股票名称','开盘价','收盘价','成交量']
}

IE_PATTERN = "{}\n\n提取上述句子中的{}的实体，并按照json的格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','隔开。"

#提供少量的示例
ie_examples = {
    '金融':[
        {'content':'2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，一度飙升至105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。',
        'answers':{
            '日期':['2023-01-10'],
            '股票名称':['古哥-D[EOOE]美股'],
            '开盘价':['100美元'],
            '收盘价':['102美元'],
            '成交量':['520000']
            }
        }
    ]
}


def init_prompts():
    ie_pre_history = [{'role':'system','content':'你是一个信息抽取助手。'}]
    for tp,example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            properties_str = ','.join(schema[tp])
            schema_str_list = f'"{tp}"({properties_str})'
            sentence_with_prompt = IE_PATTERN.format(sentence,schema_str_list)
            ie_pre_history.append({'role':'user','content':f'{sentence_with_prompt}'})
            ie_pre_history.append({'role':'assistant','content':f"{json.dumps(example['answers'],ensure_ascii=False)}"})
    return {'ie_pre_history':ie_pre_history}

def clean_response(response):
    #处理结果数据
    if '```json' in response:
        #从response这个字符串中，提取出所有被 ```json 和 ``` 包围起来的内容，并取出里面的内容
        res = re.findall(r'```json(.*?)```',response,re.DOTALL)
        print(f're------>{re}')
        if len(res) and res[0]:
            response = res[0]
    response.replace('、',',')
    try:
        return json.loads(response)
    except:
        return response

def inference(
        sentences:list,
        custom_settings:dict
              ):
    '''
    推理函数：先接收输入的数据，构造实例对象，调用模型，输出数据，再处理数据

    '''
    for sentence in sentences:
        cls_res = '金融'
        #安全措施，出现多个实体类型的时候，需要进行判断
        if cls_res not in schema:
            print(f'The type model inferences {cls_res} which is not in schema dict,exited.')
            exit()
        properties_str = ','.join(schema[cls_res])
        schema_str_list = f'"{cls_res}"({properties_str})'
        sentence_with_prompt = IE_PATTERN.format(sentence,schema_str_list)
        messages = [*custom_settings['ie_pre_history'],{'role':'user','content':sentence_with_prompt}]
        response = ollama.chat(model='qwen2:1.5b',messages=messages)
        res_content = response['message']['content']
        ie_res = clean_response(res_content)
        print(f'sentence:{sentence}')
        print(f'inference answer:{ie_res}')


if __name__ == '__main__':
    #初始化句子和自定义设置
    sentences = [
        '2023-02-15，寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。',
        '2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，投资者保持观望态度。',
    ]
    #自定义设置
    custom_settings = init_prompts()
    #开始推理
    inference(sentences, custom_settings)