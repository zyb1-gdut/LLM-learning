#导入相关的包
#from urllib import response

import ollama
from pprint import pprint

#from 私有化部署大模型聊天机器人.chat_v2.my_chat import message

#few-shot 示例
class_examples = {
    '新闻报道': '今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。',
    '财务报告': '本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康发展奠定了基础。',
    '公司公告': '本公司高兴地宣布成功完成最新一轮并购交易，收购了一家在人工智能领域领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力',
    '分析师报告': '最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算、人工智能和数字化转型被认为是引领行业发展的关键因素，投资者应关注这些趋势'
}

def init_prompts():
    #获取类别列表
    class_list = list(class_examples.keys())
    pre_history = [{"role":'system',"content":f'假设你是一个文本分类器，你需要按照{class_list}中的类别划分文本。'}]

    for _type,example in class_examples.items():
        pre_history.append({'role':'user','content':f'{example}是{class_list}中哪个类别的？'})
        pre_history.append({'role':'assistant','content':f'{_type}'})

    return {'class_list':class_list,'pre_history':pre_history}

def inference(
        sentences:list,
        custom_settings:dict
):
    response = []
    i = 0
    for sentence in sentences:
        sentence_with_prompt = f"“{sentence}”是{custom_settings['class_list']}里的哪一个类别？"
        #注意这里的sentence左右两边的是中文引号，用来括起句子。
        classify = ollama.chat(model='deepseek-r1:8b',
                               messages =[*custom_settings['pre_history'],
                                          {'role':'user','content':sentence_with_prompt}]
                               )
        '''
        #应对非法类别的方法
        
        text_label = [
            "新闻报道",
            "\n新闻报道\n",
            "这是新闻报道，因为它描述了央行降息。",
            "财经新闻"
        ]
        #处理非法类别
        raw_output = text_label[i]

        label = raw_output.strip()
        i += 1
        if label in custom_settings['class_list']:
            response.append({
                'text': sentence,
                'label': label,
                'status':'成功',
                'raw_output':raw_output
            })
        else:
            response.append({
                'text': sentence,
                'label': None,
                'status':'待人工确认',
                'raw_output':raw_output
            })'''
        response.append({
            'text':sentence,
            'label':classify['message']['content']
        })

        #print(response)
        #print(custom_settings['pre_history'])
        #print('-'*40)
    return response

if __name__ == '__main__':
    sentences = [
        "今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。",
        "本公司宣布成功收购一家在创新科技领域领先的公司，这一战略性收购将有助于公司拓展技术能力和加速产品研发。",
        "公司资产负债表显示，公司偿债能力强劲，现金流充足，为未来投资和扩张提供了坚实的财务基础。",
        "最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长，投资者应该关注这一领域的投资机会",
    ]

    custom_settings = init_prompts()
    results = inference(
        sentences=sentences,
        custom_settings=custom_settings
    )
    pprint(results,width=120,sort_dicts=False)

    # print("结果数量：", len(results))
    # print("第一条结果的类型：", type(results[0]))
    # print("第一条原文：", results[0]["text"])
    # print("第一条分类：", results[0]["label"])