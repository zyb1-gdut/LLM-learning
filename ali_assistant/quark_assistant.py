import json
import sys
from http import HTTPStatus
import yaml
import dashscope

def read_conf(path='conf_ali.yml'):
    with open(path,'r',encoding='utf-8') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    return conf

conf = read_conf()

from dashscope import Assistants, Threads, Messages, Runs

dashscope.api_key = conf['API_KEY']
def create_assistant():
    # create assistant with information
    assistant = dashscope.Assistants.create(
    # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model='qwen-max',
        name='水果店财务助手',
        description='水果店财务助手，用在水果销售过程中计算营业额',
        instructions='你是一个理财能手，根据每类水果的售出量以及单价，统计其成本和收入，计算出总利润',  # noqa E501
        tools=[{
            'type': 'quark_search'
        }],
    )

    return assistant


def verify_status_code(res):
    if res.status_code != HTTPStatus.OK:
        print('Failed: ')
        print(res)
        sys.exit(res.status_code)


if __name__ == '__main__':
    # create assistant
    assistant = create_assistant()
    print(f'assistant:{assistant}')
    verify_status_code(assistant)
    print('*' * 100)
    # create a thread.
    thread = dashscope.Threads.create()
    # print(thread)
    print(f'thread:{thread}')
    verify_status_code(thread)
    print('*' * 100)
    # create a message.
    message = dashscope.Messages.create(thread.id,
                                        content='香蕉成本价2元一斤，售价为3元一斤。橘子成本价1.5元一斤，售价为2.5元一斤；苹果成本价3元一斤，售价为3.5元一斤；芒果成本价5元一斤，售价为6元一斤；葡萄成本价2元一斤，售价为4元一斤。我卖了2斤葡萄，3.5斤的香蕉，2斤苹果，计算下总成本和总收入，给出具体的计算过程')
    print(f'message:{message}')
    verify_status_code(message)
    print('*' * 100)
    # create a new run to run message
    stream_iterator = dashscope.Runs.create(thread.id,
                                        assistant_id=assistant.id,
                                        stream=True)
    for event, msg in stream_iterator:
        print(event)
        print(msg)
    verify_status_code(msg)

    # get run statue
    # run_9fa03862-aa36-4e1c-b2a7-9fdd91cb9a1d
    run = dashscope.Runs.get(msg.id, thread_id=thread.id)
    print(f'run:{run}')
    verify_status_code(run)
    # print run status, to verify run is completed.
    print(run.status)
    print('*' * 100)
    run_steps = dashscope.Steps.list(run.id, thread_id=thread.id)

    print(f'run_steps:{run_steps}')
    verify_status_code(run_steps)
    print('*' * 100)
    # get the thread messages.
    msgs = dashscope.Messages.list(thread.id)
    print(f'msgs:{msgs}')
    print('=' * 100)
    print(json.dumps(msgs, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False))