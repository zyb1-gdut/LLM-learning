
import yaml
import dashscope
def read_conf(path='conf_ali.yml'):
    with open(path,'r',encoding='utf-8') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    return conf

conf = read_conf()

from dashscope import Assistants, Threads, Messages, Runs

dashscope.api_key = conf['API_KEY']

import json
import sys
from http import HTTPStatus

import dashscope


def create_assistant():
    # create assistant with information
    assistant = dashscope.Assistants.create(
    # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model='qwen-max',
        name='smart helper',
        description='A tool helper.',
        instructions='You are a helpful assistant. When asked a question, use tools wherever possible.',  # noqa E501
        tools=[{
            'type': 'text_to_image'
        }],
    )

    return assistant


def verify_status_code(res):
    if res.status_code != HTTPStatus.OK:
        sys.exit(res.status_code)


if __name__ == '__main__':
    # create assistant
    assistant = create_assistant()
    print(assistant)
    verify_status_code(assistant)

    # create a thread.
    thread = dashscope.Threads.create()
    print(thread)
    verify_status_code(thread)

    # create a message.
    message = dashscope.Messages.create(thread.id, content='让鳄鱼在草地上跑步，我要真实图像，不要动漫场景')
    print(message)
    verify_status_code(message)

    # create a new run to run message
    message_run = dashscope.Runs.create(thread.id, assistant_id=assistant.id)
    print(message_run)
    verify_status_code(message_run)

    # get run statue
    run = dashscope.Runs.get(message_run.id, thread_id=thread.id)
    print(run)
    verify_status_code(run)
    # print run status, to verify run is completed.
    print(run.status)

    # wait for run completed or requires_action
    run = dashscope.Runs.wait(run.id, thread_id=thread.id)
    print(run)

    run = dashscope.Runs.get(run.id, thread_id=thread.id)
    print(run)
    verify_status_code(run)

    run_steps = dashscope.Steps.list(run.id, thread_id=thread.id)

    print(run_steps)
    verify_status_code(run_steps)

    # get the thread messages.
    msgs = dashscope.Messages.list(thread.id)
    print(msgs)
    print(json.dumps(msgs, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False))