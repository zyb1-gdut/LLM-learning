# 导入SparkLLM_Thread模块，用于与星火大模型进行交互
import SparkLLM_Thread
# 导入streamlit模块，用于创建网页界面
import streamlit as st
# 导入streamlit_chat模块中的message函数，用于在界面中显示消息
from streamlit_chat import message

# 页面提示语, 开场白
st.markdown("#### 您好, 我是多风格翻译官小星, 很荣幸为您服务。 :sunglasses:")
# 文本输入框
user_input = st.text_input("请输入您需要翻译的英文文本:", key='input')
# 设置一些风格按钮选项, 来设置不同的翻译风格
# horizontal: 水平排列选项
but = st.radio(
	"翻译风格:",
	('默认风格', '古文风格', '学术风格', '琼瑶风格','莎士比亚风格'), horizontal=True)

# 根据用户选择的翻译风格，设定相应的风格参数
if but == '默认风格':
	style = '。 '
elif but == '古文风格':
	style = ', 请按照古文风格进行翻译, 用古诗词的行文风格, 做到辞藻精炼, 可用典故。'
elif but == '学术风格':
	style = ', 请按照学术风格进行翻译, 保持严谨认真的风格。'
elif but == '琼瑶风格':
	style = ', 请按照琼瑶风格进行翻译, 意境优美, 充满诗情画意, 或多愁善感, 或心花怒放。'
elif but == '莎士比亚风格':
	style = ',请按照莎士比亚风格进行翻译，意境优美，带有浓厚的英伦腔。'
else:
	style = '。 '

# 用于判断模型生成内容是否存在, 不存在则创建列表
if 'generated' not in st.session_state:
	st.session_state['generated'] = []

# 用于判断用户输入内容是否存在, 不存在则创建列表
if 'past' not in st.session_state:
	st.session_state['past'] = []

# 当用户输入内容时，进行翻译处理
if user_input:
	# 组装prompt, 最终传入大模型的是text内容
	text = user_input + "\n请将上述英文内容翻译为中文" + style

	# 保存用户输入到列表, 用于后续页面展示
	st.session_state['past'].append(user_input)

	# 向星火模型发出请求, 其中appid, api_key, api_secret 获取地址: https://console.xfyun.cn/services/bm3
	output =	'''
	调用SparkLLM_Thread类的main方法发起AI对话请求
	
	该函数通过WebSocket连接讯飞星火大模型API，发送用户问题并获取AI回复结果。
	
	参数说明:
		uid (str): 用户唯一标识符，用于区分不同用户
		chat_id (str): 聊天会话ID，用于标识不同的对话场景
		appid (str): 应用ID，用于API鉴权认证
		api_key (str): API密钥，用于接口访问权限验证
		api_secret (str): API密钥对应的密钥，用于生成请求签名
		gpt_url (str): WebSocket接口地址，指向星火大模型v3.1版本的聊天接口
		question (list): 包含对话历史的消息列表，每个元素为包含role和content字段的字典
		
	返回值:
		AI模型的响应结果，具体格式取决于SparkLLM_Thread.main方法的实现
	'''
	output = SparkLLM_Thread.main(uid='itcast',
								  chat_id='itcast',
								  appid='b42adf8d',
								  api_key='9e5f130ec4d853664c96061a2da369d8',
								  api_secret='N2ZmNjM4MTExYWM1ZGRmM2JjYmRlZWE0',
								  gpt_url='wss://spark-api.xf-yun.com/v3.1/chat',
								  question=[{"role": "user", "content": text}])

	# 保存大模型输出到列表, 用于后续页面展示
	st.session_state['generated'].append(output)

# 在前端页面展示列表中的内容
if st.session_state['generated']:
	for i in range(len(st.session_state['generated']) - 1, -1, -1):
		message(st.session_state["generated"][i], key=str(i))
		message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
