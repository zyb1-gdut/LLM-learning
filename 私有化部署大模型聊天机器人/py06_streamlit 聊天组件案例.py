#导入streamlit库。Streamlit是一个用于快速组件数据应用的Python库
import streamlit as st

#使用st.chat_input()创建一个聊天的输入框，提示用户输入问题
prompt = st.chat_input("请输入您的问题：")

st.write(prompt)

#使用st.chat_message 创建一个用户消息容器，用于显示用户的消息
#'user'表示这是用户发送的消息
with st.chat_message('user'):
    #在用户消息容器显示文本
    st.write('HELLO')

#使用st.chat_message 创建一个消息容器，用于回复消息
message = st.chat_message('assistant')
#在消息容器中显示文本“HELLO HUMAN”，模拟助手的回复
message.write('HELLO HUMAN')