#1.导入相关包
import streamlit as st
from my_utils import get_response

#2.主界面主标题
st.title("有礼智聊机器人")

#3.创建一个对话窗口
prompt = st.chat_input("请输入您的问题：")


#4.如果输入框有数据，则进行问答
if prompt :
    st.write(prompt)
    get_response(prompt)
    result = get_response(prompt)
    st.write(result)
