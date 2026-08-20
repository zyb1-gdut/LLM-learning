#安装模块包，pip install streamlit
import streamlit as st
import pandas as pd
from docutils.nodes import paragraph

#设置标题
st.title('有礼智聊机器人')

#段落 write
st.write('欢迎来到有礼智聊机器人')

#markdown格式，#号越多，标题越小
"# 1级标题"
"## 2级标题"
"### 3级标题"
"#### 4级标题"
"##### 5级标题"
"###### 6级标题"

#渲染照片，st.image()
st.image('qxq.jpg',width=500)

#静态表格使用 `st.table()` 渲染，出来的效果就是 `HTML` 的 `<table>`。
#`st.table()` 支持传入字典、`pandas.DataFrame` 等数据。
st.write('dict字典形式的静态表格')
st.table(data={
    'name': ['张三', '李四', '王五'],
    'age': [18, 20, 22],
    'gender': ['男', '女', '男']
})

# 下面我们没学过pandas 可以先了解能这样用.不用练习!!!
st.write('pandas中dataframe形式的静态表格')

df = pd.DataFrame(
    {
        'name': ['张三', '李四', '王五'],
        'age': [18, 20, 22],
        'gender': ['男', '女', '男']
    }
)
st.dataframe(df)

#分割线
st.divider()

#输入框
name = st.text_input("请输入您的名字：")
psw = st.text_input("请输入密码：",type="password")
if name:
    st.write(f'你好，{name}')

age = st.number_input("请输入您的年龄：",step = 1, value=0,min_value = 0,max_value = 200)
st.write(f"您的年龄是{age}岁。")

#输入多行内容
paragraph = st.text_area("多行内容：")

#Chat Element
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")