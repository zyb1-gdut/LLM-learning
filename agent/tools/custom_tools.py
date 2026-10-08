# 导包
from crewai.tools import tool
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib


# TODO 1 此处添加自定义工具:功能是保存文章内容到本地磁盘文件上
@tool("将文本写入txt文档中")
def store_poesy_to_txt(content: str) -> str:
    """
    将编辑后的书信文本内容自动保存到txt文档中
    """
    try:
        filename = "情书.txt"
        # 将文本写入txt文档中,如果文件不存在w模式下会自动创建
        with open(filename, 'w', encoding="utf-8") as file:
            file.write(content)

        # 返回结果展示：文件已经写入
        return f"内容已经写入到{filename}文件"
    except Exception as e:
        return f"文件写入失败,原因{e}"


# TODO 2 此处添加自定义工具:功能发送书信到指定邮箱
@tool("将书信发送到指定邮箱")
def send_email():
    """
    将编辑后的书信文本内容自动发送到指定邮箱
    """
    # 邮箱配置
    sender_email = "15830091676@163.com"
    sender_password = "BRrvcF42nVw88bnw"
    receiver_email = "15830091676@163.com"
    # 设置标题
    subject = "书信❤"
    # 获取邮件内容
    with open("情书.txt", encoding="utf-8") as f:
        body = f.read()
    # 设置邮件格式 MIMEMultipart代表构建复杂的邮件格式
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        # 连接smtp服务器并发送邮件
        # 创建一个安全的ssl连接
        smtp_server = smtplib.SMTP_SSL('smtp.163.com', 465)
        # 建立连接
        smtp_server.connect('smtp.163.com', 465)
        # 登录邮箱(传入发送人的邮箱名和授权码)
        smtp_server.login(sender_email, sender_password)
        # 发送邮件
        smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
        print("邮件发送成功!!!")
    except Exception as e:
        print("邮件发送失败!!!", e)
    finally:
        smtp_server.quit()
