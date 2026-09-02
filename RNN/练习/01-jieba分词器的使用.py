import jieba #导入jieba中文分词库
#jieba 是老式的常用的分词库，负责把没有空格的中文句子切成词语，让后续程序可以给每个词编号，再转换玮词向量输入到RN

if __name__ == '__main__':
    #定义待分词的中文文本内容
    content = "我要学习大语言模型，并且在大四的时候拿到字节的大模型部门的实习。"
    text = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"
    #掌握
    #使用jieba的lcut方法进行精确模式分词，返回结果是列表
    #words = jieba.lcut(content)
    words = jieba.lcut(text)
    #打印分词结果和数据类型
    print(words, type(words))

    #了解
    #使用jieba的lcut_for_search 方法进行搜索引擎模式分词，适用于搜索引擎建立索引
   # words_search = jieba.lcut_for_search(content)
    words_search = jieba.lcut_for_search(text)
    #打印
    print(words_search, type(words_search))

