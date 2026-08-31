import jieba  # 导入jieba中文分词库

if __name__ == '__main__':  # 程序入口判断，确保只在直接运行脚本时执行
    # 定义待分词的中文文本内容
    content = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"

    # 掌握
    # 使用jieba的lcut方法进行精确模式分词，返回结果是词汇列表
    words = jieba.lcut(content)  # 切分得到的结果是单词组成的List列表
    # 打印分词结果和数据类型
    print(words, type(words))

    # 了解
    # 使用jieba的lcut_for_search方法进行搜索引擎模式分词，适合用于搜索引擎建立索引
    words_search = jieba.lcut_for_search(content)
    # 打印搜索引擎模式分词结果和数据类型
    print(words_search, type(words_search))
