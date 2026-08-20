#1.定义一个类
class Person():
    pass #空语句，保证函数体的完整性
#2.创建对象
p1 = Person()

#对象名后添加属性
p1.name = '张三'
p1.age = 20
p1.address = '北京海淀'
#3.访问对象属性
print(p1.name) #输出：张三
print(p1.age) #输出：20
print(p1.address) #输出：北京海淀


