#定义类
class Person():
    def __init__(self,name,age):#初始化函数，创建对象时候执行
        print('创建对象，开始初始化……')
        self.name = name
        self.age = age
        print('初始化完成。')


#创建对象
p1 = Person('张三',18)
print(p1.name)
print(p1.age)

p2 = Person('李四',20)
print(p2.name)
print(p2.age)

