'''
类的定义
1： class 类名():
    属性
    功能(方法)
2:  class 类名:
    属性
    功能
3:  class 类名(object):
    属性
    功能

'''

#猫的定义
class Cat():
    #属性
    Name = 'tom'

    #函数
    def eat(self):
        print('猫吃鱼……')

    def sleep(self):
        print('猫睡觉……')

#创建一个对象
#语法结构：
#对象 = 类名()
tom = Cat()
tom.eat()

#访问属性的时候，直接 “对象名.属性”
#访问函数的时候，'对象名.函数()'，若使用“对象名.函数”-->会输出函数的内存地址
print(tom.Name)


Jerry =  Cat()
Jerry.sleep()
