#定义类
class Adder():
    def __init__(self,value=0):
        self.date = value
#创建默认函数
    def __call__(self,x):
        return self.date + x

#创建对象
A = Adder()
#无需调用函数，即可直接赋值
print(A(1)) # 1
print(A(2)) # 2

#对比
class Adder2():
    def __init__(self,value=0):
        self.date = value
#创建默认函数
    def add(self,x):
        return self.date + x

B = Adder2()
#print(B(1))#'Adder2' object is not callable
print(B.add(1))#1
print(B.add(2))#2
