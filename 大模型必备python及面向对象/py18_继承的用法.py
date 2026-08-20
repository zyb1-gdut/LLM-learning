#定义一个父类
class Animal(object):
    #定义函数
    def eat(self):
        print("i can eat")

    def sound(self):
        print("i can make a sound")
#定义一个猫类
class Cat(Animal):
    pass
#定义一个狗类
class Dog(Animal):
    pass
#创建对象
dog = Dog()
dog.eat()
dog.sound()
#创建对象
cat = Cat()
cat.eat()
cat.sound()
