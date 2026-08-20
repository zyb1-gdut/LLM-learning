class Cat(object):

    def eat(self):
        print(f'self的引用值：{self}')#对象是谁，引用值就与对象一样。self 是类的对象
        print('吃鱼……')

c1 = Cat()
print(f'c1的引用值：{c1}')
c1.eat()

c2 = Cat()
print(f'c2的引用值：{c2}')
c2.eat()
