#定义类
class Car(object):
    def __init__ (self,brand,model,color):
        self.brand = brand
        self.model = model
        self.color = color

    #定义函数
    def run (self):
        print('i can run ')

#定义子类 燃油车
class GasolineCar(Car):
    pass

#定义子类 电车
class ElectricCar(Car):
     def __init__(self,brand,model,color,battery):
         super().__init__(brand,model,color)
         self.battery = battery

#创建对象
c1 = GasolineCar('ford','mustang','red')
print(c1.brand)
print(c1.model)
print(c1.color)
c1.run()
print('-'*40)
c2 = ElectricCar('ford','mustang','blue','100kw/h')
print(c2.brand)
print(c2.model)
print(c2.color)
print(c2.battery)
c2.run()

