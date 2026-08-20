'''
    匿名函数：
        语法：
            lambda参数列表函数体

        要求：
            函数体只能有一行代码。不能换行
'''

#使用def 定义求和函数

def get_sum():
    a = 10
    b = 20
    return a+b

result = get_sum()
print(f'求和后的结果为{result}')

#使用lambda 定义求和函数
#使用无参数的匿名函数
result2 = lambda :20+20
print(f'求和后的结果为{result2()}')

#使用有参数的匿名函数
result3 = lambda a,b: a + b
#打印函数的内存地址
print(f'求和后的结果为{result3}')
print(f'求和后的结果为{result3(30,20)}')

#使用有默认值的匿名函数

result4 = lambda a=10,b=20: a+b
print(f'求和后的结果{result4()}')
print(f'求和后的结果{result4(100,200)}')#修改了默认值

#使用包裹位置的匿名函数
result5 = lambda *args : sum(args)
print(f'求和的结果为：{result5(1,2,3,4,5)}')

#使用包裹关键词的lambda表达式
result6 = lambda **kwargs : sum(kwargs.values())
print(f'求和的结果为：{result6(a=1,b=2,c=3,d=4,e=5)}')


