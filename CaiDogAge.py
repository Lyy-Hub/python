#!/usr/bin/env python
#! -*- coding:utf-8 -*-
import random #导入随机数模块
Age=random.randrange(10)#随机生成一个小于10的整数（0-9，不包括负数），并赋值给Age
#print("我的狗的随机年龄为："+bytes(Age)+"岁")
for i in range(3):
    if i < 2:
        guess_number=int(input("请输入我的狗的年龄你猜:\n"))
        if guess_number > Age:
            print("你猜的年龄有点大，想小一些吧！\n")
        elif guess_number < Age:
            print("你猜的年龄有点小，想大一点！\n")
        else:
            print("Bingo, 你猜对了，祝贺你！\n")
            break
    else:
        guess_number=int(input("请输入我的狗的年龄你猜:\n"))
        if guess_number == Age:
            print("Bingo, 你收到号码了，祝贺你！\n")
        else:
            print("你只是运气不好，来试试吧，你能行！我的狗的实际年龄是 %d...\n"% Age)