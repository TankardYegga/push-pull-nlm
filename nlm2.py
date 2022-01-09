# -*- encoding: utf-8 -*-
"""
@File    : nlm2.py
@Time    : 11/21/2021 4:40 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import numpy as np
import math

img = cv2.imread('tree.png')
img = cv2.resize(img, (500, 500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray.astype('float32') / 255
size = img.shape  # h w c

r = 1
search_range = 10
h = 0.05

sr = search_range
h2 = h * h
dest = gray.copy()  # python中的变量更像指针，不可直接赋值
div = -1 * (2 * r + 1) * (2 * r + 1) * h2;

for y in range(r, size[0] - r):
    print(y)
    for x in range(r, size[1] - r):

        srcblock = gray[y - r:y + r + 1, x - r:x + r + 1]  # 是y,x

        # 限制了搜索范围，不然实在太慢了，做个试验而已
        y_start = max(y - search_range, r)
        x_start = max(x - search_range, r)
        y_end = min(y + search_range, size[0] - r - 1)
        x_end = min(x + search_range, size[1] - r - 1)

        w = np.zeros([y_end - y_start + 1, x_end - x_start + 1])

        for yi in range(y_start, y_end + 1):
            for xi in range(x_start, x_end + 1):
                # 运动估计简化计算？
                refblock = gray[yi - r:yi + r + 1, xi - r:xi + r + 1]

                delta = np.sum(np.square(srcblock - refblock))
                # print(delta)
                w[yi - y_start, xi - x_start] = math.exp(delta / div)
                # print(yi,xi)
                # time.sleep(1)

        dest[y, x] = np.sum(w * gray[y_start:y_end + 1, x_start:x_end + 1]) / np.sum(w)

cv2.imshow('result', dest)
cv2.imwrite('tree_res.png', dest)