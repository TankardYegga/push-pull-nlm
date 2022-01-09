# -*- encoding: utf-8 -*-
"""
@File    : nlm.py
@Time    : 11/21/2021 4:36 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import numpy as np


# f为相似窗口的半径, t为搜索窗口的半径, h为高斯函数平滑参数(一般取为相似窗口的大小)
def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1), np.float32)
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))

    return kernel / kernel.sum()


def NLmeans_filter2(src, f, t, h):
    H, W, C = src.shape
    out = np.zeros((H, W, C), np.uint8)
    pad_length = f + t
    one, two, three = cv2.split(src)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])
    kernel = make_kernel(f)
    h2 = h * h

    for i in range(H):
        for j in range(W):
            for ch in range(C):
                i1 = i + f + t  # 将未填充图像上的点坐标映射到填充后图像上的对应坐标
                j1 = j + f + t
                W1 = src_padding[i1 - f:i1 + f + 1, j1 - f:j1 + f + 1, ch]  # 领域窗口W1
                w_max = 0
                aver = 0
                weight_sum = 0
                # 在搜索窗口进行搜索相似图像快
                for r in range(i1 - t, i1 + t + 1):
                    for c in range(j1 - t, j1 + t + 1):
                        if (r == i1) and (c == j1):  # 自身图像块领域先不处理
                            continue
                        else:
                            W2 = src_padding[r - f:r + f + 1, c - f:c + f + 1, ch]  # 搜索区域内的相似窗口W2
                            Dist2 = (kernel * (W1 - W2) * (W1 - W2)).sum()
                            w = np.exp(-Dist2 / h2)
                            if w > w_max:
                                w_max = w
                            weight_sum += w
                            aver += w * src_padding[r, c, ch]
                aver += w_max * src_padding[i1, j1, ch]  # 自身领域取最大的权重
                weight_sum += w_max
                out[i, j, ch] = aver / weight_sum

    return out


img = cv2.imread("lena.bmp")
out = NLmeans_filter2(img, 2, 5, 10)
cv2.imwrite("result2.bmp", out)

