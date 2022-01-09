# -*- encoding: utf-8 -*-
"""
@File    : pull_push_nlm2.py
@Time    : 11/29/2021 5:10 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
# -*- encoding: utf-8 -*-
"""
@File    : pull_push_nlm.py
@Time    : 11/22/2021 3:16 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import os
import time

import cv2
import numpy as np
import math


def upfuse():
    pass


"""
首先输入应该是图片的路径，搜索窗口的大小，相似窗口的大小，高斯平滑参数

读取图片，获取图片的shape
直接根据原来的这个shape/2 来得到新的shape


"""


def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1), np.float32)
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))

    return kernel / kernel.sum()


def downfuse_in_one_layer(img_arr, patch_win=1, sigma=10):
    height, width, depth = img_arr.shape
    new_height, new_width = height // 2, width // 2
    new_img_arr = np.zeros((new_height, new_width, depth))
    per_pixel_weight = np.zeros_like(new_img_arr)

    pad_length = patch_win
    one, two, three = cv2.split(img_arr)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])
    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma

    for i in range(new_height):
        for j in range(new_width):
            for k in range(depth):
                lt_i = i * 2 + patch_win
                lt_j = j * 2 + patch_win
                lt_k = k

                W1 = src_padding[lt_i - patch_win:lt_i + patch_win + 1,
                     lt_j - patch_win:lt_j + patch_win + 1, lt_k]  # 领域窗口W1
                w_max = 0
                aver = 0
                weight_sum = 0
                reliability_square_sum = 0

                rt_i = lt_i
                rt_j = lt_j + 1
                rt_k = lt_k
                W2 = src_padding[rt_i - patch_win:rt_i + patch_win + 1,
                     rt_j - patch_win:rt_j + patch_win + 1, rt_k]  # 搜索区域内的相似窗口W2

                lb_i = lt_i + 1
                lb_j = lt_j
                lb_k = lt_k
                W3 = src_padding[lb_i - patch_win:lb_i + patch_win + 1,
                     lb_j - patch_win:lb_j + patch_win + 1, lb_k]

                rb_i = lt_i + 1
                rb_j = lt_j + 1
                rb_k = lt_k
                W4 = src_padding[rb_i - patch_win:rb_i + patch_win + 1,
                     rb_j - patch_win:rb_j + patch_win + 1, rb_k]

                dist2 = (kernel * (W1 - W2) * (W1 - W2)).sum()
                dist3 = (kernel * (W1 - W3) * (W1 - W3)).sum()
                dist4 = (kernel * (W1 - W4) * (W1 - W4)).sum()

                w2 = np.exp( -dist2 / square_sigma)
                w3 = np.exp( -dist3 / square_sigma)
                w4 = np.exp( -dist4 / square_sigma)

                w_max = max(max(w2, w3), w4)
                weight_sum = w2 + w3 + w4 + w_max

                aver += w2 * src_padding[rt_i, rt_j, rt_k]
                aver += w3 * src_padding[lb_i, lb_j, lb_k]
                aver += w4 * src_padding[rb_i, rb_j, rb_k]
                aver += w_max * src_padding[lt_i, lt_j, lt_k]

                w2_norm = w2 / weight_sum
                w3_norm = w3 / weight_sum
                w4_norm = w4 / weight_sum
                w_max_norm = w_max / weight_sum
                reliability_square_sum = math.pow(w2_norm, 2) + math.pow(w3_norm, 2) +\
                                        math.pow(w4_norm, 2) + math.pow(w_max_norm, 2)
                # print('-'*20)
                # print(w2_norm, " ", w3_norm, " ", w4_norm, " ", w_max_norm)
                # print(reliability_square_sum)

                new_img_arr[i, j, k] = aver / weight_sum
                per_pixel_weight[i, j, k] = float(1.0 / reliability_square_sum)
    return new_img_arr, per_pixel_weight


def downfuse_in_one_layer_later(img_arr, last_layer_pixel_weight, patch_win=1, sigma=10):
    height, width, depth = img_arr.shape
    new_height, new_width = height // 2, width // 2
    new_img_arr = np.zeros((new_height, new_width, depth))
    per_pixel_weight = np.zeros_like(new_img_arr)

    pad_length = patch_win
    one, two, three = cv2.split(img_arr)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])
    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma

    for i in range(new_height):
        for j in range(new_width):
            for k in range(depth):
                lt_i = i * 2 + patch_win
                lt_j = j * 2 + patch_win
                lt_k = k

                W1 = src_padding[lt_i - patch_win:lt_i + patch_win + 1,
                     lt_j - patch_win:lt_j + patch_win + 1, lt_k]  # 领域窗口W1
                w_max = 0
                aver = 0
                weight_sum = 0
                reliability_square_sum = 0

                rt_i = lt_i
                rt_j = lt_j + 1
                rt_k = lt_k
                W2 = src_padding[rt_i - patch_win:rt_i + patch_win + 1,
                     rt_j - patch_win:rt_j + patch_win + 1, rt_k]  # 搜索区域内的相似窗口W2

                lb_i = lt_i + 1
                lb_j = lt_j
                lb_k = lt_k
                W3 = src_padding[lb_i - patch_win:lb_i + patch_win + 1,
                     lb_j - patch_win:lb_j + patch_win + 1, lb_k]

                rb_i = lt_i + 1
                rb_j = lt_j + 1
                rb_k = lt_k
                W4 = src_padding[rb_i - patch_win:rb_i + patch_win + 1,
                     rb_j - patch_win:rb_j + patch_win + 1, rb_k]

                dist2 = (kernel * (W1 - W2) * (W1 - W2)).sum()
                dist3 = (kernel * (W1 - W3) * (W1 - W3)).sum()
                dist4 = (kernel * (W1 - W4) * (W1 - W4)).sum()

                w2 = np.exp( -dist2 / square_sigma) * last_layer_pixel_weight[i*2, j*2, k] \
                        * last_layer_pixel_weight[i*2, j*2+1, k]
                w3 = np.exp( -dist3 / square_sigma) * last_layer_pixel_weight[i*2, j*2, k] \
                        * last_layer_pixel_weight[i*2+1, j*2, k]
                w4 = np.exp( -dist4 / square_sigma) * last_layer_pixel_weight[i*2, j*2, k] \
                        * last_layer_pixel_weight[i*2+1, j*2+1, k]

                w_max = max(max(w2, w3), w4)
                weight_sum = w2 + w3 + w4 + w_max

                w2_norm = w2 / weight_sum
                w3_norm = w3 / weight_sum
                w4_norm = w4 / weight_sum
                w_max_norm = w_max / weight_sum
                reliability_square_sum = math.pow(w2_norm, 2) + math.pow(w3_norm, 2) + \
                                         math.pow(w4_norm, 2) + math.pow(w_max_norm, 2)

                aver += w2_norm * src_padding[rt_i, rt_j, rt_k]
                aver += w3_norm * src_padding[lb_i, lb_j, lb_k]
                aver += w4_norm * src_padding[rb_i, rb_j, rb_k]
                aver += w_max_norm * src_padding[lt_i, lt_j, lt_k]
                # print('-' * 20)
                # print(w2_norm, " ", w3_norm, " ", w4_norm, " ", w_max_norm)
                # print(reliability_square_sum)
                weight_sum = w2_norm + w3_norm + w4_norm + w_max_norm
                print('weight_sum is:', weight_sum)

                new_img_arr[i, j, k] = aver
                per_pixel_weight[i, j, k] = float(1.0 / reliability_square_sum)
    return new_img_arr, per_pixel_weight


def downfuse_in_multiple_layers(img_path, num_down_layers=1):
    img_arr = cv2.imread(img_path)
    img_name = os.path.basename(img_path).split('.')[0]
    cv2.imwrite(img_name + '_down' + '_0.png', img_arr)

    img_arr_by_layer = []
    per_pixel_weight_by_layer = []
    img_arr_by_layer.append(img_arr)

    patch_win = 1
    sigma = 10

    for downfuse_time in range(1):
        img_arr = img_arr_by_layer[-1]
        new_img_arr, per_pixel_weight = downfuse_in_one_layer(img_arr, patch_win, sigma)
        img_arr_by_layer.append(new_img_arr)
        per_pixel_weight_by_layer.append(per_pixel_weight)
        print(downfuse_time, ":", new_img_arr.shape)
        print("max:", np.max(per_pixel_weight))
        print("min:", np.min(per_pixel_weight))
        cv2.imwrite(img_name + '_down' + '_1.png', new_img_arr)

    for downfuse_time in range(num_down_layers - 1):
        img_arr = img_arr_by_layer[-1]
        last_layer_pixel_weight = per_pixel_weight_by_layer[-1]
        new_img_arr, per_pixel_weight = downfuse_in_one_layer_later(img_arr, last_layer_pixel_weight, patch_win, sigma)
        img_arr_by_layer.append(new_img_arr)
        per_pixel_weight_by_layer.append(per_pixel_weight)
        print(downfuse_time, ":", new_img_arr.shape)
        cv2.imwrite(img_name + '_down_' + str(downfuse_time + 2) + '.png', new_img_arr)
        print("max:", np.max(per_pixel_weight))
        print("min:", np.min(per_pixel_weight))

    return img_arr_by_layer, per_pixel_weight_by_layer


def upfuse_in_one_layer(cur_layer, last_layer, last_layer_weight, patch_win=1, sigma=10):
    pad_length = patch_win
    one, two, three = cv2.split(cur_layer)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])

    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma
    updated_cur_layer = np.zeros_like(cur_layer).astype('float32')

    p_threshold = 3
    for i in range(cur_layer.shape[0]):
        for j in range(cur_layer.shape[1]):
            for k in range(cur_layer.shape[2]):
                # print(f'{i}:{j}:{k}')
                aver = 0
                weight_sum = 0

                region_i = i // 2
                region_j = j // 2
                if region_i >= last_layer_weight.shape[0] or \
                    region_j >= last_layer_weight.shape[1]:
                    continue

                region_i = region_i * 2
                region_j = region_j * 2

                cur_i = i + patch_win
                cur_j = j + patch_win
                is_center = False
                cur_W = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                        cur_j - patch_win:cur_j + patch_win + 1, k]
                if i - 1 == region_i and j - 1 == region_j:
                    # print('case 1')
                    W1 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j - 1 - patch_win:cur_j - 1 + patch_win + 1, k]
                    W2 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j - patch_win:cur_j + patch_win + 1, k]
                    W3 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j - 1 - patch_win:cur_j - 1 + patch_win + 1, k]
                    v1 = cur_layer[ i - 1, j - 1, k]
                    v2 = cur_layer[ i - 1, j, k]
                    v3 = cur_layer[ i, j - 1, k]
                elif i - 1 == region_i:
                    # print('case 2')
                    W1 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j - patch_win: cur_j + patch_win + 1, k]
                    W2 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    W3 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    v1 = cur_layer[ i - 1, j, k]
                    v2 = cur_layer[ i - 1, j + 1, k]
                    v3 = cur_layer[ i, j + 1, k]
                elif j - 1 == region_j:
                    # print('case 3')
                    W1 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j - 1 - patch_win: cur_j - 1 + patch_win + 1, k]
                    W2 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j - 1 - patch_win: cur_j - 1 + patch_win + 1, k]
                    W3 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j - patch_win: cur_j + patch_win + 1, k]
                    v1 = cur_layer[i, j - 1, k]
                    v2 = cur_layer[i + 1, j - 1, k]
                    v3 = cur_layer[i + 1, j, k]
                else:
                    # print('case 4')
                    # print(f'{i}:{j}', "|", f'{region_i}:{region_j}')
                    W1 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    W2 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j - patch_win: cur_j + patch_win + 1, k]
                    W3 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    v1 = cur_layer[i, j + 1, k]
                    v2 = cur_layer[i + 1, j, k]
                    v3 = cur_layer[i + 1, j + 1, k]
                    is_center = True

                dist1 = ( kernel * (cur_W - W1) * (cur_W - W1)).sum()
                dist2 = ( kernel * (cur_W - W2) * (cur_W - W2)).sum()
                dist3 = ( kernel * (cur_W - W3) * (cur_W - W3)).sum()
                w1 = np.exp( -dist1 / square_sigma)
                w2 = np.exp( -dist2 / square_sigma)
                w3 = np.exp( -dist3 / square_sigma)
                w_max = max(max(w1, w2), w3)

                weight_sum += w1 + w2 + w3 + w_max
                aver += (w1 * v1 + w2 * v2 + w3 * v3 + w_max * cur_layer[i, j, k])

                coarser_w_max = -1
                value = -1
                print('----------------'*5)
                for m in range(last_layer_weight.shape[0]):
                    for n in range(last_layer_weight.shape[-1]):
                        coarser_W = src_padding[2 * m + patch_win - patch_win: 2 * m + patch_win + patch_win + 1,
                                    2 * n + patch_win - patch_win: 2 * n + patch_win + patch_win + 1, k]
                        coarser_v = last_layer[m, n, k]
                        dist_w = ( kernel * (cur_W - coarser_W) * (cur_W - coarser_W)).sum()
                        w = np.exp( -dist_w / square_sigma)
                        # print("weight:", last_layer_weight[m,n,k])

                        # w *= max( p_threshold - last_layer_weight[m, n, k], 0)
                        w *= max(last_layer_weight[m,n,k] - p_threshold, 0)
                        # w *= min( p_threshold - last_layer_weight[m,n,k], 0)

                        # if 2 * m == i and 2 * n == j:
                        #     is_center = True
                        #     value = coarser_v
                        #     continue
                        #
                        # if w > coarser_w_max:
                        #     coarser_w_max = w
                        weight_sum += w
                        aver += w * coarser_v

                # if is_center:
                #     weight_sum += coarser_w_max
                #     aver += coarser_w_max * value

                # weight_sum = w1 + w2 + w3 + w4 + w_max
                # aver = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w_max * cur_layer[i, j, k]
                # updated_cur_layer[i, j, k] = aver / weight_sum

                updated_cur_layer[i, j, k] = aver / weight_sum

    return updated_cur_layer


def upfuse_in_one_layer_2(cur_layer, last_layer, last_layer_weight, patch_win=1, sigma=10):
    pad_length = patch_win
    one, two, three = cv2.split(cur_layer)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])

    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma
    updated_cur_layer = np.zeros_like(cur_layer).astype('float32')

    p_threshold = 3
    for i in range(cur_layer.shape[0]):
        for j in range(cur_layer.shape[1]):
            for k in range(cur_layer.shape[2]):
                # print(f'{i}:{j}:{k}')
                aver = 0
                weight_sum = 0

                region_i = i // 2
                region_j = j // 2
                if region_i >= last_layer_weight.shape[0] or \
                    region_j >= last_layer_weight.shape[1]:
                    continue

                region_i = region_i * 2
                region_j = region_j * 2

                cur_i = i + patch_win
                cur_j = j + patch_win
                is_center = False
                cur_W = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                        cur_j - patch_win:cur_j + patch_win + 1, k]
                if i - 1 == region_i and j - 1 == region_j:
                    # print('case 1')
                    W1 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j - 1 - patch_win:cur_j - 1 + patch_win + 1, k]
                    W2 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j - patch_win:cur_j + patch_win + 1, k]
                    W3 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j - 1 - patch_win:cur_j - 1 + patch_win + 1, k]
                    v1 = cur_layer[ i - 1, j - 1, k]
                    v2 = cur_layer[ i - 1, j, k]
                    v3 = cur_layer[ i, j - 1, k]
                elif i - 1 == region_i:
                    # print('case 2')
                    W1 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j - patch_win: cur_j + patch_win + 1, k]
                    W2 = src_padding[ cur_i - 1 - patch_win: cur_i - 1 + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    W3 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    v1 = cur_layer[ i - 1, j, k]
                    v2 = cur_layer[ i - 1, j + 1, k]
                    v3 = cur_layer[ i, j + 1, k]
                elif j - 1 == region_j:
                    # print('case 3')
                    W1 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j - 1 - patch_win: cur_j - 1 + patch_win + 1, k]
                    W2 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j - 1 - patch_win: cur_j - 1 + patch_win + 1, k]
                    W3 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j - patch_win: cur_j + patch_win + 1, k]
                    v1 = cur_layer[i, j - 1, k]
                    v2 = cur_layer[i + 1, j - 1, k]
                    v3 = cur_layer[i + 1, j, k]
                else:
                    # print('case 4')
                    # print(f'{i}:{j}', "|", f'{region_i}:{region_j}')
                    W1 = src_padding[ cur_i - patch_win: cur_i + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    W2 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j - patch_win: cur_j + patch_win + 1, k]
                    W3 = src_padding[ cur_i + 1 - patch_win: cur_i + 1 + patch_win + 1,
                         cur_j + 1 - patch_win: cur_j + 1 + patch_win + 1, k]
                    v1 = cur_layer[i, j + 1, k]
                    v2 = cur_layer[i + 1, j, k]
                    v3 = cur_layer[i + 1, j + 1, k]
                    is_center = True

                dist1 = ( kernel * (cur_W - W1) * (cur_W - W1)).sum()
                dist2 = ( kernel * (cur_W - W2) * (cur_W - W2)).sum()
                dist3 = ( kernel * (cur_W - W3) * (cur_W - W3)).sum()
                w1 = np.exp( -dist1 / square_sigma)
                w2 = np.exp( -dist2 / square_sigma)
                w3 = np.exp( -dist3 / square_sigma)
                w_max = max(max(w1, w2), w3)

                weight_sum += w1 + w2 + w3 + w_max
                aver += (w1 * v1 + w2 * v2 + w3 * v3 + w_max * cur_layer[i, j, k])

                coarser_w_max = -1
                value = -1
                print('----------------'*5)

                # if is_center:
                #     weight_sum += coarser_w_max
                #     aver += coarser_w_max * value

                # weight_sum = w1 + w2 + w3 + w4 + w_max
                # aver = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w_max * cur_layer[i, j, k]
                # updated_cur_layer[i, j, k] = aver / weight_sum

                updated_cur_layer[i, j, k] = aver / weight_sum

    return updated_cur_layer


def upfuse_in_multiple_layers(img_arr_by_layer, per_pixel_weight_by_layer, img_name, num_down_layers=3):
    bottom_layer = img_arr_by_layer[-1]
    # print('bottom layer shape:', bottom_layer.shape)
    layer_range = num_down_layers
    # layer_range = 2
    for upfuse_time in range(layer_range):
        cur_layer = img_arr_by_layer[- 2 - upfuse_time]
        last_layer = img_arr_by_layer[- 1 - upfuse_time]
        last_layer_weight = per_pixel_weight_by_layer[- 1 - upfuse_time]
        updated_cur_layer = upfuse_in_one_layer(cur_layer, last_layer, last_layer_weight)
        img_arr_by_layer[- 2 - upfuse_time] = updated_cur_layer
        # print("updated_cur_layer.shape:", updated_cur_layer.shape)
        cv2.imwrite(img_name + '_up_' + str((num_down_layers - 1 - upfuse_time)) + '.png', updated_cur_layer)


if __name__ == '__main__':

    # img_path = 'base.png'
    # img_path = 'banana7.png'
    img_path = 'belly.png'
    img_name = os.path.basename(img_path).split('.')[0]
    num_layers = 5

    start_t = time.time()
    img_arr_by_layer, per_pixel_weight_by_layer = downfuse_in_multiple_layers(img_path, num_down_layers=num_layers)
    print(type(img_arr_by_layer))
    print(type(per_pixel_weight_by_layer))

    for i in range(len(img_arr_by_layer)):
        np.save('img_arr_by_layer_' + str(i), img_arr_by_layer[i])
    for i in range(len(per_pixel_weight_by_layer)):
        np.save('per_pixel_weight_by_layer_' + str(i+1), per_pixel_weight_by_layer[i])

    end_t = time.time()
    print('down fuse total time:', end_t - start_t)

    img_arr_by_layer_loaded = [None] * (num_layers + 1)
    per_pixel_weight_by_layer_loaded = [None] * num_layers
    for i in range(num_layers + 1):
        img_arr_by_layer_loaded[i] = np.load('img_arr_by_layer_' + str(i) + '.npy')
    for j in range(num_layers):
        print('+' * 100)
        per_pixel_weight_by_layer_loaded[j] = np.load('per_pixel_weight_by_layer_' + str(j+1) + '.npy')
        print(np.min(per_pixel_weight_by_layer_loaded[j]))
        print(np.max(per_pixel_weight_by_layer_loaded[j]))

    upfuse_in_multiple_layers(img_arr_by_layer_loaded, per_pixel_weight_by_layer_loaded,
                              img_name, num_down_layers=num_layers)

    # upfuse_in_multiple_layers(img_arr_by_layer, per_pixel_weight_by_layer,
    #                           img_name, num_down_layers=5)



