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
from numba import jit

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


# @jit
def downfuse_in_one_layer(img_arr, search_win, patch_win=3, sigma=10):
    # img_arr = cv2.imread(img_path)
    height, width, depth = img_arr.shape
    new_height, new_width = height // 2, width // 2
    new_img_arr = np.zeros((new_height, new_width, depth))
    per_pixel_weight = np.zeros_like(new_img_arr)

    pad_length = search_win + patch_win
    one, two, three = cv2.split(img_arr)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])
    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma

    print(new_height)
    print(new_width)
    print(depth)
    for i in range(new_height):
        for j in range(new_width):
            for k in range(depth):
                original_i = i * 2
                original_j = j * 2
                original_k = k
                i1 = original_i + search_win + patch_win
                j1 = original_j + search_win + patch_win
                W1 = src_padding[i1 - patch_win:i1 + patch_win + 1,
                     j1 - patch_win:j1 + patch_win + 1, original_k]  # 领域窗口W1
                w_max = 0
                aver = 0
                weight_sum = 0
                reliability_square_sum = 0
                weights_list = []

                # 在搜索窗口进行搜索相似图像快
                for r in range(i1 - search_win, i1 + search_win + 1):
                    for c in range(j1 - search_win, j1 + search_win + 1):
                        if (r == i1) and (c == j1):  # 自身图像块领域先不处理
                            continue
                        else:
                            W2 = src_padding[r - patch_win:r + patch_win + 1,
                                 c - patch_win:c + patch_win + 1, original_k]  # 搜索区域内的相似窗口W2
                            Dist2 = (kernel * (W1 - W2) * (W1 - W2)).sum()
                            w = np.exp(-Dist2 / square_sigma)
                            if w > w_max:
                                w_max = w
                            weight_sum += w
                            weights_list.append(w)
                            aver += w * src_padding[r, c, original_k]
                aver += w_max * src_padding[i1, j1, original_k]  # 自身领域取最大的权重
                weight_sum += w_max
                weights_list.append(w_max)
                new_img_arr[i, j, k] = aver / weight_sum
                reliability_square_sum += math.pow(w_max, 2)
                sum_w_norm = 0
                for i in range(len(weights_list)):
                    # print('w val:', weights_list[i])
                    w_norm = weights_list[i] / weight_sum
                    sum_w_norm += w_norm
                    reliability_square_sum += math.pow(w_norm, 2)
                # print('score square sum:', reliability_square_sum)
                # print('score sum:', sum_w_norm)
                print('reliability square sum:', float(1.0 / reliability_square_sum))
                print("i is", i)
                print(per_pixel_weight.shape)
                per_pixel_weight[i, j, k] = float(1.0 / reliability_square_sum)
    print('---------FINISHED-----------------')
    return new_img_arr, per_pixel_weight


# @jit
def downfuse_in_multiple_layers(img_path, num_down_layers=2):
    img_arr = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    img_arr_by_layer = []
    per_pixel_weight_by_layer = []
    img_arr_by_layer.append(img_arr)

    search_win = 5
    patch_win = 3
    sigma = 10

    for downfuse_time in range(num_down_layers):
        img_arr = img_arr_by_layer[-1]
        new_img_arr, per_pixel_weight = downfuse_in_one_layer(img_arr, search_win, patch_win, sigma)
        img_arr_by_layer.append(new_img_arr)
        per_pixel_weight_by_layer.append(per_pixel_weight)
        print(downfuse_time, ":", new_img_arr.shape)

    print('*'*100)
    for i in range(num_down_layers+1):
        if os.path.exists(img_name + '_down_' + str(i) + '.png'):
            continue
        cv2.imwrite(img_name + '_down_' + str(i) + '.png', img_arr_by_layer[i])
        print('save')

    return img_arr_by_layer, per_pixel_weight_by_layer

# @jit
def upfuse_in_one_layer(cur_layer, last_layer, last_layer_weight, search_win=5, patch_win=3, sigma=10):
    pad_length = search_win + patch_win
    one, two, three = cv2.split(cur_layer)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])

    pad_length = patch_win
    one, two, three = cv2.split(last_layer)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype('float32')
    last_src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])

    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma
    updated_cur_layer = np.zeros_like(cur_layer).astype('float32')

    p_threshold = 100
    for i in range(cur_layer.shape[0]):
        for j in range(cur_layer.shape[1]):
            for k in range(cur_layer.shape[2]):
                padding_i = i + search_win + patch_win
                padding_j = j + search_win + patch_win
                padding_k = k
                cur_window = src_padding[padding_i - patch_win:padding_i + patch_win + 1,
                             padding_j - patch_win:padding_j + patch_win + 1, padding_k]
                weights_sum = 0
                w_max = -1
                weighted_pixel_val_sum = 0
                for r in range(padding_i - search_win, padding_i + search_win + 1):
                    for c in range(padding_j - search_win, padding_j + search_win + 1):
                        if r == padding_i and c == padding_j:
                            continue
                        else:
                            cmp_window = src_padding[r - patch_win:r + patch_win + 1,
                                         c - patch_win:c + patch_win + 1, padding_k]
                            patch_dist = (kernel * (cur_window - cmp_window) * (cur_window - cmp_window)).sum()
                            w = np.exp( - patch_dist / square_sigma)
                            weights_sum += w
                            if w > w_max:
                                w_max = w
                            weighted_pixel_val_sum += w * src_padding[r, c, padding_k]
                            # print('w is', w)
                            # print('weighted_pixel_val_sum is', weighted_pixel_val_sum)
                            # print('res is', w * src_padding[r, c, padding_k])
                weights_sum += w_max
                weighted_pixel_val_sum += w_max * src_padding[padding_i, padding_j, padding_k]
                #
                # print('val sum:', weighted_pixel_val_sum)
                # print('weight sum:', weights_sum)

                for m in range(last_layer.shape[0]):
                    for n in range(last_layer.shape[-1]):
                        reliability_score = last_layer_weight[m][n][padding_k]
                        w_scale_factor = max(reliability_score - p_threshold, 0)
                        # if reliability_score <= p_threshold:
                        #     w_scale_factor = 0
                        # else:
                        #     w_scale_factor = reliability_score
                        #
                        # if w_scale_factor == 0:
                        #     continue
                        padding_m = m * 2 + patch_win + search_win
                        padding_n = n * 2 + patch_win + search_win
                        cmp_window = src_padding[padding_m - patch_win:padding_m + patch_win + 1,
                                     padding_n - patch_win:padding_n + patch_win + 1, padding_k]
                        patch_dist = (kernel * (cur_window - cmp_window) * (cur_window - cmp_window)).sum()
                        w = np.exp( -patch_dist / square_sigma)
                        w = w * w_scale_factor
                        weights_sum += w
                        weighted_pixel_val_sum += (w * last_layer[m, n, padding_k])

                # print('val sum:', weighted_pixel_val_sum)
                # print('weight sum:', weights_sum)
                updated_cur_layer[i, j, k] = weighted_pixel_val_sum / float(weights_sum)

    return updated_cur_layer

# @jit
def upfuse_in_one_layer_2(cur_layer, last_layer, last_layer_weight, search_win=5, patch_win=3, sigma=10):
    pad_length = search_win + patch_win
    one, two, three = cv2.split(cur_layer)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])

    pad_length = search_win
    one, two, three = cv2.split(last_layer)
    src_padding_one = np.pad(one, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_two = np.pad(two, (pad_length, pad_length), mode='symmetric').astype('float32')
    src_padding_three = np.pad(three, (pad_length, pad_length), mode='symmetric').astype('float32')
    last_src_padding = cv2.merge([src_padding_one, src_padding_two, src_padding_three])

    kernel = make_kernel(patch_win)
    square_sigma = sigma * sigma
    updated_cur_layer = np.zeros_like(cur_layer).astype('float32')

    p_threshold = 3
    for i in range(cur_layer.shape[0]):
        for j in range(cur_layer.shape[1]):
            for k in range(cur_layer.shape[2]):
                padding_i = i + search_win + patch_win
                padding_j = j + search_win + patch_win
                padding_k = k
                cur_window = src_padding[padding_i - patch_win:padding_i + patch_win + 1,
                             padding_j - patch_win:padding_j + patch_win + 1, padding_k]
                weights_sum = 0
                w_max = -1
                weighted_pixel_val_sum = 0
                for r in range(padding_i - search_win, padding_i + search_win + 1):
                    for c in range(padding_j - search_win, padding_j + search_win + 1):
                        if r == padding_i and c == padding_j:
                            continue
                        else:
                            cmp_window = src_padding[r - patch_win:r + patch_win + 1,
                                         c - patch_win:c + patch_win + 1, padding_k]
                            patch_dist = (kernel * (cur_window - cmp_window) * (cur_window - cmp_window)).sum()
                            w = np.exp( - patch_dist / square_sigma)
                            weights_sum += w
                            if w > w_max:
                                w_max = w
                            weighted_pixel_val_sum += w * src_padding[r, c, padding_k]
                            # print('w is', w)
                            # print('weighted_pixel_val_sum is', weighted_pixel_val_sum)
                            # print('res is', w * src_padding[r, c, padding_k])
                weights_sum += w_max
                weighted_pixel_val_sum += w_max * src_padding[padding_i, padding_j, padding_k]
                #
                # print('val sum:', weighted_pixel_val_sum)
                # print('weight sum:', weights_sum)

                for m in range( i//2 - search_win , i//2 + search_win + 1):
                    for n in range( j//2 - search_win, j//2 + search_win + 1):
                        reliability_score = last_layer_weight[m][n][padding_k]
                        w_scale_factor = max(reliability_score - p_threshold, 0)
                        # if reliability_score <= p_threshold:
                        #     w_scale_factor = 0
                        # else:
                        #     w_scale_factor = reliability_score
                        #
                        # if w_scale_factor == 0:
                        #     continue
                        print('m', m)
                        print('n', n)
                        padding_m = m * 2 + patch_win + search_win
                        padding_n = n * 2 + patch_win + search_win
                        print('padding m', padding_m)
                        print('padding n', padding_n)
                        cmp_window = src_padding[padding_m - patch_win:padding_m + patch_win + 1,
                                     padding_n - patch_win:padding_n + patch_win + 1, padding_k]
                        assert cmp_window.shape == cur_window.shape
                        patch_dist = (kernel * (cur_window - cmp_window) * (cur_window - cmp_window)).sum()
                        w = np.exp( -patch_dist / square_sigma)
                        w = w * w_scale_factor
                        weights_sum += w
                        weighted_pixel_val_sum += (w * last_src_padding[m + patch_win, n + patch_win, padding_k])

                # print('val sum:', weighted_pixel_val_sum)
                # print('weight sum:', weights_sum)
                updated_cur_layer[i, j, k] = weighted_pixel_val_sum / float(weights_sum)

    return updated_cur_layer

# @jit
def upfuse_in_multiple_layers(img_arr_by_layer, per_pixel_weight_by_layer, img_name, num_down_layers=2):
    bottom_layer = img_arr_by_layer[-1]
    print('bottom layer shape:', bottom_layer.shape)
    for upfuse_time in range(num_down_layers):
        if os.path.exists(img_name + '_up_' + str(upfuse_time) + '.png'):
            continue
        cur_layer = img_arr_by_layer[- 2 - upfuse_time]
        last_layer = img_arr_by_layer[- 1 - upfuse_time]
        last_layer_weight = per_pixel_weight_by_layer[- 1 - upfuse_time]
        updated_cur_layer = upfuse_in_one_layer(cur_layer, last_layer, last_layer_weight)
        img_arr_by_layer[- 2 - upfuse_time] = updated_cur_layer
        print("updated_cur_layer.shape:", updated_cur_layer.shape)
        cv2.imwrite(img_name + '_up_' + str((num_down_layers - 1 - upfuse_time)) + '.png', updated_cur_layer)


if __name__ == '__main__':

    # img_path = 'base.png'
    # img_path = 'banana6.png'
    # img_path = 'banana6.png_up_0.png'
    # img_path = 'banana6.png_up_0.png_up_0.png'
    img_path = 'base3.png'
    img_name = os.path.basename(img_path).split('.')[0]
    print(img_name)

    start_t = time.time()
    img_arr_by_layer, per_pixel_weight_by_layer = downfuse_in_multiple_layers(img_path, num_down_layers=3)
    # end_t = time.time()
    # print('down fuse total time:', end_t - start_t)
    # upfuse_in_multiple_layers(img_arr_by_layer, per_pixel_weight_by_layer, img_name, num_down_layers=2)

