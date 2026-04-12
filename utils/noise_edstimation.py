#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-07 14:36:55


import numpy as np
from cv2 import imread
from skimage import img_as_float
import time

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)


if __name__ == '__main__':
    im = imread('/mnt/sunhao/gt.png')
    img_float = img_as_float(im)
    noise_level = noise_estimate(img_float)

    method = 'gaussian'
    # 根`据不同方法进行去噪
    if method == 'gaussian':
        # 高斯滤波，sigma根据估计的噪声水平调整
        sigma = noise_level * 2
        denoised = gaussian_filter(img_float, sigma=sigma)
    
    elif method == 'nlm':
        # Non-local means去噪
        h = noise_level * 0.8
        if img_float.ndim == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_as_ubyte(img_float),
                None,
                h=int(h*255),
                hColor=int(h*255),
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                img_as_ubyte(img_float),
                None,
                h=int(h*255),
                templateWindowSize=7,
                searchWindowSize=21
            )
        denoised = img_as_float(denoised)
    
    elif method == 'bilateral':
        # 双边滤波
        sigma_color = noise_level * 255
        sigma_space = noise_level * 50
        denoised = cv2.bilateralFilter(
            img_as_ubyte(img_float),
            d=5,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )
        denoised = img_as_float(denoised)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")