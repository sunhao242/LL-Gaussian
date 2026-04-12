#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.visualize_utils import minmax_normalize
from audtorch.metrics.functional import pearsonr

def l1_loss(network_output, gt):
    return (network_output - gt)

def l1_loss_mask(network_output, gt, mask):
    return torch.abs((network_output - gt) * mask).mean()

def l1_plus_loss(network_output , gt, phi=1e-3,alpha=1):
    # illumination_image = illumination_image.detach()
    weight1 = 1 / ( alpha * network_output + phi) 
    weight = weight1.detach()
    # weight = torch.exp(-network_output * 2.65).detach()
    loss = weight * (network_output - gt) 
    return loss
    # weight = 1/(1+torch.exp(-0.4*(1/illumination_image-1/0.2)))
    # weight = weight.detach()
# def l1_plus_loss(network_output, gt, delta=0.05, phi=5e-3):

#     weight1 = 1 / (network_output + phi)
#     weight1 = weight1.detach()
#     weight2 = weight1 ** 2
    
#     # 计算差异
#     diff = network_output - gt
#     abs_diff = torch.abs(diff)
    
#     # Huber 损失
#     huber_loss = torch.where(abs_diff < delta, 
#                              0.5 * diff**2 * weight2, 
#                              delta * (abs_diff - 0.5 * delta) * weight2)
    
#     return huber_loss

def l2_plus_loss(network_output , gt):
    phi = 5e-3
    weight1 = 1/(network_output + phi)
    weight1 = weight1.detach()
    return (((network_output - gt) * weight1) ** 2).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def L_Smooth(illumination_image, image, kernel_size=9):
    image = image.detach()
    # 将图像转换为灰度图像
    gray_image = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
    gray_image = gray_image.unsqueeze(0)  # 添加通道维度
    # 创建高斯滤波器窗口
    window_size = kernel_size
    channel = gray_image.size(-3)
    window = create_window(window_size, channel)
    
    if gray_image.is_cuda:
        window = window.cuda(image.get_device())
    window = window.type_as(image)
    
    # 对图像进行高斯滤波
    gray_image = F.conv2d(gray_image, window, padding=window_size//2, groups=channel)
    weight_x = torch.abs(gray_image[:,:-1,:-1] - gray_image[:,1:,:-1])
    weight_y = torch.abs(gray_image[:,:-1,:-1] - gray_image[:,:-1,1:])
    grad_x = torch.abs((illumination_image[:,:-1,:-1] - illumination_image[:,1:,:-1]) ) / (weight_x + 1e-6)
    # 计算y方向的梯度
    grad_y = torch.abs((illumination_image[:,:-1,:-1] - illumination_image[:,:-1,1:]) ) / (weight_y + 1e-6)
    # 计算梯度幅度
    grad_image = grad_x ** 2 + grad_y ** 2 + 1e-10

    return torch.sqrt(grad_image).mean()

# def L_Smooth(illumination_image, image):
#     image = image.detach().unsqueeze(0)
#     illumination_image = illumination_image.unsqueeze(0)
#     # 转换为灰度图 (保持4D张量 BCHW)
#     gray_image = 0.299 * image[:,0] + 0.587 * image[:,1] + 0.114 * image[:,2]
#     gray_image = gray_image.unsqueeze(0)  # [B,1,H,W]
    
#     # 高斯滤波
#     window_size = 9
#     window = create_window(window_size, 1).to(image.device)
#     gray_blur = F.conv2d(gray_image, window, padding=window_size//2, groups=1)
#     # 计算图像梯度权重 (使用sobel算子更合理)
#     weight_x = F.conv2d(gray_blur, torch.Tensor([[-1,0,1]]).view(1,1,1,3).to(image.device), padding=(0,1))
#     weight_y = F.conv2d(gray_blur, torch.Tensor([[-1],[0],[1]]).view(1,1,3,1).to(image.device), padding=(1,0))
#     # 计算光照图梯度
    
#     grad_x = (illumination_image[:,:,:-1,:-1] - illumination_image[:,:,:-1,1:]) / (weight_x[:,:,:-1,:-1] + 1e-3)  # 宽方向
#     grad_y = (illumination_image[:,:,:-1,:-1] - illumination_image[:,:,1:,:-1]) / (weight_y[:,:,:-1,:-1] + 1e-3)  # 高方向
    
#     # 加权梯度损失
#     grad_loss = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).mean()
    
#     return grad_loss

def Ll1_Residual(residual_image, image, clear_image, threadhold=0.05):
    image = image.detach()
    # 将图像转换为灰度图像
    gray_image = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
    gray_image = gray_image.unsqueeze(0)  # 添加通道维度
    # 创建高斯滤波器窗口
    window_size = 9
    channel = gray_image.size(-3)
    window = create_window(window_size, channel)
    
    if gray_image.is_cuda:
        window = window.cuda(image.get_device())
    window = window.type_as(image)
    
    # 对图像进行高斯滤波
    gray_image_filtered = F.conv2d(gray_image, window, padding=window_size//2, groups=channel)
    weight_x = torch.abs(gray_image[:,:-1,:-1] - gray_image[:,1:,:-1])
    weight_y = torch.abs(gray_image[:,:-1,:-1] - gray_image[:,:-1,1:])
    weight = torch.sqrt(weight_x ** 2 + weight_y ** 2 + 1e-10)
    return torch.abs(torch.clamp((residual_image + clear_image - image), -threadhold, threadhold) * weight).mean()


def L_Reflectance_Smooth(reflectance_image, illumination_image):
    illumination_image = illumination_image.detach()
    grad_x = torch.abs((reflectance_image[:,:-1,:-1] - reflectance_image[:,1:,:-1]) ) 
    # 计算y方向的梯度
    grad_y = torch.abs((reflectance_image[:,:-1,:-1] - reflectance_image[:,:-1,1:]) ) 
    weight = minmax_normalize(1/ (illumination_image[:,:-1,:-1] * (grad_x * grad_y) ** 2 + 1e-10))
    weight = weight.detach()
    # 计算梯度幅度
    grad_image = weight * torch.sqrt((grad_x ** 2 + grad_y ** 2) + 1e-10)

    return grad_image.mean()

def L_Feat_Smooth(feature_image, image, mask, depth_image):
    image = image.detach()
    # 将图像转换为灰度图像
    gray_image = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
    gray_image = gray_image.unsqueeze(0)  # 添加通道维度
    # 创建高斯滤波器窗口
    window_size = 11
    channel = gray_image.size(-3)
    window = create_window(window_size, channel)
    
    if gray_image.is_cuda:
        window = window.cuda(image.get_device())
    window = window.type_as(image)
    
    # 对图像进行高斯滤波
    gray_image = F.conv2d(gray_image, window, padding=window_size//2, groups=channel)

    # 使用pad操作来保持尺寸一致
    padded_gray = F.pad(gray_image, (0, 1, 0, 1), mode='replicate')
    padded_gray = padded_gray.repeat(feature_image.shape[0], 1, 1)
    padded_feature = F.pad(feature_image, (0, 1, 0, 1), mode='replicate')

    weight_x = torch.abs(padded_gray[:,:-1,:-1] - padded_gray[:,1:,:-1])
    weight_y = torch.abs(padded_gray[:,:-1,:-1] - padded_gray[:,:-1,1:])

    grad_x = torch.abs((padded_illumination[:,:-1,:-1] - padded_illumination[:,1:,:-1]) ) / (weight_x + 1e-8)
    # 计算y方向的梯度
    grad_y = torch.abs((padded_illumination[:,:-1,:-1] - padded_illumination[:,:-1,1:]) ) / (weight_y + 1e-8)
    # 计算梯度幅度
    grad_image = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    weight = (depth_image ** 2).repeat(grad_image.shape[0],1,1) * mask
    weight = weight.detach()
    return torch.mean(grad_image * weight)

def L_Depth_Smooth(depth_src,depth_target):
    
    img_grad_x = torch.abs(depth_target[:,:-1,:-1] - depth_target[:,1:,:-1])
    img_grad_y = torch.abs(depth_target[:,:-1,:-1] - depth_target[:,:-1,1:])
    weight_x = torch.exp(-img_grad_x.mean(1).unsqueeze(1))
    weight_y = torch.exp(-img_grad_y.mean(1).unsqueeze(1))
    grad_x = torch.abs((depth_src[:,:-1,:-1] - depth_src[:,1:,:-1]) )
    # 计算y方向的梯度
    grad_y = torch.abs((depth_src[:,:-1,:-1] - depth_src[:,:-1,1:]) ) 
    # 计算梯度幅度
    loss = ((grad_x * weight_x).sum() +
            (grad_y * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss


def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def L_Illu(gt_image, illumination_image, phi=0.1/255):
    gt_image = torch.max(gt_image, dim=0, keepdim=True)[0].repeat(3, 1, 1) + phi
    return torch.abs(gt_image - illumination_image).mean()


def pearson_depth_loss(depth_src, depth_target, eps=1e-6):
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()
    src_std = torch.sqrt(torch.mean(src ** 2) + eps)
    target_std = torch.sqrt(torch.mean(target ** 2) + eps)  
    src = src / src_std
    target = target / target_std
    co = (src * target).mean()

    return torch.clamp(1 - co , min=0.0, max=1.0)

# def pearson_depth_loss(depth_src, depth_target):
#     # 中心化
#     x = depth_src - depth_src.mean()
#     y = depth_target - depth_target.mean()
    
#     # 直接计算相关系数
#     r = torch.sum(x * y) / (torch.sqrt(torch.sum(x * x) * torch.sum(y * y)) + 1e-6)
    
#     return 1 - torch.clamp(r, min=-1.0, max=1.0)

def L_Depth_similarity(depth_src, depth_target, box_p, p_corr):
    num_box_h = depth_src.shape[0] // box_p
    num_box_w = depth_src.shape[1] // box_p
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p
    n_corr = int(p_corr * num_box_h * num_box_w)

    x_0 = torch.randint(0, max_h, (n_corr,), device='cuda')
    y_0 = torch.randint(0, max_w, (n_corr,), device='cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p

    _loss = sum(
        pearson_depth_loss(
            depth_src[x0:x1, y0:y1].reshape(-1),
            depth_target[x0:x1, y0:y1].reshape(-1)
        )
        for x0, x1, y0, y1 in zip(x_0, x_1, y_0, y_1)
    )

    return _loss / n_corr

def constancy_loss(x):
    Consis_rg = torch.pow(x[0]-x[1], 2)
    Consis_rb = torch.pow(x[1]-x[2], 2)
    Consis_gb = torch.pow(x[2]-x[0], 2)
    loss = (torch.pow(torch.pow(Consis_rg, 2) + torch.pow(Consis_rb, 2) + torch.pow(Consis_gb, 2), 0.5)).mean()
    return loss

def local_degree_loss(x, y, enhance_degree):
    y = y.detach()
    # loss = torch.abs((torch.log(1+x) - torch.log(1+y * enhance_degree))).mean()
    loss = torch.pow(torch.pow(x - y * enhance_degree, 2) + 1e-8, 0.5).mean()
    return loss

def global_degree_loss(x, enhance_degree):
    loss = torch.abs(torch.log(x.mean()) - torch.log(enhance_degree)).mean()
    return loss

def consistency_loss(illumination_enhance_image, illumination_image):
    illumination_image = illumination_image.detach()
    ratio = illumination_enhance_image.mean() / (illumination_image.mean() + 1e-4)
    weight_x = illumination_image[:,:-1,:-1] - illumination_image[:,1:,:-1]
    weight_y = illumination_image[:,:-1,:-1] - illumination_image[:,:-1,1:]
    grad_x = (illumination_enhance_image[:,:-1,:-1] - illumination_enhance_image[:,1:,:-1]) 
    grad_y = (illumination_enhance_image[:,:-1,:-1] - illumination_enhance_image[:,:-1,1:]) 
    # 计算梯度幅度
    grad_image = (grad_x - weight_x * ratio) ** 2+ (grad_y - weight_y * ratio) ** 2 + 1e-8

    return torch.sqrt(grad_image).mean()

# Gray World Colour Constancy
def L_Gray(image):
    RG = (image[0]-image[1]) ** 2
    RB = (image[1]-image[2]) ** 2
    GB = (image[2]-image[0]) ** 2
    k = torch.sqrt(RG + GB + RB + 1e-8)
    return k.mean()