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
import math
# from .graphics_utils import fov2focal
import torch.nn.functional as F
import kornia

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

def ciede2000_batch(img1, img2):
    """
    Args:
        img1, img2: [B, 3, H, W] - RGB tensors in range [0, 1] or [0, 255]
    Returns:
        ΔE00: [B, 1] - Mean CIEDE2000 difference per image
    """
    # Normalize to [0,1] float
    if img1.max() > 1.0:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    B, _, H, W = img1.shape
    img1_np = img1.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]
    img2_np = img2.permute(0, 2, 3, 1).cpu().numpy()

    delta_e_list = []
    for b in range(B):
        lab1 = rgb2lab(img1_np[b])  # [H, W, 3]
        lab2 = rgb2lab(img2_np[b])

        # 批量计算 ΔE00
        delta_e_map = deltaE_ciede2000(lab1, lab2)  # [H, W]
        delta_e_list.append(delta_e_map.mean())

    return torch.tensor(delta_e_list).view(B, 1)

def eval_depth(pred, target, min_depth=1e-2):
    assert pred.shape == target.shape
    # Step 1: 创建 mask
    mask_pred = pred < min_depth
    mask_target = target < min_depth
    mask = mask_pred | mask_target  # 截断

    pred[mask] = min_depth
    target[mask] = min_depth

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

def best_fit_affine(x: torch.Tensor, y: torch.Tensor, axis: int) -> torch.Tensor:
    """Computes best fit a, b such that a * x + b = y, in a least square sense."""
    x_m = x.mean(dim=axis, )
    y_m = y.mean(dim=axis)
    xy_m = (x * y).mean(dim=axis)
    xx_m = (x * x).mean(dim=axis)
    # slope a = Cov(x, y) / Cov(x, x).
    a = (xy_m - x_m * y_m) / (xx_m - x_m * x_m )
    b = y_m - a * x_m
    return a, b

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std
import torch
from torch import Tensor

def color_correct(img: Tensor, ref: Tensor, num_iters: int = 5, eps: float = 10/255) -> Tensor:
    """Warp `img` to match the colors in `ref_img`.
    
    Args:
        img: Input image tensor of shape (N, C, H, W)
        ref: Reference image tensor of shape (N, C, H, W)
        num_iters: Number of iterations for color correction
        eps: Small value for numerical stability
    """
    if img.shape[1] != ref.shape[1]:
        raise ValueError(
            f'img\'s {img.shape[1]} and ref\'s {ref.shape[1]} channels must match'
        )
    # Reshape to (N, C, -1) and then transpose to (N, -1, C)
    N, C = img.shape[:2]
    img_mat = img.reshape(N, C, -1).transpose(1, 2)  # (N, HW, C)
    ref_mat = ref.reshape(N, C, -1).transpose(1, 2)  # (N, HW, C)
    
    def is_unclipped(z: Tensor) -> Tensor:
        return (z >= eps) & (z <= (1 - eps))
    
    mask0 = is_unclipped(img_mat)  # (N, HW, C)
    
    # Process each batch independently
    corrected_mats = []
    for n in range(N):
        img_mat_n = img_mat[n]  # (HW, C)
        ref_mat_n = ref_mat[n]  # (HW, C)
        mask0_n = mask0[n]      # (HW, C)
        
        # Iterative optimization
        for _ in range(num_iters):
            # Construct quadratic expansion
            a_mat = []
            for c in range(C):
                # Quadratic term
                a_mat.append(img_mat_n[:, c:c+1] * img_mat_n[:, c:])
            # Linear term
            a_mat.append(img_mat_n)
            # Bias term
            a_mat.append(torch.ones_like(img_mat_n[:, :1]))
            a_mat = torch.cat(a_mat, dim=-1)  # (HW, num_coeffs)
            
            warp = []
            for c in range(C):
                # Get reference color channel
                b = ref_mat_n[:, c]  # (HW,)
                
                # Create mask for valid pixels
                mask = mask0_n[:, c] & is_unclipped(img_mat_n[:, c]) & is_unclipped(b)
                
                # Apply mask
                ma_mat = torch.where(mask.unsqueeze(1), a_mat, torch.zeros_like(a_mat))
                mb = torch.where(mask, b, torch.zeros_like(b))
                
                try:
                    # 使用更稳定的求解方法
                    w = torch.pinverse(ma_mat) @ mb.unsqueeze(1)
                except RuntimeError:
                    # 如果失败，使用一个简单的缩放方案
                    w = torch.zeros((a_mat.shape[1], 1), device=img.device)
                    if mask.any():
                        scale = (mb[mask].mean() / (ma_mat[mask].mean() ))
                        w[0] = scale
                
                warp.append(w)
            
            warp = torch.cat(warp, dim=1)  # (num_coeffs, C)
            
            # Apply color transformation
            img_mat_n = torch.matmul(a_mat, warp).clamp(0, 1)  # (HW, C)
        
        corrected_mats.append(img_mat_n)
    
    # Stack batches and reshape back to (N, C, H, W)
    corrected_mat = torch.stack(corrected_mats, dim=0)  # (N, HW, C)
    corrected_img = corrected_mat.transpose(1, 2).reshape(img.shape).contiguous()  # (N, C, H, W)
    return corrected_img      
          

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size) 

    # return torch.clamp(normalized_feat * style_std.expand(size) + style_mean.expand(size), 0, 0.99)
    return torch.clamp(content_feat * style_mean / content_mean, 0, 1)

def match_images_adain(est_list: list, gt_list: list, axis: tuple = (-2, -1)) -> list:
    est_matched_list = []
    for i in range(len(est_list)):
        # Mapping is computed gt->est to be robust since `est` may be very noisy.
        est_matched = adaptive_instance_normalization(est_list[i], gt_list[i])
        est_matched_list.append(est_matched)
    return est_matched_list
def match_images_affine(est_list: list, gt_list: list, axis: tuple = (-2, -1)) -> list:
    """Computes affine best fit of gt->est for a list of est tensors, then maps est back to match gt.
    input:
        est_list: list of est tensors, which are of shape (1, 3, H, W)
        gt_list: list of gt tensors, which are of shape (1, 3, H, W)
    """
    
    est_matched_list = []
    for i in range(len(est_list)):
        # Mapping is computed gt->est to be robust since `est` may be very noisy.
        a, b = best_fit_affine( est_list[i].squeeze(0), gt_list[i].squeeze(0), axis=axis)
        a = a.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,est_list[i].shape[-2], est_list[i].shape[-1])
        b = b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,est_list[i].shape[-2], est_list[i].shape[-1])

        # Inverse mapping back to gt ensures we use a consistent space for metrics.
        est_matched = torch.clamp(a * est_list[i] + b, 0, 1)
        # gt_matched = gt_list[i] * a + b
        print("gt.mean().item():", gt_list[i].mean().item())
        print("est_matched.mean().item():", est_matched.mean().item())
        print("est_list[i].mean().item():", est_list[i].mean().item())
        est_matched_list.append(est_matched)
        print("affine_matched, a, b:", a[0,0,0,0].item(), b[0,0,0,0].item())
    return est_matched_list


import cv2
import numpy as np
def normalize_brightness(renders, gts):
    normalized_list = []
    for i in range(len(renders)):
        render = renders[i].squeeze(0).permute(1,2,0).cpu().numpy() * 255
        gt = gts[i].squeeze(0).cpu().permute(1,2,0).numpy() * 255
        normalized_render = align_brightness(render, gt)
        # normalized_render = render * gt.mean() / render.mean()
        normalized_render =  torch.clamp(torch.from_numpy(normalized_render/255.0).permute(2,0,1).unsqueeze(0).cuda(), 0, 1)
        normalized_list.append(normalized_render)
    return normalized_list

# def normalize_brightness_to_gt_color(image, gt_image):
#     """
#     将输入彩色图像的亮度归一化到与GT彩色图像的亮度一致
#     :param image: 输入的彩色图像
#     :param gt_image: GT彩色图像
#     :return: 归一化后的彩色图像
#     """
#     # 转换到 YCbCr 空间
#     ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     ycrcb_gt = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)
    
#     # 分离 Y 通道
#     y, cr, cb = cv2.split(ycrcb_image)
#     y_gt, _, _ = cv2.split(ycrcb_gt)
    
#     # 计算GT图像的平均亮度
#     gt_brightness = np.mean(y_gt)
    
#     # 计算当前图像的平均亮度
#     current_brightness = np.mean(y)
    
#     # 调整 Y 通道的亮度
#     y_normalized = y * (gt_brightness / current_brightness)
    
#     # 裁剪像素值，确保在 [0, 255] 范围内
#     y_normalized = np.clip(y_normalized, 0, 255).astype(cr.dtype)
    
#     # 合并调整后的 Y 通道和原始 Cr、Cb 通道
#     ycrcb_normalized = cv2.merge([y_normalized, cr, cb])
    
#     # 转换回 BGR 空间
#     normalized_image = cv2.cvtColor(ycrcb_normalized, cv2.COLOR_YCrCb2BGR)
    
#     return normalized_image

import cv2
import numpy as np

def align_brightness(enhanced_img, gt_img, color_space='LAB'):
    # 转换到指定色彩空间并提取亮度通道
    if enhanced_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (enhanced_img.shape[1], enhanced_img.shape[0]))
    if color_space not in ['YUV', 'LAB']:
        raise ValueError(f"Invalid color space: {color_space}")
    if enhanced_img.dtype != np.uint8 or gt_img.dtype != np.uint8:
        enhanced_img = enhanced_img.astype(np.uint8)
        gt_img = gt_img.astype(np.uint8)

    # Color space conversion
    if color_space == 'YUV':
        enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2YUV)
        reference = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YUV)
    else:  # LAB
        enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)
        reference = cv2.cvtColor(gt_img, cv2.COLOR_BGR2LAB)

    # Extract luminance channels
    y_enhanced = enhanced[..., 0].astype(np.float32)
    y_ref = reference[..., 0].astype(np.float32)

    # Compute statistics with stability
    mu_e, mu_gt = np.mean(y_enhanced), np.mean(y_ref)
    sigma_e, sigma_gt = np.std(y_enhanced), np.std(y_ref)
    min_std = 5.0  # prevent division by near-zero
    sigma_e = max(sigma_e, min_std) + 1e-9
    sigma_gt = max(sigma_gt, min_std) + 1e-9


    # Linear correction
    y_aligned = (y_enhanced - mu_e) * (sigma_gt / sigma_e) + mu_gt
    y_aligned = np.clip(y_aligned, 0, 255).astype(np.uint8)

    # Merge channels and convert back
    enhanced[..., 0] = y_aligned
    if color_space == 'YUV':
        aligned = cv2.cvtColor(enhanced, cv2.COLOR_YUV2BGR)
    else:
        aligned = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return np.clip(aligned, 0, 255).astype(np.uint8)


def normalize_brightness_to_gt_color(image, gt_image):
    """
    将输入彩色图像的亮度归一化到与GT彩色图像的亮度一致（LAB颜色空间）
    :param image: 输入的彩色图像
    :param gt_image: GT彩色图像
    :return: 归一化后的彩色图像
    """
    # 转换到 LAB 空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_gt = cv2.cvtColor(gt_image, cv2.COLOR_BGR2LAB)
    
    # 分离 L, A, B 通道
    l, a, b = cv2.split(lab_image)
    l_gt, _, _ = cv2.split(lab_gt)
    
    # 计算GT图像的平均亮度
    gt_brightness = np.mean(l_gt)
    
    # 计算当前图像的平均亮度
    current_brightness = np.mean(l)
    
    # 调整 L 通道的亮度
    l_normalized = l * (gt_brightness / current_brightness)
    
    # 裁剪像素值，确保在 [0, 255] 范围内
    l_normalized = np.clip(l_normalized, 0, 255).astype(a.dtype)
    
    # 合并调整后的 L 通道和原始 A、B 通道
    lab_normalized = cv2.merge([l_normalized, a, b])
    
    # 转换回 BGR 空间
    normalized_image = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    
    return normalized_image



def match_images_affine_simplify(est_list: list, gt_list: list, axis: tuple = (-2, -1)) -> list:
    """Computes global affine best fit for all images using least squares method.
    input:
        est_list: list of est tensors, which are of shape (1, 3, H, W)
        gt_list: list of gt tensors, which are of shape (1, 3, H, W)
    """
    est_matched_list = []
    for i in range(len(est_list)):
        est_mean = est_list[i].mean()
        gt_mean = gt_list[i].mean()
        est_matched = torch.clamp(gt_mean / est_mean  * est_list[i], 0, 1)
        est_matched_list.append(est_matched)
    return est_matched_list

def Camera_Reprojection(image, closest_image, closest_depth, cam, closest_cam):
    R = torch.tensor(cam.R, device=closest_image.device, dtype=torch.float)
    T = torch.tensor(cam.T, device=closest_image.device, dtype=torch.float)
    closest_R = torch.tensor(closest_cam.R, device=closest_image.device, dtype=torch.float)
    closest_T = torch.tensor(closest_cam.T, device=closest_image.device, dtype=torch.float)
    fx = fov2focal(cam.FoVx, cam.image_width)
    fy = fov2focal(cam.FoVy, cam.image_height)
    cx = torch.tensor(cam.image_width / 2, device=closest_image.device, dtype=torch.float)
    cy = torch.tensor(cam.image_height / 2, device=closest_image.device, dtype=torch.float)
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], device=closest_image.device, dtype=torch.float)
    K_inv = torch.inverse(K)
    # 将closest_image转换到image像素空间
    
    # 1. 创建像素坐标网格
    h, w = closest_image.shape[1:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).to(torch.float).to(closest_image.device)
    
    # 2. 将像素坐标转换为相机坐标
    cam_coords = torch.matmul(K_inv, pixels.reshape(-1, 3).T).T
    cam_coords *= closest_depth.reshape(-1, 1)
    
    # # 3. 将相机坐标转换为世界坐标
    world_coords = torch.matmul(closest_R.T, cam_coords.T).T + closest_T.unsqueeze(0)

    
    # 4. 将世界坐标转换为目标相机坐标
    target_cam_coords = torch.matmul(R, (world_coords - T.unsqueeze(0)).T).T

    # 3. 将相机坐标转换为世界坐标
    # world_coords = torch.matmul(closest_R, (cam_coords - closest_T.unsqueeze(0)).T).T
    
    # # 4. 将世界坐标转换为目标相机坐标
    # target_cam_coords = torch.matmul(R.T, world_coords.T).T + T.unsqueeze(0)
    

    # 5. 将目标相机坐标投影到目标图像平面
    target_pixels = torch.matmul(K, target_cam_coords.T).T
    target_pixels = target_pixels[:, :2] / target_pixels[:, 2:3]
    
    # 6. 使用网格采样获取重投影的像素值
    target_pixels = target_pixels.reshape(h, w, 2)
    target_pixels = target_pixels * 2 / torch.tensor([w-1, h-1]).to(target_pixels.device) - 1
    reprojected_image = F.grid_sample(closest_image.unsqueeze(0), target_pixels.unsqueeze(0), mode='nearest', align_corners=True)[0]
    # 获取重投影对应的深度值
    reprojected_depth = F.grid_sample(closest_depth.unsqueeze(0), target_pixels.unsqueeze(0), mode='nearest', align_corners=True)[0]

    
    return reprojected_image, reprojected_depth

def Camera_Reprojection_inverse(source_depth, source_cam, target_image, target_cam):
    """
    Reprojects target image into source view using source depth map
    Args:
        source_depth: depth map tensor of shape [H, W]
        source_cam: source camera parameters
        target_image: target image tensor of shape [C, H, W]
        target_cam: target camera parameters
    Returns:
        reprojected_image: target image reprojected to source view [C, H, W]
    """
    device = target_image.device
    
    # Extract camera parameters and transformation matrices
    source_R = torch.tensor(source_cam.R, device=device, dtype=torch.float)
    source_T = torch.tensor(source_cam.T, device=device, dtype=torch.float)
    target_R = torch.tensor(target_cam.R, device=device, dtype=torch.float)
    target_T = torch.tensor(target_cam.T, device=device, dtype=torch.float)
    
    # Get camera intrinsics
    fx_source = fov2focal(source_cam.FoVx, source_cam.image_width)
    fy_source = fov2focal(source_cam.FoVy, source_cam.image_height)
    cx_source = torch.tensor(source_cam.image_width / 2, device=device, dtype=torch.float)
    cy_source = torch.tensor(source_cam.image_height / 2, device=device, dtype=torch.float)
    
    fx_target = fov2focal(target_cam.FoVx, target_cam.image_width)
    fy_target = fov2focal(target_cam.FoVy, target_cam.image_height)
    cx_target = torch.tensor(target_cam.image_width / 2, device=device, dtype=torch.float)
    cy_target = torch.tensor(target_cam.image_height / 2, device=device, dtype=torch.float)
    
    K_source = torch.tensor([[fx_source, 0, cx_source],
                            [0, fy_source, cy_source],
                            [0, 0, 1]], device=device, dtype=torch.float)
    K_target = torch.tensor([[fx_target, 0, cx_target],
                            [0, fy_target, cy_target],
                            [0, 0, 1]], device=device, dtype=torch.float)
    
    K_source_inv = torch.inverse(K_source)
    
    # Create pixel coordinate grid
    h, w = target_image.shape[1:]
    y, x = torch.meshgrid(torch.arange(h, device=device), 
                         torch.arange(w, device=device))
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()
    
    # Convert source pixels to camera coordinates
    cam_coords = torch.matmul(K_source_inv, pixels.reshape(-1, 3).T).T
    cam_coords *= source_depth.reshape(-1, 1)
    
    # Convert camera coordinates to world coordinates
    world_coords = torch.matmul(source_R.T, cam_coords.T).T + source_T.unsqueeze(0)
    
    # Convert world coordinates to target camera coordinates
    target_cam_coords = torch.matmul(target_R, (world_coords - target_T.unsqueeze(0)).T).T
    
    # Project target camera coordinates to target image plane
    target_pixels = torch.matmul(K_target, target_cam_coords.T).T
    target_pixels = target_pixels[:, :2] / target_pixels[:, 2:3]
    
    # Reshape to [H, W, 2]
    target_pixels = target_pixels.reshape(h, w, 2)
    
    # Normalize coordinates to [-1, 1] range for grid_sample
    target_pixels = target_pixels * 2 / torch.tensor([target_cam.image_width - 1, 
                                                     target_cam.image_height - 1],
                                                    device=device) - 1
    
    # Sample from target image
    reprojected_image = F.grid_sample(
        target_image.unsqueeze(0),
        target_pixels.unsqueeze(0),
        align_corners=True,
        mode='nearest'
    )[0]
    
    return reprojected_image

def map_pixels_between_views(source_image, source_depth, source_cam, target_cam):
    """
    Maps pixels from source view to target view
    Args:
        source_image: source image tensor of shape [3, H, W]
        source_depth: depth map tensor of shape [H, W]
        source_cam: source camera parameters
        target_cam: target camera parameters
    Returns:
        target_pixels: mapped pixel coordinates in target view of shape [H, W, 2]
    """
    device = source_image.device
    
    # Extract camera parameters and transformation matrices
    source_R = torch.tensor(source_cam.R, device=device, dtype=torch.float)
    source_T = torch.tensor(source_cam.T, device=device, dtype=torch.float)
    target_R = torch.tensor(target_cam.R, device=device, dtype=torch.float)
    target_T = torch.tensor(target_cam.T, device=device, dtype=torch.float)
    
    # Get camera intrinsics
    fx_source = fov2focal(source_cam.FoVx, source_cam.image_width)
    fy_source = fov2focal(source_cam.FoVy, source_cam.image_height)
    cx_source = torch.tensor(source_cam.image_width / 2, device=device, dtype=torch.float)
    cy_source = torch.tensor(source_cam.image_height / 2, device=device, dtype=torch.float)
    
    fx_target = fov2focal(target_cam.FoVx, target_cam.image_width)
    fy_target = fov2focal(target_cam.FoVy, target_cam.image_height)
    cx_target = torch.tensor(target_cam.image_width / 2, device=device, dtype=torch.float)
    cy_target = torch.tensor(target_cam.image_height / 2, device=device, dtype=torch.float)
    
    K_source = torch.tensor([[fx_source, 0, cx_source],
                            [0, fy_source, cy_source],
                            [0, 0, 1]], device=device, dtype=torch.float)
    K_target = torch.tensor([[fx_target, 0, cx_target],
                            [0, fy_target, cy_target],
                            [0, 0, 1]], device=device, dtype=torch.float)
    
    K_source_inv = torch.inverse(K_source)
    
    # Create pixel coordinate grid
    h, w = source_image.shape[1:]
    y, x = torch.meshgrid(torch.arange(h, device=device), 
                         torch.arange(w, device=device))
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()
    
    # Convert source pixels to camera coordinates
    cam_coords = torch.matmul(K_source_inv, pixels.reshape(-1, 3).T).T
    cam_coords *= source_depth.reshape(-1, 1)
    
    # Convert camera coordinates to world coordinates
    world_coords = torch.matmul(source_R.T, cam_coords.T).T + source_T.unsqueeze(0)
    
    # Convert world coordinates to target camera coordinates
    target_cam_coords = torch.matmul(target_R, (world_coords - target_T.unsqueeze(0)).T).T
    
    # Project target camera coordinates to target image plane
    target_pixels = torch.matmul(K_target, target_cam_coords.T).T
    target_pixels = target_pixels[:, :2] / target_pixels[:, 2:3]
    
    # Reshape to [H, W, 2]
    target_pixels = target_pixels.reshape(h, w, 2)
    
    return target_pixels

def fill_source_from_target(source_shape, target_image, target_pixels):
    """
    将target图像的像素值填充回source图像坐标系
    Args:
        source_shape: source图像的形状 (C, H, W)
        target_image: target图像张量 (C, H, W)
        target_pixels: 映射坐标 (H, W, 2)，从map_pixels_between_views获得
    Returns:
        filled_image: 填充后的图像 (C, H, W)
    """
    device = target_image.device
    
    # target_pixels已经是[-1,1]范围的映射坐标
    # 直接使用target_pixels进行采样
    filled_image = F.grid_sample(
        target_image.unsqueeze(0),  # (1, C, H, W)
        target_pixels.unsqueeze(0),  # (1, H, W, 2)
        align_corners=True,
        mode='nearest'
    )[0]  # (C, H, W)
    
    return filled_image


# def fill_source_from_target(source_shape, target_image, target_pixels):
#     """
#     将target图像的像素值填充回source图像坐标系，使用最近邻采样而非插值
#     Args:
#         source_shape: source图像的形状 (C, H, W)
#         target_image: target图像张量 (C, H, W)
#         target_pixels: 映射坐标 (H, W, 2)
#     Returns:
#         filled_image: 填充后的图像 (C, H, W)
#     """
#     device = target_image.device
#     C, H, W = source_shape
    
#     # 创建输出图像
#     filled_image = torch.zeros(source_shape, device=device)
    
#     # 将target_pixels转换到像素坐标
#     target_x = (target_pixels[..., 0] * (W-1) / 2 + (W-1) / 2).round().long()
#     target_y = (target_pixels[..., 1] * (H-1) / 2 + (H-1) / 2).round().long()
    
#     # 创建有效mask（在图像范围内的像素）
#     valid_mask = (target_x >= 0) & (target_x < W) & (target_y >= 0) & (target_y < H)
    
#     # 对每个通道进行采样
#     for y in range(H):
#         for x in range(W):
#             if valid_mask[y, x]:
#                 tx, ty = target_x[y, x], target_y[y, x]
#                 filled_image[:, y, x] = target_image[:, ty, tx]
    
#     return filled_image

# from skimage.color import lab2rgb
# # 标准 Lab 测试对
# rgb1 = np.array([[[0, 0.2, 0.2]]])  # [1,1,3]
# rgb2 = np.array([[[0, 0.8, 0.1]]])


# # 转换为 PyTorch 格式 [B, 3, H, W]
# img1 = torch.tensor(rgb1).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, 1, 1]
# img2 = torch.tensor(rgb2).permute(2, 0, 1).unsqueeze(0).float()

# # 使用你实现的函数进行测试
# de = ciede2000_batch(img1, img2)
# print(f"ΔE00 (Computed): {de.item():.4f}, Expected: 2.0425")