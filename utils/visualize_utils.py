from typing import Callable, Dict, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.colors as mcolors

def visualize_camera_trajectories(render_cameras, train_cameras, save_path="camera_trajectories.png"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制训练相机轨迹（蓝色）
    train_positions = []
    train_directions = []
    for cam in train_cameras:
        pos = cam.T
        forward = -cam.R[:, 2]  # 相机朝向（z轴负方向）
        train_positions.append(pos)
        train_directions.append(forward)
    
    train_positions = np.array(train_positions)
    train_directions = np.array(train_directions)
    
    # 绘制渲染相机轨迹（红色）
    render_positions = []
    render_directions = []
    for cam in render_cameras:
        pos = cam.T
        forward = -cam.R[:, 2]
        render_positions.append(pos)
        render_directions.append(forward)
    
    render_positions = np.array(render_positions)
    render_directions = np.array(render_directions)
    
    # 绘制相机位置
    ax.scatter(train_positions[:, 0], train_positions[:, 1], train_positions[:, 2], 
            c='blue', label='Train Cameras', s=50)
    ax.scatter(render_positions[:, 0], render_positions[:, 1], render_positions[:, 2], 
            c='red', label='Test Cameras', s=50)
    
    # 绘制相机朝向（使用箭头）
    scale = 0.1  # 调整箭头长度
    for pos, direction in zip(train_positions, train_directions):
        ax.quiver(pos[0], pos[1], pos[2],
                direction[0], direction[1], direction[2],
                color='blue', alpha=0.5, length=scale)
    
    for pos, direction in zip(render_positions, render_directions):
        ax.quiver(pos[0], pos[1], pos[2],
                direction[0], direction[1], direction[2],
                color='red', alpha=0.5, length=scale)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 调整坐标轴比例使其相等
    max_range = np.array([
        train_positions.max(axis=0) - train_positions.min(axis=0),
        render_positions.max(axis=0) - render_positions.min(axis=0)
    ]).max() / 2.0
    
    mid_x = (train_positions[:, 0].mean() + render_positions[:, 0].mean()) / 2
    mid_y = (train_positions[:, 1].mean() + render_positions[:, 1].mean()) / 2
    mid_z = (train_positions[:, 2].mean() + render_positions[:, 2].mean()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def add_label_centered(
    img: np.ndarray,
    text: str,
    font_scale: float = 1.0,
    thickness: int = 2,
    alignment: str = "top",
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, font_scale, thickness=thickness)[0]
    img = img.astype(np.uint8).copy()

    if alignment == "top":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, 50),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    elif alignment == "bottom":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, img.shape[0] - textsize[1]),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        raise ValueError("Unknown text alignment")

    return img

def tensor2rgbjet(
    tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    return cv2.applyColorMap(tensor2rgb(tensor, x_max=x_max, x_min=x_min), cv2.COLORMAP_JET)


def tensor2rgb(
    tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    x = tensor.data.cpu().numpy()
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    gain = 255 / np.clip(x_max - x_min, 1e-3, None)
    x = (x - x_min) * gain
    x = x.clip(0.0, 255.0)
    x = x.astype(np.uint8)
    return x


def tensor2image(
    tensor: th.Tensor,
    x_max: Optional[float] = 1.0,
    x_min: Optional[float] = 0.0,
    mode: str = "rgb",
    mask: Optional[th.Tensor] = None,
    label: Optional[str] = None,
) -> np.ndarray:

    tensor = tensor.detach()

    # Apply mask
    if mask is not None:
        tensor = tensor * mask

    if len(tensor.size()) == 2:
        tensor = tensor[None]

    # Make three channel image
    assert len(tensor.size()) == 3, tensor.size()
    n_channels = tensor.shape[0]
    if n_channels == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif n_channels != 3:
        raise ValueError(f"Unsupported number of channels {n_channels}.")

    # Convert to display format
    img = tensor.permute(1, 2, 0)

    if mode == "rgb":
        img = tensor2rgb(img, x_max=x_max, x_min=x_min)
    elif mode == "jet":
        # `cv2.applyColorMap` assumes input format in BGR
        img[:, :, :3] = img[:, :, [2, 1, 0]]
        img = tensor2rgbjet(img, x_max=x_max, x_min=x_min)
        # convert back to rgb
        img[:, :, :3] = img[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"Unsupported mode {mode}.")

    if label is not None:
        img = add_label_centered(img, label)

    return img
    
# d: b x 1 x H x W
# screenCoords: b x 2 x H X W
# focal: b x 2 x 2
# princpt: b x 2
# out: b x 3 x H X W
def depthImgToPosCam_Batched(d, screenCoords, focal, princpt):
    p = screenCoords - princpt[:, :, None, None]
    x = (d * p[:, 0:1, :, :]) / focal[:, 0:1, 0, None, None]
    y = (d * p[:, 1:2, :, :]) / focal[:, 1:2, 1, None, None]
    return th.cat([x, y, d], dim=1)

# p: b x 3 x H x W
# out: b x 3 x H x W
def computeNormalsFromPosCam_Batched(p):
    p = F.pad(p, (1, 1, 1, 1), "replicate")
    d0 = p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]
    d1 = p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]
    n = th.cross(d0, d1, dim=1)
    norm = th.norm(n, dim=1, keepdim=True)
    norm = norm + 1e-5
    norm[norm < 1e-5] = 1  # Can not backprop through this
    return -n / norm

def visualize_normal(inputs, depth_p):
    # Normals
    uv = th.stack(
        th.meshgrid(
            th.arange(depth_p.shape[2]), th.arange(depth_p.shape[1]), indexing="xy"
        ),
        dim=0,
    )[None].float().cuda()
    position = depthImgToPosCam_Batched(
        depth_p[None, ...], uv, inputs["focal"], inputs["princpt"]
    )
    normal = 0.5 * (computeNormalsFromPosCam_Batched(position) + 1.0)
    normal = normal[0, [2, 1, 0], :, :]  # legacy code assumes BGR format
    normal_p = tensor2image(normal, label="normal_p")

    return normal_p

def minmax_normalize(tensor):

    # 初始化一个新的张量用于存储归一化后的结果
    normalized_tensor = th.zeros_like(tensor)
    
    # 分别对每个通道进行归一化
    min_val = tensor[0].min()
    max_val = tensor[0].max()
    
    # 避免除以零的情况
    if max_val != min_val:
        normalized_tensor[0] = (tensor[0] - min_val) / (max_val - min_val)
    else:
        normalized_tensor[0] = tensor[0]  # 如果max和min相等，不进行归一化

    return normalized_tensor



def visualize_anchor(anchor_points, save_path):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    # 获取点云数据
    points = anchor_points
    
    # 计算点云的主方向（使用PCA）
    center = points.mean(axis=0)
    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 根据主方向计算最佳视角
    principal_direction = eigenvectors[:, -1]  # 最大特征值对应的特征向量
    azim = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
    elev = np.arctan2(principal_direction[2], np.sqrt(principal_direction[0]**2 + principal_direction[1]**2)) * 180 / np.pi
    
    # 调整视角，确保在合理范围内
    azim = (azim + 90) % 360  # 调整方位角到合适区间
    elev = np.clip(elev, 0, 89)  # 限制仰角在0-89度之间

    # 根据深度计算颜色
    depths = points[:, 2] - points[:, 2].min()
    depths = depths / depths.max()
    # 使用自定义colormap
    plt.cm.viridis(depths)
    # 绘制高质量散点图
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=depths, 
                        cmap='viridis',
                        s=0.5,  # 点大小
                        alpha=0.2,  # 透明度
                        linewidth=0)

    # 设置视角
    ax.view_init(elev=elev, azim=azim)
    
    # 设置坐标轴范围,确保比例一致
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                        points[:, 1].max()-points[:, 1].min(),
                        points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置轴标签和标题
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    plt.title(f'Point Cloud Visualization\n{points.shape[0]} points')
    
    # 调整布局
    plt.tight_layout()

    # 添加标题显示当前点数
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_anchor_with_camera(render_cameras, train_cameras, anchor_points, save_path):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ################ camera trajectory ################
    # 绘制训练相机轨迹（蓝色）
    train_positions = []
    train_directions = []
    for cam in train_cameras:
        pos = cam.T
        forward = -cam.R[:, 2]  # 相机朝向（z轴负方向）
        train_positions.append(pos)
        train_directions.append(forward)
    
    train_positions = np.array(train_positions)
    train_directions = np.array(train_directions)
    
    # 绘制渲染相机轨迹（红色）
    render_positions = []
    render_directions = []
    for cam in render_cameras:
        pos = cam.T
        forward = -cam.R[:, 2]
        render_positions.append(pos)
        render_directions.append(forward)
    
    render_positions = np.array(render_positions)
    render_directions = np.array(render_directions)
    
    # 绘制相机位置
    ax.scatter(train_positions[:, 0], train_positions[:, 1], train_positions[:, 2], 
            c='blue', label='Train Cameras', s=10)
    ax.scatter(render_positions[:, 0], render_positions[:, 1], render_positions[:, 2], 
            c='red', label='Test Cameras', s=10)
    
    # 绘制相机朝向（使用箭头）
    scale = 0.1  # 调整箭头长度
    for pos, direction in zip(train_positions, train_directions):
        ax.quiver(pos[0], pos[1], pos[2],
                direction[0], direction[1], direction[2],
                color='blue', alpha=0.5, length=scale)
    
    for pos, direction in zip(render_positions, render_directions):
        ax.quiver(pos[0], pos[1], pos[2],
                direction[0], direction[1], direction[2],
                color='red', alpha=0.5, length=scale)
    
    ################ anchor points ################
    # 获取点云数据
    points = anchor_points.detach().cpu().numpy()
    # 去除过远点
    mask = (np.abs(points[:, 2]) < 10) & (np.abs(points[:, 1]) < 10) & (np.abs(points[:, 0]) < 10)
    points = points[mask]
    # 计算点云的主方向（使用PCA）
    center = points.mean(axis=0)
    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 根据主方向计算最佳视角
    principal_direction = eigenvectors[:, -1]  # 最大特征值对应的特征向量
    azim = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
    elev = np.arctan2(principal_direction[2], np.sqrt(principal_direction[0]**2 + principal_direction[1]**2)) * 180 / np.pi
    
    # 调整视角，确保在合理范围内
    azim = (azim + 90) % 360  # 调整方位角到合适区间
    elev = np.clip(elev, 0, 89)  # 限制仰角在0-89度之间

    # 根据深度计算颜色
    depths = points[:, 2] - points[:, 2].min()
    depths = depths / depths.max()
    # 使用自定义colormap
    plt.cm.viridis(depths)
    # 绘制高质量散点图
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=depths, 
                        cmap='viridis',
                        s=0.5,  # 点大小
                        alpha=0.2,  # 透明度
                        linewidth=0)

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    
    # 设置坐标轴范围,确保比例一致
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                        points[:, 1].max()-points[:, 1].min(),
                        points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置轴标签和标题
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    plt.title(f'Point Cloud Visualization\n{points.shape[0]} points')
    
    # 调整布局
    plt.tight_layout()

    # 添加标题显示当前点数
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_heatmap(loss_image, iteration, save_dir):
    # 创建一个包含四个子图的图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始热力图绘制 (在第一个子图)
    # import ipdb; ipdb.set_trace()
    cmap = plt.get_cmap('coolwarm')
    # cmap = plt.get_cmap('Reds')
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    if np.max(loss_image) > 0 and np.min(loss_image) < 0:
        norm = mcolors.TwoSlopeNorm(vmin=np.min(loss_image), vcenter=0, vmax=np.max(loss_image))
    elif np.min(loss_image) > 0 and np.max(loss_image) > 0:
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=np.max(loss_image))
    else:
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax1.imshow(loss_image, cmap=cmap, norm=norm, interpolation='nearest')
    # im = ax1.imshow(Ll1_image_np, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax1)
    ax1.set_title(f"Ll1 Heatmap_{iteration}")
    
    # 直方图绘制 (在第二个子图)
    flattened_data = loss_image.flatten()
    ax2.hist(flattened_data, bins=50, color='skyblue', edgecolor='black')
    ax2.set_title("Loss Distribution Histogram")
    ax2.set_xlabel("Loss Value")
    ax2.set_ylabel("Frequency")
    
    # 添加一些统计信息作为文本
    stats_text = f'Mean: {np.mean(flattened_data):.4f}\n'
    stats_text += f'Std: {np.std(flattened_data):.4f}\n'
    stats_text += f'Min: {np.min(flattened_data):.4f}\n'
    stats_text += f'Max: {np.max(flattened_data):.4f}'
    ax2.text(0.95, 0.95, stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 正值热力图 (在第三个子图)]
    # import ipdb; ipdb.set_trace()
    positive_data = np.maximum(loss_image, 0)
    im_pos = ax3.imshow(positive_data, cmap='Reds', interpolation='nearest')
    plt.colorbar(im_pos, ax=ax3)
    ax3.set_title("Positive Values Heatmap")
    
    # 负值热力图 (在第四个子图)
    negative_data = (np.log(np.abs(np.minimum(loss_image, 0))))
    im_neg = ax4.imshow(negative_data, cmap='Blues', interpolation='nearest')
    plt.colorbar(im_neg, ax=ax4)
    ax4.set_title("Negative Values Heatmap")
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # # 保存图像
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"loss_heatmap_{iteration}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()





# from mpl_toolkits.mplot3d import Axes3D
# import torch
# def plot_point_cloud_projection(points, viewpoint_cam, output_path,alpha=0.5):
#     """
#     绘制点云投影图并保存为图像文件。
    
#     参数:
#     - gaussians: GaussianModel对象，包含点云的xyz坐标。
#     - viewpoint_cam: 相机对象，包含相机信息。
#     - output_path: 输出图像文件的路径。
#     """
#     # 获取点云的xyz坐标
#     # 获取相机的外参矩阵（RT矩阵）
#     # import pdb;pdb.set_trace()
#     from utils.camera_utils import camera_project2, camera_project2_with_full_proj
#     # import pdb;pdb.set_trace()
#     xy = camera_project2(viewpoint_cam, points)
#     # xy = camera_project2_with_full_proj(viewpoint_cam, points)
#     u = xy[:,0].detach().cpu().numpy() 
#     v = xy[:,1].detach().cpu().numpy()

#     # --- 绘制投影图 ---
#     width = viewpoint_cam.image_width
#     height = viewpoint_cam.image_height
#     # 计算英寸尺寸（假设DPI为100）
#     dpi = 100  # 可以根据需要调整DPI
#     figsize = (width * 1.5 / dpi, height * 1.5 / dpi)

#     plt.figure(figsize=figsize, dpi=dpi)
#     plt.scatter((u + width ) / 3 ,(v +height  ) / 3 , s=1, c='y', alpha=alpha)

#     plt.xlim(0, width)
#     plt.ylim(0, height)
#     plt.gca().invert_yaxis()  # 图像坐标系原点在左上
#     plt.axis('off')  # 关闭坐标轴
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除边距
#     # 保存图像，背景透明
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
#     plt.close()
    # import pdb;pdb.set_trace()

from mpl_toolkits.mplot3d import Axes3D
import torch
def plot_point_cloud_projection(points, viewpoint_cam, output_path,alpha=0.5):
    """
    绘制点云投影图并保存为图像文件。
    
    参数:
    - gaussians: GaussianModel对象，包含点云的xyz坐标。
    - viewpoint_cam: 相机对象，包含相机信息。
    - output_path: 输出图像文件的路径。
    """
    # 获取点云的xyz坐标
    # 获取相机的外参矩阵（RT矩阵）
    # import pdb;pdb.set_trace()
    from utils.camera_utils import camera_project2, camera_project2_with_full_proj
    # import pdb;pdb.set_trace()
    xy = camera_project2(viewpoint_cam, points)
    # xy = camera_project2_with_full_proj(viewpoint_cam, points)
    u = xy[:,0].detach().cpu().numpy() 
    v = xy[:,1].detach().cpu().numpy()

    # --- 绘制投影图 ---
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    # 计算英寸尺寸（假设DPI为100）
    dpi = 100  # 可以根据需要调整DPI
    figsize = (width / dpi , height / dpi )

    plt.figure(figsize=figsize , dpi=dpi * 3)
    plt.scatter(u ,v , s=0.1, c='y', alpha=alpha)

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()  # 图像坐标系原点在左上
    plt.axis('off')  # 关闭坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除边距
    # 保存图像，背景透明
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)
            

def visualize_cmap(value,
                weight,
                colormap,
                lo=None,
                hi=None,
                percentile=99.,
                curve_fn=lambda x: x,
                modulus=None,
                matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized


def visualize_scene_and_cameras(points, viewpoint_cams, colors=None, point_size=1.0,
                                 active_camera_index=None, save_path=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    # 1. 获取所有相机中心
    cam_centers = []
    cam_dirs = []
    for cam in viewpoint_cams:
        R = np.array(cam.R)
        T = -np.array(cam.T).reshape(3)
        center = -R.T @ T
        cam_centers.append(center)
        cam_dirs.append(R.T[:, 2])  # 相机Z轴朝向（朝外）

    cam_centers = np.stack(cam_centers, axis=0)  # (M, 3)
    cam_dirs = np.stack(cam_dirs, axis=0)        # (M, 3)

    # 2. 计算平均朝向 & 平移中心
    mean_dir = np.mean(cam_dirs, axis=0)
    mean_dir /= np.linalg.norm(mean_dir)  # 单位化
    center_mean = np.mean(cam_centers, axis=0)

    # 3. 构建新坐标系 (Z: mean_dir, Y: arbitrary vertical, X: cross)
    up_vector = np.array([0, 1, 0])
    if np.abs(np.dot(mean_dir, up_vector)) > 0.99:
        up_vector = np.array([1, 0, 0])
    x_axis = np.cross(up_vector, mean_dir)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(mean_dir, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = mean_dir

    R_align = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3)

    # 4. 对所有坐标变换：居中 + 朝向对齐
    points = (points - center_mean) @ R_align
    cam_centers = (cam_centers - center_mean) @ R_align

    if colors is not None:
        colors = colors

    # 点云距离裁剪（以cam_centers为中心）
    dists = np.linalg.norm(points[:, None, :] - cam_centers[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    distance_threshold = 1.0
    mask = min_dists < distance_threshold
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    # 5. 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    if colors is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c='gray', alpha=0.05)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, alpha=0.05)

    # 绘制相机
    for i, cam in enumerate(viewpoint_cams):
        R = np.array(cam.R)
        T = np.array(cam.T).reshape(3)
        center = -R.T @ T
        center = (center - center_mean) @ R_align
        R_cam = R.T @ R_align  # 对齐后的相机方向

        axis_length = 0.05
        x_axis = center + R_cam[:, 0] * axis_length
        y_axis = center + R_cam[:, 1] * axis_length
        z_axis = center + R_cam[:, 2] * axis_length

        lw = 2.0 if i == active_camera_index else 0.5
        ax.plot([center[0], x_axis[0]], [center[1], x_axis[1]], [center[2], x_axis[2]], c='r', linewidth=lw)
        ax.plot([center[0], y_axis[0]], [center[1], y_axis[1]], [center[2], y_axis[2]], c='g', linewidth=lw)
        ax.plot([center[0], z_axis[0]], [center[1], z_axis[1]], [center[2], z_axis[2]], c='b', linewidth=lw)
        # if i == active_camera_index:
        #     ax.text(*center, f"Cam {i}", color='gold', fontsize=8)

    # 连线所有相机中心形成轨迹线
    # ax.plot(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2], color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel("Aligned X")
    ax.set_ylabel("Aligned Y")
    ax.set_zlabel("Aligned Z")
    ax.set_title("Camera-Aligned Point Cloud Visualization")
    view_dir = z_axis  # = mean_dir
    azim = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    hyp = np.linalg.norm(view_dir[:2])
    elev = np.degrees(np.arctan2(view_dir[2], hyp))

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()  # 隐藏坐标轴线、刻度和标签
    ax.grid(False)     # 隐藏网格线
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()