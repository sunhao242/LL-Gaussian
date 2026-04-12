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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
import scipy
import matplotlib.pyplot as plt
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_project(cameras, xyz):
    # import pdb; pdb.set_trace()
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    pose = cameras.world_view_transform.transpose(0, 1)
   
    origins = pose[:3, 3]
    rotation = pose[:3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_models, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx = fov2focal(cameras.FoVx, cameras.image_width)
    fy = fov2focal(cameras.FoVy, cameras.image_height)
    cx = torch.tensor(cameras.image_width / 2, device=xyz.device, dtype=torch.float)
    cy = torch.tensor(cameras.image_height / 2, device=xyz.device, dtype=torch.float)
    

    x = fx * x + cx
    y = fy * y + cy
    return torch.stack((x, y), -1)

def camera_project2(camera, xyz):
    # import pdb; pdb.set_trace()
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    origins = - torch.tensor(camera.T).cuda()
    rotation = torch.tensor(camera.R).cuda()
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_models, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = torch.tensor(camera.image_width / 2, device=xyz.device, dtype=torch.float)
    cy = torch.tensor(camera.image_height / 2, device=xyz.device, dtype=torch.float)
    

    x = fx * x + cx
    y = fy * y + cy

    
    return torch.stack((x, y), -1)


def camera_project2_with_full_proj(cameras, xyz):
    # 添加齐次坐标
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [..., 4]
    
    # 使用full_proj_transform进行变换
    clip_coords = (xyz_h @ cameras.full_proj_transform.T)  # [..., 4]
    
    # 透视除法
    clip_coords = clip_coords[..., :3] / clip_coords[..., 3:]  # [..., 3]
    
    # 裁剪空间到像素空间转换
    # 假设裁剪空间范围为[-1,1]
    import pdb;pdb.set_trace()
    x = (clip_coords[..., 0]) * 0.5 * cameras.image_width
    y = (clip_coords[..., 1]) * 0.5 * cameras.image_height
    
    return torch.stack((x, y), -1)

def visualizer(camera_poses, colors, save_path="/mnt/data/1.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for pose, color in zip(camera_poses, colors):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]  # Corrected to use 3D translation component
        camera_positions = np.einsum(
            "...ij,...j->...i", np.linalg.inv(rotation), -translation
        )

        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")

    plt.savefig(save_path)
    plt.close()

    return save_path


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)
def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    ###  Additional operation
    # inter_poses = []
    # for pose in poses:
    #     tmp_pose = np.eye(4)
    #     tmp_pose[:3] = np.concatenate([pose.R.T, pose.T[:, None]], 1)
    #     tmp_pose = np.linalg.inv(tmp_pose)
    #     tmp_pose[:, 1:3] *= -1
    #     inter_poses.append(tmp_pose)
    # inter_poses = np.stack(inter_poses, 0)
    # poses, transform = transform_poses_pca(inter_poses)
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points) 



