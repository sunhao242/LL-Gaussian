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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.pose_utils import get_tensor_from_camera, get_camera_from_tensor
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from utils.graphics_utils import get_uniform_points_on_sphere_fibonacci
from tqdm import tqdm
from utils.camera_utils import camera_project
from utils.visualize_utils import  visualize_anchor, plot_point_cloud_projection

@torch.no_grad()
def get_sky_points(num_points, points3D, cameras):
    xnp = torch
    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=xnp)
    points = points.to(points3D.device)
    mean = points3D.mean(0)[None]
    sky_distance = xnp.quantile(xnp.linalg.norm(points3D - mean, 2, -1), 0.97) * 10
    points = points * sky_distance
    points = points + mean
    gmask = torch.zeros((points.shape[0],), dtype=xnp.bool, device=points.device)
    for cam in tqdm(cameras, desc="Generating skybox"):
        uv = camera_project(cam, points[xnp.logical_not(gmask)])
        mask = xnp.logical_not(xnp.isnan(uv).any(-1))
        # Only top 2/3 of the image
        assert cam.image_height is not None
        mask = xnp.logical_and(mask, uv[..., -1] < 2/3 * cam.image_height)
        gmask[xnp.logical_not(gmask)] = xnp.logical_or(gmask[xnp.logical_not(gmask)], mask)
    return points[gmask], sky_distance / 2

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_residual_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_reflectance_dist : bool = False,
                 add_illumination_dist : bool = False,
                 add_residual_dist : bool = False,
                 use_residual : bool = False,
                 use_3D_filter : bool = False,
                 use_undependent_illumination : bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_residual_dim = appearance_residual_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_reflectance_dist = add_reflectance_dist
        self.add_illumination_dist = add_illumination_dist
        self.add_residual_dist = add_residual_dist

        self.use_3D_filter = use_3D_filter

        self.use_undependent_illumination = use_undependent_illumination
        
        ## residual
        self.use_residual = use_residual

        self.render_enhancement = False

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank: # weight of anchor
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        ## residual
        if self.use_residual:
            self.n_offsets_residual = int(n_offsets )
            self._offset_residual = torch.empty(0)
            self._anchor_feat_residual = torch.empty(0)
            self._scaling_residual = torch.empty(0)



        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0 # take distant as input or not 
        self.mlp_opacity = nn.Sequential(  # output n_offsets's opacity
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        if self.use_residual:
            self.mlp_opacity_residual = nn.Sequential(  # output n_offsets's opacity
                nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, self.n_offsets_residual),
                nn.Sigmoid()
            ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0 # take distant as input or not 
        self.mlp_cov = nn.Sequential( # output n_offsets's covriance
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        if self.use_residual:
            self.mlp_cov_residual = nn.Sequential( # output n_offsets's covriance
                nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 7*self.n_offsets_residual),
            ).cuda()
        self.reflectance_dist_dim = 1 if self.add_reflectance_dist else 0 # take distant as input or not
        self.mlp_reflectance = nn.Sequential(
            nn.Linear(feat_dim + self.reflectance_dist_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        self.illumination_dist_dim = 1 if self.add_illumination_dist else 0 # take distant as input or not
        self.mlp_illumination= nn.Sequential(
            nn.Linear(feat_dim// 2+3+self.illumination_dist_dim, feat_dim // 2),
            nn.ReLU(True),
            nn.Linear(feat_dim// 2, 1*self.n_offsets)
        ).cuda()
        if self.use_residual:
            self.residual_dist_dim = 1 if self.add_residual_dist else 0 # take distant as input or not
            self.residual_net= nn.Sequential(
                nn.Linear(feat_dim+3+self.residual_dist_dim+self.appearance_residual_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*self.n_offsets_residual),
                nn.Sigmoid()
            ).cuda()

        self.enhancement_net = nn.Sequential(
                nn.Linear(feat_dim + 1*self.n_offsets, feat_dim // 2),
                nn.ReLU(True),
                nn.Linear(feat_dim // 2, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        # self.mlp_color.eval()
        self.mlp_illumination.eval()
        self.mlp_reflectance.eval()
        self.enhancement_net.eval()
        if self.use_residual:
            self.residual_net.eval()
            self.mlp_cov_residual.eval()
            self.mlp_opacity_residual.eval()
        if self.appearance_residual_dim > 0: # use appearance embedding or not
            self.embedding_appearance.eval()
        if self.use_feat_bank: # use anchor feature or not (mutil-resolution)
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        # self.mlp_color.train()
        self.mlp_illumination.train()
        self.mlp_reflectance.train()
        self.enhancement_net.train()
        if self.use_residual:
            self.residual_net.train()
            self.mlp_cov_residual.train()
            self.mlp_opacity_residual.train()
        if self.appearance_residual_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        if self.use_residual:
            return (
                self._anchor,
                self._anchor_feat_residual,
                self._offset,
                self._offset_residual,
                self._local, # ?
                self._scaling,
                self._scaling_residual,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.denom, 
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self._anchor,
                self._offset,
                self._local, 
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.denom, 
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
    
    def restore(self, model_args, training_args):
        if self.use_residual:
            (self.active_sh_degree, 
            self._anchor, 
            self._anchor_feat_residual,
            self._offset,
            self._offset_residual,
            self._local,
            self._scaling, 
            self._scaling_residual,
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            self.training_setup(training_args)
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
        else:
            (self.active_sh_degree, 
            self._anchor, 
            self._offset,
            self._local,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            self.training_setup(training_args)
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    def set_appearance_residual(self, num_cameras):
        if self.appearance_residual_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_residual_dim).cuda()

    @property
    def get_appearance_residual(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    

    def get_scaling_with_3D_filter(self, scales, visible_mask):
        scales = torch.square(scales) + torch.square(self.filter_3D[visible_mask])
        scales = torch.sqrt(scales)
        return scales  

    @property
    def get_scaling_residual(self):
        return 1.0*self.scaling_activation(self._scaling_residual)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    

    def get_opacity_with_3D_filter(self, opacity, visible_mask):
        # apply 3D filter
        scales = self.get_scaling[visible_mask]
        scales = scales.repeat(self.n_offsets, 1)
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D[visible_mask].repeat(self.n_offsets, 1)) 
        det2 = scales_after_square.prod(dim=1) 

        eps = 1e-10  # 小的常数值
        det1 = torch.clamp_min(det1, eps)
        det2 = torch.clamp_min(det2, eps)
        coef = torch.sqrt(det1 / det2)
        if torch.any(torch.isnan(coef)) or torch.any(torch.isinf(coef)):
            import pdb;pdb.set_trace()
        return opacity * coef[..., None]

    @property
    def get_opacity_residual_mlp(self):
        return self.mlp_opacity_residual

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_cov_residual_mlp(self):
        return self.mlp_cov_residual
    # @property
    # def get_color_mlp(self):
    #     return self.mlp_color

    @property
    def get_reflectance_mlp(self):
        return self.mlp_reflectance

    @property
    def get_enhancement_net(self):
        return self.enhancement_net

    @property
    def get_illumination_mlp(self):
        return self.mlp_illumination
    
    @property
    def get_residual_net(self):
        return self.residual_net
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

###

    def init_RT_seq(self, cam_list):
        poses = []
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1))
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)

    def get_RT(self, idx):
        pose = self.P[idx]
        return pose

    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose
    
    def get_closest_RT(self, pose):
        poses_list = self.P
        distances = torch.norm(poses_list[:, 4:] - pose[4:].unsqueeze(0), dim=1)
        index = torch.randint(1, 3, (1,)).item() 
        min_distance_idx = torch.argmin(distances)
        distances[min_distance_idx] = float('inf')
        
        # 选择最近的姿态
        closest_idx = torch.argmin(distances)
        return self.P[closest_idx], closest_idx
#####


    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    # def get_covariance_residual(self, scaling_modifier = 1):
    #     return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation_residual)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_anchor
        print("points number:", xyz.shape[0])
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size # resize to voxel space and remove the repeat parts
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, num_sky_gaussians=0, cameras=None, prune_ratio : float = 0.05,model_path=None, beta=1):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points # 
        os.makedirs( os.path.join(model_path, 'dust3r'), exist_ok=True)
        plot_point_cloud_projection(torch.tensor(pcd.points).cuda(), cameras.copy()[0], os.path.join(model_path, 'dust3r', f"anchor_in_view0_dust3r.png"), alpha=0.2)
        # for i in range(len(cameras)):
        #     plot_point_cloud_projection(torch.tensor(pcd.points).cuda(), cameras.copy()[i], os.path.join(model_path, 'dust3r', f"anchor_in_view{i}_dust3r.png"), alpha=0.2)



        if self.voxel_size <= 0: # auto-obtain the voxel_size
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        ## dust3r + downsampling
        # points = self.voxelize_sample(points, voxel_size=self.voxel_size ) # resize to voxel space, turn to the voxel (N, x, y)
        # down_size = self.voxel_size
        # # points = self.voxelize_sample(points, voxel_size=down_size * 4) #50,4
        # while len(points) > 200000:
        #     down_size *= 1.5  # 逐步增大体素
        #     points = self.voxelize_sample(points, voxel_size=down_size) #50



        ## dust3r+ FPS(Farthest Point Sampling)
        # def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
        #     """
        #     xyz: (N, 3) input point cloud
        #     n_samples: number of points to sample
        #     return: (n_samples,) indices of sampled points
        #     """
        #     N, _ = xyz.shape
        #     centroids = torch.zeros(n_samples, dtype=torch.long, device=xyz.device)
        #     distance = torch.ones(N, device=xyz.device) * 1e10
        #     farthest = torch.randint(0, N, (1,), device=xyz.device).item()
        #     for i in range(n_samples):
        #         centroids[i] = farthest
        #         centroid = xyz[farthest].unsqueeze(0)  # (1, 3)
        #         dist = torch.sum((xyz - centroid) ** 2, dim=1)
        #         mask = dist < distance
        #         distance[mask] = dist[mask]
        #         farthest = torch.max(distance, dim=0)[1].item()
        #     return centroids
        # fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        # original_points_size = fused_point_cloud.shape[0]
        # target_points = int(original_points_size * prune_ratio)

        # if target_points < fused_point_cloud.shape[0]:
        #     print(f"Original point cloud size: {original_points_size}, Target size after FPS: {target_points}")
        #     sampled_indices = farthest_point_sampling(fused_point_cloud, target_points)
        #     fused_point_cloud = fused_point_cloud[sampled_indices]
        #     print(f"Sampled point cloud size: {fused_point_cloud.shape[0]}")


        # LLGIM
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        original_points_size = fused_point_cloud.shape[0]
        print(f"original points size: {original_points_size}")
        # import pdb; pdb.set_trace()
        # 根据距离随机裁剪点云
        tau= 1
        visualize_anchor(fused_point_cloud.detach().cpu().numpy(), os.path.join(model_path, 'anchor_without_prune.png'))
        print("prune_ratio:", prune_ratio)
        while fused_point_cloud.shape[0] > original_points_size * prune_ratio and prune_ratio < 1:  
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
            # tau = torch.max(torch.tensor(1), tau * torch.exp(- torch.tensor(fused_point_cloud.shape[0]/original_points_size)))
            tau *= torch.exp( 1.0 * torch.tensor(beta * fused_point_cloud.shape[0]/original_points_size))
            dist2_threshold = torch.tensor(self.voxel_size * tau)
            print("tau:", tau)
            
            
            # 计算保留概率,距离越小概率越小
            probs = dist2 / dist2_threshold
            probs = torch.clamp(probs, 0.5, 1)
        
            # # 随机采样生成mask
            rand = torch.rand_like(dist2)
            # rand_idx = torch.randint(0, dist2.shape[0], (dist2.shape[0]*9//10,))
            mask = rand < probs
            # mask[rand_idx] = True
            dist2 = dist2[mask]
            fused_point_cloud = fused_point_cloud[mask]
            print(fused_point_cloud.shape[0])
        




        

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        visualize_anchor(fused_point_cloud.detach().cpu().numpy(), os.path.join(model_path, 'anchor.png'))
        if num_sky_gaussians:
            th_cameras = cameras
            skybox, self._sky_distance = get_sky_points(num_sky_gaussians, fused_point_cloud, th_cameras)
            skybox = skybox
            print(f"Adding skybox with {skybox.shape[0]} points")
            fused_point_cloud = torch.cat((fused_point_cloud, skybox), dim=0)
            opacities = torch.cat((opacities, inverse_sigmoid(torch.ones((skybox.shape[0], 1), dtype=torch.float, device="cuda"))), dim=0)

        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda() # use to caculate the position of 3d gaussians
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001) # get the distance of the voxel center and prune the overlapping voxel
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6) 
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        

        if self.use_residual:
            anchors_feat_residual = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
            offsets_residual = torch.zeros((fused_point_cloud.shape[0], self.n_offsets_residual, 3)).float().cuda()
            scales_residual = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6) 


        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        if self.use_residual:
            self._anchor_feat_residual = nn.Parameter(anchors_feat_residual.requires_grad_(True))
            self._offset_residual = nn.Parameter(offsets_residual.requires_grad_(True))
            self._scaling_residual = nn.Parameter(scales_residual.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")




    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")


        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                # {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.mlp_reflectance.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_reflectance"},
                {'params': self.mlp_illumination.parameters(), 'lr': training_args.mlp_color_lr_init , "name": "mlp_illumination"},
                {'params': self.enhancement_net.parameters(), 'lr': training_args.mlp_enhance_lr_init  , "name": "enhancement_net"},
                # {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_residual_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                # {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.mlp_reflectance.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_reflectance"},
                {'params': self.mlp_illumination.parameters(), 'lr': training_args.mlp_color_lr_init , "name": "mlp_illumination"},
                {'params': self.enhancement_net.parameters(), 'lr': training_args.mlp_enhance_lr_init , "name": "enhancement_net"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                # {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.mlp_reflectance.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_reflectance"},
                {'params': self.mlp_illumination.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_illumination"},
                {'params': self.enhancement_net.parameters(), 'lr': training_args.mlp_enhance_lr_init  , "name": "enhancement_net"},
            ]
        if self.use_residual:
            l.append({'params': [self._anchor_feat_residual], 'lr': training_args.feature_lr , "name": "anchor_feat_residual"})
            l.append({'params': [self._offset_residual], 'lr': training_args.offset_lr_init * self.spatial_lr_scale  , "name": "offset_residual"})
            l.append({'params': [self._scaling_residual], 'lr': training_args.scaling_lr, "name": "scaling_residual"})
            l.append({'params': self.residual_net.parameters(), 'lr': training_args.mlp_color_lr_init , "name": "residual_net"})
            l.append({'params': self.mlp_opacity_residual.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity_residual"})
            l.append({'params': self.mlp_cov_residual.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov_residual"})
            
#####
        l_cam = [{'params':[self.P], 'lr':training_args.pose_lr_init, "name": "pose"},]
        
        l += l_cam
#####
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale ,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        # self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
        #                                             lr_final=training_args.mlp_color_lr_final,
        #                                             lr_delay_mult=training_args.mlp_color_lr_delay_mult,
        #                                             max_steps=training_args.mlp_color_lr_max_steps)
        self.mlp_reflectance_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init ,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        self.mlp_illumination_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init ,
                                                    lr_final=training_args.mlp_color_lr_final ,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        self.enhancement_net_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_enhance_lr_init ,
                                                    lr_final=training_args.mlp_enhance_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_residual:
            self.residual_net_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
            
            self.offset_residual_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale * 5,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale * 5,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
            
            self.mlp_opacity_residual_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init ,
                                                        lr_final=training_args.mlp_opacity_lr_final ,
                                                        lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                        max_steps=training_args.mlp_opacity_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_residual_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
            
        self.pose_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_lr_init,
                                                    lr_final=training_args.pose_lr_final,
                                                    lr_delay_mult=training_args.pose_lr_delay_mult,
                                                    max_steps=training_args.pose_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            # if param_group["name"] == "mlp_color":
            #     lr = self.mlp_color_scheduler_args(iteration)
            #     param_group['lr'] = lr
            if param_group["name"] == "mlp_reflectance":
                lr = self.mlp_reflectance_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_illumination":
                lr = self.mlp_illumination_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "enhancement_net":
                lr = self.enhancement_net_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_residual_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group["name"] == "offset_residual":
                lr = self.offset_residual_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group["name"] == "residual_net":
                lr = self.residual_net_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group['name'] == "mlp_opacity_residual":
                lr = self.mlp_opacity_residual_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_residual and param_group['name'] == "mlp_cov_residual":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group['name'] == "pose":
                lr = self.pose_scheduler_args(iteration)
                param_group['lr'] = lr


    def freeze(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] != "enhancement_net" and param_group["name"] != "mlp_reflectance":
                param_group['lr'] = 0

                

    ### need debug
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        if self.use_residual:
            for i in range(self._anchor_feat_residual.shape[1]):
                l.append('r_anchor_feat_residual_{}'.format(i))
            for i in range(self._scaling_residual.shape[1]):
                l.append('r_scale_residual_{}'.format(i))
            for i in range(self._offset_residual.shape[1]*self._offset_residual.shape[2]):
                l.append('r_offset_residual_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filter_3D = self.filter_3D.detach().cpu().numpy()
        if self.use_residual:
            anchor_feat_residual = self._anchor_feat_residual.detach().cpu().numpy()
            scale_residual = self._scaling_residual.detach().cpu().numpy()
            offset_residual = self._offset_residual.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation, filter_3D), axis=1)
        if self.use_residual:
            attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation, filter_3D, anchor_feat_residual, scale_residual, offset_residual), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis].astype(np.float32)
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        if self.use_residual:
            anchor_feat_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("r_anchor_feat_residual")]
            anchor_feat_residual_names = sorted(anchor_feat_residual_names, key = lambda x: int(x.split('_')[-1]))
            anchor_feat_residuals = np.zeros((anchor.shape[0], len(anchor_feat_residual_names)))
            for idx, attr_name in enumerate(anchor_feat_residual_names):
                anchor_feat_residuals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            
            offset_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("r_offset_residual")]
            offset_residual_names = sorted(offset_residual_names, key = lambda x: int(x.split('_')[-1]))
            offset_residuals = np.zeros((anchor.shape[0], len(offset_residual_names)))
            for idx, attr_name in enumerate(offset_residual_names):
                offset_residuals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            offset_residuals = offset_residuals.reshape((offset_residuals.shape[0], 3, -1))
            
            scaling_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("r_scale_residual")]
            scaling_residual_names = sorted(scaling_residual_names, key = lambda x: int(x.split('_')[-1]))
            scaling_residuals = np.zeros((anchor.shape[0], len(scaling_residual_names)))
            for idx, attr_name in enumerate(scaling_residual_names):
                scaling_residuals[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

            

            

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = nn.Parameter(torch.tensor(filter_3D, dtype=torch.float, device="cuda"))

        if self.use_residual:
            self._anchor_feat_residual = nn.Parameter(torch.tensor(anchor_feat_residuals, dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling_residual = nn.Parameter(torch.tensor(scaling_residuals, dtype=torch.float, device="cuda").requires_grad_(True))
            self._offset_residual = nn.Parameter(torch.tensor(offset_residuals, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'residual_net' in group['name'] or \
                'embedding' in group['name'] or \
                'enhancement_net' in group['name'] or \
                'pose' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        if not hasattr(self, 'grad_variance'):
            self.grad_variance = torch.zeros_like(self.offset_gradient_accum)
            self.grad_mean = torch.zeros_like(self.offset_gradient_accum)

        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach() # [N * n_offsets, 1]
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets]) # [N, n_offsets]
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True) # [N, 1]
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1 # add for visiting anchor

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1) # [N * n_offsets]
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask # only update the visiable gaussians in the visiable anchors
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter 
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1 # add for visiting gaussians

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'residual_net' in group['name'] or \
                'enhancement_net' in group['name'] or \
                'embedding' in group['name'] or \
                'pose' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                try:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                except:
                    print(group['name'])
                    import pdb; pdb.set_trace()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_residual:
            self._anchor_feat_residual = optimizable_tensors["anchor_feat_residual"]
            self._scaling_residual = optimizable_tensors["scaling_residual"]
            self._offset_residual = optimizable_tensors["offset_residual"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0: # if increased anchor number is zero, skip to next turn
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0) # add the increased anchor id

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1) # get all gaussians's location
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask] # get selected gaussians's location
            selected_grid_coords = torch.round(selected_xyz / cur_size).int() # get selected guassian's voxel location

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0) # get selected anchor voxel set, which will be used for the generation of new ancher's feature


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096 * 5
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
            
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates] # ues the big grad anchors to grow the new feature of the new anchors

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                if self.use_residual:
                    new_scaling_residual = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size
                    new_scaling_residual = torch.log(new_scaling_residual)
                    new_anchor_feat_residual = self._anchor_feat_residual.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                    new_anchor_feat_residual = scatter_max(new_anchor_feat_residual, inverse_indices.unsqueeze(1).expand(-1, new_anchor_feat_residual.size(1)), dim=0)[0][remove_duplicates] 

                    new_offset_residual = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets_residual,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                if self.use_residual:
                    d["scaling_residual"] = new_scaling_residual
                    d["offset_residual"] = new_offset_residual
                    d["anchor_feat_residual"] = new_anchor_feat_residual

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                if self.use_residual:
                    self._scaling_residual = optimizable_tensors["scaling_residual"]
                    self._offset_residual = optimizable_tensors["offset_residual"]
                    self._anchor_feat_residual = optimizable_tensors["anchor_feat_residual"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, mode="train", phi=0.5):
        # # adding anchors
        if mode =="warmup":
            old_anchor_num = self.anchor_demon.shape[0]
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold * phi).squeeze(dim=1) # choose the neural gaussians with high seen ratio as growing anchor candidates
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)  # choose the anchors with low opacity as prune candidates
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1] # choose the 
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 

        if mode == "warmup":
            unvisibility_mask = (self.anchor_demon == 0).squeeze(dim=1) 
            unvisibility_mask[old_anchor_num:] = False

            print("removed unvisibility anchor: ", sum(unvisibility_mask))
            prune_mask = torch.logical_or(prune_mask, unvisibility_mask)

            
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            # self.mlp_color.eval()
            # color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            # color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            # self.mlp_color.train()

            self.mlp_reflectance.eval()
            reflectance_mlp = torch.jit.trace(self.mlp_reflectance, (torch.rand(1, self.feat_dim + self.reflectance_dist_dim).cuda()))
            reflectance_mlp.save(os.path.join(path, 'reflectance_mlp.pt'))
            self.mlp_reflectance.train()

            self.mlp_illumination.eval()
            illumination_mlp = torch.jit.trace(self.mlp_illumination, (torch.rand(1, self.feat_dim//2+3+self.illumination_dist_dim).cuda()))
            illumination_mlp.save(os.path.join(path, 'illumination_mlp.pt'))
            self.mlp_illumination.train()


            self.enhancement_net.eval()
            enhancement_net = torch.jit.trace(self.enhancement_net, (torch.rand(1, self.feat_dim + self.n_offsets).cuda()))
            enhancement_net.save(os.path.join(path, 'enhancement_net.pt'))
            self.enhancement_net.train()

            if self.use_residual:
                self.residual_net.eval()
                residual_net = torch.jit.trace(self.residual_net, (torch.rand(1, self.feat_dim+3+self.residual_dist_dim + self.appearance_residual_dim).cuda()))
                residual_net.save(os.path.join(path, 'residual_net.pt'))
                self.residual_net.train()

                self.mlp_cov_residual.eval()
                cov_residual_mlp = torch.jit.trace(self.mlp_cov_residual, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
                cov_residual_mlp.save(os.path.join(path, 'cov_residual_mlp.pt'))
                self.mlp_cov_residual.train()

                self.mlp_opacity_residual.eval()
                opacity_residual_mlp = torch.jit.trace(self.mlp_opacity_residual, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
                opacity_residual_mlp.save(os.path.join(path, 'opacity_residual_mlp.pt'))
                self.mlp_opacity_residual.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_residual_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    # 'color_mlp': self.mlp_color.state_dict(),
                    'reflectance_mlp': self.mlp_reflectance.state_dict(),
                    'illumination_mlp': self.mlp_illumination.state_dict(),
                    'enhancement_net': self.enhancement_net.state_dict(),
                    'residual_net': self.residual_net.state_dict(),
                    'cov_residual_mlp': self.mlp_cov_residual.state_dict(),
                    'opacity_residual_mlp': self.mlp_opacity_residual.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_residual_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    # 'color_mlp': self.mlp_color.state_dict(),
                    'reflectance_mlp': self.mlp_reflectance.state_dict(),
                    'illumination_mlp': self.mlp_illumination.state_dict(),
                    'enhancement_net': self.enhancement_net.state_dict(),
                    'residual_net': self.residual_net.state_dict(),
                    'cov_residual_mlp': self.mlp_cov_residual.state_dict(),
                    'opacity_residual_mlp': self.mlp_opacity_residual.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    # 'color_mlp': self.mlp_color.state_dict(),
                    'reflectance_mlp': self.mlp_reflectance.state_dict(),
                    'illumination_mlp': self.mlp_illumination.state_dict(),
                    'enhancement_net': self.enhancement_net.state_dict(),
                    'residual_net': self.residual_net.state_dict(),
                    'cov_residual_mlp': self.mlp_cov_residual.state_dict(),
                    'opacity_residual_mlp': self.mlp_opacity_residual.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            # self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            self.mlp_reflectance = torch.jit.load(os.path.join(path, 'reflectance_mlp.pt')).cuda()
            self.mlp_illumination = torch.jit.load(os.path.join(path, 'illumination_mlp.pt')).cuda()
            self.enhancement_net = torch.jit.load(os.path.join(path, 'enhancement_net.pt')).cuda()
            if self.use_residual:
                self.residual_net = torch.jit.load(os.path.join(path, 'residual_net.pt')).cuda()
                self.mlp_cov_residual = torch.jit.load(os.path.join(path, 'cov_residual_mlp.pt')).cuda()
                self.mlp_opacity_residual = torch.jit.load(os.path.join(path, 'opacity_residual_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_residual_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            # self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            self.mlp_reflectance.load_state_dict(checkpoint['reflectance_mlp'])
            self.mlp_illumination.load_state_dict(checkpoint['illumination_mlp'])
            self.enhancement_net.load_state_dict(checkpoint['enhancement_net'])
            if self.use_residual:
                self.residual_net.load_state_dict(checkpoint['residual_net'])
                self.mlp_cov_residual.load_state_dict(checkpoint['cov_residual_mlp'])
                self.mlp_opacity_residual.load_state_dict(checkpoint['opacity_residual_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_residual_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
        
