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
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_residual import GaussianRasterizationSettings_Residual, GaussianRasterizer_Residual
from diff_gaussian_rasterization_fast import GaussianRasterizationSettings_Fast, GaussianRasterizer_Fast
from scene.gaussian_model import GaussianModel
from utils.pose_utils import get_camera_from_tensor, quadmultiply




def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]


    if pc.use_residual:
        anchor_residual = pc.get_anchor[visible_mask]
        anchor_feat_residual = pc._anchor_feat_residual[visible_mask]
        grid_offsets_residual = pc._offset_residual[visible_mask]
        grid_scaling_residual = pc.get_scaling_residual[visible_mask]
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]
        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    cat_local_view_woview = torch.cat([feat, ob_dist], dim=1) # [N, c+1]
    cat_local_view_woview_wodist = torch.cat([feat], dim=1) # [N, c]

    ## for illumination
    cat_local_view_illumination = torch.cat([feat[:, pc.feat_dim//2:], ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_illumination_wodist = torch.cat([feat[:, pc.feat_dim//2:], ob_view], dim=1) # [N, c+3]



    if pc.use_residual:
        cat_local_view_residual = torch.cat([anchor_feat_residual, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist_residual = torch.cat([anchor_feat_residual, ob_view], dim=1) # [N, c+3]
        if pc.appearance_residual_dim > 0:
            camera_indicies = torch.ones_like(cat_local_view_residual[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
            appearance_residual = pc.get_appearance_residual(camera_indicies)
    # get offset's opacity
    if pc.add_opacity_dist: 
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    if pc.use_residual:
        if pc.add_opacity_dist:
            neural_opacity_residual = pc.get_opacity_residual_mlp(cat_local_view_residual)
        else:
            neural_opacity_residual = pc.get_opacity_residual_mlp(cat_local_view_wodist_residual)
        # neural_opacity_residual = torch.ones_like(neural_opacity_residual) * 0.005
        neural_opacity_residual = neural_opacity_residual.reshape([-1, 1]) 
        mask_residual = (neural_opacity_residual>0.0)
        mask_residual = mask_residual.view(-1)
        opacity_residual = neural_opacity_residual[mask_residual]

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    if pc.use_3D_filter:
        neural_opacity = pc.get_opacity_with_3D_filter(neural_opacity, visible_mask)
        grid_scaling = pc.get_scaling_with_3D_filter(grid_scaling, visible_mask)
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]


    # get offset's illumination
    # if pc.appearance_residual_dim > 0:
    #     # noise = pc.get_noise_net(torch.cat([cat_local_view, appearance], dim=1))
    #     if pc.add_illumination_dist:
    #         illumination = pc.get_illumination_mlp(torch.cat([cat_local_view_illumination, appearance], dim=1))
    #     else:
    #         illumination = pc.get_illumination_mlp(torch.cat([cat_local_view_illumination_wodist, appearance], dim=1))

    # else:
        # noise = pc.get_noise_net(cat_local_view)
    if pc.add_illumination_dist:
        illumination_feat = pc.get_illumination_mlp(cat_local_view_illumination)
        illumination = torch.nn.Sigmoid()(illumination_feat)

    else:
        
        illumination_feat = pc.get_illumination_mlp(cat_local_view_illumination_wodist)
        illumination = torch.nn.Sigmoid()(illumination_feat)
    illumination_enhanced = pc.get_enhancement_net(torch.cat([feat.detach(), illumination_feat.detach()], dim=1))
    if pc.use_residual and pc.appearance_residual_dim>0:
        if pc.add_residual_dist:
            color_residual = pc.get_residual_net(torch.cat([cat_local_view_residual, appearance_residual], dim=1))
        else:
            color_residual = pc.get_residual_net(torch.cat([cat_local_view_wodist_residual, appearance_residual], dim=1))
        color_residual = color_residual.reshape([anchor.shape[0]*pc.n_offsets_residual, 3]) # [mask]
    elif pc.use_residual:
        if pc.add_residual_dist:
            color_residual = pc.get_residual_net(cat_local_view_residual)
        else:
            color_residual = pc.get_residual_net(cat_local_view_wodist_residual)
        color_residual = color_residual.reshape([anchor.shape[0]*pc.n_offsets_residual, 3]) # [mask]
    # get offset's reflectance

    if pc.add_reflectance_dist:
        reflectance = pc.get_reflectance_mlp(cat_local_view_woview)
    else:
        reflectance = pc.get_reflectance_mlp(cat_local_view_woview_wodist)

    # color = illumination.repeat(1, 1, 3) * reflectance
    # color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask] 
    illumination = illumination.reshape([anchor.shape[0]*pc.n_offsets, 1]) # [mask]
    reflectance = reflectance.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]
    illumination_enhanced = illumination_enhanced.reshape([anchor.shape[0]*pc.n_offsets, 3])

    

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
        if pc.use_residual:
            scale_rot_residual = pc.get_cov_residual_mlp(cat_local_view_residual)
            scale_rot_residual = scale_rot_residual.reshape([anchor.shape[0]*pc.n_offsets_residual, 7]) # [mask]
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
        if pc.use_residual:
            scale_rot_residual = pc.get_cov_residual_mlp(cat_local_view_wodist_residual)
            scale_rot_residual = scale_rot_residual.reshape([anchor.shape[0]*pc.n_offsets_residual, 7]) # [mask]
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    if pc.use_residual:
        offsets_residual = grid_offsets_residual.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    # concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    # concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # # concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    # masked = concatenated_all[mask]
    # scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, reflectance, illumination, illumination_enhanced, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]

    scaling_repeat, repeat_anchor, reflectance, illumination, illumination_enhanced, scale_rot, offsets = masked.split([6, 3, 3, 1, 3, 7, 3], dim=-1)
    

    if pc.use_residual:
        concatenated_residual = torch.cat([grid_scaling_residual, anchor_residual], dim=-1)
        concatenated_repeated_residual = repeat(concatenated_residual, 'n (c) -> (n k) (c)', k=pc.n_offsets_residual)
        feat_repeated_residual = repeat(anchor_feat_residual, 'n (c) -> (n k) (c)', k=pc.n_offsets_residual)
        concatenated_all_residual = torch.cat([concatenated_repeated_residual, color_residual, scale_rot_residual, offsets_residual, feat_repeated_residual], dim=-1)# [[6,3],3,7,3,32]
        masked_residual = concatenated_all_residual[mask_residual]
        scaling_repeat_residual, repeat_anchor_residual, color_residual, scale_rot_residual, offsets_residual, feat_repeated_residual = masked_residual.split([6, 3, 3, 7, 3, anchor_feat_residual.shape[1]], dim=-1)
        scaling_residual = scaling_repeat_residual[:, 3:] * torch.sigmoid(scale_rot_residual[:, :3])
        rot_residual = pc.rotation_activation(scale_rot_residual[:, 3:7])
        offsets_residual = offsets_residual * scaling_residual[:,:3]
        xyz_residual = repeat_anchor_residual + offsets_residual
        # try:
        #     feat_downsampled_residual, _, _ = torch.pca_lowrank(torch.abs(feat_repeated_residual), q=3)
        # except:
        #     feat_downsampled_residual = torch.zeros(feat_repeated_residual.shape[0], 3, device=feat_repeated_residual.device)
    else:
        xyz_residual = None
        scaling_residual = None
        rot_residual = None
        opacity_residual = None
        mask_residual = None
        color_residual = None
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    # 方案1：使用更稳定的SVD方法
    # U, S, V = torch.svd(feat_repeated)
    # feat_selected_tmp = torch.matmul(U[:, :1], torch.diag(S[:1]))
    # selected_channels = torch.randperm(feat_repeated.shape[1])[:2]
    # feat_selected = torch.cat([feat_selected_tmp, feat_repeated[:, selected_channels]], dim=1)
    # try:
    #     feat_downsampled, _, _ = torch.pca_lowrank(torch.abs(feat_repeated), q=3)
    # except:
    #     feat_downsampled = torch.zeros(feat_repeated.shape[0], 3, device=feat_repeated.device)

    # feat_selected = feat_repeated.detach()÷
    # feat_downsampled = feat_repeated.detach()

    if is_training:
        return xyz, reflectance, illumination, illumination_enhanced, opacity, scaling, rot, neural_opacity, mask, xyz_residual, color_residual, scaling_residual, rot_residual, opacity_residual
    else:
        return xyz, reflectance, illumination, illumination_enhanced, opacity, scaling, rot, xyz_residual, color_residual, scaling_residual, rot_residual, opacity_residual

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, camera_pose=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_illumination_mlp.training
    # is_enhancing = pc.render_enhancement
        
    if is_training:
        xyz, reflectance, illumination, illumination_enhanced, opacity, scaling, rot, neural_opacity, mask, xyz_residual, color_residual, scaling_residual, rot_residual, opacity_residual = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, reflectance, illumination, illumination_enhanced, opacity, scaling, rot, xyz_residual, color_residual, scaling_residual, rot_residual, opacity_residual = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    # print("ill_shape:", illumination.shape)
    # print("ref_shape:", reflectance.shape)

    illumination = illumination.repeat(1, 3) # align illumination dimension with the reflectance

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # xyz_dist = torch.norm(xyz - viewpoint_camera.camera_center, dim=-1, keepdim=True)
    # xyz_dist = xyz_dist.repeat([1, 3])
    # import pdb;pdb.set_trace()
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

####
    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
####

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=1,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug
    )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = xyz
    gaussians_rot = rot

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points


    # input_concated = torch.cat([reflectance, illumination, feat_repeated.detach(), feat_downsampled.detach()], dim=1)


    # rendered_image, radii, depth_map = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = illumination * reflectance,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)


    rendered_reflectance,radii,depth_map= rasterizer(
        # means3D = xyz,
        means3D = means3D,
        # means2D = screenspace_points,
        means2D = means2D,
        shs = None,
        colors_precomp = reflectance,
        opacities = opacity,
        scales = scaling,
        # rotations = rot,
        rotations = gaussians_rot_trans,
        cov3D_precomp = None)  

    rendered_illumination,_,_ = rasterizer(
        # means3D = xyz,
        means3D = means3D,
        # means2D = screenspace_points,
        means2D = means2D,
        shs = None,
        colors_precomp = illumination,
        opacities = opacity,
        scales = scaling,
        # rotations = rot,
        rotations = gaussians_rot_trans,
        cov3D_precomp = None)

    rendered_illumination_enhanced, _, _= rasterizer(
        # means3D = xyz,
        means3D = means3D.detach(),
        # means2D = screenspace_points,
        means2D = means2D,
        shs = None,
        colors_precomp = illumination_enhanced,
        opacities = opacity,
        scales = scaling,
        # rotations = rot,
        rotations = gaussians_rot_trans,
        cov3D_precomp = None)

    # rendered_feat,_,_ = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = feat_repeated,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)

    # rendered_feat_downsampled,_,_ = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = feat_downsampled,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)
    # rendered_reflectance = rendered_combined[:3,:,:]
    # rendered_illumination = rendered_combined[3:6,:,:]
    # rendered_image = rendered_reflectance * rendered_illumination
    # rendered_feat = rendered_combined[6:-3,:,:]
    # rendered_feat_downsampled = rendered_combined[-3:,:,:]
    if pc.use_residual:
        # raster_settings_residual = GaussianRasterizationSettings_Residual(
        #     image_height=int(viewpoint_camera.image_height),
        #     image_width=int(viewpoint_camera.image_width),
        #     tanfovx=tanfovx,
        #     tanfovy=tanfovy,
        #     bg=bg_color,
        #     scale_modifier=scaling_modifier,
        #     # viewmatrix=viewpoint_camera.world_view_transform,
        #     # projmatrix=viewpoint_camera.full_proj_transform,
        #     viewmatrix=w2c.detach(),
        #     projmatrix=projmatrix.detach(),
        #     sh_degree=1,
        #     # campos=viewpoint_camera.camera_center,
        #     campos=camera_pos.detach(),
        #     prefiltered=False,
        #     debug=pipe.debug
        # )
        raster_settings_residual = GaussianRasterizationSettings_Residual(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            # viewmatrix=view point_camera.world_view_transform,
            # projmatrix=viewpoint_camera.full_proj_transform,
            viewmatrix=w2c,
            projmatrix=projmatrix,
            sh_degree=1,
            # campos=viewpoint_camera.camera_center,
            campos=camera_pos.detach(),
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer_residual = GaussianRasterizer_Residual(raster_settings=raster_settings_residual)
        screenspace_points_residual = torch.zeros_like(xyz_residual, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
        gaussians_xyz_residual = xyz_residual
        gaussians_rot_residual = rot_residual

        xyz_ones_residual = torch.ones(gaussians_xyz_residual.shape[0], 1).cuda().float()
        xyz_homo_residual = torch.cat((gaussians_xyz_residual, xyz_ones_residual), dim=1)
        gaussians_xyz_residual_trans = (rel_w2c @ xyz_homo_residual.T).T[:, :3]
        gaussians_rot_residual_trans = quadmultiply(camera_pose[:4], gaussians_rot_residual)

        means3D_residual = gaussians_xyz_residual_trans
        means2D_residual = screenspace_points_residual
        # input_concated_residual = torch.cat([color_residual, torch.zeros(color_residual.shape[0], input_concated.shape[1]-color_residual.shape[1]-feat_downsampled_residual.shape[1]).cuda().detach(), feat_downsampled_residual.detach()], dim=1)
        # print("reflectance:", reflectance.mean(), "color_residual:", color_residual.mean())
       
        rendered_residual,_ = rasterizer_residual(
            # means3D = xyz_residual,
            # means2D = screenspace_points_residual,
            means3D = means3D_residual,
            means2D = means2D_residual,
            shs = None,
            colors_precomp = color_residual,
            opacities = opacity_residual,
            scales = scaling_residual,
            rotations = gaussians_rot_residual_trans,
            # rotations = rot_residual,
            cov3D_precomp = None)
        
        # rendered_feat_downsampled_residual,_ = rasterizer_residual(
        #     means3D = means3D_residual,
        #     means2D = means2D_residual,
        #     shs = None,
        #     colors_precomp = feat_downsampled_residual,
        #     opacities = opacity_residual,
        #     scales = scaling_residual,
        #     rotations = gaussians_rot_residual_trans,
        #     cov3D_precomp = None)
        
        
        # rendered_residual = rendered_combined_residual[:3,:,:]
        # rendered_feat_downsampled_residual = rendered_combined_residual[6:-3,:,:]
        # print("rendered_reflectance:", rendered_reflectance.mean(), "rendered_residual:", rendered_residual.mean())
        # import pdb; pdb.set_trace()
        if is_training:
            return {"render": rendered_reflectance * rendered_illumination,
                    "render_reflectance":rendered_reflectance,
                    "render_illumination":rendered_illumination,
                    "render_illumination_enhanced":rendered_illumination_enhanced,
                    "render_depth":depth_map,
                    # "render_depth_variance":depth_variance_map,
                    "render_residual":rendered_residual * 0.1 ,
                    # "render_feat":rendered_feat,
                    # "render_feat_downsampled":rendered_feat_downsampled,
                    # "render_feat_downsampled_residual":rendered_feat_downsampled_residual,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "selection_mask": mask,
                    "neural_opacity": neural_opacity,
                    "scaling": scaling,
                    "scaling_residual":scaling_residual
                    }
        else:
            return {"render": rendered_reflectance * rendered_illumination,
                    "render_reflectance":rendered_reflectance,
                    "render_illumination":rendered_illumination,
                    "render_illumination_enhanced":rendered_illumination_enhanced,
                    "render_residual":rendered_residual * 0.1 ,
                    "render_depth":depth_map,
                    # "render_depth_variance":depth_variance_map,
                    # "render_residual":rendered_residual,
                    # "render_feat":rendered_feat,
                    # "render_feat_downsampled":rendered_feat_downsampled,
                    # "render_feat_downsampled_residual":rendered_feat_downsampled_residual,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    }

    # rendered_image, radii, depth_map = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = reflectance * illumination,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)

    # rendered_reflectance,_,_ = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = reflectance,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)
    
    # rendered_illumination,_,_ = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = illumination,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)
    # # import ipdb; ipdb.set_trace()
    # # feat_repeated = torch.sigmoid(feat_repeated)
    # # padding_size = 3 - feat_repeated.shape[1] % 3
    # # if padding_size != 0:
    # #     feat_repeated = torch.cat([feat_repeated, torch.zeros_like(feat_repeated[:, :padding_size])], dim=1)
    # # rendered_feat = None
    # # for i in range(0, feat_repeated.shape[1], 3):
    # rendered_feat,_,_ = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = feat_repeated,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)
    #     # if rendered_feat is None:
    #     #     rendered_feat = rendered_feat_tmp
    #     # else:
    #     #     rendered_feat = torch.cat([rendered_feat, rendered_feat_tmp], dim=0)
    # # rendered_feat = rendered_feat[:feat_repeated.shape[1]-padding_size, :, :]
    # rendered_feat_downsampled,_,_ = rasterizer(
    #     # means3D = xyz,
    #     means3D = means3D,
    #     # means2D = screenspace_points,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = feat_downsampled,
    #     opacities = opacity,
    #     scales = scaling,
    #     # rotations = rot,
    #     rotations = gaussians_rot_trans,
    #     cov3D_precomp = None)
    
    # rendered_residual, radii = rasterizer(
    #     means3D = xyz,
    #     means2D = screenspace_points,
    #     shs = None,
    #     colors_precomp = residual,
    #     opacities = opacity,
    #     scales = scaling,
    #     rotations = rot,
    #     cov3D_precomp = None)
    # # import pdb; pdb.set_trace()
    # if pc.appearance_residual_dim > 0:
    #     camera_indicies = torch.ones_like(rendered_image.permute(1,2,0)[:,:,0], dtype=torch.long, device=rendered_image.device) * viewpoint_camera.uid
    #     # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
    #     appearance = pc.get_appearance(camera_indicies)
    #     noise = pc.get_noise_net(torch.cat([rendered_image.detach(), appearance.permute(2,0,1)], dim=0).unsqueeze(dim=0))
    # else:
    #     noise = pc.get_noise_net(rendered_image.detach()).unsqueeze(dim=0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.

    # import pdb; pdb.set_trace()
    if is_training:
        return {"render": rendered_reflectance * rendered_illumination,
                "render_reflectance":rendered_reflectance,
                "render_illumination":rendered_illumination,
                "render_illumination_enhanced":rendered_illumination_enhanced,
                "render_depth":depth_map,
                # "render_depth_variance":depth_variance_map,
                # "render_residual":rendered_residual
                # "render_feat":rendered_feat,
                # "render_feat_downsampled":rendered_feat_downsampled,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_reflectance * rendered_illumination,
                "render_reflectance":rendered_reflectance,
                "render_illumination":rendered_illumination,
                "render_illumination_enhanced":rendered_illumination_enhanced,
                "render_depth":depth_map,
                # "render_depth_variance":depth_variance_map,
                # "render_residual":rendered_residual,
                # "render_feat":rendered_feat,
                # "render_feat_downsampled":rendered_feat_downsampled,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }




def generate_neural_gaussians_fast(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist


    # cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    # cat_local_view_woview = torch.cat([feat, ob_dist], dim=1) # [N, c+1]
    cat_local_view_woview_wodist = torch.cat([feat], dim=1) # [N, c]

    ## for illumination
    # cat_local_view_illumination = torch.cat([feat[:, pc.feat_dim//2:], ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_illumination_wodist = torch.cat([feat[:, pc.feat_dim//2:], ob_view], dim=1) # [N, c+3]




    # get offset's opacity

    neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)


    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    if pc.use_3D_filter:
        neural_opacity = pc.get_opacity_with_3D_filter(neural_opacity, visible_mask)
        grid_scaling = pc.get_scaling_with_3D_filter(grid_scaling, visible_mask)
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]




    illumination_feat = pc.get_illumination_mlp(cat_local_view_illumination_wodist)
    illumination = torch.nn.Sigmoid()(illumination_feat)
    illumination_enhanced = pc.get_enhancement_net(torch.cat([feat.detach(), illumination_feat.detach()], dim=1))


    # get offset's reflectance

    reflectance = pc.get_reflectance_mlp(cat_local_view_woview_wodist)

    # color = illumination.repeat(1, 1, 3) * reflectance
    # color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask] 
    illumination = illumination.reshape([anchor.shape[0]*pc.n_offsets, 1]) # [mask]
    reflectance = reflectance.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]
    illumination_enhanced = illumination_enhanced.reshape([anchor.shape[0]*pc.n_offsets, 3])

    


    scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    

    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, reflectance, illumination, illumination_enhanced, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]

    scaling_repeat, repeat_anchor, reflectance, illumination, illumination_enhanced, scale_rot, offsets = masked.split([6, 3, 3, 1, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets


    if is_training:
        return xyz, reflectance*illumination_enhanced, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, reflectance*illumination_enhanced, opacity, scaling, rot

def render_fast(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, camera_pose=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = False
    pc.use_residual = False

    xyz, color, opacity, scaling, rot = generate_neural_gaussians_fast(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    # print("ill_shape:", illumination.shape)
    # print("ref_shape:", reflectance.shape)

    # illumination = illumination.repeat(1, 3) # align illumination dimension with the reflectance

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # xyz_dist = torch.norm(xyz - viewpoint_camera.camera_center, dim=-1, keepdim=True)
    # xyz_dist = xyz_dist.repeat([1, 3])
    # import pdb;pdb.set_trace()
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

####
    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    # import pdb; pdb.set_trace()
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
####

    raster_settings = GaussianRasterizationSettings_Fast(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=1,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug
    )


    rasterizer_fast = GaussianRasterizer_Fast(raster_settings=raster_settings)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = xyz
    gaussians_rot = rot

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points

    rendering,_,_= rasterizer_fast(
        # means3D = xyz,
        means3D = means3D,
        # means2D = screenspace_points,
        means2D = means2D,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        # rotations = rot,
        rotations = gaussians_rot_trans,
        cov3D_precomp = None)  
       
    return {"render": rendering}




def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, camera_pose=None):

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

####
    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose

    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
####


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=1,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_anchor


#####
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc.get_anchor.clone()
    gaussians_rot = pc.get_rotation.clone()

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
#####

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
###
    pipe.compute_cov3D_python = False
###
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        scales = pc.get_scaling_with_3D_filter(scales,torch.ones(scales.shape[0]).bool().cuda())
        # rotations = pc.get_rotation
        rotations = gaussians_rot_trans

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
