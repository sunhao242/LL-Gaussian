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
import os
from os import makedirs
import torch

import numpy as np

from pathlib import Path
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import imageio
from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel,render_fast
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.visualize_utils import minmax_normalize, visualize_cmap
from utils.pose_utils import get_tensor_from_camera
import sys
from utils.camera_utils import visualizer, generate_interpolated_path
from scene.dataset_readers import loadCameras
import matplotlib.cm as cm
from time import perf_counter
from utils.loss_utils import l1_plus_loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pose(path, train_cams):
    w2c_list = np.load(path)
    quat_pose = []
    for i in range(len(w2c_list)):
        bb= w2c_list[i]
        bb = get_tensor_from_camera(bb)
        quat_pose.append(bb)
    quat_pose = torch.stack(quat_pose).to('cuda')

    return quat_pose

def save_interpolate_pose(model_path, iter, n_views):

    org_pose = np.load(model_path / f"pose/pose_{iter}.npy")
    
       # 添加相机位置重排序逻辑
    positions = org_pose[:, :3, 3]  # 提取相机位置
    sorted_indices = []
    current_idx = 0  # 从第一个相机开始
    sorted_indices.append(current_idx)
    
    # 贪心算法：每次选择距离当前相机最近的下一个相机
    remaining_indices = set(range(n_views))
    remaining_indices.remove(current_idx)
    k = None
    while remaining_indices:
        current_pos = positions[current_idx]
        # 找到距离当前相机最近的下一个相机
        # import pdb; pdb.set_trace()
        if k is None:
            next_idx = min(remaining_indices, 
                      key=lambda i: np.linalg.norm(positions[i] - current_pos))
        else:
            next_idx = min(remaining_indices, 
                      key=lambda i: np.linalg.norm(positions[i] - current_pos))
        sorted_indices.append(next_idx)
        remaining_indices.remove(next_idx)
        current_idx = next_idx
        
    # 重新排序相机位姿
    org_pose = org_pose[sorted_indices] 
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/poses.png")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):

        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2,:3,:], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/poses_interpolated.png")
    np.save(model_path / f"pose/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)


def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_reflectance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_reflectances")
    render_illumination_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations")
    render_enhanced_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_enhanceds")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
    render_residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_residuals")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(render_reflectance_path, exist_ok=True)
    makedirs(render_illumination_path, exist_ok=True)
    makedirs(render_enhanced_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_residual_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

    gaussians._anchor.requires_grad_(False)
    gaussians._offset.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians.eval()




    from utils.pose_utils import get_tensor_from_camera
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = 50
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))


        camera_tensor_T = camera_pose[-3:].requires_grad_(True)
        camera_tensor_q = camera_pose[:4].requires_grad_(True)
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.003},
            {"params": [camera_tensor_q], "lr": 0.001}
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=num_iter, eta_min=0.0001)
        with tqdm(total=num_iter, desc=f"Tracking Time Step: {idx+1}", leave=True) as progress_bar:
            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            current_min_loss = float(1e20)
            gt = view.original_image[0:3, :, :]
            initial_loss = None

            for iteration in range(num_iter):
                # rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
                render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size, visible_mask=voxel_visible_mask, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))
                rendering = render_pkg["render"]
                black_hole_threshold = 0.0
                mask = (rendering > black_hole_threshold).float()
                loss = torch.abs(l1_plus_loss(rendering, gt) * mask).mean()
                loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if iteration == 0:
                        initial_loss = loss.item()  # Capture initial loss

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_q = camera_tensor_q.clone().detach()
                        candidate_T = camera_tensor_T.clone().detach()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), initial_loss=initial_loss)
                scheduler.step()

            camera_tensor_q = candidate_q
            camera_tensor_T = candidate_T
        with torch.no_grad():
            optimal_pose = torch.cat([camera_tensor_q, camera_tensor_T])
            # print("optimal_pose-camera_pose: ", optimal_pose-camera_pose)
            #rendering_opt = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)["render"]
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size, camera_pose=optimal_pose)
            render_pkg_opt = render(view, gaussians, pipeline, background, kernel_size=kernel_size, visible_mask=voxel_visible_mask, camera_pose=optimal_pose)

        
            
            rendering = torch.clamp(render_pkg_opt["render"], 0.0, 1.0)
            rendering_reflectance = torch.clamp(render_pkg_opt["render_reflectance"], 0.0, 1.0)
            rendering_illumination = torch.clamp(render_pkg_opt["render_illumination"] , 0.0, 1.0)
            rendering_enhanced = torch.clamp(render_pkg["render_illumination_enhanced"] * render_pkg["render_reflectance"], 0.0, 1.0)
            rendering_depth = 1 - minmax_normalize(render_pkg["render_depth"])
            if 'render_residual' in render_pkg:
                rendering_residual = torch.clamp(render_pkg["render_residual"], 0.0, 1.0)
                torchvision.utils.save_image(rendering_residual, os.path.join(render_residual_path, view.image_name + ".png"))



            # gts
            gt = view.original_image[0:3, :, :]
            
            # error maps
            errormap = (rendering - gt).abs()


            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering_reflectance, os.path.join(render_reflectance_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering_illumination, os.path.join(render_illumination_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering_enhanced, os.path.join(render_enhanced_path, view.image_name + ".png"))

            torchvision.utils.save_image(rendering_depth, os.path.join(render_depth_path, view.image_name + ".png"))
            torchvision.utils.save_image(errormap, os.path.join(error_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

            depth_est = 1 - rendering_depth.squeeze().cpu().numpy()
            depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
            depth_est = torch.as_tensor(depth_est).permute(2,0,1)
            torchvision.utils.save_image(depth_est, os.path.join(render_depth_path, 'color_{0:05d}'.format(idx) + ".png"))




def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):


    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_path_enhanced = os.path.join(model_path, name, "ours_{}".format(iteration), "renders(enhanced)")
    render_reflectance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_reflectances")
    render_illumination_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations")
    render_illumination_path_enhanced = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations(enhanced)")
    render_illumination_path_enhance = os.path.join(model_path, name, "ours_{}".format(iteration), "render_illuminations_enhance")
    render_enhanced_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_enhanceds")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
    render_residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_residuals")
    render_residual_path_fast = os.path.join(model_path, name, "ours_{}".format(iteration), "render_residuals(enhanced)")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(render_path_enhanced, exist_ok=True)
    makedirs(render_reflectance_path, exist_ok=True)
    makedirs(render_illumination_path, exist_ok=True)
    makedirs(render_illumination_path_enhanced, exist_ok=True)
    makedirs(render_illumination_path_enhance, exist_ok=True)
    makedirs(render_enhanced_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_residual_path, exist_ok=True)
    makedirs(render_residual_path_fast, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    time_consume = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        
        if name == "interp":
            pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        else:
            pose = gaussians.get_RT(view.uid)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size,camera_pose=pose)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, kernel_size=kernel_size,camera_pose=pose)
        torch.cuda.synchronize(); t1 = time.time()
        time_consume += t1 - t0

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendering_enhance = torch.clamp(render_pkg["render"] * 30, 0.0, 1.0)
        rendering_reflectance = torch.clamp(render_pkg["render_reflectance"], 0.0, 1.0)
        rendering_illumination = torch.clamp(render_pkg["render_illumination"] , 0.0, 1.0)
        rendering_illumination_enhanced = torch.clamp(render_pkg["render_illumination"] * 30 , 0.0, 1.0)
        rendering_illumination_enhance = torch.clamp(render_pkg["render_illumination_enhanced"] , 0.0, 1.0)
        rendering_enhanced = torch.clamp(render_pkg["render_illumination_enhanced"] * render_pkg["render_reflectance"], 0.0, 1.0)
        rendering_depth = 1 - minmax_normalize(render_pkg["render_depth"])
        if "render_residual" in render_pkg:
             rendering_residul_image = torch.clamp(render_pkg["render_residual"] * 30, 0.0, 1.0)
             torchvision.utils.save_image(rendering_residul_image, os.path.join(render_residual_path, view.image_name + ".png"))
             rendering_residul_image_enhance = torch.clamp(render_pkg["render_residual"] * 30, 0.0, 1.0)
             torchvision.utils.save_image(rendering_residul_image_enhance, os.path.join(render_residual_path_fast, view.image_name + ".png"))

        if name != "interp":
            gt = view.original_image[0:3, :, :]
        name_list.append(view.image_name + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_enhance, os.path.join(render_path_enhanced, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_reflectance, os.path.join(render_reflectance_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination, os.path.join(render_illumination_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination_enhance , os.path.join(render_illumination_path_enhance, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination_enhanced, os.path.join(render_illumination_path_enhanced, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_enhanced, os.path.join(render_enhanced_path, view.image_name + ".png"))

        torchvision.utils.save_image(rendering_depth, os.path.join(render_depth_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
        depth_est = 1 - rendering_depth.squeeze().cpu().numpy()
        depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        depth_est = torch.as_tensor(depth_est).permute(2,0,1)
        torchvision.utils.save_image(depth_est, os.path.join(render_depth_path, 'color_{0:05d}'.format(idx) + ".png"))
        if name != "interp":
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

    img_num = idx + 1
    fps = img_num / time_consume
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_optimize, infer_video : bool):
    with torch.no_grad():

        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_residual_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_reflectance_dist, dataset.add_illumination_dist, dataset.add_residual_dist, dataset.use_residual, dataset.use_3D_filter)
        scene = Scene(dataset, gaussians, depth_piror_model=None, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)

    if not skip_test:
        gaussians.init_RT_seq(scene.test_cameras)
        if skip_optimize:
            render_set_optimize(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)

    with torch.no_grad():
        if infer_video :
            gaussians.use_residual = False
            save_interpolate_pose(Path(args.model_path), scene.loaded_iter, len(scene.getTrainCameras()))
            interp_pose = np.load(Path(args.model_path) / 'pose' / 'pose_interpolated.npy')
            viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
            render_set(
                dataset.model_path,
                "interp",
                scene.loaded_iter,
                viewpoint_stack,
                gaussians,
                pipeline,
                background,
                dataset.kernel_size
            )
            image_folder = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/render_enhanced')
            output_video_file = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/interp_enhanced_view.mp4')
            images_to_video(image_folder, output_video_file, fps=30)
            image_folder = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/renders')
            output_video_file = os.path.join(dataset.model_path, f'interp/ours_{scene.loaded_iter}/interp_view.mp4')
            images_to_video(image_folder, output_video_file, fps=30)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--dataset_path", default='None', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_optimize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--infer_video", action="store_true")

    args = get_combined_args(parser)
    if args.dataset_path:
        args.source_path = args.dataset_path
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_optimize, args.infer_video)