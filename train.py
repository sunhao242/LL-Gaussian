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
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from time import perf_counter
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import sys
import torchvision.transforms.functional as tf

sys.modules['torchvision.transforms.functional_tensor'] = tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("./submodules/StableSR")
sys.path.append("./submodules/Depth-Anything-V2")

# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_plus_loss, L_Smooth, L_Illu, L_Gray, L_Depth_similarity, L_Reflectance_Smooth, L_Depth_Smooth, pearson_depth_loss
from gaussian_renderer import prefilter_voxel, render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, Camera_Reprojection, Camera_Reprojection_inverse
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.visualize_utils import minmax_normalize, visualize_camera_trajectories, visualize_anchor_with_camera, visualize_heatmap, plot_point_cloud_projection, visualize_cmap
import numpy as np
import cv2
from utils.pose_utils import save_pose, load_pose
from torchvision import transforms
import matplotlib.cm as cm

from depth_anything_v2.dpt import DepthAnythingV2
from utils.StableSR_utlis import get_SRModel,SD_refine
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


def depth_piror_Model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model_encoder = 'vitl'
    depth_anything = DepthAnythingV2(**model_configs[model_encoder])
    depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{model_encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to('cuda').eval()
    return depth_anything



def depth_piror_generator(model, scene):
    depth_piror_dict = dict()
    for camera in scene.getTrainCameras().copy():
        gt_path = os.path.join(args.source_path,'images',camera.image_name+'.png')
        gt_image = cv2.imread(gt_path)
        depth_piror = model.infer_image(gt_image).unsqueeze(0)
        idx = camera.uid
        depth_piror_dict[idx] = depth_piror
    return depth_piror_dict

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)
    log_dir = pathlib.Path(__file__).parent.resolve()
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    print('Backup Finished!')

def get_closest_uid(cameras, uid):
    other_cameras = [(cam.uid, cam.camera_center) for cam in cameras if cam.uid != uid]
    camera_uids, camera_centers = zip(*other_cameras)
    distances = torch.norm(torch.stack(camera_centers) - cameras[uid].camera_center, dim=1)
    return camera_uids[torch.argmin(distances).item()]


def setup_closest_pose_dic(cameras, closest_pose_dic):
    for cam in cameras:
        closest_idx = get_closest_uid(cameras, cam.uid)
        closest_pose_dic[cam.uid] = closest_idx
    return closest_pose_dic

class LinearDecayWeight:
    def __init__(self, 
                 initial_weight: float = 1.0, 
                 final_weight: float = 0.5, 
                 total_steps: int = 10000):
        self.initial = initial_weight
        self.final = final_weight
        self.total_steps = total_steps
        self.decay_rate = (initial_weight - final_weight) / total_steps
    
    def __call__(self, step: int):
        weight = self.initial - self.decay_rate * step
        return max(weight, self.final)  # 确保不低于最终值


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, mode="train"):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_residual_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_reflectance_dist, dataset.add_illumination_dist, dataset.add_residual_dist, dataset.use_residual, dataset.use_3D_filter)
    depth_piror_model = depth_piror_Model()
    if mode == "warmuped":
        scene = Scene(dataset, gaussians, depth_piror_model, ply_path=ply_path, shuffle=False, load_iteration=-1 , only_ply=True)
        gaussians.train()
    elif mode == "train" or mode == "warmup":
        scene = Scene(dataset, gaussians, depth_piror_model, ply_path=ply_path, shuffle=False)
    depth_piror_dict = scene.depth_piror_dict
    gaussians.training_setup(opt)

    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + '/pose', exist_ok=True)
    save_pose(scene.model_path + '/pose' + "/pose_org.npy", gaussians.P, train_cams_init)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras)
    ## setup the closest pose dictionary    
    closest_pose_dic = {}
    setup_closest_pose_dic(scene.getTrainCameras().copy(), closest_pose_dic)

####
    if args.optim_pose == False:
        gaussians.P.requires_grad_(False)
####
    if mode=="warmup":
        gaussians.P.requires_grad_(False)

    weight_scheduler = LinearDecayWeight(initial_weight=2, final_weight=0.5,total_steps=opt.iterations)
    weight_scheduler2 = LinearDecayWeight(initial_weight=5e-4, final_weight=1e-3,total_steps=opt.update_until)

    if mode == "warmup":
        gaussians.P.requires_grad_(False)
        os.makedirs( os.path.join(scene.model_path, 'view'), exist_ok=True)
        # for i in range(len(scene.getTrainCameras())):
        #     torchvision.utils.save_image(scene.getTrainCameras().copy()[i].original_image * 20, os.path.join(scene.model_path, 'view', f"view{i}.png"))

        visualize_camera_trajectories(scene.getTrainCameras().copy(), scene.getTestCameras().copy (), os.path.join(scene.model_path, "camera_pose_downsampled.png"))
        visualize_anchor_with_camera(scene.getTrainCameras().copy(), scene.getTrainCameras().copy(), gaussians.get_anchor, os.path.join(scene.model_path, "anchor_with_camera_downsampled.png"))
        os.makedirs( os.path.join(scene.model_path, 'fitered'), exist_ok=True)
        plot_point_cloud_projection(gaussians._anchor, scene.getTrainCameras().copy()[0], os.path.join(scene.model_path, 'fitered', f"anchor_in_view0.png"))
        # for i in range(len(scene.getTrainCameras())):
        #     plot_point_cloud_projection(gaussians._anchor, scene.getTrainCameras().copy()[i], os.path.join(scene.model_path, 'fitered', f"anchor_in_view{i}.png"))

    if mode == "train" or mode == "warmuped":
        visualize_camera_trajectories(scene.getTrainCameras().copy(), scene.getTestCameras().copy (), os.path.join(scene.model_path, "camera_pose_warmuped.png"))
        visualize_anchor_with_camera(scene.getTrainCameras().copy(), scene.getTrainCameras().copy(), gaussians.get_anchor, os.path.join(scene.model_path, "anchor_with_camera_downsampled.png"))
        os.makedirs( os.path.join(scene.model_path, 'warmuped'), exist_ok=True)
        plot_point_cloud_projection(gaussians._anchor, scene.getTrainCameras().copy()[0], os.path.join(scene.model_path, 'warmuped', f"anchor_in_view0.png"))
        # for i in range(len(scene.getTrainCameras())):
        #     plot_point_cloud_projection(gaussians._anchor, scene.getTrainCameras().copy()[i], os.path.join(scene.model_path, 'warmuped', f"anchor_in_view{i}.png"))
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    enhanced_image_dict = {}
    refined_image_dict = {}


    cams = scene.getTrainCameras().copy()
    means = torch.zeros(3, device="cuda")
    for cam in cams:
        means += cam.original_image.mean(dim=(1,2))
    mean = (means / len(cams)).mean()
    enhance_ratio = int(0.45/mean)
    first_iter += 1

    ## diffusion prior setup
    diff_path = os.path.join(dataset.source_path, "diffusion_prior_" + str(enhance_ratio))
    os.makedirs(diff_path, exist_ok=True)
    if os.listdir(diff_path):  # Check if the directory is not empty
        print(f"[INFO] Loading existing diffusion prior images from {diff_path}")
        for path_name in os.listdir(diff_path):
            uid = int(path_name.split('.')[0])
            refined_image_dict[uid] = transforms.ToTensor()(Image.open(os.path.join(diff_path, path_name)).convert("RGB"))
    else:
        cams = scene.getTrainCameras().copy()
        print(f"[INFO] Loading SD models...")
        sd_model, vq_model, sd_opt = get_SRModel()
        print(f"[INFO] SD models loaded!")

        print("SD rendering progress")
        progress_bar = tqdm(cams, desc="SD rendering progress")

        for cam in progress_bar:
            uid = cam.uid
            input_image = cam.original_image.cuda() * enhance_ratio
            # # Save input image for debugging
            # if not os.path.exists("./input_image"):
            #     os.makedirs("./input_images")
            # torchvision.utils.save_image(input_image, os.path.join("input_images", f"input_image_{uid}.png"))

            # Process the image
            input_image = input_image.unsqueeze(0)
            output = SD_refine(sd_model, vq_model, input_image, sd_opt)
            output_image = output.squeeze(0)
            refined_image_dict[cam.uid] = output_image
            # Save refined image
            torchvision.utils.save_image(output_image, os.path.join(diff_path, f"{uid}.png"))
            # # Save refined image for debugging
            # if not os.path.exists("./refined_image"):
            #     os.makedirs("./refined_images")
            # torchvision.utils.save_image(output_image, os.path.join("refined_images", f"refined_image_{uid}.png"))

            # Update progress bar
            progress_bar.set_description(f"SD rendering progress (UID: {uid})")

        # Clean up
        sd_model.to('cpu')
        del sd_model  
        del vq_model  
        del sd_opt  
        progress_bar.close()
          
    timing_stats = {}
    total_start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background,  scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None
        t0 = time.time()
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        pose = gaussians.get_RT(viewpoint_cam.uid) 
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        t1 = time.time()
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background, dataset.kernel_size, camera_pose=pose)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, visible_mask=voxel_visible_mask, retain_grad=retain_grad, camera_pose=pose)  
        timing_stats['render_time'] = timing_stats.get('render_time', 0) + (time.time() - t1)

        t1 = time.time()
        reflectance_image, illumination_image, illumination_enhanced_image, depth_image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity= render_pkg["render_reflectance"], render_pkg["render_illumination"], render_pkg["render_illumination_enhanced"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
        

        gt_image = viewpoint_cam.original_image.cuda()
        depth_piror_norm = depth_piror_dict[viewpoint_cam.uid]

        if mode == "warmup":
            image_tmp = torch.clamp(reflectance_image * illumination_image, 0.0, 1.0)
        else:
            if "render_residual" in render_pkg:
                scaling_residual = render_pkg["scaling_residual"]
                residual_image = render_pkg["render_residual"]
                image_tmp = torch.clamp(reflectance_image * illumination_image + residual_image, 0.0, 1.0) # edit
            else:
                residual_image = torch.zeros_like(gt_image) 
                image_tmp = torch.clamp(reflectance_image * illumination_image, 0.0, 1.0)
        timing_stats["data_time"] = timing_stats.get("data_time", 0) + (time.time() - t1)
        t2 = time.time()

        Ll1_value = l1_plus_loss(image_tmp, gt_image, phi=0.5/255)
        Ll1 = torch.abs(Ll1_value).mean()
        L_smooth =  L_Smooth(illumination_image, gt_image, kernel_size=9) * 1e-3
        L_illu = L_Illu(gt_image, illumination_image) 
        L_depth_similarity = (L_Depth_similarity(1 - minmax_normalize(depth_image).squeeze(0), depth_piror_norm.squeeze(0), 128, 0.5) ) * 0.15
        
        if FUSED_SSIM_AVAILABLE:
            ssim_loss = fused_ssim((reflectance_image * illumination_image).unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_loss = ssim(reflectance_image * illumination_image, gt_image)
        ssim_loss = 1.0 - ssim_loss
        scaling_reg = scaling.prod(dim=1).mean()

        if torch.isnan(scaling_reg) or torch.isinf(scaling_reg):
            print("Warning: scaling_reg is nan or inf")
            print("scaling_reg:", scaling_reg.item())

        if mode == "warmup":
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + L_illu + 0.01 * scaling_reg 
            if iteration >= opt.update_from:
                loss += L_smooth * 0.1 +  L_depth_similarity 
        else:
            loss = (1.0 - opt.lambda_dssim ) * Ll1 + opt.lambda_dssim *  ssim_loss + L_illu + 0.01 * scaling_reg  

            
            if iteration >= opt.update_from:
                loss +=  L_smooth + L_depth_similarity

            L_diff = 0
            if iteration >= opt.update_from:
                L_degree = torch.abs((illumination_enhanced_image.mean(0) - torch.clamp(illumination_image.mean(0).detach() * enhance_ratio, 0, 1))).mean() * 0.2 + torch.abs(illumination_enhanced_image.mean() - illumination_image.mean().detach() * enhance_ratio) * 0.05
                L_smooth_enhancement = L_Smooth(illumination_enhanced_image/enhance_ratio, gt_image, kernel_size=9) * 5e-4
                loss += L_degree + L_smooth_enhancement
            if iteration >= opt.update_from * 2:
                L_diff =  torch.abs(illumination_enhanced_image * reflectance_image.detach() - refined_image_dict[viewpoint_cam.uid].cuda()).mean() + torch.abs(illumination_enhanced_image.detach() * reflectance_image - refined_image_dict[viewpoint_cam.uid].cuda()).mean() * 0.2
                loss += L_diff
            if dataset.use_residual:
                scaling_residual_reg = scaling_residual.prod(dim=1).mean()
                L_residual_reg = torch.mean(residual_image) * weight_scheduler(iteration)  
                loss += L_residual_reg + 0.05 * scaling_residual_reg 
 
        timing_stats['loss_time'] = timing_stats.get('loss_time', 0) + (time.time() - t2)


        t1 = time.time()
        loss.backward()
        timing_stats['backward_time'] = timing_stats.get('backward_time', 0) + (time.time() - t1)

        t1 = time.time()
        iter_end.record()
        
        with torch.no_grad():

            if mode=="warmup" or not dataset.use_residual:
                wandb.log({'loss':loss, 'iteration':iteration})
            else:
                wandb.log({'loss':loss, 'iteration':iteration})
            if (iteration - 1) % 600 == 0:
                gt_image = torch.clamp(gt_image * enhance_ratio, 0.0, 1.0)
                image = torch.clamp(image_tmp * enhance_ratio, 0.0, 1.0)
                enhanced_image = torch.clamp(reflectance_image * illumination_image * enhance_ratio, 0.0, 1.0)
                illumination_image = torch.clamp(illumination_image * enhance_ratio, 0.0, 1.0)
                if not dataset.use_residual or mode=="warmup" : residual_image = torch.zeros_like(image)
                residual_image = torch.clamp(residual_image * enhance_ratio, 0.0, 1.0)
                enhanced_image_pil = torchvision.transforms.ToPILImage()(enhanced_image)
                

                wandb.log({'loss':loss, 'iteration':iteration,
                        'gt_image':wandb.Image(torchvision.transforms.ToPILImage()(gt_image)),
                        'image':wandb.Image(torchvision.transforms.ToPILImage()(image)),
                        'depth':wandb.Image(1-minmax_normalize(depth_image)),
                        'residual_image':wandb.Image(torchvision.transforms.ToPILImage()(residual_image)),
                        'illumination':wandb.Image(torchvision.transforms.ToPILImage()(illumination_image)),
                        'reflectance':wandb.Image(torchvision.transforms.ToPILImage()(reflectance_image)),
                        'depth_piror_image':wandb.Image(torchvision.transforms.ToPILImage()(depth_piror_norm)),
                        'clear_image':wandb.Image(enhanced_image_pil),
                        'illumination_enhanced':wandb.Image(torchvision.transforms.ToPILImage()(illumination_enhanced_image)),
                        'image_enhanced':wandb.Image(torchvision.transforms.ToPILImage()(illumination_enhanced_image * reflectance_image)),
                        'refined_image':wandb.Image(torchvision.transforms.ToPILImage()(refined_image_dict[viewpoint_cam.uid].cuda())),
                })

            # debug
            if iteration % 600 == 0:
                print("reflectance_image", reflectance_image.mean())
                print("illumination_image", illumination_image.mean())
                print("residual_image", residual_image.mean())
                print("image", image.mean())
                print("gt_image", gt_image.mean())

            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                if mode=="warmup" or not dataset.use_residual:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}","L1": f"{Ll1:.{4}f}", "L_smooth": f"{L_smooth:.{5}f}","L_depth_similarity": f"{L_depth_similarity:.{5}f}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}","L1": f"{Ll1:.{4}f}", "L_illu": f"{L_illu:.{5}f}", "L_smooth": f"{L_smooth:.{5}f}","L_depth_similarity": f"{L_depth_similarity:.{5}f}", "L_residual_reg": f"{L_residual_reg:.{5}f}"})
                progress_bar.update(10)
            

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save"
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, dataset.kernel_size)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + '/pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            timing_stats['log_time'] = timing_stats.get('log_time', 0) + (time.time() - t1)
            # densification
            t1 = time.time()
            if mode == "warmup":
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, mode=mode)
                        gaussians.compute_3D_filter(cameras=trainCameras)
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
            else:
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, mode=mode)
                        gaussians.compute_3D_filter(cameras=trainCameras)
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
                if iteration % 100 == 0 and iteration > opt.update_until:
                    if iteration < opt.iterations - 100:
                        # don't update in the end of training
                        gaussians.compute_3D_filter(cameras=trainCameras)

            
            timing_stats['densification_time'] = timing_stats.get('densification_time', 0) + (time.time() - t1)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
           
            timing_stats['total_time'] = timing_stats.get('total_time', 0) + (time.time() - t0)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, kernel_size=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                ##
                    pose = scene.gaussians.get_RT(viewpoint.uid)
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs,  kernel_size=kernel_size, camera_pose=pose)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask,  kernel_size=kernel_size, camera_pose=pose)["render"], 0.0, 1.0)
                ##
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            # errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()



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
        num_iter = 500
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        camera_tensor_T = camera_pose[-3:].requires_grad_(True)
        camera_tensor_q = camera_pose[:4].requires_grad_(True)
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.0003},
            {"params": [camera_tensor_q], "lr": 0.0001}
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

            depth_est = rendering_depth.squeeze().cpu().numpy()
            depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
            depth_est = torch.as_tensor(depth_est).permute(2,0,1)
            torchvision.utils.save_image(depth_est, os.path.join(render_depth_path, 'color_{0:05d}'.format(idx) + ".png"))


    with torch.no_grad():
        print(">>> Calculate FPS: ")
        fps_list = []
        for _ in range(1000):
            start = perf_counter()
            _ = render(view, gaussians, pipeline, background, kernel_size=kernel_size, camera_pose=optimal_pose)
            end = perf_counter()
            fps_list.append(end - start)        
        fps_list.sort()
        fps_list = fps_list[100:900]
        fps = 1 / (sum(fps_list) / len(fps_list))
        print(">>> FPS = ", fps)
        with open(f"{model_path}/total_fps.json", 'a') as fp:
            json.dump(f'{fps}', fp, indent=True)
            fp.write('\n')

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
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
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        ####
        pose = gaussians.get_RT(view.uid)
        ####
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, kernel_size=kernel_size, camera_pose=pose)
        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size, visible_mask=voxel_visible_mask, camera_pose=pose)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendering_reflectance = torch.clamp(render_pkg["render_reflectance"], 0.0, 1.0)
        rendering_illumination = torch.clamp(render_pkg["render_illumination"] , 0.0, 1.0)
        rendering_enhanced = torch.clamp(render_pkg["render_illumination_enhanced"] * render_pkg["render_reflectance"], 0.0, 1.0)
        rendering_depth = 1 - minmax_normalize(render_pkg["render_depth"])
        if 'render_residual' in render_pkg:
            rendering_residual = torch.clamp(render_pkg["render_residual"], 0.0, 1.0)
            torchvision.utils.save_image(rendering_residual, os.path.join(render_residual_path, view.image_name + ".png"))
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append(view.image_name + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_reflectance, os.path.join(render_reflectance_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_illumination, os.path.join(render_illumination_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering_enhanced, os.path.join(render_enhanced_path, view.image_name + ".png"))

        torchvision.utils.save_image(rendering_depth, os.path.join(render_depth_path, view.image_name + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        depth_est = rendering_depth.squeeze().cpu().numpy()
        depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        depth_est = torch.as_tensor(depth_est).permute(2,0,1)
        torchvision.utils.save_image(depth_est, os.path.join(render_depth_path, 'color_{0:05d}'.format(idx) + ".png"))

        per_view_dict[view.image_name + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():

        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                dataset.appearance_residual_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_reflectance_dist, dataset.add_illumination_dist, dataset.add_residual_dist, dataset.use_residual, dataset.use_3D_filter)
        scene = Scene(dataset, gaussians, depth_piror_model=None, load_iteration=iteration, shuffle=False)
        # gaussians.train()
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

    if not skip_test:
        gaussians.init_RT_seq(scene.test_cameras)
        render_set_optimize(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)
        visible_count = 0
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        # if wandb is not None:
        #     wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
        #     wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
        #     wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000,10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8000,10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--optim_pose", action="store_true", default=True)
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        # import pdb;pdb.set_trace()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            # mode="online",
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training

    if args.warmup:
        import copy
        args_warmup = copy.deepcopy(args)
        args_warmup.iterations = 2000
        args_warmup.save_iterations.append(args_warmup.iterations)
        print(args.warmup)
        args_warmup.start_stat = 500
        args_warmup.update_from = 1500
        args_warmup.offset_lr_init = 0.0001
        args_warmup.opacity_lr = 0.1
        args_warmup.mlp_color_lr_init = 0.1
        args_warmup.densify_grad_threshold = 0.0002 
        args_warmup.min_opacity = 0.1
        args_warmup.success_threshold = 0.8
        training(lp.extract(args_warmup), op.extract(args_warmup), pp.extract(args_warmup), dataset,  args_warmup.test_iterations, args_warmup.save_iterations, args_warmup.checkpoint_iterations, args_warmup.start_checkpoint, args_warmup.debug_from, wandb, logger, mode="warmup")
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args_warmup.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path, mode="warmuped")

    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, mode="train")
    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
