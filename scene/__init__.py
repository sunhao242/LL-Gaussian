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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import cv2
from utils.visualize_utils import minmax_normalize, visualize_anchor
from utils.pose_utils import load_pose

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, depth_piror_model, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, only_ply=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.depth_piror_model = depth_piror_model

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(os.path.join(args.source_path, "sparse"))
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"


        self.gaussians.set_appearance_residual(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            for camera in self.test_cameras[resolution_scale]:
                print(f'camera.image_name: {camera.image_name}')

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if not only_ply:
                self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            if os.path.exists(os.path.join(self.model_path, "pose", f"pose_{self.loaded_iter}.npy")):
                pose = load_pose(os.path.join(self.model_path, "pose", f"pose_{self.loaded_iter}.npy"))
                self.gaussians.P = pose
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, num_sky_gaussians=1000, cameras=self.getTestCameras(),model_path=self.model_path, prune_ratio=args.prune_ratio, beta=args.beta)
            print(f'self.gaussians.get_anchor.shape: {self.gaussians.get_anchor.shape}')
        
####
            self.gaussians.init_RT_seq(self.train_cameras)
####
        if self.depth_piror_model:
            self.depth_piror_dict = self.depth_piror_generator(args.source_path)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def depth_piror_generator(self, source_path):
        depth_piror_dict = dict()
        for camera in self.getTrainCameras().copy():
            gt_path = os.path.join(source_path, 'images', camera.image_name + '.*')
            import glob
            gt_path = glob.glob(gt_path)[0]  # 获取匹配的第一个文件路径
            gt_image = cv2.imread(gt_path)
            depth_piror = self.depth_piror_model.infer_image(gt_image).unsqueeze(0)
            idx = camera.uid
            depth_piror_dict[idx] = minmax_normalize(depth_piror)
            # depth_piror_dict[idx] = depth_piror
        return depth_piror_dict
    