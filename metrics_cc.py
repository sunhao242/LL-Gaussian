import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import normalize_brightness, psnr
from utils.loss_utils import ssim


def read_images(folder, denoise=False, selected_names=None):
    images = []
    names = []
    if selected_names is None:
        source_names = sorted(os.listdir(folder))
    else:
        available_names = {
            Path(name).stem: name
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
        }
        source_names = []
        for selected_name in selected_names:
            matched_name = available_names.get(Path(selected_name).stem)
            if matched_name is None:
                raise FileNotFoundError(
                    f"找不到与 render 对应的 GT：{selected_name}"
                )
            source_names.append(matched_name)

    for name in source_names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue

        image = Image.open(path)
        if denoise:
            image_bgr = np.array(image)[:, :, ::-1]
            image_bgr = cv2.fastNlMeansDenoisingColored(
                image_bgr,
                None,
                h=3,
                hColor=3,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            image = Image.fromarray(image_bgr[:, :, ::-1])

        images.append(tf.to_tensor(image).unsqueeze(0)[:, :3].cuda())
        names.append(name)
    return images, names


def evaluate(render_dir, gt_dir, output=None):
    renders, image_names = read_images(render_dir)
    gts, _ = read_images(gt_dir, denoise=True, selected_names=image_names)

    for index in range(len(renders)):
        if renders[index].shape != gts[index].shape:
            renders[index] = F.interpolate(
                renders[index],
                size=(gts[index].shape[2], gts[index].shape[3]),
                mode="bilinear",
                align_corners=False,
            )

    renders_cc = normalize_brightness(renders, gts)
    metric_values = {
        "SSIM": [],
        "PSNR": [],
        "LPIPS": [],
        "SSIM_CC": [],
        "PSNR_CC": [],
        "LPIPS_CC": [],
        "delta_2000_CC": [],
    }

    for index in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        render = renders[index]
        render_cc = renders_cc[index].float().contiguous()
        gt = gts[index]

        metric_values["SSIM"].append(ssim(render, gt))
        metric_values["PSNR"].append(psnr(render, gt))
        metric_values["LPIPS"].append(lpips(render, gt, net_type="vgg"))
        metric_values["SSIM_CC"].append(ssim(render_cc, gt))
        metric_values["PSNR_CC"].append(psnr(render_cc, gt))
        metric_values["LPIPS_CC"].append(lpips(render_cc, gt, net_type="vgg"))

    results = {
        name: torch.tensor(values).mean().item()
        for name, values in metric_values.items()
    }
    per_view = {
        name: dict(zip(image_names, torch.tensor(values).tolist()))
        for name, values in metric_values.items()
    }

    for name, value in results.items():
        print(f"  {name:<13}: {value:>12.7f}")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump({"mean": results, "per_view": per_view}, file, indent=2)

    return results, per_view


if __name__ == "__main__":
    parser = ArgumentParser(description="计算单场景 render 与 GT 图像指标")
    parser.add_argument("--render", "--render_dir", "-r", dest="render_dir", required=True)
    parser.add_argument("--gt", "--gt_dir", "-g", dest="gt_dir", required=True)
    parser.add_argument("--output", "-o", help="可选的 JSON 输出路径")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    evaluate(args.render_dir, args.gt_dir, args.output)
