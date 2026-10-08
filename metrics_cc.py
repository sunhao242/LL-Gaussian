"""
Demo script for computing PSNR_CC, SSIM_CC, and LPIPS_CC metrics.

The "_CC" suffix means metrics are computed after color-correcting the
rendered images to match the ground-truth color distribution.

Usage:
    python demo_metrics_cc.py --render_dir <path_to_renders> --gt_dir <path_to_gts>

Both directories should contain matching image files (png/jpg) sorted by name.
"""

import os
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as tf
import pyiqa


def color_correct(img: torch.Tensor, ref: torch.Tensor, num_iters: int = 5, eps: float = 10 / 255) -> torch.Tensor:
    """Warp ``img`` to match the colors in ``ref``.

    Args:
        img: Input image tensor of shape (N, C, H, W).
        ref: Reference image tensor of shape (N, C, H, W).
        num_iters: Number of iterations for color correction.
        eps: Small value for numerical stability.
    """
    if img.shape[1] != ref.shape[1]:
        raise ValueError(f"Channel mismatch: img has {img.shape[1]}, ref has {ref.shape[1]}")

    N, C = img.shape[:2]
    img_mat = img.reshape(N, C, -1).transpose(1, 2)
    ref_mat = ref.reshape(N, C, -1).transpose(1, 2)

    def is_unclipped(z):
        return (z >= eps) & (z <= (1 - eps))

    mask0 = is_unclipped(img_mat)

    corrected_mats = []
    for n in range(N):
        img_mat_n = img_mat[n]
        ref_mat_n = ref_mat[n]
        mask0_n = mask0[n]

        for _ in range(num_iters):
            a_mat = []
            for c in range(C):
                a_mat.append(img_mat_n[:, c:c + 1] * img_mat_n[:, c:])
            a_mat.append(img_mat_n)
            a_mat.append(torch.ones_like(img_mat_n[:, :1]))
            a_mat = torch.cat(a_mat, dim=-1)

            warp = []
            for c in range(C):
                b = ref_mat_n[:, c]
                mask = mask0_n[:, c] & is_unclipped(img_mat_n[:, c]) & is_unclipped(b)
                ma_mat = torch.where(mask.unsqueeze(1), a_mat, torch.zeros_like(a_mat))
                mb = torch.where(mask, b, torch.zeros_like(b))

                try:
                    w = torch.pinverse(ma_mat) @ mb.unsqueeze(1)
                except RuntimeError:
                    w = torch.zeros((a_mat.shape[1], 1), device=img.device)
                    if mask.any():
                        w[0] = mb[mask].mean() / ma_mat[mask].mean()

                warp.append(w)

            warp = torch.cat(warp, dim=1)
            img_mat_n = torch.matmul(a_mat, warp).clamp(0, 1)

        corrected_mats.append(img_mat_n)

    corrected_mat = torch.stack(corrected_mats, dim=0)
    corrected_img = corrected_mat.transpose(1, 2).reshape(img.shape).contiguous()
    return corrected_img


def load_images(directory, device):
    """Load all images from a directory as (1, C, H, W) tensors."""
    tensors = []
    names = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = Image.open(os.path.join(directory, fname)).convert("RGB")
        tensors.append(tf.to_tensor(img).unsqueeze(0).to(device))
        names.append(fname)
    return tensors, names


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR_CC, SSIM_CC, LPIPS_CC")
    parser.add_argument("--render_dir", type=str, required=True, help="Directory of rendered images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of ground-truth images")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    renders, render_names = load_images(args.render_dir, device)
    gts, gt_names = load_images(args.gt_dir, device)

    if len(renders) != len(gts):
        print(f"Warning: render count ({len(renders)}) != gt count ({len(gts)}). Using min.")
        n = min(len(renders), len(gts))
        renders, gts = renders[:n], gts[:n]
        render_names = render_names[:n]

    print(f"Loaded {len(renders)} image pairs")

    ssim_metric = pyiqa.create_metric("ssim", device=device)
    psnr_metric = pyiqa.create_metric("psnr", device=device)
    lpips_metric = pyiqa.create_metric("lpips", device=device)

    psnrs, ssims, lpips_vals = [], [], []
    psnrs_cc, ssims_cc, lpips_cc = [], [], []

    for i in tqdm(range(len(renders)), desc="Evaluating"):
        render, gt = renders[i], gts[i]

        if render.shape != gt.shape:
            h = min(render.shape[2], gt.shape[2])
            w = min(render.shape[3], gt.shape[3])
            render = render[:, :, :h, :w]
            gt = gt[:, :, :h, :w]

        render_cc = color_correct(render, gt)

        ssims.append(ssim_metric(render, gt).item())
        psnrs.append(psnr_metric(render, gt).item())
        lpips_vals.append(lpips_metric(render, gt).item())

        ssims_cc.append(ssim_metric(render_cc, gt).item())
        psnrs_cc.append(psnr_metric(render_cc, gt).item())
        lpips_cc.append(lpips_metric(render_cc, gt).item())

    def avg(vals):
        return sum(vals) / len(vals)

    print("\n" + "=" * 50)
    print(f"{'Metric':<15} {'Before CC':>12} {'After CC':>12}")
    print("=" * 50)
    print(f"{'PSNR':<15} {avg(psnrs):>12.4f} {avg(psnrs_cc):>12.4f}")
    print(f"{'SSIM':<15} {avg(ssims):>12.4f} {avg(ssims_cc):>12.4f}")
    print(f"{'LPIPS':<15} {avg(lpips_vals):>12.4f} {avg(lpips_cc):>12.4f}")
    print("=" * 50)

    if args.output:
        import json
        results = {
            "PSNR": avg(psnrs), "PSNR_CC": avg(psnrs_cc),
            "SSIM": avg(ssims), "SSIM_CC": avg(ssims_cc),
            "LPIPS": avg(lpips_vals), "LPIPS_CC": avg(lpips_cc),
            "num_images": len(renders),
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
