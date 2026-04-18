# LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis （Accepted by ACM MM 2025）

[[`Project Page`](https://sunhao242.github.io/LL-Gaussian_web.github.io/)] [[`Paper`](https://dl.acm.org/doi/pdf/10.1145/3746027.3755375)] [[`arXiv`](https://arxiv.org/abs/2504.10331)]  [[`Dataset`](https://drive.google.com/file/d/1Y5lhAEXFN0lZDN-ITPPVtjm42-jKk9JR/view?usp=sharing)]

## 🚀 Overview
LL-Gaussian is a novel framework for low-light scene reconstruction and novel view synthesis. It leverages 3D Gaussian Splatting with specialized modules to enhance rendering quality in extreme lighting conditions.
 
## 👨‍💻 Authors

Hao Sun<sup>1,2</sup>, [Fenggen Yu](https://fenggenyu.github.io/)<sup>4</sup>, Huiyao Xu<sup>3</sup>, Tao Zhang<sup>5</sup>, [Changqing Zou](https://changqingzou.weebly.com/)<sup>1,3</sup>  
<sup>1</sup>Zhejiang Lab <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>State Key Lab of CAD&CG, Zhejiang University  
<sup>4</sup>Simon Fraser University <sup>5</sup>Hangzhou Dianzi University


## ✨ Key Features
- 🌙 **Low-Light Gaussian Initialization**: Robust 3D Gaussian initialization without reliance on SfM tools like COLMAP.
- 🔀 **Gaussian Decomposition**: Dual-branch modeling of intrinsic vs. transient scene components.
- 🎥 **Novel View Synthesis in the Dark**: State-of-the-art rendering quality under low-light and nighttime conditions.

## 📌 TODO

- [x] Provide LLRS dataset.
- [x] Provide inference code.
- [x] Provide training code.

---

## 📦 Installation


### 1. Clone repository

```bash
git clone --recursive https://github.com/sunhao242/LL-Gaussian.git 
cd LL-Gaussian
```

### 2. Create environment

```bash
conda create -n llgaussian python==3.10
conda activate llgaussian
pip install -r requirements.txt
```

## 📊 Dataset

- Download the LLRS-sRGB dataset and place it under:

```bash
./dataset/LLRS-sRGB
```

👉 Download from [Link](https://drive.google.com/file/d/1Y5lhAEXFN0lZDN-ITPPVtjm42-jKk9JR/view?usp=sharing).



## ⚡ Quick Inference 

### 1. Download pretrained checkpoint

Download checkpoint from [backup](https://drive.google.com/file/d/1Mf7pG5Lm5N3ybfpgNuvy9PMQgrdgaMVN/view?usp=drive_link).
Place it under:

```bash
./backup
```


### 2. Run inference


```bash
python render.py -m ./backup/LLRS-sRGB/{scene_name}/{XXXX-XX-XX_XX:XX:XX} --dataset_path ./dataset/LLRS-sRGB/{scene_name} --skip_train
```


## 🏋️ Training

Step 1: Download StableSR checkpoints

```bash
./checkpoints/
  ├── vqgan_cfw_00011.ckpt
  ├── stablesr_turbo.ckpt
```

  - Download autoencoder from [Huggingface](https://huggingface.co/Iceclear/StableSR/resolve/main/vqgan_cfw_00011.ckpt) 
  - Download StableSR-Turbo from [Huggingface](https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_turbo.ckpt) 

Step 2: Download the Depth Anything V2 

Download [Depth-Anything-V2-Large](Depth-Anything-V2-Large) to:

```bash
./checkpoints/
```

Step 3: Start training

```bash
bash scripts/single_train.sh
```


## ⚙️ Inference Arguments
| Argument            | Description                             |
| ------------------- | --------------------------------------- |
| `-m / --model_path` | Path to checkpoint                      |
| `--dataset_path`    | Dataset root                            |
| `--skip_train`      | Skip training split rendering           |
| `--skip_test`       | Skip test split rendering               |
| `--skip_optimize`   | Enable test-view rendering branch       |
| `--iteration`       | Load specific iteration (`-1` = latest) |
| `--infer_video`     | Export interpolated video               |




## 📂 Output Structure

```text
<model_path>/
  test/
    ours_<iter>/
      renders/
      render_reflectances/
      render_illuminations/
      render_enhanceds/
      render_depths/
      render_residuals/
      errors/
      gt/
      per_view_count.json
```


## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{sun2025ll,
  title={Ll-gaussian: Low-light scene reconstruction and enhancement via gaussian splatting for novel view synthesis},
  author={Sun, Hao and Yu, Fenggen and Xu, Huiyao and Zhang, Tao and Zou, Changqing},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={4261--4270},
  year={2025}
}
```
## 📜 LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## 🙏 Acknowledgement

We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work.
