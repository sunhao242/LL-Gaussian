"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything


from submodules.StableSR.ldm.util import instantiate_from_config
from submodules.StableSR.ldm.models.diffusion.ddim import DDIMSampler
from submodules.StableSR.ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import cv2
from submodules.StableSR.scripts.util_image import ImageSpliterTh
from pathlib import Path
from submodules.StableSR.scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
import gin


def load_gin_config(gin_file):
    gin.parse_config_file(gin_file)



def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def read_image(im_path):
	im = np.array(Image.open(im_path).convert("RGB"))
	im = im.astype(np.float32)/255.0
	im = im[None].transpose(0,3,1,2)
	im = (torch.from_numpy(im) - 0.5) / 0.5

	return im.cuda()

@gin.configurable
class Config:
    def __init__(self, seed, config, ckpt, ddpm_steps, dec_w, n_samples, vqgan_ckpt, colorfix_type, precision, tile_overlap, upscale, vqgantile_stride,vqgantile_size,input_size,n_iter,C,f):
        self.seed = seed
        self.ckpt = ckpt
        self.ddpm_steps = ddpm_steps
        self.dec_w = dec_w
        self.n_samples = n_samples
        self.vqgan_ckpt = vqgan_ckpt
        self.colorfix_type = colorfix_type
        self.config = config
        self.precision = precision
        self.tile_overlap = tile_overlap
        self.upscale = upscale
        self.vqgantile_stride = vqgantile_stride
        self.vqgantile_size =vqgantile_size
        self.input_size = input_size
        self.n_iter = n_iter
        self.C = C
        self.f = f

def get_SRModel():
	gin.parse_config_file('./configs/config.gin')
	opt = Config()

	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	model.configs = config

	vqgan_config = OmegaConf.load("./submodules/StableSR/configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w




	return model, vq_model, opt

def SD_refine(model, vq_model, input_image, sd_opt):
		# batch_size = opt.n_samples
	with torch.no_grad():
		input_image = (input_image - 0.5) / 0.5
		device = input_image.device
		model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
							linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
		model.num_timesteps = 1000

		sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
		sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

		use_timesteps = set(space_timesteps(1000, [sd_opt.ddpm_steps]))
		last_alpha_cumprod = 1.0
		new_betas = []
		timestep_map = []
		for i, alpha_cumprod in enumerate(model.alphas_cumprod):
			if i in use_timesteps:
				new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
				last_alpha_cumprod = alpha_cumprod
				timestep_map.append(i)
		new_betas = [beta.data.cpu().numpy() for beta in new_betas]
		model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
		model.num_timesteps = 1000
		model.ori_timesteps = list(use_timesteps)
		model.ori_timesteps.sort()
		model = model.to(device)


		precision_scope = autocast if sd_opt.precision == "autocast" else nullcontext
		niqe_list = []
		
		with precision_scope("cuda"):
			with model.ema_scope():
				size_min = min(input_image.size(-1), input_image.size(-2))
				upsample_scale = max(sd_opt.input_size/size_min, sd_opt.upscale)
				input_h, input_w = input_image.shape[2:]
				input_image = F.interpolate(
							input_image,
							size=(int(input_image.size(-2)*upsample_scale),
									int(input_image.size(-1)*upsample_scale)),
							mode='bicubic',
							)
				
				input_image = input_image.clamp(-1, 1)
				im_lq_bs = [input_image, ]  # 1 x c x h x w, [-1, 1]

				im_lq_bs = torch.cat(im_lq_bs, dim=0)
				ori_h, ori_w = im_lq_bs.shape[2:]
				ref_patch=None
				if not (ori_h % 32 == 0 and ori_w % 32 == 0):
					flag_pad = True
					pad_h = ((ori_h // 32) + 1) * 32 - ori_h
					pad_w = ((ori_w // 32) + 1) * 32 - ori_w
					im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
				else:
					flag_pad = False
				
				paded_h = im_lq_bs.shape[2]
				paded_w = im_lq_bs.shape[3]
				
				if im_lq_bs.shape[2] > sd_opt.vqgantile_size or im_lq_bs.shape[3] > sd_opt.vqgantile_size:
					if im_lq_bs.shape[2] < sd_opt.vqgantile_size:
						im_lq_bs = F.pad(im_lq_bs, pad=(0, 0, 0 , sd_opt.vqgantile_size - im_lq_bs.shape[2]), mode='reflect')
					if im_lq_bs.shape[3] < sd_opt.vqgantile_size:
						im_lq_bs = F.pad(im_lq_bs, pad=(0, sd_opt.vqgantile_size - im_lq_bs.shape[3], 0, 0), mode='reflect')
					# import pdb; pdb.set_trace()
					im_spliter = ImageSpliterTh(im_lq_bs, sd_opt.vqgantile_size, sd_opt.vqgantile_stride, sf=1)
					for im_lq_pch, index_infos in im_spliter:
						seed_everything(sd_opt.seed)
						init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  # move to latent space
						text_init = ['']*sd_opt.n_samples
						semantic_c = model.cond_stage_model(text_init)
						noise = torch.randn_like(init_latent)
						# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
						t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
						t = t.to(device).long()
						x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
						# x_T = noise
						samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=im_lq_pch.size(0), timesteps=sd_opt.ddpm_steps, time_replace=sd_opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(sd_opt.input_size/8), tile_overlap=sd_opt.tile_overlap, batch_size_sample=sd_opt.n_samples)
						_, enc_fea_lq = vq_model.encode(im_lq_pch)
						x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
						if sd_opt.colorfix_type == 'adain':
							x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
						elif sd_opt.colorfix_type == 'wavelet':
							x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
						im_spliter.update_gaussian(x_samples, index_infos)
					im_sr = im_spliter.gather()
					# import pdb ; pdb.set_trace()
					if paded_h < sd_opt.vqgantile_size:
						im_sr = im_sr[:,:, :paded_h,:]
					if paded_w < sd_opt.vqgantile_size:
						im_sr = im_sr[:,:,:,:paded_w]
					im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
				else:
					init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))  # move to latent space
					text_init = ['']*sd_opt.n_samples
					semantic_c = model.cond_stage_model(text_init)
					noise = torch.randn_like(init_latent)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
					t = t.to(device).long()
					x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
					# x_T = noise
					samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=im_lq_bs.size(0), timesteps=sd_opt.ddpm_steps, time_replace=sd_opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(sd_opt.input_size/8), tile_overlap=sd_opt.tile_overlap, batch_size_sample=sd_opt.n_samples)
					_, enc_fea_lq = vq_model.encode(im_lq_bs)
					x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
					if sd_opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
					elif sd_opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, im_lq_bs)
					im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)

				if flag_pad:
					im_sr = im_sr[:, :, :ori_h, :ori_w ]


				if upsample_scale > sd_opt.upscale:
					im_sr = F.interpolate(
								im_sr,
								size=(input_h,
										input_w),
								mode='bicubic',
								)
					im_sr = torch.clamp(im_sr, min=0.0, max=1.0)
				

				

				return im_sr.squeeze(0)

