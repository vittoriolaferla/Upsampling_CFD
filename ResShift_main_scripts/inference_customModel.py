#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.utils_ResShift.util_opts import str2bool
from basicsr.basicsr_resshift.utils.download_util import load_file_from_url

_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    'deblur': 4,
    }
_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
    'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
    'deblur': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_deblur_s4.pth',
     }

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-vp", "--vel_path", type=str, default="", help="Vel path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to your custom model checkpoint.")
    parser.add_argument("--config_path", type=str, default=None, help="Path to your custom configuration file.")
    parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="v1",
            choices=["v1", "v2", "v3"],
            help="Checkpoint version.",
            )
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="realsr",
            choices=['realsr', 'bicsr', 'inpaint_imagenet', 'inpaint_face', 'faceir', 'deblur'],
            help="Chopping forward.",
            )
    args = parser.parse_args()

    return args


def get_configs(args):
    ckpt_dir = Path('./weights')

    if args.config_path is not None:
        configs = OmegaConf.load(args.config_path)
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        configs.model.ckpt_path = str(ckpt_path)
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'

    # Adjust chopping parameters
    args.chop_size = max(args.chop_size, 12)
    args.chop_stride = args.chop_stride if args.chop_stride > 0 else args.chop_size

    return configs, args.chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            sf=args.scale,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 12),
            )


    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            args.vel_path,
            mask_path=None,
            bs=args.bs,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
