"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F
from core.wandb_logger import WandbLogger

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from class_utils.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_text_model,
    create_uncond_model,
)
from text_utils.text_diffusion import create_model_and_diffusion as text_create
from text_utils.text_diffusion import model_and_diffusion_defaults as text_defaults
from text_utils.text_diffusion import model_and_diffusion_defaults_upsampler


from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
# from train_dataset import TrainData
from valdata import  ValData

def main(run):
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    # model_sketch_weights="/cis/home/ngopala2/Works/diffusion_sketch_from_start/diif_code/weights/model014599.pt"

    resume_checkpoint1="./weights/64x64_diffusion.pt"
    resume_checkpoint2="./weights/base.pt"
    resume_checkpoint3='./weights/upsample.pt'
    # resume_checkpoint3="./new_weights_CVPR/ffhq_10m.pt"
    model_class_weights=resume_checkpoint1
    model_text_weights = resume_checkpoint2
    model_upsample_weights = resume_checkpoint3

    # model_text_weights="./new_weights_CVPR/text_celeba/model_latest.pt"
    # model_uncond_weights=resume_checkpoint3

    logger.log("creating model...")
    # model_segment, diffusion = sr_create_model_and_diffusion(
    #     **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    # )
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model_sketch, diffusion = sr_create_model_and_diffusion(
    #     **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    # )

    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = True
    options_up['timestep_respacing'] = '100' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = text_create(**options_up)


    text_options = text_defaults()
    # del text_options["timestep_respacing"]
    model_text,_ =text_create(**text_options)
    model_text.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    gt_dir='/export/io83/data/Nithin/Works/Data/CelebA_sketch_segment/CelebA-HQ-img/'
    gt_mask_dir='/export/io83/data/Nithin/Works/Data/CelebA_sketch_segment/CelebA_mask/'

    # data = load_superres_data()
    val_data = DataLoader(ValData(dataset='ffhq'), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()

    print(args)
    data = load_superres_data(
        gt_dir,
        gt_dir,
        gt_mask_dir,
        args.batch_size,
        image_size=256,
        class_cond=False
    )

    logger.log("Testing...")
    TrainLoop(
        model1=model,
        model2=model_text,
        model3=model_up,
        diffusion=diffusion,
        diffusion2=diffusion_up,
        data=data,
        val_dat=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint1=model_class_weights,
        resume_checkpoint2=model_text_weights,
        resume_checkpoint3=model_upsample_weights,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).test_diffusion(run)
# yes

def load_superres_data(data_dir,gt_dirs, gt_mask_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        gt_dir=gt_dirs,
        gt_mask_dir=gt_mask_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
    )
    for large_batch, model_kwargs in data:
        # model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs

# /media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ECCV_2022/diffusion_ema_rain_imagenet/diffusion_ema_clip/guided-diffusion-main/weights/64_256_upsampler.pt
def create_argparser():
    defaults = dict(
        data_dir='/export/io83/data/Nithin/Works/Data/CelebA_sketch_segment/',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=8,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=200,
        resume_checkpoint="/cis/home/ngopala2/Works/diffusion_sketch_start/weights/64_256_upsampler.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
# /media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ECCV_2022/ddrm-master/models/256x256_diffusion_uncond
# /media/labuser/cb8bb1ad-451a-4aa4-870c-2d3eeafe2525/ECCV_2022/guided_ema/guided_ema_imagenet/guided-diffusion-main/weights/64_256_upsampler.pt
if __name__ == "__main__":
    run=WandbLogger()
    main(run)
