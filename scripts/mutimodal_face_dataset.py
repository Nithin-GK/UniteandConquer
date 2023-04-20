import argparse
import torch.nn.functional as F

from guided_diffusion import dist_util
from guided_diffusion.resample import create_named_schedule_sampler
from face_utils.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_sketch_model
)
import os

from face_utils.diff_test import diffusion_test



def face_diffusion(text_prompt = None):
    args = create_argparser().parse_args()
    options_diffusion = args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    model, diffusion = sr_create_model_and_diffusion(
        **options_diffusion
    )
    model.to(dist_util.dev())


    sketch_model=create_sketch_model()
    sketch_model.to(dist_util.dev())
    if options_diffusion['use_fp16']:
        model.convert_to_fp16()
        sketch_model.convert_to_fp16()


    model_weights="./weights/model_latest.pt"
    sketch_weights="./weights/model_sketch.pt"
    model.load_state_dict(
        dist_util.load_state_dict(model_weights, map_location="cpu")
    )
    sketch_model.load_state_dict(
        dist_util.load_state_dict(sketch_weights, map_location="cpu")
    )

    diffusion_test(model,sketch_model,diffusion,args)

def create_argparser():
    
    defaults = dict(
        resume_checkpoint="./weights/model_latest.pt",
    )

    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    face_diffusion()
