"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util
from guided_diffusion.resample import create_named_schedule_sampler
from class_utils.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from text_utils.text_diffusion import create_model_and_diffusion as text_create
from text_utils.text_diffusion import model_and_diffusion_defaults as text_defaults
from text_utils.text_diffusion import model_and_diffusion_defaults_upsampler
from guided_diffusion.diffusion_test import test_diff
def multimodal_diff(text_prompt=None,ImageNet_class=None, n_samples=None):
    param_dict = create_argparser()
    # for _ in param_dict:
    #     print(_)
    dist_util.setup_dist()
    try:
        param_dict['imagenet_class']=int(ImageNet_class)
        param_dict['text_prompt']=text_prompt
        param_dict['n_samples']=int(n_samples)

    except:
        pass
    model1_weights="./weights/64x64_diffusion.pt"
    model2_weights="./weights/base.pt"
    model_sr_weights='./weights/upsample.pt'
    options_model1 = model_and_diffusion_defaults()
    options_model1['use_fp16'] = True
    options_model1['timestep_respacing'] = '100' # use 27 diffusion steps for very fast sampling

    model, diffusion = create_model_and_diffusion(
        **options_model1 
    )
    if options_model1['use_fp16']:
        model.convert_to_fp16()

    model.load_state_dict(
        dist_util.load_state_dict(model1_weights, map_location="cpu")
    )
    model.to(dist_util.dev())

    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = True
    options_up['timestep_respacing'] = '100' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = text_create(**options_up)
    model_up.convert_to_fp16()
    model_up.load_state_dict(
        dist_util.load_state_dict(model_sr_weights, map_location="cpu")
    )
    model_up.to(dist_util.dev())

    text_options = text_defaults()
    model_text,_ =text_create(**text_options)
    model_text.to(dist_util.dev())
    model_text.convert_to_fp16()
    model_text.load_state_dict(
        dist_util.load_state_dict(model2_weights, map_location="cpu")
    )


    model.eval()
    model_text.eval()
    model_up.eval()

    img = test_diff(model,model_text,model_up,diffusion,diffusion_up,**param_dict)
    return img

def create_argparser():
    defaults = dict(
        text_prompt='A yellow flower field',
        imagenet_class=200,
        reliability =0.6,
        guidance = 5,
        n_samples=8,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args=parser.parse_args()
    return args_to_dict(args, defaults.keys())

if __name__ == "__main__":
    multimodal_diff()
