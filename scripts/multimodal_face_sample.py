import argparse
import torch.nn.functional as F
from PIL import Image
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
import numpy as np
from face_utils.diff_test import diffusion_test
import torch as th

def preprocess_image(image):
    try:
        image = np.array(image).astype(np.float32)/127.5-1.0
        image = np.transpose(image , [2, 0, 1])
        image = np.expand_dims(image,0)
    except:
        cantwork
    return image

def list_to_bool_list(modal):
    map_dict =  ['Face_map','Hair_map','Text','Sketch']
    ret_list=[False]*4
    for i in range(len(map_dict)):
        if(map_dict[i] in modal):
            ret_list[i]=True
    return ret_list


def face_diffusion(Text = None, face_image=None,hair_image=None,sketch_image=None, modalities_use = ['Face_map','Hair_map','Sketch','Text'],num_samples=1):
    args = create_argparser()
    options_diffusion =sr_model_and_diffusion_defaults()
    model, diffusion = sr_create_model_and_diffusion(**options_diffusion)
    model.to(dist_util.dev())
    args_pass={}
    modalities_use=list_to_bool_list(modalities_use)
    try:
        # if(face_image is not None):
            args_pass['Face_map']=preprocess_image(face_image)
            args_pass['Hair_map']=preprocess_image(hair_image)
            args_pass['Sketch']=preprocess_image(sketch_image)
            args_pass['Text']=Text
            args_pass['num_samples']=num_samples
            args_pass['modalities']=np.array(modalities_use,dtype=bool)

        # else:
        #     gotoexcept
    except:

        face_image =  Image.open(args['Face_path']).convert("RGB")
        hair_image =  Image.open(args['Hair_path']).convert("RGB")
        sketch_image =  Image.open(args['sketch_path']).convert("RGB")
        args_pass['Text']= "This person wears eyeglasses."
        args_pass['Face_map']=preprocess_image(face_image)
        args_pass['Hair_map']=preprocess_image(hair_image)
        args_pass['Sketch']=preprocess_image(sketch_image)
        args_pass['num_samples']=num_samples
        args_pass['modalities']=np.array(modalities_use,dtype=bool)

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
    result = diffusion_test(model,sketch_model,diffusion,**args_pass)
    return result
def create_argparser():
    defaults = dict(
        face_path='./data/face_map/10008.jpg',
        hair_path='./data/hair_map/10008.jpg',
        sketch_path='./data/sketch/10008.jpg',
        num_samples=1
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args=parser.parse_args()
    return args_to_dict(args, defaults.keys())



if __name__ == "__main__":
    face_diffusion()
