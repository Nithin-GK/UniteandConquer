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

from class_utils.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,

)
from text_utils.text_diffusion import create_model_and_diffusion as text_create
from text_utils.text_diffusion import model_and_diffusion_defaults as text_defaults
from text_utils.text_diffusion import model_and_diffusion_defaults_upsampler
from guided_diffusion.diffusion_test import test_diff

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

class Multimodalgradio:
    def __init__(self):
        options_diffusion =sr_model_and_diffusion_defaults()
        self.face_multimodal,self.face_diffusion = sr_create_model_and_diffusion(**options_diffusion)
        self.face_multimodal.to(dist_util.dev())       
        self.face_multimodal.convert_to_fp16()
        self.load_model(self.face_multimodal,"./weights/model_latest.pt")


        self.sketch_model=create_sketch_model()
        self.sketch_model.to(dist_util.dev())
        self.sketch_model.convert_to_fp16()
        self.load_model(self.sketch_model,"./weights/model_sketch.pt")

        options_model1 = model_and_diffusion_defaults()
        options_model1['use_fp16'] = True
        options_model1['timestep_respacing'] = '100' # use 27 diffusion steps for very fast sampling
        self.class_model, self.class_diffusion = create_model_and_diffusion(
        **options_model1 
        )    
        self.class_model.to(dist_util.dev())
        self.load_model(self.class_model,"./weights/64x64_diffusion.pt")
        self.class_model.convert_to_fp16()


        text_options = text_defaults()
        self.model_text,_ =text_create(**text_options)
        self.model_text.to(dist_util.dev())
        self.model_text.convert_to_fp16()
        self.load_model(self.model_text,"./weights/base.pt")


        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = True
        options_up['timestep_respacing'] = '100' 
        self.model_up, self.diffusion_up = text_create(**options_up)
        self.model_up.convert_to_fp16()
        self.load_model(self.model_up,'./weights/upsample.pt')
        self.model_up.to(dist_util.dev())

        self.face_multimodal.eval()
        self.sketch_model.eval()
        self.class_model.eval()
        self.model_text.eval()
        self.model_up.eval()

    def load_model(self,model,path):
        model.load_state_dict(
        dist_util.load_state_dict(path, map_location="cpu")
        )

    def natural_images(self,text_prompt=None,ImageNet_class=None, n_samples=None):
        param_dict = self.create_argparser()

        try:
            param_dict['imagenet_class']=int(ImageNet_class)
            param_dict['text_prompt']=text_prompt
            param_dict['n_samples']=int(n_samples)

        except:
            pass

        img = test_diff(self.class_model,self.model_text,self.model_up,self.class_diffusion,self.diffusion_up,**param_dict)
        return img


    def face_images(self, Text = None, face_image=None,hair_image=None,sketch_image=None, modalities_use = ['Face_map','Hair_map','Sketch','Text'],num_samples=1):
        args = self.create_argparser()
        args_pass={}
        modalities_use=list_to_bool_list(modalities_use)

        try:
                args_pass['Face_map']=preprocess_image(face_image)
                args_pass['Hair_map']=preprocess_image(hair_image)
                args_pass['Sketch']=preprocess_image(sketch_image)
                args_pass['Text']=Text
                args_pass['num_samples']=num_samples
                args_pass['modalities']=np.array(modalities_use,dtype=bool)

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
        result = diffusion_test(self.face_multimodal,self.sketch_model,self.face_diffusion,**args_pass)
        return result

    def create_argparser(self):
        defaults = dict(
            text_prompt='A yellow flower field',
            imagenet_class=200,
            reliability =0.6,
            guidance = 5,
            n_samples=8,
            face_path='./data/face_map/10008.jpg',
            hair_path='./data/hair_map/10008.jpg',
            sketch_path='./data/sketch/10008.jpg',
            num_samples=1
        )
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        args=parser.parse_args()
        return args_to_dict(args, defaults.keys())