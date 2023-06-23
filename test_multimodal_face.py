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
        print("path not found")
        exit()
    return image

def list_to_bool_list(modal):
    map_dict =  ['Face_map','Hair_map','Text','Sketch']
    ret_list=[False]*4
    for i in range(len(map_dict)):
        if(map_dict[i] in modal):
            ret_list[i]=True
    return ret_list

class Multimodalface:
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

        self.face_multimodal.eval()
        self.sketch_model.eval()

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
    
def find_modalities_use(path):
    modalities=[]
    if(path[0] is not None):
        modalities.append('Face_map')
    if(path[1] is not None):
        modalities.append('Hair_map')
    if(path[2] is not None):
        modalities.append('Sketch')
    if(path[3] is not None):
        modalities.append('Text')
    return modalities

def list_files(path):
    if path is not None:
        files = os.listdir(path)
        files.sort()
        return files
    

def path_to_data(path):
    if path is not None:
        files = list_files(path)
        return files
    else:
        return None
    

def file_or_none(path1,path2):
        try:
            face_file=os.path.join(path1,path2)
            if os.path.exists(face_file) ==False:
                return None
            else:
                return face_file
        except:
            return None

def paired_data_loader(face_path=None,hair_path=None,text_path=None,sketch_path=None):
    face_files = path_to_data(face_path)
    hair_files = path_to_data(hair_path)
    sketch_files = path_to_data(sketch_path)

    if text_path is not None:
        text_dict={}
        with open(text_path) as f:
            text = f.readlines()
        for _ in text:
            text = _.strip('\n').split(':')
            text_dict[text[0]]=text[1]
    
    paired_data=[]


    for i in range(len(face_path)):
            face_file=file_or_none(face_path,face_path[i])
            hair_file=file_or_none(hair_path,face_path[i])
            sketch_file=file_or_none(sketch_path,sketch_path[i])
            try:
                text=text_dict[face_path[i]]
            except:
                text=None
            data_points=[face_file,hair_file,sketch_file,text]
            modalities=find_modalities_use(data_points)

            paired_data.append([data_points, modalities])

    return paired_data

import argparse

if __name__ == "__main__":
    multimodal = Multimodalface()



    parser = argparse.ArgumentParser(description='Multimodal face generation')
    parser.add_argument('--hair_dir', type=str, default=None, help='Input hair semantic maps')
    parser.add_argument('--face_dir', type=str, default=None,help='Input face semantic maps')
    parser.add_argument('--sketch_dir', type=str, default=None,help='Input sketch')
    parser.add_argument('--text_dir', type=str, default=None,help='Input text conditoning')
    parser.add_argument('--num_samples', type=int, default=1,help='Input face semantic maps')


    args = parser.parse_args()
    data = paired_data_loader(args.hair_dir,args.face_dir,args.text_dir,args.sketch_dir)
    for i in range(len(data)):
        multimodal.face_images(data[i][0][3],data[i][0][0],data[i][0][1],data[i][0][2],data[i][1],args.num_samples)
