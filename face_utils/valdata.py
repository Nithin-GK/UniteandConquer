
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
# --- Training dataset --- #
import torch as th
import cv2
import sys
sys.path.append('..')

class ValData(data.Dataset):
    def __init__(self, crop_size=[256,256]):
        super().__init__()
        self.face_dir =  'data/face_map/'
        self.hair_dir =  'data/hair_map/'
        self.sketch_dir =  'data/sketch/'

        self.input_images=sorted(os.listdir(self.face_dir))
    def get_images(self, index):

        image_name = self.input_images[index]
        clear_image     =Image.open(os.path.join(self.sketch_dir,image_name)).convert("RGB")
        face_image      =Image.open(os.path.join(self.face_dir,image_name)).convert("RGB")
        hair_image      =Image.open(os.path.join(self.hair_dir,image_name)).convert("RGB")
        sketch_image    =Image.open(os.path.join(self.sketch_dir,image_name)).convert("RGB")
    


        clear_image     = np.array(clear_image).astype(np.float32)/127.5-1.0
        face_image      = np.array(face_image).astype(np.float32)/127.5-1.0
        hair_image      = np.array(hair_image).astype(np.float32)/127.5-1.0
        sketch_image    = np.array(sketch_image).astype(np.float32)/127.5-1.0


        out_dict = {}
        clear_image   = np.transpose(clear_image , [2, 0, 1])
        face_image    = np.transpose(face_image  , [2, 0, 1])
        hair_image    = np.transpose(hair_image  , [2, 0, 1])
        sketch_image  = np.transpose(sketch_image, [2, 0, 1])

        SR=np.concatenate((face_image ,hair_image ),0)

        out_dict={'SR': SR,'HR': clear_image, 'sketch': sketch_image ,'Index': image_name,'embed_token': 0}

        return  out_dict

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_images)
