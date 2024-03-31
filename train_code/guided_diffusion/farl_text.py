import torch
# from CLIP import clip
import clip
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
import torchvision.transforms.functional as Func
# from optimization.augmentations import ImageAugmentations_paired
# from optimization.augmentations import ImageAugmentations

import torch.nn as nn



def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance



class Get_embeds(nn.Module):
    def __init__(self):
        super(Get_embeds, self).__init__()
        self.model = (
            clip.load("ViT-B/16", device='cuda', jit=False)[0].eval().requires_grad_(False)
        )
        self.model = self.model.cuda()
        farl_state=torch.load('/data/ngopala2/works/CVPR_multimodal_ours/celeba_text2im/weights_cond/FaRL-Base-Patch16-LAIONFace20M-ep64.pth') # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
        self.model.load_state_dict(farl_state["state_dict"],strict=False)
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.clip_size =self.model.visual.input_resolution
        self.model.eval()


    def encode_image(self,x_in):
        x_in = x_in.add(1).div(2)
        norm_xin = self.clip_normalize(x_in)
        farl_in = Func.resize(norm_xin, [self.clip_size, self.clip_size])
        with torch.no_grad():
            image_embeds = self.model.encode_image(farl_in).float()
        return image_embeds

    def forward_text(self,text_in):
        text = clip.tokenize(text_in).cuda()
        with torch.no_grad():
            enc_text= self.model.encode_text(text).float()
        return enc_text