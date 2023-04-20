import numpy as np
import cv2
import torch as th
from guided_diffusion import dist_util
import torch.distributed as dist
from tqdm import tqdm
import os
from .farl_text import Get_embeds 
from .valdata import  ValData

from torch.utils.data import DataLoader

def create_dir(names):
    for _ in names:
        if(os.path.exists(_)==False):
            os.makedirs(_)

def save_images(images,direcs, name):
    for img,d1 in zip(images,direcs):
        img_name = os.path.join(d1,name[0])
        # print(img_name)
        cv2.imwrite(img_name,img[:,:,::-1])

def diffusion_test(model1, model2,diffusion, args):
        text_embedder = Get_embeds()
        batch_size=1
        val_data = DataLoader(ValData(), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()
        with th.no_grad():
            for data_var in tqdm(val_data):
                model_kwargs={}
                model_kwargs1 = data_var
                for k, v in model_kwargs1.items():
                    if('Index' in k):
                        img_name=v
                    else:
                        model_kwargs[k]= v.to(dist_util.dev())
                text=['This person wears eyeglasses'] #args['text_prompt']
                micro_embed =text_embedder.forward_text(text)
                micro_embed /= micro_embed.norm(dim=-1, keepdim=True)
                rt=th.clone(micro_embed)
                LR=model_kwargs['SR']
                LR1=th.clone(model_kwargs['SR'])
                LR2=th.clone(model_kwargs['SR'])
                LR3=th.clone(model_kwargs['SR'])
                batch_size=1
                sketch=th.clone(model_kwargs['sketch'])
                LR1[:,:3]=0
                LR2[:,3:]=0
                LR3 =LR3*0
                LR_cat=th.cat([LR1,LR2,LR3,LR3],0)
                image_token_net = th.cat([rt*0,rt*0,rt,rt*0],0)
                embed_tokens_net=th.tensor([0,0,1,0]).cuda()
                model_kwargs['image_embed']=image_token_net
                model_kwargs['embed_embed']=embed_tokens_net

                def model_fn(x_t, ts, **kwargs):

                    half = x_t[: batch_size]
                    skts=ts
                    ts=th.cat([ts,ts,ts,ts],0)
                    half1=th.cat([half,half,half, half],0)
                    model_in1=th.cat([half1,LR_cat],1)
                    model_in2=th.cat([half,sketch],1)
                    model_out1 = model1(model_in1,ts, **kwargs)
                    model_out2 = model2(model_in2,skts, **kwargs)
                    model1_eps,model1_var= model_out1[:,:3],model_out1[:,3:]
                    model2_eps,model2_var= model_out2[:,:3],model_out2[:,3:]
                    f_c,h_c,t_c,u_c= th.split(model1_eps, len(model1_eps) // 4, dim=0)
                    _,var1,_,_=th.split(model1_var, len(model1_eps) // 4, dim=0)
                    s_c = model2_eps
                    guidance=[1,1,1.25,0.65]
              
                    cond_eps = guidance[0]*f_c+guidance[1]*h_c+guidance[2]*t_c +guidance[3]*s_c
                    uncond_eps = u_c
                    half_eps = cond_eps- (sum(guidance)-1)*uncond_eps
                    out = th.cat([half_eps,var1], dim=1)
                    return out

                img1 = th.randn((batch_size,3,256,256)).cuda()
                sample = diffusion.p_sample_loop(
                                    model_fn,
                                    (batch_size,3,256,256),
                                    noise=img1,
                                    device='cuda',
                                    clip_denoised=True,
                                    progress=True,
                                    model_kwargs=model_kwargs,
                                    cond_fn=None,
                                    )[:batch_size]
                       
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                blur_image = ((model_kwargs['SR']+1)* 127.5).clamp(0, 255).to(th.uint8)
                # sketch_image=((model_kwargs['sketch']+1)* 127.5).clamp(0, 255).to(th.uint8)
                clean_image = ((model_kwargs['HR']+1)* 127.5).clamp(0, 255).to(th.uint8)
                blur_image= blur_image.permute(0, 2, 3, 1)
                blur_image = blur_image.contiguous().cpu().numpy()
                clean_image= clean_image.permute(0, 2, 3, 1)
                clean_image= clean_image.contiguous().cpu().numpy()
                sketch_image = ((model_kwargs['sketch']+1)* 127.5).clamp(0, 255).to(th.uint8)

                sketch_image= sketch_image.permute(0, 2, 3, 1)
                sketch_image= sketch_image.contiguous().cpu().numpy()
                data_path ='./results/face/'

                i=0
                sample=np.concatenate((sketch_image[i,:,:,:3],blur_image[i,:,:,:3],blur_image[i,:,:,3:],sample[0].cpu().numpy(), clean_image[i]), axis=1)
    
                create_dir([data_path])
                save_images([sample],[data_path],img_name)
    