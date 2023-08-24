from posixpath import split
from tkinter.tix import Tree
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

def save_images(images,direcs, name_text, modality):
    count=0

    grid_folder = os.path.join('./results/face/grid/', modality)
    nongrid_folder = os.path.join('./results/face/non_grid', modality)

    gridfold = os.path.join(grid_folder,name_text)
    nongridfold = os.path.join(nongrid_folder,name_text)

    if(os.path.exists(gridfold)==False):
        os.makedirs(gridfold)


    if(os.path.exists(nongridfold)==False):
        os.makedirs(nongridfold)

    count=0
    imgname=os.path.join(gridfold,str(count)+'.png')
    imggridname=os.path.join(nongridfold,str(count)+'.png')

    while(os.path.exists(imgname)==True):
        count=count+1
        imgname=os.path.join(gridfold,str(count)+'.png')


    while(os.path.exists(imggridname)==True):
        count=count+1
        imggridname=os.path.join(nongridfold,str(count)+'.png')

    cv2.imwrite(imgname,images[1][:,:,::-1])
    cv2.imwrite(imggridname,images[0][:,:,::-1])


def map_to_map(model_kwargs,key,noise):
    if key in model_kwargs:
        return model_kwargs[key]

    else:
            return noise


def diffusion_test(model1, model2,diffusion, **args):
        text_embedder = Get_embeds()
        batch_size=1
        with th.no_grad():
                model_kwargs={}
                for k, v in args.items():
                    if('Text' in k):
                        text=v       
                    else:
                        text=''
                        model_kwargs[k]= th.tensor(v).to(dist_util.dev())
                micro_embed =text_embedder.forward_text(text)
                micro_embed /= micro_embed.norm(dim=-1, keepdim=True)
                rt=th.clone(micro_embed)

                shape=(1,3,256,256)
                noise=th.zeros(shape).cuda()
         
                face_map=map_to_map(model_kwargs,'Face_map', noise)
                hair_map=map_to_map(model_kwargs,'Hair_map', noise)
                sketch_map=map_to_map(model_kwargs,'Sketch', noise)

                batch_size=args['num_samples']
                hair_input =th.cat([hair_map*0, hair_map],1)
                face_input =th.cat([face_map, face_map*0],1)
                uncond_input =th.cat([face_map*0, hair_map*0],1)

                LR_cat=th.cat([face_input,hair_input,uncond_input], 0 )
                image_token_net = th.cat([rt*0,rt*0,rt],0) 


                LR_cat=LR_cat[args["modalities"][:3]]

                image_token_net = image_token_net[args["modalities"][:3]]
                embed_tokens_net=th.tensor([0,0,1]).cuda()
                embed_tokens_net=embed_tokens_net[args["modalities"][:3]]


                if(args['modalities'][-1] or sum(args['modalities'][:-1])>1.5):
                    LR_cat=th.cat([LR_cat,uncond_input],0)
                    image_token_net=th.cat([image_token_net,rt*0],0)


                model_kwargs['image_embed']=image_token_net
                model_kwargs['embed_token']=embed_tokens_net

                def model_fn(x_t, ts, **kwargs):

                    half = x_t[: batch_size]
                    skts=ts
                    ts=th.cat([ts,ts,ts],0)
                    ts=ts[args["modalities"][:3]]
                    add_val=0
                    if(args['modalities'][-1] or sum(args['modalities'][:-1])>1.5):
                        ts=th.cat([ts,skts],0)
                        add_val=1
                    half1=th.cat([half]*(sum(args['modalities'][:3])+add_val),0)
                    model_in1=th.cat([half1,LR_cat],1)
                    model_in2=th.cat([half,sketch_map],1)
                    model_out1 = model1(model_in1,ts, **kwargs)
                    model1_eps,model1_var= model_out1[:,:3],model_out1[:,3:]

                    eps_split= th.split(model1_eps, 1, dim=0)
                    var_split=th.split(model1_var, 1, dim=0)
                    s_c=0
                    guidance=np.array([1.0]*sum(args['modalities'][:-1])+[0.60])

                    if(args['modalities'][-1]):
                        model_out2 = model2(model_in2,skts, **kwargs)
                        model2_eps,model2_var= model_out2[:,:3],model_out2[:,3:]
                        s_c = model2_eps
                        # print("high")
                    else:
                        guidance=[1.0]*sum(args['modalities'])+[0]
                        
                    if(sum(args['modalities'][:-1])>=3):
                    # if((np.sum(guidance[:-1]))==0):
                        guidance[-2]=1.25
                    elif(sum(args['modalities'][:-1])>0 and args['modalities'][-1]):
                        guidance=np.array([1.20]*sum(args['modalities'][:-1])+[0.60])

                    if(sum(args['modalities'][:-1])==0):
                        guidance[-1]=1.0


                    # if(sum(args['modalities'][:3])<2.5 and sum(args['modalities'][-1])==1 ):
                    # # if((np.sum(guidance[:-1]))==0):
                    #     guidance[-2]=1

                    cond_eps=0
                    for eps, weight in zip(eps_split[:-1],guidance[:-1]):
                        cond_eps=cond_eps+weight*eps


                    cond_eps = cond_eps + guidance[-1]*s_c
                    uncond_eps = eps_split[-1]
                    half_eps = cond_eps- (sum(guidance)-1)*uncond_eps
                    out = th.cat([half_eps,var_split[0]], dim=1)
                    return out

                img1 = th.randn((batch_size,3,256,256)).cuda()
                sample = diffusion.ddim_sample_loop(
                                    model_fn,
                                    (batch_size,3,256,256),
                                    noise=img1,
                                    device='cuda',
                                    clip_denoised=True,
                                    progress=True,
                                    model_kwargs=model_kwargs,
                                    cond_fn=None,
                                    eta=0.0
                                    )[:batch_size]
                       
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                blur_image = ((hair_map+1)* 127.5).clamp(0, 255).to(th.uint8)
                # sketch_image=((model_kwargs['sketch']+1)* 127.5).clamp(0, 255).to(th.uint8)
                clean_image = ((face_map+1)* 127.5).clamp(0, 255).to(th.uint8)
                blur_image= blur_image.permute(0, 2, 3, 1)
                blur_image = blur_image.contiguous().cpu().numpy()
                clean_image= clean_image.permute(0, 2, 3, 1)
                clean_image= clean_image.contiguous().cpu().numpy()
                sketch_image = ((sketch_map+1)* 127.5).clamp(0, 255).to(th.uint8)

                sketch_image= sketch_image.permute(0, 2, 3, 1)
                sketch_image= sketch_image.contiguous().cpu().numpy()
                data_path ='./results/face/'

                i=0 
                arr=sketch_image[0]*0
                if(args['modalities'][-1]):
                    arr=np.concatenate((arr,sketch_image[i,:,:,:3]),axis=1)
                if(args['modalities'][0]):
                    arr=np.concatenate((arr,blur_image[i,:,:,:3]),axis=1)
                if(args['modalities'][1]):
                    arr=np.concatenate((arr,clean_image[i,:,:,:3]),axis=1)
                # print(arr.shape)
                sample_save=np.concatenate((arr,sample[0].cpu().numpy()), axis=1)
                # print(sample.shape)
                sample_save=sample_save[:,256:,:]
                sample=sample[0].cpu().numpy()

                create_dir([data_path])
                modality=find_modalities(args['modalities'])
                save_images([sample_save, sample],data_path,text, modality)

                return sample
    

def find_modalities(use):
        modality=""
        if(use[0]==True):
            modality=modality+"Face_"

        if(use[1]==True):
            modality=modality+"Hair_"
   
        if(use[2]==True):
            modality=modality+"Text_"
        
        if(use[3]==True):
            modality=modality+"Sketch_"
    
        modality=modality[:-1]
        if(use[0] and use[1] and use[2] and use[3] ):
            modality='all'
        return modality

