


from dataclasses import field
from random import randint
import torch as th
import numpy as np
import os
import cv2
import numpy as np
from . import dist_util
import torch.distributed as dist
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image

def create_dir(names):
    for _ in names:
        if(os.path.exists(_)==False):
            os.makedirs(_)

def save_images(images,direcs, name):
    sample = ((images[0] + 1) * 127.5).clamp(0, 255)
    grid = make_grid(sample, nrow=8)
    grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))

    count=0
    img_grid_fol = os.path.join(direcs[0],'grid',name[0][:-4])
    if(os.path.exists(img_grid_fol)==False):
        os.makedirs(img_grid_fol)


    img_name = os.path.join(img_grid_fol,str(count)+'.png')
    img_name_sub = os.path.join(img_grid_fol+'_'+str(count)+'.png')

    while os.path.exists(img_name) == True:
        count=count+1
        img_name =os.path.join(img_grid_fol,str(count)+'.png')
        img_name_sub = os.path.join(img_grid_fol+'_'+str(count)+'.png')

    img.save(img_name)
    img.save(img_name_sub)

    non_grid_images_fold = os.path.join(direcs[0],'nongrid',name[0][:-4])

    if(os.path.exists(non_grid_images_fold)==False):
        os.makedirs(non_grid_images_fold)

    count=0

    img_name=os.path.join(non_grid_images_fold , str(count)+'.png')
    while os.path.exists(img_name) == True:
        count=count+1

        img_name = os.path.join(non_grid_images_fold , str(count)+'.png')

    sample_np=sample.contiguous().permute(0, 2, 3, 1).cpu().numpy()

    sample_np=np.uint8(sample_np)
    for i in range(sample.shape[0]):
        sample_save=Image.fromarray(sample_np[i])
        sample_save.save(img_name)
        count=count+1
        img_name = os.path.join(non_grid_images_fold , str(count)+'.png')

    return img


def  get_model_kwargs_from_prompts(model , text_ele, imagenet_class, batch_size,imagenetzero_class=0):
                    tokens = model.tokenizer.encode(text_ele)
                    tokens, mask = model.tokenizer.padded_tokens_and_mask(
                        tokens, 128
                    )
                    full_batch_size = batch_size * 2
                    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
                        [], 128
                    )
                    device='cuda'
                    model_kwargs2 = dict(
                        tokens=th.tensor(
                            [tokens]* batch_size   + [uncond_tokens]* batch_size  , device=device
                        ),
                        mask=th.tensor(
                            [mask] * batch_size  + [uncond_mask]* batch_size   ,
                            dtype=th.bool,
                            device=device,
                        ),
                        y=th.tensor(
                            [imagenet_class]* batch_size +[imagenetzero_class]* batch_size ,
                            dtype=th.int64,
                            device=device,
                        )        
                    )
                    mask=th.tensor(
                            [mask] * batch_size  + [uncond_mask]* batch_size   ,
                            dtype=th.bool,
                            device=device,
                        ) 
                    return model_kwargs2


def test_diff(model1,model2,model3,diffusion1,diffusion2, **args):

            # " A yellow flower field","Road leading into mountains",
                model3.eval()
                model1.eval()
                model2.eval()
                batch_size=args["n_samples"]
                classes2=[200,291,850]
                guidance=th.tensor([5,5]).cuda()
                betas=th.tensor([5.0/9.0,4.0/9.0]).cuda()

                text_ele = args['text_prompt']
                imagenet_class=args['imagenet_class']
                device='cuda'
                # batch_size=1
                indices = list(range(100))[::-1]
                img1 = th.randn((batch_size,3,64,64)).cuda()
                noise_img1=th.clone(th.cat([img1,img1],0))
                noise_up = th.randn((batch_size,3,256,256), device=device)
                model_kwargs_a = get_model_kwargs_from_prompts(model2,text_ele, imagenet_class,batch_size)
                models=[model1,model2]

                def model_fn(x_t, ts, **kwargs):
                    half = x_t[: len(x_t) // 2]
                    combined = th.cat([half, half], dim=0)
                    model_out_dict=[]
                    for model in models:
                        model_out = model(combined, ts, **kwargs)
                        model_out_dict.append(model_out)

                    eps, variances=[],[]
                    for _ in model_out_dict:
                        eps.append(_[:,:3])
                        variances.append(_[:,3:])

                    cond_eps,uncond_eps=0,0
                    for g,_,beta in zip(guidance,eps,betas):
                        c, uc = th.split(_, len(_) // 2, dim=0)
                        cond_eps=cond_eps + c*g
                        uncond_eps=uncond_eps + uc*beta

                    half_eps = cond_eps- (sum(guidance)-1)*uncond_eps
                    rest=variances[0]
                    eps = th.cat([half_eps, half_eps], dim=0)
                    return th.cat([eps, rest], dim=1)



                with th.no_grad():
                    img2=th.clone(noise_img1) #img2_cated                                               
                    model2.del_cache()
                    out = diffusion1.ddim_sample_loop(
                                model_fn,
                                (batch_size*2,3,64,64),
                                noise=img2,
                                device=device,
                                clip_denoised=True,
                                progress=True,
                                model_kwargs=model_kwargs_a,
                                cond_fn=None,
                                eta=0.0
                                )[:batch_size]
              
                
                    model2.del_cache()
                    upsample_temp = 0.997
                    tokens = model3.tokenizer.encode(text_ele)
                    tokens, mask = model3.tokenizer.padded_tokens_and_mask(
                            [],128
                            )
                    model_kwargs = dict(
                    low_res=out[:batch_size],
                    tokens=th.tensor(
                            [tokens] * batch_size, device=device
                            ),
                    mask=th.tensor(
                            [mask] * batch_size,
                            dtype=th.bool,
                            device='cuda',
                            ),
                        )
                    model3.del_cache()
                    up_shape = (batch_size, 3, 256, 256)
                    up_samples = diffusion2.ddim_sample_loop(
                                            model3,
                                            up_shape,
                                            noise=th.clone(noise_up) * upsample_temp,
                                            device=device,
                                            clip_denoised=True,
                                            progress=True,
                                            model_kwargs=model_kwargs,
                                            cond_fn=None,
                                            eta=0.0
                                            )[:batch_size]
                    model3.del_cache()
                                             

                    foldero= './results/text_and_class/'
                    create_dir([foldero])
                    img_name= str(imagenet_class)+'_'+str(text_ele) + '.png'
                    img = save_images([up_samples],[foldero],[img_name])
                    return img

