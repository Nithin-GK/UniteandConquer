import numpy as np
import cv2
import torch as th
from . import dist_util, logger
import torch.distributed as dist
from tqdm import tqdm
def diffusion_test(val_data,model, diffusion,run,text_embedder):
        number=0
        batch_id1=0
        with th.no_grad():
            for data_var in tqdm(val_data):
                batch_id1=batch_id1+1
                model_kwargs={}
                model_kwargs1 = data_var
                ct=0         
                for k, v in model_kwargs1.items():
                    if('Index' in k):
                        img_name=v
                    else:
                        model_kwargs[k]= v.to(dist_util.dev())
                LR=model_kwargs['HR']
                text=['This person has big nose and has bags under eyes']
                micro_embed =text_embedder.forward_text(text)
                micro_embed /= micro_embed.norm(dim=-1, keepdim=True)
                model_kwargs['image_embed']=micro_embed
                sample = diffusion.p_sample_loop(
                                    model,
                                    LR.shape,
                                    clip_denoised=True,
                                    model_kwargs=model_kwargs,
                                    ct=ct
                    )
                                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(all_samples, sample)  # gather not supported with NCCL
    
                blur_image = ((model_kwargs['SR']+1)* 127.5).clamp(0, 255).to(th.uint8)
                clean_image = ((model_kwargs['HR']+1)* 127.5).clamp(0, 255).to(th.uint8)
                blur_image= blur_image.permute(0, 2, 3, 1)
                blur_image = blur_image.contiguous().cpu().numpy()
                clean_image= clean_image.permute(0, 2, 3, 1)
                clean_image= clean_image.contiguous().cpu().numpy()
                data_path ='./ddpm_ffhq/'

                for i in range(LR.shape[0]):
                    # print(blur_image[i,:,:,:3].shape,blur_image[i,:,:,3:].shape,sample[i].cpu().numpy().shape, clean_image[i].shape)
                    img_disp=np.concatenate((blur_image[i,:,:,:3],blur_image[i,:,:,3:],sample[i].cpu().numpy(), clean_image[i]), axis=1)
                    if dist.get_rank() == 0:
                        cv2.imwrite(data_path+img_name[i],img_disp[:,:,::-1])
                        run.log_image(
                                                        'validation_' +str(batch_id1), 
                                                        np.concatenate((blur_image[i,:,:,:3],blur_image[i,:,:,3:],sample[i].cpu().numpy(), clean_image[i]), axis=1)
                        )