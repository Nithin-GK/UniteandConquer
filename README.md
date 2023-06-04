# [CVPR'23] Unite and Conquer: Plug & Play Multi-Modal Synthesis using Diffusion Models

 Paper: [``arXiv``](https://arxiv.org/abs/2211.09120v1)

### [Project Page](https://nithin-gk.github.io/projectpages/Multidiff/index.html) | [Paper](https://arxiv.org/abs/2212.00793) | [Github](https://github.com/Nithin-GK/UniteandConquer) | [Huggingface](https://huggingface.co/spaces/gknithin/MultimodalDiffusion)



### Contributions:

- We propose a diffusion-based solution for image generation under the presence of multimodal priors.
- We tackle the problem of need for paired data for multimodal synthesis by deriving upon the flexible property of diffusion models.
- Unlike existing methods, our method is easily scalable and can be incorporated with off-the-shelf models to add additional constraints

# Simple Instructions for Running


## Environment setup 

```
conda env create -f environment.yml
```
##  Demo 


```
python gradio_set.py

```


Once you perform the above  steps, the models will automatically get downloaded to your Directory. One that is finished, the models will be automatically downloaded you will get a local demo link which can be used to tey the generation models on your own. More details about internal components of the code will be uploaded shortyl


## Citation
5. If you use our work, please use the following citation
```
@article{nair2022unite,
  title={Unite and Conquer: Cross Dataset Multimodal Synthesis using Diffusion Models},
  author={Nair, Nithin Gopalakrishnan and Bandara, Wele Gedara Chaminda and Patel, Vishal M},
  journal={arXiv preprint arXiv:2212.00793},
  year={2022}
}
```

This code is reliant on:
```
https://github.com/openai/guided-diffusion
https://github.com/openai/glide-text2im
https://github.com/FacePerceiver/facer
```
