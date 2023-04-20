
import gradio as gr

from scripts.multimodal_sample import multimodal_diff 
from scripts.multimodal_face_sample import face_diffusion
import numpy as np
def greet(name):
    return "Hello " + name + "!"

import json
import os
idx2label = []
cls2label = {}
from scripts.multimodal_gradio import Multimodalgradio
from download_models import download_files

download_files()
uniconquer =Multimodalgradio()

with open("class_utils/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

def class_find(label):
    return idx2label[label]

examples=[["A yellow flower field", 358],
  ["A road leading to mountains", 850],
  ["Photo of a beach", 200],
  ["A wheat field", 291],
  ["A garden of cherry blossom trees", 292]]
# print(os.path.exists("./data/face_map/00008.jpg"))
images=sorted(os.listdir('./data/face_map'))
examples_face=[]
examples_text=["A person with blonde hair","A person with black hair","A person with eyeglasses"]
for _ in images:
    textind=np.random.randint(3)
    text=examples_text[textind]
    examples_face.append(["./data/face_map/"+_, "./data/hair_map/"+_,"./data/sketch/"+_,text])

with gr.Blocks(css=".gradio-container {background-color: white}") as demo:
    gr.Markdown(
            "#                            Unite and Conquer: Plug & Play Multimodal Synthesis using Diffusion Models"
    )
    gr.Markdown(
        """
        Enjoy composite synthesis (Scroll down for examples)

        """
    )
    with gr.Tab("Multimodal Generic"):
        with gr.Row(variant='default'):

            with gr.Column(variant='panel'):
                text_input=gr.Textbox(label="Step 1:Type a text prompt here")
                class_input=gr.Slider(minimum=0,maximum=999,step=1,interactive=True,label="Step 2: Slide to select an ImageNet Class")
                class_name=gr.Textbox(label="Corresponding ImageNet Class name(default: Tibetian Terrier)")
                class_button=gr.Button("Find class")
                sample_input=gr.Slider(minimum=1,maximum=32,step=1,interactive=True,label="How many samples do you need?")

                synth_button=gr.Button("Generate")
            diff_output=gr.Image(shape=(50,50))
        gr.Examples(examples,[text_input,class_input])

        synth_button.click(fn=uniconquer.natural_images,inputs=[text_input,class_input, sample_input],outputs=[diff_output])
        class_button.click(fn=class_find,inputs=[class_input],outputs=[class_name])

    with gr.Tab("Multimodal Face"):
        with gr.Row(variant='default'):

            with gr.Column(variant='panel'):
                text_input_face =   gr.Textbox(label="Type a text prompt here")
                modality_input     =   gr.CheckboxGroup(["Face_map","Hair_map","Text","Sketch"],label="Step 2: Enter modalities required(Always check these boxes)")
                gr.Markdown('Input Images(Check out examples at the footer!)(You need to give all three modes)')
                with  gr.Row(variant='panel'):
                    face_input= gr.Image(shape=(256,256), label='Input face semantic map')
                    hair_input= gr.Image(shape=(256,256), label='Input hair semantic map')
                    sketch_input= gr.Image(shape=(256,256), label='Input sketch map')

                face_synth_button=gr.Button("Generate Face")
            diff_output=gr.Image(shape=(50,50))

        face_synth_button.click(fn=uniconquer.face_images,inputs=[text_input_face,face_input,hair_input,sketch_input,modality_input],outputs=[diff_output])
        # print(examples_face)
        gr.Markdown(
                "###       Examples "
        )


        gr.Examples(examples_face,[face_input,hair_input,sketch_input,text_input_face])



demo.launch(share=True)   
# examples_face_map=['./data/face_map/10008.jpg','./data/face_map/10004.jpg','./data/face_map/10005.jpg']
# examples_hair_map=['./data/hair_map/10008.jpg','./data/hair_map/10004.jpg','./data/hair_map/10005.jpg']
# examples_sketch_map=['./data/sketch/10008.jpg','./data/sketch/10004.jpg','./data/sketch/10005.jpg']
# examples_text=["A person with blonde hair","A person with black hair","A person with eyeglasses"]
        # with gr.Row(variant='panel'):
        #     gr.Examples(examples_face_map,[face_input], label="Face Map")
        #     gr.Examples(examples_hair_map,[hair_input], label="Hair Map")
        #     gr.Examples(examples_sketch_map,[sketch_input], label="Sketch")
        #     gr.Examples(examples_text,[text_input_face], label="Text")