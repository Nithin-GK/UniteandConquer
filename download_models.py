
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
from download_models_func import download_files

download_files()