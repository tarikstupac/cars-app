from fastai import *
from fastai.vision.all import *
import torch
from torchvision import transforms
import urllib.request
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO
from utils import *
import timm
import wwf


def opt_func(ps, lr=defaults.lr): return Lookahead(RAdam(ps, lr=lr))

MODEL_URL = "https://www.dropbox.com/s/18z2ogv2nreu68m/test_model.pkl?dl=1"
urllib.request.urlretrieve(MODEL_URL,"test_model.pkl")

learn = load_learner("test_model.pkl")

st.title("Klasifikacija Auta")

st.write("")

uploaded = st.file_uploader("Uploadaj sliku", type="jpg")

if uploaded is not None:
    img = PILImage.create(uploaded)
    st.image(img, caption='Dodana Slika', use_column_width=True)
    st.write("")
    with st.spinner('Analiziram...'):
        time.sleep(5)
   
    pred_y,pred_idx,probs=learn.predict(img)
    pred_vocab=int(np.argmax(pred_idx))
    prediction=learn.dls.vocab[pred_vocab]
    
    if prediction!="":
        st.success("Auto sa slike je : " + str(prediction))
    
