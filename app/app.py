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
import cv2


def opt_func(ps, lr=defaults.lr): return Lookahead(RAdam(ps, lr=lr))

MODEL_URL = "https://www.dropbox.com/s/18z2ogv2nreu68m/test_model.pkl?dl=1"
if os.path.exists('test_model.pkl') is False :
    urllib.request.urlretrieve(MODEL_URL,"test_model.pkl")

learn = load_learner("test_model.pkl")

#st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache(supress_st_warning = True)
#@st.cache(persist = True)

st.title("Klasifikacija Auta")
st.text('Odaberite jednu od opcija sa strane : Dodaj sliku ili Uslikaj sa web kamere!')
st.write("")

def predict(img):
    st.image(img, caption='Dodana Slika', use_column_width=True)
    st.write("")
    with st.spinner('Analiziram...'):
        time.sleep(5)
   
    pred_y,pred_idx,probs=learn.predict(img)
    pred_vocab=int(np.argmax(pred_idx))
    prediction=learn.dls.vocab[pred_vocab]
    
    if prediction!="":
        st.success("Auto sa slike je : " + str(prediction))

def main():
    menu = ['Dodaj sliku', 'Uslikaj sa web kamere']
    choice = st.sidebar.selectbox('Meni', menu)
    if choice == 'Dodaj sliku':
        uploaded = st.file_uploader("Uploadaj sliku", type="jpg")
        if uploaded is not None:
            img = PILImage.create(uploaded)
            predict(img)

    elif choice == 'Uslikaj sa web kamere':
        img = 0
        cam = cv2.VideoCapture(0)
        webcam_window = st.image([])
        #cv2.namedWindow('Webcam')
        btn = st.button("Uslikaj","three")
        while True :
            ret, frame = cam.read()
            if not ret:
                st.text('Greska : Kamera se ne moze pokrenuti!')
                break
            #cv2.imshow('Webcam', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_window.image(frame)
            #k = cv2.waitKey(1)

            #if k%256 == 32:
            if btn:
                #img_name = 'cam_img.jpg'
                #cv2.imwrite(img_name, frame)
                #temp = cv2.imread(img_name)
                #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                img = PILImage.create(frame)
                break
        cam.release()
        cv2.destroyAllWindows()
        time.sleep(2)
        predict(img)

if __name__ == '__main__':
    main()




    
