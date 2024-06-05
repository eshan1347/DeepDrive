import streamlit as st
import pandas as pd
import numpy as np
# from pred import predict
# from pred import predict0
from pred0 import predict1
from pred0 import predict0
from tempfile import NamedTemporaryFile
import glob
import re

st.title('Road Quality Assesment')
st.markdown('Model to detect faults and check the quality of roads')

# st.markdown()
opt0 = st.selectbox(
   "Select Media Type",
   ("Image", "Video"),
   index=None,
   placeholder="Image",
)

if opt0 == 'Image':
    imgs = st.file_uploader('Upload Road images :',type=['jpg'],accept_multiple_files=False)
# if len(imgs) != 0:
    # for img in imgs:
    if imgs is not None:
        st.image(imgs)

if opt0 == 'Video':
    imgs = st.file_uploader('Upload Road images :',type=['avi'],accept_multiple_files=False)
# if len(imgs) != 0:
    # for img in imgs:
    if imgs is not None:
        st.video(imgs)

option = st.selectbox(
   "Select Model",
   ("Model1", "InstanceModel"),
   index=None,
   placeholder="Model1",
)

st.write('Selected:', option)

def extract_number(folder_name):
    # Using regular expression to extract the numerical part from the folder name
    match = re.search(r'\d+', folder_name)
    if match:
        return int(match.group())
    else:
        return 0

if option == 'Model1':
    model = '/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/Road_ds/best (1).pt'
    model0 = '/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/DATASET_W_w.pt'
else:
    model0 = '/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/Runs/segment/train3/weights/best.pt'

# st.button('Run Model')
# print(f'IMAGES FILE : {imgs}')
if st.button('Run Model'):
    with NamedTemporaryFile(suffix=".jpg") as temp:
        temp.write(imgs.getvalue())
        temp.seek(0)
        roadArea,pts = predict1(model,temp.name)
        cn, path, score , areas = predict0(model0,temp.name, roadArea, pts)
        st.write(cn)
        st.write(f"Road Quality Score: {score}")
    # for i in cn:
    #     st.markdown(f'Classnames: {i}')
# for img in imgs:
    folders = glob.glob('pred/images*')
    # folders=sorted(folders, key = lambda ele: (ele.isnumeric(), int(ele) if ele.isnumeric() else ele))
    folders = sorted(folders, key=extract_number)
    folders1 = glob.glob('pred/RoadImages*')
    folders1 = sorted(folders1, key=extract_number)
    # st.write(folders)
    # num = list(folders[-1])[-1]
    num=re.findall(r'\d+',folders[-1])
    num=int(num[0])
    st.write(num)

    num1=re.findall(r'\d+',folders1[-1])
    num1=int(num1[0])
    st.write(num1)
    st.image(f'/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/pred/RoadImages{num1}/{path.split("/")[-1]}')
    st.image(f'/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/pred/images{num}/{path.split("/")[-1]}')