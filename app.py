import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors



feature_list1=np.array(pickle.load(open("embeddings_0.pkl","rb")))
feature_list2=np.array(pickle.load(open("embeddings_1.pkl","rb")))
feature_list3=np.array(pickle.load(open("embeddings_2.pkl","rb")))
feature_list4=np.array(pickle.load(open("embeddings_3.pkl","rb")))
feature_list=np.concatenate((feature_list1, feature_list2, feature_list3, feature_list4))
filenames=np.array(pickle.load(open("filenames.pkl","rb")))

model=ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
model.trainable=False

model=keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array, axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result=result/norm(result)

    return normalized_result

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def recommend(features, feature_list):

    neighbors=NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices= neighbors.kneighbors([features])
    
    return indices


st.title("Fashion Recommender System")


# 1.) Image Upload by User
uploaded_file=st.file_uploader("Choose An Image")

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file):

        display_image=Image.open(uploaded_file)
        st.image(display_image)

        # 2.) Feature Extraction of Uploaded File
        features=extract_features(os.path.join("uploads",uploaded_file.name), model)

        # 3.) Recommendation
        indices=recommend(features, feature_list)

        # 4.) Showing Recommended Image.
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            img1=Image.open(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+filenames[indices[0][0]])
            st.image(img1)
        with col2:
            img2=Image.open(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+filenames[indices[0][1]])
            st.image(img2)
        with col3:
            img3=Image.open(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+filenames[indices[0][2]])
            st.image(img3)
        with col4:
            img4=Image.open(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+filenames[indices[0][3]])
            st.image(img4)
        with col5:
            img5=Image.open(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+filenames[indices[0][4]])
            st.image(img5)


    else:
        st.header("Some Error Occured in file upload")






