# import tensorflow
import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

model=ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
model.trainable=False

model=keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary()) ---> To have a look at the Number of Parameters and Output shape.

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array, axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result=result/norm(result)

    return normalized_result


filenames=[]
for file in os.listdir(r"D:\CNN_Project\archive\fashion-dataset\images"):

    filenames.append(os.path.join(file))

# print(filenames[0:10])


for j in range(4):

    feature_list=[]

    if j==0:
        for file in tqdm(filenames[0:10001]):
            feature_list.append(extract_features(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+file, model))
        pickle.dump(feature_list, open(f"embeddings_{j}.pkl", "wb"))

    if j==1:
        for file in tqdm(filenames[10001:20001]):
            feature_list.append(extract_features(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+file, model))
        pickle.dump(feature_list, open(f"embeddings_{j}.pkl", "wb"))

    elif j==2:
        for file in tqdm(filenames[20001:30001]):
            feature_list.append(extract_features(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+file, model))
        pickle.dump(feature_list, open(f"embeddings_{j}.pkl", "wb"))

    if j==3:
        for file in tqdm(filenames[30001:]):
            feature_list.append(extract_features(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+file, model))
        pickle.dump(feature_list, open(f"embeddings_{j}.pkl", "wb"))


print(np.array(feature_list).shape) # Just to check shape of output array.

pickle.dump(filenames, open("filenames.pkl", "wb"))
