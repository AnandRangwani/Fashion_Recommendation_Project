import keras
import pickle 
import numpy as np
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

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


img = image.load_img("D:\CNN_Project\sample\images (2).jpeg", target_size=(224,224)) 
img_array = image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array, axis=0)
preprocessed_img=preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result=result/norm(result)

neighbors=NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
neighbors.fit(feature_list)
distances, indices= neighbors.kneighbors([normalized_result])
print(indices)


for file in indices[0]:
    temp_img=cv2.imread(r"D:\CNN_Project\archive\fashion-dataset\images"+"\\"+filenames[file])
    cv2.imshow("output", cv2.resize(temp_img, (512,512)))

    cv2.waitKey(0)







