import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
#descriptions = pickle.load(open('descriptions.pkl', 'rb'))  


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')


def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5 , algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# Load image data from CSV
data = pd.read_csv('processed_dataset.csv')
image_paths = [os.path.join('images', f) for f in data['image_filename']]
descriptions = data['image_url'].tolist()
product_display_names = data['productDisplayName'].tolist()  # List of names



num_cols = min(2, len(image_paths))
cols1 = st.columns(num_cols)
for i, (image_path, description, product_display_name) in enumerate(zip(image_paths, descriptions, product_display_names)):
    
    image = Image.open(image_path)
    with cols1[min(1,i % num_cols)]:

        st.image(image, use_column_width = True)  
        st.write(f"Name: {product_display_name}")
        st.write(f"Link: {description}")

        if st.button(f"Recommend for {product_display_name}", key=f"recommend_{i}"):
            features = feature_extraction(image_path, model)
            indices = recommend(features, feature_list)


            rec_container = st.sidebar.container(border = True)
            rec_container.markdown("Recommendations")  


            num_cols = 5  

#
            rec_cols = rec_container.columns(num_cols)
            for j in range(1, len(indices[0])):
                with rec_container:
                #with rec_cols[j % num_cols]:
                    recommended_image = Image.open(filenames[indices[0][j]])
                    st.image(recommended_image, caption=indices[0][j])
                    name_to_display = "Not available"
                    #if product_display_names:
                    #if len(product_display_names) > indices[0][j]:
                    name_to_display = product_display_names[indices[0][j]]
                    st.write(f"Product Name: {name_to_display}")
                    st.write(f"Link: {descriptions[indices[0][j]]}")

        
                #if st.__version__ >= '0.70.0':
                    #st.image(recommended_image, use_column_width = True)  
               # else:
                    #st.image(recommended_image, caption=descriptions[indices[0][j]])

        
                #name_to_display = "Not available"
                #if product_display_names:
                 #   if len(product_display_names) > indices[0][j]:
                  #      name_to_display = product_display_names[indices[0][j]]
                #st.write(f"Product Name: {name_to_display}")
                #st.write(f"Description: {descriptions[indices[0][j]]}")
