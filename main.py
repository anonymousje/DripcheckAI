import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-computed data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define the model (same as before)
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
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# Get all images from a folder (replace 'images' with your folder path)
image_paths = [os.path.join('images', f) for f in os.listdir('images') if f.endswith(('.jpg', '.png'))]


# Display images with click event for recommendations
num_cols = min(3, len(image_paths))  # Adjust number of columns based on image count
cols = st.columns(num_cols)
image_list = []
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path)
    image_list.append(image)
    with cols[i % num_cols]:
        st.image(image, use_column_width=True)
        if st.button(f"Recommend for {os.path.basename(image_path)}", key=f"recommend_{i}"):
            features = feature_extraction(image_path, model)
            indices = recommend(features, feature_list)

            # Show recommended images in new columns (replace 3 with desired number)
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            for j in range(1, len(indices[0])):
                with rec_col1 if j % 3 == 1 else (rec_col2 if j % 3 == 2 else rec_col3):
                    st.image(filenames[indices[0][j]])


