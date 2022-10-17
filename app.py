import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image

model = tf.keras.models.load_model('model_imp.h5')

st.title("Mask Detection")
st.text("Welcome to the mask detection page. In here, you can drop any image of a person to detect whether a person is wearing a mask or not.")

img = st.file_uploader("Upload an image of a person...", type="jpg")

if st.button('Submit'):

    img2 = Image.open(img)
    st.image(img2)
    img3 = np.array(img2)
    img_resize = cv2.resize(img3,(150, 150))    
    img_reshape = img_resize/255
    
    prediction = model.predict(np.array([img_reshape]))
    fin = np.argmax(prediction)
        
    if fin is None:
        st.text("Please upload an image file")
    elif fin == 0:
        st.text("Masked")
    elif fin == 1:
        st.text("Unmasked")        
    else:
        st.text("Invalid")
