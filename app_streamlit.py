import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

def predictionflowerS(test_image):
    model = tf.keras.models.load_model(
        r'C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition\model.h5')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Daisy'
        return prediction
    elif result[0][1] == 1:
        prediction = 'Dandelion'
        return prediction
    elif result[0][2] == 1:
        prediction = 'Rose'
        return prediction
    elif result[0][3] == 1:
        prediction = 'Sunflower'
        return prediction
    else:
        prediction = 'Tulip'
        return prediction
def load_image(image_file):
    img = Image.open(image_file)
    newsize=(128,128)
    img=img.resize(newsize)
    return img

st.title("Flower Classification Application")

# uploaded_file = st.sidebar.file_uploader("Choose a file")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if image_file is not None:
    # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                  "filesize":image_file.size}
    st.write(file_details)
    img=load_image(image_file)
    # To View Uploaded Image
    st.image(img,width=250)
    classifier=predictionflowerS(img)
    # st.write('Predicted:', classifier)
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 62px;">Predicted: {classifier}</p>'.format(classifier=classifier)
    st.markdown(new_title, unsafe_allow_html=True)