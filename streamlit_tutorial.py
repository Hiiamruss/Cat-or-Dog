import streamlit as st
from fastai.vision.all import *

st.title("Cat vs Dog Classifier")
st.text("Built by Russell S.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def is_cat(f):
    return f[0].isupper()

cat_vs_dog_model = load_learner("cat-vs-dog1.pkl")


def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = cat_vs_dog_model.predict(img)
    likehood_is_cat = outputs[1]
    if likehood_is_cat >= 9:
        return "Cat"
    elif likehood_is_cat <= 0.1:
        return "Dog"
    else:
        return "Try Another Picture"
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=200)
st.video("https://www.ayclogic.com/wp-content/uploads/2025/07/Crossing-Street.mp4")

