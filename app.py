# app.py
import streamlit as st
import numpy as np
import json
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load labels mapping
labels_path = os.path.join("artifacts", "labels.json")
if not os.path.exists(labels_path):
    st.error("labels.json not found. Run training first (train.py) and ensure artifacts/labels.json exists.")
    st.stop()

with open(labels_path, "r") as f:
    class_indices = json.load(f)
# invert mapping: idx -> class_name
idx_to_class = {v: k for k, v in class_indices.items()}

# Load model
model_path = "artifacts/best_model.h5" if os.path.exists("artifacts/best_model.h5") else "artifacts/cattle_breed_model.h5"
if not os.path.exists(model_path):
    st.error(f"Model not found. Expected {model_path}. Train the model first.")
    st.stop()

@st.cache_resource
def load_the_model(path):
    return load_model(path)
model = load_the_model(model_path)

st.title("🐄 Cattle Breed Identifier")

# Show supported breeds under the title
st.subheader("Supported breeds")
try:
    # Build ordered list by index (0..n-1)
    num_classes = len(class_indices)
    labels_ordered = [idx_to_class[i] for i in range(num_classes)]
    st.write(f"{num_classes} breeds supported:")
    for breed in labels_ordered:
        st.markdown(f"- {breed}")
except Exception:
    # Fallback: if something unexpected happens, show a simple message
    st.info("Breed list not available")


uploaded_file = st.file_uploader("Upload an image of cattle", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Identify Breed"):
        # preprocess
        x = image.img_to_array(img)            # shape (224,224,3), dtype float32 with values 0-255
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)               # same as training

        preds = model.predict(x)[0]           # probs array
        top3_idx = preds.argsort()[-3:][::-1]

        # Map indices to class names & show confidences
        st.subheader("Top predictions")
        for i in top3_idx:
            class_name = idx_to_class[int(i)]
            confidence = preds[int(i)]
            st.write(f"{class_name}: {confidence*100:.2f}%")

        predicted_idx = int(np.argmax(preds))
        predicted_class = idx_to_class[predicted_idx]
        st.success(f"Predicted Breed: **{predicted_class}** ({preds[predicted_idx]*100:.2f}%)")

        # Show bar chart for all classes (sorted)
        labels = [idx_to_class[i] for i in range(len(preds))]
        probs = preds
        # use st.bar_chart with dict-of-values
        chart_data = {labels[i]: float(probs[i]) for i in range(len(labels))}
        st.bar_chart(chart_data)
