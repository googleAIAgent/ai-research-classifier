import streamlit as st
import pickle
import numpy as np
from google.cloud import aiplatform
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_ID = "southern-splice-419102"
REGION = "australia-southeast1"
ENDPOINT_ID = "6264775362509537280"

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# Load tokenizer
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pickle", "rb") as f:
    label_tokenizer = pickle.load(f)

def preprocess_input(text, maxlen=30):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')
    return padded.tolist()

st.set_page_config(page_title="AI Research Classifier", layout="centered")
st.title("📘 AI Research Abstract Classifier")
st.write("Enter your research abstract below to get its predicted category using a deployed AI model.")

user_input = st.text_area("Enter abstract here:", height=200)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Sending to Vertex AI..."):
            processed = preprocess_input(user_input)
            prediction = endpoint.predict(instances=processed)
            pred_index = np.argmax(prediction.predictions[0])
            pred_label = label_tokenizer.inverse_transform([pred_index])[0]
            st.success("🎯 Predicted Label: " + pred_label)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + Google Vertex AI")
