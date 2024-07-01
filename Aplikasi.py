import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SpatialDropout1D

model = load_model('lstm_model.h5', custom_objects={'SpatialDropout1D': SpatialDropout1D})
# Define tokenizer parameters (must be the same as used during training)
max_len = 150
oov_tok = "<OOV>"
vocab_size = 450

# Load the dataset
data = pd.read_csv('sampilng.csv')  # Replace with actual file path

# Dummy data to fit tokenizer (the same data used during training)
X_train = data['tweet']

# Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

# Predicting new text
def predict_text(text, threshold=0.5):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='pre', truncating='pre')
    pred = model.predict(padded)
    class_pred = (pred >= threshold).astype(int)
    return class_pred

# Streamlit app
st.title('Text Classification App')

input_text = st.text_input("Enter text for classification:")

if st.button('Predict'):
    if input_text:
        prediction = predict_text(input_text)
        st.write(f"Prediction: {prediction[0][0]}")
        if prediction[0][0] == 0:
            st.write("Negative")
        else:
            st.write("Positive")
    else:
        st.write("Please enter some text to classify.")
