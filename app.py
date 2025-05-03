# app.py
import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
from textblob import TextBlob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load tokenizer and encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the trained model
model = load_model("best_model.h5")

# Text pre-processing + sentiment
def preprocess_input(text):
    sentiment = TextBlob(text).sentiment.polarity
    sequence = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequence, maxlen=100, padding='post')
    return padded_seq, np.array([[sentiment]])

# Prediction function
def predict_emotion(text):
    padded_text, sentiment_score = preprocess_input(text)
    pred_probs = model.predict([padded_text, sentiment_score])
    pred_class = np.argmax(pred_probs, axis=1)
    emotion = label_encoder.inverse_transform(pred_class)[0]
    return emotion

# Metrics function for ROC & Confusion Matrix images
def show_metrics():
    return "ROC_curve.png", "CFMat.png"

# Gradio layout with Tabs
with gr.Blocks() as demo:
    gr.Markdown("# Emotion Classification App")

    with gr.Tab("Predict Emotion"):
        text_input = gr.Textbox(label="Enter your text")
        predict_button = gr.Button("Predict")
        output_label = gr.Text(label="Predicted Emotion")

        predict_button.click(fn=predict_emotion, inputs=text_input, outputs=output_label)

    with gr.Tab("Model Performance"):
        gr.Markdown("## ROC Curve and Confusion Matrix")
        roc_image = gr.Image(label="ROC Curve")
        cm_image = gr.Image(label="Confusion Matrix")
        metrics_button = gr.Button("Load Metrics")

        metrics_button.click(fn=show_metrics, inputs=[], outputs=[roc_image, cm_image])

# Launch app
demo.launch()
