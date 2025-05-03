# Emotion Classification from Text using Dual-Input BiLSTM with Sentiment Embedding

##  Overview
This project classifies emotions from raw text by combining traditional word embeddings with sentiment polarity scores using a dual-input Bidirectional LSTM (BiLSTM) architecture. It is built with TensorFlow and deployed via a Gradio-powered web interface.

 *Dataset:* https://www.kaggle.com/datasets/kosweet/cleaned-emotion-extraction-dataset-from-twitter

## Model Highlights
- Dual-input architecture: One input for tokenized text and one for sentiment score.
- BiLSTM for textual representation.
- Residual dense layers for improved gradient flow.
- Achieves **90.63% test accuracy** and **0.2136 test loss**.


## How to Run the Project

1. Clone the repository:
   git clone https://github.com/yourusername/emotion-classifier.git
   cd emotion-classifier
   
Install dependencies:


pip install -r requirements.txt


Launch the Gradio app:

python app.py
 Live Demo
Hosted on Hugging Face Spaces or localhost (Gradio). Enter a sentence, and it will return the predicted emotion label.
Link: https://huggingface.co/spaces/bhuvann13/Twitter_emotional_analysis

*Dependencies*
See requirements.txt for exact versions.


*Future Enhancements*
Use Transformer-based encoders (e.g., BERT, RoBERTa)

Add emoji-to-emotion interpretation

Extend emotion categories (e.g., surprise, disgust)

Add REST API or Docker support

