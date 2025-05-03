# Emotion Classification from Text using Dual-Input BiLSTM with Sentiment Embedding

##  Overview
This project classifies emotions from raw text by combining traditional word embeddings with sentiment polarity scores using a dual-input Bidirectional LSTM (BiLSTM) architecture. It is built with TensorFlow and deployed via a Gradio-powered web interface.

 *Dataset:* https://www.kaggle.com/datasets/kosweet/cleaned-emotion-extraction-dataset-from-twitter

## Model Highlights
- Dual-input architecture: One input for tokenized text and one for sentiment score.
- BiLSTM for textual representation.
- Residual dense layers for improved gradient flow.
- Achieves **90.63% test accuracy** and **0.2136 test loss**.

##  Project Structure

emotion_classifier/
├── app.py # Gradio app code
├── best_model.h5 # Trained Keras model
├── tokenizer.pkl # Tokenizer used during training
├── label_encoder.pkl # LabelEncoder for mapping predictions
├── ROC_curve.png # ROC curve visualization
├── CFMat.png # Confusion matrix
├── requirements.txt # Python dependencies
└── README.md # Project description



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

*Evaluation Summary*
Emotion	Precision	Recall	F1-Score	Support
angry	0.83	0.93	0.88	30,234
disappointed	0.91	0.86	0.88	31,318
happy	1.00	0.93	0.96	30,106
Accuracy			0.91	91,658

*Future Enhancements*
Use Transformer-based encoders (e.g., BERT, RoBERTa)

Add emoji-to-emotion interpretation

Extend emotion categories (e.g., surprise, disgust)

Add REST API or Docker support

