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

```bash
git clone https://github.com/bhuvannv13/Twitter-Emotional-Classification.git
cd Twitter-Emotional-Classification
pip install -r requirements.txt
python app.py
```

The Gradio app loads `best_model.h5`, `tokenizer.pkl` and `label_encoder.pkl` from the repository folder. Enter a sentence and it returns the predicted emotion.

## Live Demo

[Hugging Face Space](https://huggingface.co/spaces/bhuvann13/Twitter_emotional_analysis)

## Future Enhancements

- Use Transformer-based encoders (e.g., BERT, RoBERTa)
- Add emoji-to-emotion interpretation
- Extend emotion categories (e.g., surprise, disgust)
- Add REST API or Docker support
