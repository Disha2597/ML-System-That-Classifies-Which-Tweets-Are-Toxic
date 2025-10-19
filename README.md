# ML-System-That-Classifies-Which-Tweets-Are-Toxic:

🧠 Toxic Tweet Classification with BERT + TensorFlow

This project demonstrates how to build a binary text classification model that automatically flags harmful (toxic) tweets using a BERT-based preprocessing pipeline and a TensorFlow sequential model.

📘 Problem Overview

The goal is to classify tweets as either toxic or non-toxic, making this a supervised text classification task. The dataset contains around 56,000 labeled tweets, providing a solid foundation for training a deep learning model that can detect harmful content.

📊 Data Exploration

The dataset was examined for class balance and cleaned by removing unnecessary columns. Understanding the label distribution helped identify potential class imbalance, which is important for model performance and fairness.

🔧 Preprocessing

Instead of manually cleaning and tokenizing the text, the BERT tokenizer was used for subword tokenization and text vectorization. This approach efficiently converts raw text into numerical representations that deep learning models can understand, while preserving contextual meaning.

⚙️ Model Architecture

The model was built using TensorFlow’s Sequential API and includes:

Embedding Layer – Converts words into dense numerical vectors that capture semantic meaning.

Bidirectional LSTM – Processes each tweet in both forward and backward directions to capture contextual information.

Dense Layers with Nonlinear Activations – Learn complex relationships and output a probability indicating tweet toxicity.

🚀 Training Pipeline

A TensorFlow data pipeline was created using the tf.data API to efficiently handle batching, caching, shuffling, and prefetching. This setup optimizes GPU utilization and ensures smooth training performance.

📈 Evaluation

The trained model was evaluated on a separate test set using accuracy, precision, and recall. These metrics provide insight into how well the model identifies toxic content, especially when dealing with imbalanced data.

🧩 Key Takeaways

Developed an end-to-end NLP classification pipeline using TensorFlow and Transformers.

Leveraged BERT tokenization for efficient and context-aware text preprocessing.

Applied data pipeline optimizations to improve model training performance.

Evaluated the model using key metrics to ensure balanced and reliable performance.
