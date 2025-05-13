
# Sentiment Analysis and IMDb Rating Prediction System

## Project Overview
This project implements a sophisticated natural language processing (NLP) system that performs dual tasks:
1. Classifying movie reviews as positive or negative (sentiment analysis)
2. Predicting the numerical IMDb rating based on review content

The system leverages deep learning techniques, specifically LSTM (Long Short-Term Memory) networks, alongside pre-trained GloVe word embeddings to understand and quantify the sentiment expressed in movie reviews.

## Dataset Description
The IMDb dataset consists of three primary columns:
- Movie Name: The title of the film being reviewed
- Movie Review: The textual content of the review
- IMDb Rating: The numerical rating (typically on a scale of 1-10)

This dataset provides a rich source of labeled data for both sentiment classification and rating prediction tasks.

## Implementation Architecture

### 1. Data Preprocessing Pipeline
The preprocessing stage transforms raw review text into a format suitable for deep learning models:

- **Text Cleaning**:
  - Removal of HTML tags and special characters
  - Removal of punctuation marks and digits
  - Conversion to lowercase for uniformity
  - Elimination of excessive whitespace

- **Text Tokenization**:
  - Splitting reviews into individual words/tokens
  - Removal of stopwords (common words like "the", "is", "and")
  - Setting a maximum sequence length for uniformity

- **Data Splitting**:
  - Division of dataset into training (70%), validation (15%), and test (15%) sets
  - Stratification by rating to ensure balanced distribution

### 2. Feature Engineering with GloVe Embeddings
- Used pre-trained GloVe (Global Vectors for Word Representation) embeddings (100-dimensional vectors)
- Created an embedding matrix mapping each word in the vocabulary to its corresponding GloVe vector
- Words not found in GloVe were initialized with random vectors
- Implemented an embedding layer in the model that leverages these pre-trained vectors

### 3. Model Architecture
The system implements multiple neural network architectures:

- **Main LSTM Model**:
  - Embedding Layer: Initialized with pre-trained GloVe vectors
  - LSTM Layer (128 units): Captures sequential relationships in text
  - Dropout Layer (0.5): Prevents overfitting
  - Dense Layer: With ReLU activation
  - Output Layers:
    - Binary classification head for sentiment (sigmoid activation)
    - Regression head for rating prediction (linear activation)

- **Bidirectional LSTM Variant**:
  - Similar to the main model but uses Bidirectional LSTM to capture context from both directions
  - Improved performance on complex sentiment expressions
  - Used for comparison against the standard LSTM

### 4. Training Process
- **Loss Functions**:
  - Binary cross-entropy for sentiment classification
  - Mean squared error for rating prediction
  - Combined loss with weighted contributions from both tasks

- **Optimization**:
  - Adam optimizer with learning rate of 0.001
  - Batch size of 64 for efficient training
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau

- **Training Duration**:
  - 15 epochs with early stopping patience of 3
  - Approximately 2 hours on GPU hardware

### 5. Model Evaluation
- **Sentiment Classification Metrics**:
  - Accuracy: 84.7%
  - Precision: 86.2%
  - Recall: 83.9%
  - F1-Score: 85.0%

- **Rating Prediction Metrics**:
  - Mean Absolute Error (MAE): 0.72
  - Root Mean Squared Error (RMSE): 0.94
  - R² Score: 0.67


## Future Enhancements
- Implementation of attention mechanisms to focus on sentiment-heavy portions of reviews
- Experimentation with transformer-based models like BERT for potentially higher accuracy
- Development of a genre-specific rating prediction system to account for genre-based rating variations
- Incorporation of additional features like review length, publication date, and reviewer history

