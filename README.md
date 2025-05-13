# Sentiment Analysis and IMDb Rating Prediction Project Steps

## 1. Data Loading and Exploration
- Loaded the IMDb dataset containing movie reviews and their sentiment labels
- Performed exploratory data analysis (EDA) to check class distribution
- Found balanced dataset with approximately 50% positive and 50% negative reviews
- Analyzed text characteristics by calculating:
  * Number of characters per review
  * Number of words per review (using NLTK tokenizer)
  * Number of sentences per review
- Created correlation heatmap to identify relationships between features

## 2. Text Preprocessing
- Implemented comprehensive text cleaning pipeline:
  * Removed HTML tags using regex
  * Converted all text to lowercase
  * Removed punctuation and numbers
  * Eliminated single character words
  * Removed stopwords (common words like "the", "and", "is")
  * Stripped extra whitespace
- Encoded sentiment labels (negative=0, positive=1)
- Split dataset into training (80%) and testing (20%) sets

## 3. Feature Engineering
- Created a tokenizer trained on the vocabulary of the training set
- Converted text reviews to sequences of numbers (word indices)
- Padded sequences to ensure uniform length (100 tokens)
- Loaded pre-trained GloVe word embeddings (100-dimensional vectors)
- Created an embedding matrix mapping each word in our vocabulary to its GloVe vector

## 4. Model Development
Implemented and compared multiple neural network architectures:

### 4.1 Simple Neural Network
- Used GloVe embeddings (non-trainable)
- Flattened the embedding outputs
- Added a dense layer with sigmoid activation
- Achieved 74.7% test accuracy

### 4.2 Convolutional Neural Network (CNN)
- Added a 1D convolutional layer (128 filters)
- Used Global Max Pooling to reduce dimensionality
- Achieved 84.9% test accuracy

### 4.3 Long Short-Term Memory (LSTM) Network
- Implemented LSTM layer with 128 units
- Added dropout (0.4) to prevent overfitting
- Trained for 20 epochs
- Achieved 84.7% test accuracy

### 4.4 Bi-Directional LSTM (not fully implemented in the snippet)

## 5. Model Improvement Techniques
- Applied dropout regularization
- Considered model depth increases (adding more LSTM layers)
- Experimented with different hyperparameters

## 6. Evaluation on Live Data
- Loaded unseen IMDb reviews
- Applied the same preprocessing pipeline
- Made predictions using the trained LSTM model
- Scaled sentiment predictions to align with IMDb's 10-point rating scale
- Created a comparison dataframe with movies, reviews, actual IMDb ratings, and predicted sentiments

The project successfully demonstrates how to build a sentiment analysis system that not only classifies reviews as positive or negative but also predicts numerical ratings based on text content.
