# Tweet Sentiment Classifier with Machine Learning Models

This project aims to develop a sentiment classifier model using an equally distributed subset of kaggle’s [Twitter 
Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) dataset. The objective is to classify each tweet into one of the sentiment classes using various 
machine learning models, and optimize their performance through hyperparameter tuning and evaluation techniques.


## Project Overview
 The key objectives were:

1. **Load and Preprocess Data**: We utilized Kaggle’s Twitter Sentiment Analysis Dataset, balanced the dataset, and 
applied text preprocessing techniques including lemmatization and special token replacement.
2. **TF-IDF Vectorization**: Conversion of text data into numerical features using TF-IDF, focusing on unigram and 
bigram features, in order to prepare the data for classification.
3. **Classification Models**: Implementation and evaluation of several classifiers, including:
   - Dummy Classifier (Baseline)
   - Naive Bayes
   - Logistic Regression
   - Multi-Layer Perceptron (MLP)
   - k-Nearest Neighbors (k-NN)
   - Support Vector Machines (SVM)
4. **Hyperparameter Tuning**: Utilized Grid Search to optimize the Logistic Regression model.
5. **Evaluation Tools**: Applied learning curves and precision-recall curves to evaluate model performance.

## Dataset

The dataset used for this project is a [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) from Kaggle, which contains 100,000 entries. 
We balanced the dataset to include 5,000 instances each for positive and negative sentiments.

## Preprocessing

The preprocessing steps included:
- **Tokenization and Lemmatization**: Standard NLP techniques to reduce the dimensionality of text data.
- **Special Token Replacement**: Replaced URLs, usernames, hashtags, stopwords, etc., with placeholders.
- **TF-IDF Vectorization**: Transformed the text data into numerical features using the `TfidfVectorizer` from scikit-learn.

## Classification Models

### Dummy Classifier
A baseline model that predicts the most frequent class. This model serves as a benchmark for comparison with more 
sophisticated classifiers and also helps in identifying class imbalances.

### Naive Bayes
A probabilistic classifier based on applying Bayes' theorem. Used both a simple implementation and a version with 
cross-validation.

### Logistic Regression
A linear model for binary classification. Used both simple Logistic Regression and a version with cross-validation, 
leading to slight improvements in F1 scores.

### Multi-Layer Perceptron (MLP)
A neural network model that was evaluated for its ability to classify the sentiment.

### k-Nearest Neighbors (k-NN)
A non-parametric classification algorithm based on majority voting among the k nearest data points.

### Support Vector Machines (SVM)
A classifier that finds the hyperplane that best separates the classes in a high-dimensional space. Also applied 
Singular Value Decomposition (SVD) for dimensionality reduction.

## Hyperparameter Tuning

For Logistic Regression, we conducted hyperparameter tuning using Grid Search, optimizing parameters like the solver, 
penalty, and the number of features in the TF-IDF vectorizer.

## Evaluation

### Learning Curves
Used to analyze how the model's performance improves with more data, helping to detect underfitting or overfitting.

### Precision-Recall Curves
Evaluated the performance of models, particularly useful for imbalanced datasets. The area under the precision-recall curve (AUC-PR) was used as a summary metric.
