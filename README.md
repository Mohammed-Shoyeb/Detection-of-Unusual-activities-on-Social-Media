# Detection-of-Unusual-activities-on-Social-Media


## Table of Contents
1. [Overview](#overview)
2. [Introduction](#introduction)
3. [System Requirements](#system-requirements)
4. [Technologies Used](#technologies-used)
5. [Tools of Implementation](#tools-of-implementation)
6. [Spam Detection Techniques](#spam-detection-techniques)
7. [Machine Learning Algorithms](#machine-learning-algorithms)
8. [Test Cases](#test-cases)
9. [Conclusion and Future Works](#conclusion-and-future-works)

---

## Overview
This project focuses on detecting spammers and fake user identification on Twitter using various machine learning algorithms. The detection involves analyzing tweet text, URLs, and account information to identify fraudulent activities and predict whether an account is genuine or fake.

## Introduction
Twitter is a widely used social media platform, but it is also prone to spam and fake accounts. This project aims to classify Twitter accounts and content as either genuine or spam by utilizing multiple machine learning techniques. The system processes datasets of tweets, extracts relevant features, and applies algorithms such as Naive Bayes, Random Forest, and Extreme Machine Learning to perform accurate classifications.

## System Requirements
- **OS**: Windows 10 / Mac OS X / Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Processor**: Intel i5 or equivalent
- **Python**: Version 3.8 or higher
- **Libraries**: NumPy, pandas, Scikit-learn, TensorFlow, Matplotlib
- **IDE**: Jupyter Notebook or any Python IDE

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Libraries**: Scikit-learn, TensorFlow
- **Data Visualization**: Matplotlib
- **Data Processing**: Pandas, NumPy
- **Development Environment**: Jupyter Notebook

## Tools of Implementation
- **AWS S3**: For dataset storage.
- **QuickSight**: For visualization of results.
- **Python**: As the core programming language.
- **Scikit-learn and TensorFlow**: For machine learning model development.
  
## Spam Detection Techniques
The project employs a range of techniques categorized into the following:

- **Fake Content Detection**: Identifying tweets that contain misleading or fraudulent information.
- **URL-Based Spam Detection**: Analyzing URLs within tweets to detect potential spam links.
- **Spam Detection in Trending Topics**: Monitoring trending topics for the presence of spam-related content.
- **Fake User Detection**: Utilizing features from user accounts to determine the likelihood of them being fake or spam.

Here's an enhanced version of the **Machine Learning Algorithms** section with detailed descriptions of each algorithm's role and implementation in the project:

---

## Machine Learning Algorithms

The project implements four key machine learning algorithms to detect spam and fake accounts on Twitter. Below is a breakdown of each algorithm and its contribution to the project:

### 1. **Naive Bayes**
   - **What it does**: Naive Bayes is a probabilistic classifier based on Bayes' Theorem with an assumption of independence among features.
   - **Implementation in the Project**: 
     - This algorithm is primarily used for analyzing tweet text and URLs.
     - It helps in detecting fake content, spam URLs, and spam trends within trending topics.
     - After extracting the features from the tweet text and URLs, Naive Bayes calculates the likelihood of the content being spam or fake, providing an efficient baseline for spam detection.
   - **Accuracy**: 66.66%
   - **Precision**: 55%

### 2. **Random Forest**
   - **What it does**: Random Forest is an ensemble learning method that operates by constructing multiple decision trees and outputting the majority vote from individual trees.
   - **Implementation in the Project**:
     - Used for fake account detection by analyzing user account features, such as follower count, tweet frequency, and user bio information.
     - It trains on historical labeled data to classify accounts as either spam or genuine.
     - Random Forest was chosen for its ability to handle large datasets and complex feature interactions.
     - After training the model on 80% of the data, it was tested on the remaining 20%, achieving a reasonable classification performance.
   - **Accuracy**: 60%
   - **Precision**: 53.4%

### 3. **Support Vector Machine (SVM)**
   - **What it does**: SVM is a supervised learning algorithm that finds a hyperplane in an N-dimensional space to classify data points into different categories.
   - **Implementation in the Project**:
     - SVM was employed to analyze the tweet data to detect fake accounts based on user behavior and content features.
     - It is particularly effective in cases with complex data distributions and is used here to classify user accounts into spam or genuine.
     - The model was able to detect patterns in user activity and content distribution, providing a more refined classification with higher accuracy than Random Forest.
   - **Accuracy**: 86.66%
   - **Precision**: 43.33%

### 4. **Extreme Machine Learning (EML)**
   - **What it does**: Extreme Machine Learning (EML) is a faster variant of traditional neural networks. It selects random hidden nodes and determines the output weights in a single step.
   - **Implementation in the Project**:
     - EML was implemented as the primary method for detecting spam and fake accounts due to its speed and accuracy.
     - It processes the same features used by Random Forest and SVM but performs better because of its ability to generalize better over large datasets.
     - EML produced the best results in this project, outperforming the other algorithms in terms of both accuracy and precision.
     - This algorithm trains the model on the dataset and then predicts the likelihood of an account being spam based on learned patterns from previous fake accounts.
   - **Accuracy**: 93.33%
   - **Precision**: 83.33%

---

These detailed descriptions highlight how each machine learning algorithm is utilized in the project and the specific results they produced. This section can now effectively showcase the comparative strengths of the algorithms and how they contribute to spam detection.

## Test Cases

| Test Case ID | Test Case Name | Description | Test Steps | Expected Result | Actual Result | Status | Priority |
|--------------|----------------|-------------|------------|-----------------|---------------|--------|----------|
| 01 | Upload Twitter JSON Format Tweets Dataset | Test whether the dataset is uploaded | Upload the dataset | Dataset uploaded successfully | Dataset uploaded | Passed | High |
| 02 | Load Naive Bayes | Load Naive Bayes algorithm to analyze tweet text or URL | Click to load Naive Bayes | Algorithm loaded successfully | Algorithm loaded | Passed | High |
| 03 | Detect Fake Content, Spam URL | Detect fake content and spam URLs in tweets | Run detection process | Fake content and spam detected | Detected successfully | Passed | High |
| 04 | Run Random Forest | Run Random Forest for fake account detection | Train model and predict | Random Forest model processed | Random Forest run successfully | Passed | High |
| 05 | Run Extreme Machine Learning | Run EML for spam detection | Train model and predict | EML model processed | EML run successfully | Passed | High |
| 06 | Accuracy Comparison | Compare accuracy of models | Generate accuracy comparison | Accuracy comparison displayed | Displayed successfully | Passed | High |

## Conclusion and Future Works
In this project, we implemented various machine learning techniques to detect spam accounts on Twitter. The Extreme Machine Learning algorithm demonstrated the highest accuracy, outperforming other models. Future work may focus on identifying false news and rumor sources on social media platforms, as well as improving real-time detection methods for spam accounts.
