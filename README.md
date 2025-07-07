## Credit-Card-Fraud-Detection-Model

*Overview:*

This project implements a machine learning-based credit card fraud detection system using Python. It processes the creditcard.csv dataset, addresses class imbalance using undersampling and oversampling (SMOTE), trains Logistic Regression and Decision Tree Classifier models, and evaluates their performance using accuracy, precision, recall, and F1-score. The final Decision Tree model is saved for real-time predictions.

*Dataset:*

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** 284,807 transactions | 492 frauds (~0.17%)
- **Target Variable:** Class (0 = Normal, 1 = Fraud)

*Prerequisites:*

**Required Python libraries:**

   pandas==2.1.4
   numpy==1.26.4
   scikit-learn==1.5.1
   imblearn==0.12.3
   seaborn==0.13.2
   matplotlib==3.9.2
   joblib==1.4.2

   Install them using:

   pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.5.1 imblearn==0.12.3 seaborn==0.13.2 matplotlib==3.9.2 joblib==1.4.2

## Project Structure:

Data Preprocessing:

  Loaded dataset with 284,807 rows and 31 columns.
  Scaled Amount column using StandardScaler (mean=0, std=1).
  Dropped Time column as it was not relevant for classification.
  Removed 1,081 duplicate rows, resulting in 283,726 unique transactions.
  Confirmed no missing values in the dataset.

Exploratory Data Analysis (EDA):

   Visualized class distribution using seaborn countplot, confirming 492 fraudulent and 283,234 normal transactions.
   Analyzed class imbalance (0.17% fraud rate).

Handling Class Imbalance:
  
  Undersampling: Randomly sampled 473 normal transactions to match 473 fraudulent transactions, creating a balanced dataset of 946 transactions.
  Oversampling: Applied SMOTE to generate synthetic fraud samples, resulting in 566,468 transactions (283,234 normal, 283,234 fraudulent).


Model Training and Evaluation:

  Split data into 80% training and 20% testing sets.
  Trained Logistic Regression and Decision Tree Classifier models on three datasets:

Original (Imbalanced):


  Logistic Regression: Accuracy ~99.9%, Precision ~85%, Recall ~60%, F1-Score ~70%.
  Decision Tree: Accuracy ~99.8%, Precision ~75%, Recall ~65%, F1-Score ~70%.


Undersampled:

   Logistic Regression: Accuracy ~94%, Precision ~95%, Recall ~93%, F1-Score ~94%.
   Decision Tree: Accuracy ~92%, Precision ~93%, Recall ~91%, F1-Score ~92%.


Oversampled (SMOTE):

  Logistic Regression: Accuracy ~97%, Precision ~98%, Recall ~96%, F1-Score ~97%.
  Decision Tree: Accuracy ~99.5%, Precision ~99%, Recall ~99%, F1-Score ~99%.
  Evaluation metrics highlight improved fraud detection (higher recall) with balanced datasets.

Model Deployment:

  Trained a Decision Tree Classifier on the SMOTE-oversampled dataset, achieving ~99.5% accuracy.
  Saved the model as credit_card_model.pkl using joblib.

  Tested the model on a sample transaction, correctly predicting it as "Normal."

## How to Run:

   Place creditcard.csv in the same directory as the script.
   Install required libraries (see Prerequisites).
   Run the script:python credit_card_fraud_detection.py
   Outputs include model performance metrics and a sample transaction prediction.

## Results:

  Original Dataset: High accuracy (~99.9%) but lower recall (~60-65%) due to class imbalance, limiting fraud detection.
  Undersampled Dataset: Balanced dataset improved recall (~91-93%) and F1-score (~92-94%).
  Oversampled Dataset (SMOTE): Best performance with ~99.5% accuracy, ~99% precision, ~99% recall, and ~99% F1-score for Decision Tree.
  Sample Prediction: Correctly classified a test transaction as "Norm






