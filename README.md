

#  Fake News Detection System

### *NLP & Machine Learning Project — 25F AI Infrastructure & Architecture*

---

##  Project Overview

This project implements a **Fake News Detection System** using natural language processing (NLP) and supervised machine learning. The model classifies online news articles as **REAL** or **FAKE** based solely on textual content.

The system includes:

* Automated text preprocessing
* TF-IDF vectorization
* Logistic Regression model training
* Classification performance evaluation
* Confusion matrix visualization
* Batch and interactive predictions
* Exportable reports and saved models

This project is designed for academic submission and CLI operation.

---

##  Dataset

**Fake and Real News Dataset (Kaggle)**
[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Supports two formats:

1. **Fake.csv + True.csv**
2. **train.csv + test.csv**

---

## ML Pipeline Summary

1. **Clean & preprocess text**

   * Remove special characters
   * Normalize whitespace
   * Lowercasing
   * Drop empty rows

2. **Feature extraction**

   * TF-IDF vectorization
   * `max_features = 5000`

3. **Model training**

   * Logistic Regression (`max_iter = 1000`)
   * 80/20 train/test split if no test file is provided

4. **Evaluation metrics**

   * Accuracy
   * Precision, Recall, F1-Score
   * Confusion matrix
   * Classification report

---

##  Features

###  Training Mode

* Load dataset
* Clean and vectorize text
* Train Logistic Regression model
* Evaluate accuracy and generate performance metrics

###  Prediction Modes

* **Custom text prediction**
* **Batch prediction** from text file
* Confidence score for every prediction

###  Reporting & Export

* `classification_report.txt`
* `confusion_matrix.png`
* `batch_predictions.txt`
* Saved model file: `fake_news_model.pkl`

###  CLI Interface

User-friendly menu-driven interface:

```
1. Train New Model
2. Load Existing Model
3. Test with Custom News Article
4. Batch Prediction from File
5. View Model Performance
6. Generate Reports
7. Generate Visualizations
8. Save Current Model
9. Exit
```

---

##  How to Run

### **1. Install Dependencies**

Inside your environment:

```bash
pip install pandas numpy scikit-learn matplotlib wordcloud colorama
```

### **2. Place Dataset in Same Directory**

Accepted files:

```
Fake.csv
True.csv
```

or

```
train.csv
test.csv
```

### **3. Run the Program**

```bash
python fakenews_scanner.py
```

---

##  Output Files (Generated Automatically)

| File                          | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| **classification_report.txt** | Accuracy, F1-Score, confusion matrix, model explanation |
| **confusion_matrix.png**      | Visualization of predictions                            |
| **batch_predictions.txt**     | Results of file-based prediction mode                   |
| **fake_news_model.pkl**       | Saved model and vectorizer                              |

---

## Technologies Used

* **Python 3.x**
* **pandas, numpy** — data processing
* **scikit-learn** — ML model + vectorization
* **matplotlib / wordcloud** — visualizations
* **colorama** — colored CLI output
* **pickle** — model saving

---

## Model Performance (Typical Results)

| Metric            | Expected Score      |
| ----------------- | ------------------- |
| Training Accuracy | 95–99%              |
| Testing Accuracy  | 92–96%              |
| Model Type        | Logistic Regression |

*(Your results may vary depending on dataset splits.)*

---

## Author

**A. Sanchez**
25F AI Infrastructure & Architecture – 01
Fake News Detection Project (10% Grade Component)

---

##  Notes

* This project is intended for academic purposes.
* Accuracy depends heavily on dataset quality and representation.
* The CLI interface ensures that even non-technical users can interact with the model.
