# Iris Flower Classification (`Iris-Flower-Classification`)

A supervised machine learning classification project in Python (`iris.ipynb`) using **Logistic Regression** on `Iris.csv` to classify iris flowers into three species (*Setosa*, *Versicolor*, *Virginica*) from sepal and petal dimensions.

---

## 1. Executive Summary & Purpose
This project demonstrates an end-to-end supervised classification pipeline—from exploratory visualization to train/test splitting and multi-metric model evaluation.

## 2. Pipeline Architecture & Layers
```mermaid
flowchart LR
    Data["Iris.csv (Pandas)"] --> EDA["EDA & Visualization (Matplotlib / Seaborn)"]
    EDA --> Split["train_test_split (scikit-learn)"]
    Split --> Model["LogisticRegression"]
    Model --> Eval["Evaluation (Accuracy, Recall, F1, Classification Report)"]
```

| Layer | Tools & Methods |
| :--- | :--- |
| **Data Ingestion & Manipulation** | `pandas`, `numpy` (`Iris.csv`) |
| **Exploratory Data Analysis** | `matplotlib.pyplot`, `seaborn` |
| **Modeling** | `sklearn.linear_model.LogisticRegression`, `train_test_split` |
| **Evaluation** | `accuracy_score`, `f1_score`, `recall_score`, `classification_report` |

## 3. Getting Started
```bash
git clone https://github.com/islamsamyy/Iris-Flower-Classification.git
cd Iris-Flower-Classification
jupyter notebook iris.ipynb
```

## 4. Business Value & Growth Opportunities
- **Reusable Classification Template**: Serves as a clean benchmark baseline for multi-class tabular classification problems (e.g., product categorization, customer tiering) and can be wrapped in a FastAPI inference endpoint for production demos.
