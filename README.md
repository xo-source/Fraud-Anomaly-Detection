# Fraud-Anomaly-Detection
Fraud detection system using unsupervised anomaly detection (Isolation Forest). Explores feature engineering, PCA, and trade-offs between recall and precision in imbalanced datasets.

## Overview
This project is an unsupervised machine learning system for detecting fraudulent credit card transactions using anomaly detection techniques. It focuses on identifying rare fraudulent behavior in highly imbalanced financial data without relying on supervised learning.

---

## Problem Statement
Credit card fraud is extremely rare compared to normal transactions, making it difficult for traditional supervised models to learn meaningful patterns. This project explores unsupervised anomaly detection methods to flag suspicious transactions based on deviations from normal behavior.

---

## Dataset

The dataset is the Kaggle Credit Card Fraud Detection dataset.

It contains 284,807 transactions with 492 fraud cases (highly imbalanced ~0.17%).

Features are anonymized (PCA-transformed V1–V28), with only:
- Time
- Amount
- Class (target)

Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Approach

### 1. Data Preprocessing
- Log transformation applied to `Amount` and `Time`
- Feature scaling using StandardScaler

### 2. Model Used
- Isolation Forest (primary model)
- PCA-based anomaly detection (experimented)
- Feature engineering attempts (rolling stats, ratios, deviations)

---

## Evaluation Strategy
Although the model is unsupervised, evaluation was performed using labeled data:

- Confusion Matrix
- Precision, Recall, F1-score
- Focus on recall due to fraud detection priority

---

## Results

### Isolation Forest Performance
- Recall (Fraud Detection Rate): ~0.58–0.60
- Precision: ~0.09–0.11
- High false positive rate, but strong fraud coverage

---

## Experiments

Several anomaly detection approaches were tested to evaluate whether performance gains could be achieved beyond Isolation Forest:

- Local Outlier Factor (LOF): struggled with scalability and high-dimensional structure, producing unstable results on the full dataset.
- Autoencoder-based anomaly detection: showed similar behavior to Isolation Forest, with no significant improvement in separation of fraud vs normal transactions.
- PCA-based reconstruction error: used as an additional anomaly scoring method, but did not significantly outperform the baseline model.

Overall, results indicate that model complexity did not significantly improve performance, suggesting that the limiting factor is feature representation rather than model choice.
## Key Insights

- Increasing model complexity (PCA, feature engineering) did not significantly improve performance
- Isolation Forest already captures most separable structure in the dataset
- Feature representation is the main limiting factor, not model choice
- Trade-off observed between recall (fraud detection) and precision (false alarms)

---

## Conclusion
This project demonstrates a practical fraud detection pipeline using unsupervised learning. It highlights the importance of trade-offs in anomaly detection systems and shows that improving feature representation is often more impactful than changing models.

---

## Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib

---

## Future Improvements
- Behavioral feature engineering (user-level patterns)
- Autoencoder-based anomaly detection
- Cost-sensitive threshold optimization
- Real-time fraud scoring system
