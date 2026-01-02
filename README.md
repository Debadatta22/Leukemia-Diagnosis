# Leukemia-Diagnosis

<p align="center">
  <a href="https://drive.google.com/file/d/1b7dr6t2s4JeqsVINFvpLiELtDlL8zOQs/view?usp=drive_link" target="_blank">
    <img src="https://img.shields.io/badge/View PY-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="View PY" />
  </a>
</p>

<p align="center">
  <a href="https://drive.google.com/file/d/1x63V5dwPSU2OJ63MvRbncqtcG5lUXLDi/view?usp=drive_link" target="_blank">
    <img src="https://img.shields.io/badge/View IPYNB-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="View IPYNB" />
  </a>
</p>

<p align="center">
  <a href="https://drive.google.com/file/d/1SIbS8ZXohXHm71G6DPXsHY3mrvz7o97B/view?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/Read Research Paper-FF9900?style=for-the-badge&logo=googlescholar&logoColor=white" alt="Read Research Paper" />
  </a>
</p>

-----------------------

# 🧬 Improving Leukemia Diagnosis using Feature Selection–Driven Machine Learning

## 📌 Project Overview
Leukemia is a life-threatening blood cancer characterized by abnormal growth of white blood cells. Accurate and early classification of leukemia subtypes is critical for effective treatment planning and improved patient survival.

This project presents a **feature selection–driven machine learning framework** for leukemia classification using **high-dimensional genomic data**. The proposed pipeline integrates **data preprocessing, feature selection, multi-criteria feature ranking (TOPSIS), and machine learning classifiers**, with a special focus on **Backpropagation Neural Networks (BPNN)**.

The system achieves a **maximum classification accuracy of 95.45%**, demonstrating the effectiveness of combining robust feature selection with neural network–based classification.

---

## 🎯 Objectives
- Handle **high-dimensional, small-sample genomic data** efficiently  
- Reduce redundancy and noise using **advanced feature selection**
- Rank features objectively using **TOPSIS (multi-criteria decision making)**
- Compare multiple machine learning models
- Identify the **best-performing classifier** for leukemia prediction
- Provide reliable evaluation using **medical performance metrics**

---

## 🧠 Key Contributions
- Hybrid feature selection using:
  - **ANOVA**
  - **Information Gain (Entropy)**
  - **Correlation Analysis**
- Feature ranking using **TOPSIS**
- Extensive comparison of **traditional ML, ensemble models, and neural networks**
- Detailed evaluation using:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - F1-Score
  - Matthews Correlation Coefficient (MCC)
- ROC curve and confusion matrix analysis
- PCA-based dimensionality reduction comparison

---

## 📂 Dataset Description
- **Dataset:** Leukemia Microarray Dataset (`leukemia.csv`)
- **Samples:** 72 patients
- **Features:** 7,130 genomic attributes
- **Classes:**  
  - AML  
  - B-Cell  
  - T-Cell  

Due to **high dimensionality and class imbalance**, advanced preprocessing and feature selection techniques are required.

---

## 🔧 Methodology Pipeline

### 1️⃣ Data Preprocessing
- Missing value handling:
  - Numerical → Mean imputation
  - Categorical → Mode imputation
- Feature scaling using **StandardScaler**
- Label encoding of target class
- Train-test split: **70% training / 30% testing**
- Class imbalance handling using **SMOTE**

---

### 2️⃣ Feature Selection Techniques
To reduce dimensionality and select biologically relevant features:

#### 🔹 ANOVA (Analysis of Variance)
Identifies features with statistically significant differences across leukemia classes.

#### 🔹 Information Gain
Measures how much information a feature contributes toward class prediction.

#### 🔹 Correlation Analysis
Removes redundant features with high inter-feature correlation and low target correlation.

---

### 3️⃣ Feature Ranking using TOPSIS
All selected features are ranked using **Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)**.

**Why TOPSIS?**
- Considers multiple selection criteria simultaneously
- Identifies features closest to the ideal solution
- Produces a stable and objective feature ranking

Top-ranked features are selected for model training.

---

### 4️⃣ Machine Learning Models Used
The following classifiers were trained and evaluated:

- Logistic Regression
- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost + SVM
- XGBoost
- Artificial Neural Network (ANN)
- **Backpropagation Neural Network (BPNN)**
- Voting / Stacking Ensembles

---

### 5️⃣ Dimensionality Reduction (Optional)
- **PCA (Principal Component Analysis)** applied
- Components selected to retain **~95% variance**
- Performance compared **with and without PCA**

---

## 📊 Performance Evaluation Metrics
Medical-grade evaluation metrics used:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **Specificity**
- **F1-Score**
- **Matthews Correlation Coefficient (MCC)**
- **Confusion Matrix**
- **ROC-AUC Curves (multi-class)**

---

## 🏆 Results Summary

| Classifier | Accuracy (%) |
|----------|--------------|
| Logistic Regression | 86.36 |
| SVM | 90.90 |
| Random Forest | 90.90 |
| ANN | 90.90 |
| Naïve Bayes | **95.45** |
| **BPNN** | **95.45 (Best)** |
| XGBoost | 81.81 |

✔ **BPNN with ReLU activation achieved the highest accuracy of 95.45%**  
✔ Strong generalization observed via ROC and MCC analysis

---

## 📈 Visual Analysis
- Confusion Matrix for best model (BPNN)
- Multi-class ROC curves
- Accuracy comparison bar plots
- Feature selection comparison graphs
- PCA vs Non-PCA performance plots

---

## 🛠️ Technologies Used
- **Python**
- **NumPy, Pandas**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib, Seaborn**
- **Google Colab**

---

## ▶️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/leukemia-diagnosis-ml.git
cd leukemia-diagnosis-ml

# Install dependencies
pip install -r requirements.txt

# Run the notebook / script
python leukemia.py

