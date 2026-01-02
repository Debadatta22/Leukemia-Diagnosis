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

## 🔹 Synthetic Minority Oversampling Technique (SMOTE)
What is SMOTE?

SMOTE (Synthetic Minority Oversampling Technique) is a data balancing technique used to handle class imbalance problems in machine learning datasets. Class imbalance occurs when one or more classes have significantly fewer samples than others, which can bias the model toward majority classes.

Instead of duplicating existing minority samples, SMOTE generates new synthetic data points by interpolating between existing minority class samples. This helps create a more balanced dataset without overfitting.

**How SMOTE Works**

Identify samples belonging to the minority class.

For each minority sample, find its k-nearest neighbors.

Randomly select one of the neighbors.

Generate a new synthetic sample along the line segment between the original sample and its neighbor.

Repeat until the desired balance is achieved.

**Why SMOTE is Important in This Project**

The leukemia dataset contains very few patient samples (72) with uneven class distribution.

Without balancing, machine learning models tend to favor dominant leukemia classes, leading to poor diagnosis of minority subtypes.

**SMOTE ensures:**

Equal learning opportunity for all leukemia classes

Improved sensitivity (recall) for minority classes

Reduced model bias

Better generalization and clinical reliability

**Impact on Results**

By applying SMOTE during preprocessing:

Classification accuracy improved

Recall and MCC scores increased

Neural networks (BPNN) learned more stable decision boundaries

## 🔹 Feature Selection
**What is Feature Selection?**

Feature selection is the process of identifying and selecting the most relevant input variables (features) that contribute significantly to the prediction task while removing irrelevant, redundant, or noisy features.

In high-dimensional datasets, such as genomic data, feature selection is critical to avoid overfitting and improve model interpretability.

Why Feature Selection is Necessary in This Project

The dataset contains 7,130 genomic features but only 72 samples

High dimensionality leads to:

Overfitting

Increased computational cost

Reduced model stability

Poor generalization

Feature selection helps in:

Reducing dimensionality

Improving learning efficiency

Enhancing classification accuracy

Identifying important biomarkers

**Feature Selection Techniques Used**

## 1️⃣ ANOVA (Analysis of Variance)

Measures statistical differences between feature values across leukemia classes

Identifies features that show significant variation among classes

Removes features with low discriminative power

Contribution to Project:

Filters out irrelevant genomic attributes

Retains biologically meaningful features

## 2️⃣ Information Gain (Entropy-Based)

Measures how much information a feature provides about the target class

Higher information gain means stronger prediction capability

Contribution to Project:

Selects features that reduce uncertainty in leukemia classification

Improves early-stage decision-making

## 3️⃣ Correlation Analysis

Measures linear dependency between features and target class

Removes:

Highly correlated redundant features

Features weakly related to the class label

Contribution to Project:

Eliminates multicollinearity

Improves model robustness and numerical stability

Overall Benefit of Feature Selection

By combining ANOVA, Information Gain, and Correlation:

Noise is reduced

Only the most relevant genomic features are retained

Models train faster and perform better

Overfitting is significantly reduced

## 🔹 TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
What is TOPSIS?

TOPSIS is a multi-criteria decision-making (MCDM) technique used to rank alternatives based on their distance from an ideal best solution and an ideal worst solution.

In this project, TOPSIS is used to rank selected features based on multiple feature selection criteria.

Why TOPSIS is Needed After Feature Selection

Each feature selection method evaluates features differently:

ANOVA focuses on variance

Information Gain focuses on entropy reduction

Correlation focuses on linear dependency

TOPSIS combines all these perspectives into a single, objective ranking.

**How TOPSIS Works**

Construct a decision matrix using feature scores from ANOVA, Information Gain, and Correlation.

Normalize the scores to bring them onto a common scale.

**Determine:**

Ideal best solution (maximum values)

Ideal worst solution (minimum values)

Calculate:

Distance of each feature from ideal best

Distance from ideal worst

**Compute TOPSIS score:**

Higher score → more important feature

Rank features accordingly.

Contribution of TOPSIS to This Project

Ensures fair and unbiased feature ranking

Prevents dominance of a single selection method

Selects features that perform consistently across criteria

Improves classification accuracy and model reliability

## 🔹 Block Diagram and Workflow Explanation

The proposed system follows a structured multi-stage machine learning pipeline, ensuring robustness and accuracy at each stage.

Step 1: Data Collection

Leukemia microarray dataset is collected

Contains genomic features and labeled leukemia classes

Step 2: Data Preprocessing

Missing values handled using statistical imputation

Feature scaling applied to normalize data

SMOTE applied to handle class imbalance

Dataset split into training and testing sets

Purpose:
Ensure clean, balanced, and standardized input data

Step 3: Feature Selection

ANOVA identifies statistically significant features

Information Gain selects entropy-reducing features

Correlation removes redundant features

Purpose:
Reduce dimensionality and remove noise

Step 4: Feature Ranking using TOPSIS

Scores from all feature selection methods are combined

TOPSIS ranks features based on overall importance

Top-ranked features are selected

Purpose:
Ensure optimal and objective feature prioritization

Step 5: Model Training

Selected features are fed into multiple classifiers:

Traditional ML models

Ensemble methods

Neural networks (BPNN)

Purpose:
Learn patterns distinguishing leukemia subtypes

Step 6: Model Evaluation

Performance measured using:

Accuracy

Precision

Recall

Specificity

F1-score

MCC

Confusion matrix and ROC curves analyzed

Purpose:
Assess clinical reliability and prediction strength

Step 7: Best Model Selection

BPNN identified as the best-performing classifier

Achieves highest accuracy and stability

## 🔹 Overall Significance of the Workflow

This workflow ensures:

Balanced learning (SMOTE)

Reduced complexity (Feature Selection)

Optimal feature prioritization (TOPSIS)

Robust classification (BPNN)

High diagnostic accuracy (95.45%)

----

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

