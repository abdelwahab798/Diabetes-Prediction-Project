# Diabetes Prediction Project

## 📋 Project Description
This project focuses on building machine learning models to predict whether a person has diabetes or not, using the **Pima Indians Diabetes Dataset**.  
We applied different models, tuned their hyperparameters, handled data imbalance, and compared the models' performances.

---

## 🛠 Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

We also applied **GridSearchCV** to find the best hyperparameters for each model.

---

## 📊 Experiments
We conducted several experiments to evaluate and improve model performance:

1. **Baseline Models:**  
   Trained Logistic Regression, Decision Tree, and Random Forest without any tuning.

2. **Hyperparameter Tuning:**  
   Applied **GridSearchCV** to optimize model performance.

3. **Handling Imbalanced Data:**  
   Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset and retrained the models.

4. **Comparison:**  
   Evaluated models based on:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

---

## ⚙️ Dataset
- **Source:** [Pima Indians Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target:**
  - Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 📈 Results Summary

- After **GridSearchCV**, the **Decision Tree** model achieved the best performance with the highest accuracy and F1-Score.
- After applying **SMOTE** to address data imbalance, the **Random Forest** model outperformed the others in terms of overall balance between Precision and Recall.
- Hyperparameter tuning was essential to boost the models' performance.
- F1-Score proved to be the most important metric, especially due to the class imbalance in the dataset.

---

## 🚀 Conclusion
- **Best model without SMOTE:** Decision Tree (after GridSearchCV)
- **Best model with SMOTE:** Random Forest (after GridSearchCV + SMOTE)

---

## 📚 What We Learned
- The importance of hyperparameter tuning for improving model performance.
- SMOTE is effective for handling imbalanced datasets.
- Choosing the right evaluation metrics is crucial when dealing with skewed classes.

---

## 📎 How to Run
1. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
    ```
2. Run the Jupyter Notebook and follow the steps to reproduce the results.

---

## ✨ Author
- **[Your Name]**

---
