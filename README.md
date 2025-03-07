# Data Mining Project Report: Heart Disease Prediction

## 1. Introduction

This report documents a data mining project focused on predicting the presence of heart disease in patients. The project utilizes a dataset containing various medical attributes to train and evaluate machine learning models. The goal is to develop a predictive model that can accurately identify individuals at risk of heart disease, enabling early intervention and preventative measures.

## 2. Dataset Description

The dataset used in this project is the "Heart Disease UCI" dataset, obtained from the UCI Machine Learning Repository. It contains the following attributes:

* **age:** Age in years.
* **sex:** (1 = male; 0 = female).
* **cp:** Chest pain type (4 values).
* **trestbps:** Resting blood pressure.
* **chol:** Serum cholesterol in mg/dl.
* **fbs:** Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
* **restecg:** Resting electrocardiographic results (values 0, 1, 2).
* **thalach:** Maximum heart rate achieved.
* **exang:** Exercise induced angina (1 = yes; 0 = no).
* **oldpeak:** ST depression induced by exercise relative to rest.
* **slope:** The slope of the peak exercise ST segment.
* **ca:** Number of major vessels (0-3) colored by fluoroscopy.
* **thal:** 3 = normal; 6 = fixed defect; 7 = reversible defect.
* **target:** 0 = no disease, 1 = disease.

## 3. Data Preprocessing

The data preprocessing steps performed in the Colab notebook include:

* **Importing Libraries:** Necessary libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib were imported.
* **Loading the Dataset:** The dataset was loaded into a Pandas DataFrame.
* **Checking for Missing Values:** The notebook confirmed that there were no missing values in the dataset.
* **Data Exploration:** Basic descriptive statistics were generated to understand the distribution of the features.
* **Feature Scaling:** `StandardScaler` was used to scale the numerical features, ensuring that all features contribute equally to the model.
* **One-Hot Encoding:** The categorical features (cp, restecg, slope, thal) were converted into numerical features using one-hot encoding.
* **Splitting the Data:** The dataset was split into training and testing sets, with 80% of the data used for training and 20% for testing.

## 4. Model Selection and Training

The following machine learning models were trained and evaluated:

* **Logistic Regression:** A linear model for binary classification.
* **K-Nearest Neighbors (KNN):** A non-parametric method used for classification and regression.
* **Support Vector Machine (SVM):** A supervised learning model with associated learning algorithms that analyze data used for classification and regression analysis.
* **Decision Tree:** A decision support tool that uses a tree-like graph or model of decisions and their possible consequences.
* **Random Forest:** An ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time.
* **Gradient Boosting Machine (GBM):** A machine learning technique used for regression and classification tasks, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

## 5. Model Evaluation

The performance of each model was evaluated using the following metrics:

* **Accuracy:** The proportion of correctly classified instances.
* **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
* **Recall:** The ratio of correctly predicted positive observations to all observations in the actual class.
* **F1-Score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** A table that visualizes the performance of a classification model.

The notebook displayed the classification report and confusion matrix for each model, showing the precision, recall, F1-score, and support for each class.

## 6. Results and Analysis

Based on the evaluation metrics, the following observations were made:

* The Random Forest and Gradient Boosting Machine (GBM) models achieved the highest accuracy scores.
* All models performed reasonably well, indicating that the features in the dataset are informative for predicting heart disease.
* The confusion matrices show the amount of true positives, true negatives, false positives, and false negatives, providing insight into the types of errors made by each model.
* The one hot encoding significantly increases the amount of columns in the dataframe.

## 7. Conclusion

The project successfully demonstrated the feasibility of using machine learning models to predict heart disease. The Random Forest and Gradient Boosting Machine models proved to be the most effective, achieving high accuracy scores. These models can be valuable tools for healthcare professionals in identifying individuals at risk of heart disease.

## 8. Future Work

* **Hyperparameter Tuning:** Fine-tune the hyperparameters of the models to further improve their performance.
* **Feature Engineering:** Explore additional feature engineering techniques to create more informative features.
* **More Advanced Models:** Test more advanced models, such as neural networks, to potentially achieve better results.
* **Cross-Validation:** Implement cross-validation techniques to ensure the robustness of the models.
* **Expand Dataset:** attempt to add more data to the set, to see if the model can be improved further.
* **Deployment:** Consider deploying the best performing model as a web application or API for practical use.

This report provides a concise overview of the data mining project. The Colab notebook contains the detailed implementation and results.
**Colab Notebook Link:** [https://colab.research.google.com/drive/1cA1xhaLfCnnaDX-Odreo4YifeeMW6fuP#scrollTo=uztRqKckG9R9](https://colab.research.google.com/drive/1cA1xhaLfCnnaDX-Odreo4YifeeMW6fuP#scrollTo=uztRqKckG9R9)
