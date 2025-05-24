## Early Stage Diabetes Risk Prediction

This repository contains a data science project focused on predicting the early stage risk of diabetes using a provided dataset. The project utilizes a Decision Tree Classifier model to analyze the data and make predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Prediction and Evaluation](#prediction-and-evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build a classification model that can identify individuals at early risk of diabetes based on various health attributes. The project workflow includes:

1. Loading and exploring the dataset.
2. Preprocessing the data, including handling missing values and encoding categorical features.
3. Splitting the data into training and testing sets.
4. Training a Decision Tree Classifier model using both Gini impurity and Entropy as criteria.
5. Making predictions on the test set.
6. Evaluating the model's performance using metrics such as Accuracy, Precision, and F1 Score.
7. Visualizing the results to compare the performance of models trained with different criteria.

## Dataset

The dataset used in this project is `diabetes_data_upload.csv`. It contains various features related to patient health and a target variable indicating whether the individual is at early risk of diabetes.

**Note:** Please ensure you have the `diabetes_data_upload.csv` file in the same directory as the notebook or provide the correct path to the file.

## Data Preprocessing

The data preprocessing steps include:

- Checking for missing values (although the code shows no missing values in this dataset).
- Separating features (`X`) and the target variable (`y`).
- Applying one-hot encoding to the categorical features to convert them into a numerical format suitable for the model. The `drop_first=True` option is used to avoid multicollinearity.

## Model Training

A Decision Tree Classifier model is trained in this project. Two models are trained with different criteria for splitting nodes:

- **Model 1:** Trained using the Gini impurity criterion.
- **Model 2:** Trained using the Entropy criterion.

The data is split into training and testing sets using `train_test_split`.

Visualizations of the trained decision trees (both Gini and Entropy based) are generated and saved as PNG files (`diabetes_gini.png` and `diabetes_entropy.png`). A text representation of the Gini-based tree is also printed and saved to a file (`decision_tree.txt`).

## Prediction and Evaluation

The trained models are used to make predictions on the test set. The performance of each model is evaluated using the following metrics:

- **Accuracy:** The proportion of correctly classified instances.
- **Precision:** The ability of the model to correctly identify positive instances among all instances predicted as positive.
- **F1 Score:** The harmonic mean of Precision and Recall, providing a balance between the two.

The evaluation metrics are calculated and printed for both the Gini-based and Entropy-based models.

## Results

A bar plot is generated to visually compare the Accuracy, Precision, and F1 Score of the Gini-based and Entropy-based models. This plot helps in understanding which criterion results in better performance for this specific dataset.

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn` (scikit-learn)

You can install these dependencies using pip:

## How to Run

1. Clone this repository to your local machine.
2. Ensure you have the `diabetes_data_upload.csv` file in the correct location.
3. Install the required dependencies.
4. Open the provided Jupyter Notebook (or Google Colab notebook).
5. Run the cells sequentially to execute the data preprocessing, model training, prediction, and evaluation steps.

## Contributing

If you would like to contribute to this project, please feel free to fork the repository and submit a pull request.

**Colab Notebook Link:** [https://colab.research.google.com/drive/1cA1xhaLfCnnaDX-Odreo4YifeeMW6fuP#scrollTo=uztRqKckG9R9](https://colab.research.google.com/drive/1cA1xhaLfCnnaDX-Odreo4YifeeMW6fuP#scrollTo=uztRqKckG9R9)
