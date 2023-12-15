# Diabetes Analysis and Prediction

## Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The aim is to diagnose whether a patient has diabetes based on diagnostic measurements.

## Content
- **Importing Libraries**: Setup and preparation with required Python libraries.
- **Loading the Dataset**: Handling the dataset and preparing it for analysis.
- **Exploring the Dataset**: Understanding the dataset's structure and contents.
- **Data Preprocessing**: Cleaning and preparing data for modeling.
- **Looking for Missing Values**: Identifying and addressing missing data.
- **Looking for Correlation**: Analyzing relationships between features.
- **Model Training**: Building and training the Logistic Regression model.
- **Prediction Making**: Utilizing the model to make predictions.
- **Checking Accuracy**: Evaluating the model's performance.
- **Dashboard**: Developing a web-based interface for users to make predictions using the model.

### Patient Selection Criteria
All individuals in this dataset are female, at least 21 years old, and of Pima Indian heritage.

### Feature Description
- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function.
- **Age**: Age in years.
- **Outcome**: Class variable (0 or 1) indicating if the patient has diabetes.

## Tools and Technologies
- **Programming Language**: Python
- **Notebook Environment**: Jupyter
- **Web Framework**: Django
- **Machine Learning Algorithm**: Logistic Regression
- **Model Validation**: train_test_split, accuracy_score
- **Data Visualization**: scatter_matrix
- **Libraries**: pandas, numpy, matplotlib, seaborn, missingno

## About the Dataset
The dataset's purpose is to diagnostically predict whether or not a patient has diabetes, based on a set of diagnostic measurements included in the dataset. The selection of instances adheres to a specific constraint, ensuring all patients are female with a minimum age of 21 years and of Pima Indian heritage.

## Column Description
The dataset comprises several medical predictor variables and one target variable, Outcome. The independent variables include:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure
- **SkinThickness**: Triceps skin fold thickness
- **Insulin**: 2-Hour serum insulin
- **BMI**: Body Mass Index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age of the patient

The dependent variable is:

- **Outcome**: Indicates whether the patient has diabetes (1) or not (0).

## Project Structure
The repository is structured as follows:
- **Data Analysis Notebooks**: Jupyter notebooks containing the exploratory data analysis, data cleaning, and preprocessing steps.
- **Prediction Application**: Django application code for the interactive prediction model.
- **Models**: Trained machine learning models used for prediction.
- **Data**: The `diabetesDatabase.csv` file used for analysis and model training.
- **Visualizations**: Graphs and charts generated during the analysis.


