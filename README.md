# NeuroFive ML Track

Exploratory Data Analysis (EDA) on the Titanic dataset as part of the NeuroFive ML learning track.

## Task 1: Initial EDA
- Data inspection using pandas (.info(), .describe(), .head())
- Identified missing values and column types

## Task 2: Data Cleaning and Visualization
- Missing value handling (fillna(), dropna())
- Outlier detection using boxplots
- Visualizations: histogram, boxplot, bar chart, correlation heatmap

## Task 3: Classification Model
- Predicting passenger survival using Logistic Regression (scikit-learn)
- Categorical columns (Sex, Embarked) encoded using pd.get_dummies()
- Data split 80/20 using train_test_split
- Final accuracy: 80%
- Evaluated using accuracy_score and a confusion matrix

## Files
- NeuroFive_ML.ipynb — main notebook with all analysis and modeling
- train.csv — Titanic dataset
