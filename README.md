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

## Task 4: Regression Model
- Predicted median house value using Linear Regression (scikit-learn)
- Dataset: California housing dataset (sklearn.datasets)
- Features used: MedInc, HouseAge, AveRooms, AveBedrms, Population
- Data split 80/20 using train_test_split
- RMSE: 0.802 (~$80,200 average prediction error)
- R² Score: 0.509 (model explains ~51% of price variation)
- Predicted vs actual values visualized with a scatter plot

## Files
- NeuroFive_ML.ipynb — Titanic EDA, cleaning, and classification model
- train.csv — Titanic dataset
- NeuroFive_Regression.ipynb — California housing regression model
