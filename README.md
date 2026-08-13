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

  ## Task 5: Model Evaluation & Tuning

- Calculated Precision, Recall, F1-score using classification_report
- Explained why accuracy alone is misleading for imbalanced data
- Tuned hyperparameters (C, solver) using GridSearchCV with 5-fold CV
- Compared original vs tuned model performance
- Original Accuracy: 80%

## Task 6: Customer Churn Prediction

- Dataset: Telco Customer Churn (Kaggle)
- Compared Decision Tree and Logistic Regression classifiers
- Handled categorical variables with pd.get_dummies()
- Noted class imbalance in Churn column (~73% No, ~27% Yes)
- Top churn drivers: Contract type, Tenure, Monthly Charges
- Business summary written for non-technical stakeholders

 ## Task 7: ML Pipeline

- Built a scikit-learn Pipeline using ColumnTransformer (StandardScaler + OneHotEncoder)
- Combined preprocessing and Logistic Regression into one pipeline object
- Added 2 engineered features: FamilySize and IsAlone
- Compared pipeline accuracy with and without engineered features
- Saved the final trained pipeline using joblib
  
## Files
- NeuroFive_ML.ipynb — Titanic EDA, cleaning, and classification model
- train.csv — Titanic dataset
- NeuroFive_Regression.ipynb — California housing regression model
- NeuroFive_Churn.ipynb — Telco customer churn prediction model
- WA_Fn-UseC_-Telco-Customer-Churn.csv — Telco churn dataset
