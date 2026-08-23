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

  ## Task 8: Ensemble Methods

- Trained and compared Logistic Regression, Random Forest, and XGBoost on Titanic data
- Compared feature importances between Random Forest and XGBoost
- Explained the difference between bagging (Random Forest) and boosting (XGBoost)

| Model | Accuracy |
|---|---|
| Logistic Regression | 81.01% |
| Random Forest | 82.12% |
| XGBoost | 80.45% |

  
## Task 9: Handling Imbalanced Data

- Dataset: Telco Customer Churn (reused from Task 6)
- Checked and visualized class balance (~73% No Churn, ~27% Churn)
- Fixed data leakage issue by properly excluding target columns from features
- Applied class_weight='balanced' to Logistic Regression to address imbalance
- Compared Precision/Recall/F1 before and after balancing

| Metric (Churn class) | Original Model | Balanced Model |
|---|---|---|
| Precision | 0.69 | 0.52 |
| Recall | 0.60 | 0.84 |
| F1-score | 0.64 | 0.64 |
| Overall Accuracy | 82% | 75% |

Class weighting significantly improved recall for the Churn class 
(60% to 84%), meaning the model now catches far more at-risk customers, 
at the cost of some precision and overall accuracy. For churn prediction, 
this trade-off is worthwhile since missing an actual churner is more 
costly than a false alarm.

## Live Demo
Try the app here: https://neurofive-ml-track-sngfcd4vcu8f8aremqz5nj.streamlit.app

## Files
- NeuroFive_ML.ipynb — Titanic EDA, cleaning, classification model, and tuning
- train.csv — Titanic dataset
- NeuroFive_Regression.ipynb — California housing regression model
- NeuroFive_Churn.ipynb — Telco customer churn prediction and ensemble models
- NeuroFive_Imbalance.ipynb — Handling imbalanced churn data
- WA_Fn-UseC_-Telco-Customer-Churn.csv — Telco churn dataset
