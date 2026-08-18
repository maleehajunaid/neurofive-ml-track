import streamlit as st
import joblib
import pandas as pd

model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Passenger details daalo, predict karo survive hoga ya nahi.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare Paid", 0.0, 512.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]],
                          columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.success(f"✅ Likely to Survive (Probability: {probability:.1%})")
    else:
        st.error(f"❌ Unlikely to Survive (Probability: {probability:.1%})")
