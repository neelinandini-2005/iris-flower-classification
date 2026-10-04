import streamlit as st
import joblib
import pandas as pd

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Classification 🌸")
st.write("Enter the flower measurements to predict its species.")

sepal_length = st.number_input(
    "Sepal length (cm)", min_value=0.0, value=5.1
)

sepal_width = st.number_input(
    "Sepal width (cm)", min_value=0.0, value=3.5
)

petal_length = st.number_input(
    "Petal length (cm)", min_value=0.0, value=1.4
)

petal_width = st.number_input(
    "Petal width (cm)", min_value=0.0, value=0.2
)

if st.button("Predict Flower"):

    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=model.feature_names_in_
    )

    prediction = model.predict(features)[0]

    st.success(f"Predicted flower species: {prediction}")
