import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Movie Rating Prediction",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Rating Prediction")
st.write("Predict the IMDb rating of a movie using Machine Learning.")

# Load Dataset
df = pd.read_csv("movratpre.csv")
df.dropna(inplace=True)

genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(df["Genre"])
df["Director"] = director_encoder.fit_transform(df["Director"])

df.drop("Movie", axis=1, inplace=True)

X = df.drop("Rating", axis=1)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.header("Enter Movie Details")

genre = st.selectbox("Genre", list(genre_encoder.classes_))

director = st.selectbox("Director", list(director_encoder.classes_))

year = st.slider("Release Year", 1990, 2025, 2024)

duration = st.slider("Duration (minutes)", 80, 220, 140)

votes = st.number_input(
    "IMDb Votes",
    min_value=1000,
    max_value=5000000,
    value=500000,
    step=1000
)

if st.button("Predict Rating"):

    genre_encoded = genre_encoder.transform([genre])[0]
    director_encoded = director_encoder.transform([director])[0]

    movie = pd.DataFrame({
        "Genre": [genre_encoded],
        "Director": [director_encoded],
        "Year": [year],
        "Duration": [duration],
        "Votes": [votes]
    })

    prediction = model.predict(movie)[0]

    prediction = max(0, min(10, prediction))

    st.success(f"⭐ Predicted Movie Rating: **{prediction:.2f}/10**")

    if prediction >= 8.5:
        st.balloons()
        st.info("Excellent movie! 🌟")
    elif prediction >= 7:
        st.info("Good movie! 👍")
    elif prediction >= 5:
        st.warning("Average movie 🙂")
    else:
        st.error("Below average movie 👎")

st.markdown("---")
st.caption("Developed by AJAY DIVVI")