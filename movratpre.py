import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Model Performance")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score:", round(r2, 2))

genre = input("Enter Genre: ")
director = input("Enter Director: ")
year = int(input("Enter Release Year: "))
duration = int(input("Enter Duration (minutes): "))
votes = int(input("Enter Number of Votes: "))

if genre not in genre_encoder.classes_:
    genre = genre_encoder.classes_[0]

if director not in director_encoder.classes_:
    director = director_encoder.classes_[0]

genre_encoded = genre_encoder.transform([genre])[0]
director_encoded = director_encoder.transform([director])[0]

new_movie = pd.DataFrame({
    "Genre": [genre_encoded],
    "Director": [director_encoded],
    "Year": [year],
    "Duration": [duration],
    "Votes": [votes]
})

predicted_rating = model.predict(new_movie)

print("Predicted Movie Rating:", round(predicted_rating[0], 2))