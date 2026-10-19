import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create an example dataset
data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 8, 3, 5, 7, 9, 1, 4, 6, 10
    ],
    "attendance": [
        50, 55, 60, 65, 70, 75, 80, 85, 60, 70,
        80, 90, 65, 75, 85, 95, 45, 68, 78, 98
    ],
    "marks": [
        25, 32, 38, 45, 52, 60, 68, 76, 35, 48,
        63, 82, 42, 56, 73, 90, 20, 46, 62, 95
    ]
}

df = pd.DataFrame(data)

# 2. Select input features and target
X = df[["study_hours", "attendance"]]
y = df["marks"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on the test set
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print(f"Mean Absolute Error: {mae:.2f} marks")
print(f"R² score: {r2:.2f}")

# 6. Try a prediction
new_student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [80]
})

predicted_marks = model.predict(new_student)[0]

print(f"Predicted marks for the example student: {predicted_marks:.2f}")


joblib.dump(model, "student_marks_model.pkl")

print("Trained model saved successfully!")