import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("hdi_sample_dataset.csv")

# Input features
X = df[[
    "Life_Expectancy",
    "Mean_Years_of_Schooling",
    "Expected_Years_of_Schooling",
    "GNI_Per_Capita"
]]

# Target
y = df["HDI"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")
print("✅ model.pkl created")
