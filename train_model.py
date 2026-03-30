import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv('dataset/energydata_complete.csv')
# Clean data
data = data.dropna()

# Feature Engineering
data["temp_avg"] = (data["T1"] + data["T2"] + data["T_out"]) / 3
data["humidity_avg"] = (data["RH_1"] + data["RH_2"]) / 2
data["temp_diff"] = data["T1"] - data["T_out"]
data["energy_interaction"] = data["lights"] * data["T_out"]

# Select important columns
features = [
    "lights", "T1", "RH_1", "T2", "RH_2",
    "T_out", "Press_mm_hg", "Windspeed",
    "temp_avg", "humidity_avg", "temp_diff","energy_interaction"
]

X = data[features]
y = data["Appliances"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
   "Random Forest": RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
),
    "Gradient Boosting": GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6
)
}

best_model = None
best_error = float("inf")
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    error = mean_absolute_error(y_test, predictions)
    
    print(f"{name} MAE:", error)
    
    if error < best_error:
        best_error = error
        best_model = model
        best_model_name = name

print("\nBest Model:", best_model_name)
print("Best MAE:", best_error)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Save best model
joblib.dump(best_model, "model/energy_model.pkl")


print("Best model saved successfully!")