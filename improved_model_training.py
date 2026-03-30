import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== IMPROVED MODEL TRAINING FOR HIGHER ACCURACY ===\n")

# Load and analyze data
data = pd.read_csv('dataset/energydata_complete.csv')
print(f"Original dataset shape: {data.shape}")
print(f"Missing values: {data.isnull().sum().sum()}")

# Data cleaning
data = data.dropna()
print(f"After removing missing values: {data.shape}")

# Convert date to datetime and extract time features
data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# Advanced feature engineering
print("\n=== ADVANCED FEATURE ENGINEERING ===")

# Temperature-based features
data['temp_range'] = data['T1'] - data['T2']
data['temp_out_diff'] = data['T1'] - data['T_out']
data['temp_avg_all'] = (data['T1'] + data['T2'] + data['T3'] + data['T4'] + data['T5'] + 
                        data['T6'] + data['T7'] + data['T8'] + data['T9']) / 9

# Humidity-based features
data['humidity_range'] = data['RH_1'] - data['RH_2']
data['humidity_avg_all'] = (data['RH_1'] + data['RH_2'] + data['RH_3'] + data['RH_4'] + 
                           data['RH_5'] + data['RH_6'] + data['RH_7'] + data['RH_8'] + data['RH_9']) / 9

# Interaction features
data['temp_humidity_interaction'] = data['temp_avg_all'] * data['humidity_avg_all']
data['lights_temp_interaction'] = data['lights'] * data['T_out']
data['pressure_temp_interaction'] = data['Press_mm_hg'] * data['temp_avg_all']
data['wind_temp_interaction'] = data['Windspeed'] * data['temp_avg_all']

# Time-based features
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Cyclical features for energy patterns
data['is_peak_hour'] = ((data['hour'] >= 18) & (data['hour'] <= 22)).astype(int)
data['is_night'] = ((data['hour'] >= 0) & (data['hour'] <= 6)).astype(int)

# Weather comfort index
data['comfort_index'] = (data['temp_avg_all'] - 20) / 10 + (data['humidity_avg_all'] - 50) / 20

# Energy efficiency indicators
data['lights_per_temp'] = data['lights'] / (data['temp_avg_all'] + 1)
data['energy_efficiency_score'] = data['lights'] / (data['Appliances'] + 1)

# Select all relevant features
feature_columns = [
    # Original features
    'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T_out', 'Press_mm_hg', 'Windspeed',
    # Engineered features
    'temp_range', 'temp_out_diff', 'temp_avg_all', 'humidity_range', 'humidity_avg_all',
    'temp_humidity_interaction', 'lights_temp_interaction', 'pressure_temp_interaction',
    'wind_temp_interaction', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_peak_hour', 'is_night', 'is_weekend', 'comfort_index', 'lights_per_temp'
]

X = data[feature_columns]
y = data['Appliances']

print(f"Total features created: {len(feature_columns)}")
print(f"Training data shape: {X.shape}")

# Remove outliers using IQR method
print("\n=== OUTLIER REMOVAL ===")
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (y >= lower_bound) & (y <= upper_bound)
X_clean = X[outlier_mask]
y_clean = y[outlier_mask]

print(f"Removed {len(y) - len(y_clean)} outliers ({(len(y) - len(y_clean))/len(y)*100:.1f}%)")
print(f"Clean dataset shape: {X_clean.shape}")

# Feature selection
print("\n=== FEATURE SELECTION ===")
selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(X_clean, y_clean)
selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]

print(f"Selected top {len(selected_features)} features:")
for i, feature in enumerate(selected_features):
    print(f"{i+1:2d}. {feature}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_clean, test_size=0.2, random_state=42
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define improved models with hyperparameter tuning
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
}

# Train and evaluate models
print("\n=== MODEL TRAINING AND EVALUATION ===")
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    if name in ['Linear Regression', 'Ridge Regression']:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    accuracy = (1 - mae / y_test.mean()) * 100
    
    results[name] = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy,
        'model': model
    }
    
    print(f"  MAE: {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  R²: {r2:.4f}")
    print(f"  Accuracy: {accuracy:.1f}%")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
print(f"Best Accuracy: {best_accuracy:.1f}%")
print(f"Best MAE: {results[best_model_name]['mae']:.2f} Wh")
print(f"Best R²: {results[best_model_name]['r2']:.4f}")

# Save the best model and preprocessing objects
joblib.dump(best_model, 'model/improved_energy_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(selector, 'model/feature_selector.pkl')
joblib.dump(selected_features, 'model/selected_features.pkl')

print("\n=== IMPROVED MODEL SAVED ===")
print("Files saved:")
print("- improved_energy_model.pkl")
print("- scaler.pkl")
print("- feature_selector.pkl")
print("- selected_features.pkl")

# Compare with original model
print(f"\n=== ACCURACY IMPROVEMENT ===")
original_accuracy = 64.5
improvement = best_accuracy - original_accuracy
print(f"Original Model Accuracy: {original_accuracy:.1f}%")
print(f"Improved Model Accuracy: {best_accuracy:.1f}%")
print(f"Improvement: +{improvement:.1f}% points ({improvement/original_accuracy*100:.1f}% relative improvement)")

if improvement > 5:
    print("🎉 SIGNIFICANT IMPROVEMENT ACHIEVED!")
elif improvement > 2:
    print("✅ GOOD IMPROVEMENT ACHIEVED!")
else:
    print("⚠️  Limited improvement - consider more advanced techniques")
