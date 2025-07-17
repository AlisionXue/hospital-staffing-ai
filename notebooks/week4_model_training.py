import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load preprocessed data
df = pd.read_csv('data/hospital_week4_timeseries_lagged.csv')

# 2. Define features and target
features = ['treatment_count_lag1', 'efficiency_lag1']
target = 'treatment_count'

# 3. Split data into train, val, and test sets
train = df[df['dataset_split'] == 'train']
val   = df[df['dataset_split'] == 'val']
test  = df[df['dataset_split'] == 'test']

# 4. Prepare feature matrices and target vectors
X_train = train[features]
y_train = train[target]
X_val   = val[features]
y_val   = val[target]
X_test  = test[features]
y_test  = test[target]

# 5. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# 7. Evaluate performance using RMSE and R²
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))  # RMSE for validation set
val_r2   = r2_score(y_val, val_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))  # RMSE for test set
test_r2   = r2_score(y_test, test_pred)

# 8. Print evaluation results
print("📊 Validation Set:")
print(f"RMSE: {val_rmse:.2f}")
print(f"R²:   {val_r2:.2f}")

print("\n📉 Test Set:")
print(f"RMSE: {test_rmse:.2f}")
print(f"R²:   {test_r2:.2f}")

# 9. Visualize prediction vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(test_pred, label='Predicted', marker='x')
plt.title('Test Set Prediction vs Actual')
plt.xlabel('Sample Index')
plt.ylabel('Treatment Count')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 10. Save plot to file
plt.savefig('docs/week4_model_test_plot.png')
plt.show()
