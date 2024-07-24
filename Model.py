 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("housingdata.csv")

# Replace NaN values with the mean of their respective columns
df_filled = df.fillna(df.mean())

# Split the dataset into features (X) and target variable (y)
X = df_filled.drop(columns=['MEDV'])  # Features
y = df_filled['MEDV']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train and evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, mae, r2

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=0.1),
    "Elastic Net Regression": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Evaluate each model
results = {}
for name, model in models.items():
    y_pred, rmse, mae, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {
        "Predictions": y_pred,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }

# Print the evaluation results
results_df = pd.DataFrame(results).T
print(results_df[["RMSE", "MAE", "R²"]])

# Plotting the results
for name, result in results.items():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, result["Predictions"], color='blue', label='Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
    plt.title(f'{name}')
    plt.xlabel('Actual MEDV')
    plt.ylabel('Predicted MEDV')
    plt.legend()
    plt.show()

# Comparison Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["RMSE", "MAE", "R²"]
for i, metric in enumerate(metrics):
    axes[i].bar(results.keys(), results_df[metric])
    axes[i].set_title(f'Model Comparison: {metric}')
    axes[i].set_ylabel(metric)
    axes[i].set_xticklabels(results.keys(), rotation=45)

plt.tight_layout()
plt.show()
