#Build Linear Regression Model and Apply Regularization –Ridge, Lasso,and Elastic Net Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("housingdata.csv")

# Replace NaN values with the mean of their respective columns
df_filled = df.fillna(df.mean())

# Split the dataset into features (X) and target variable (y)
X = df_filled.drop(columns=['MEDV'])  # Features
y = df_filled['MEDV']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
print("Simple Linear Regression RMSE:", lr_rmse)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_rmse = mean_squared_error(y_test, lasso_pred, squared=False)
print("Lasso Regression RMSE:", lasso_rmse)

# Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_pred, squared=False)
print("Ridge Regression RMSE:", ridge_rmse)

# Elastic Net Regression
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
elasticnet_pred = elasticnet_model.predict(X_test)
elasticnet_rmse = mean_squared_error(y_test, elasticnet_pred, squared=False)
print("Elastic Net Regression RMSE:", elasticnet_rmse)
import matplotlib.pyplot as plt

# Scatter plot for Simple Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Simple Linear Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()

# Scatter plot for Lasso Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lasso_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Lasso Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()

# Scatter plot for Ridge Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, ridge_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Ridge Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()

# Scatter plot for Elastic Net Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, elasticnet_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Elastic Net Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()
