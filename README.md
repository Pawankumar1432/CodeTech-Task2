# CodeTech-Task2
# Housing Price Prediction with Regression Models
## Overview
This project focuses on building and evaluating multiple regression models to predict housing prices. We use Linear Regression, Lasso Regression, Ridge Regression, and Elastic Net Regression and compare their performances using various evaluation metrics.

## Dataset
The dataset used in this project is `housingdata.csv`, which contains features related to housing and the target variable `MEDV` (Median value of owner-occupied homes in $1000s).

## Project Structure
- `housingdata.csv`: The dataset used for training and testing the models.
- `model_comparison.py`: The main script for training, evaluating, and comparing the regression models.
- `README.md`: Project documentation.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

You can install the required packages using:
pip install pandas numpy scikit-learn matplotlib

##Usage
1.Clone the repository:

git clone  https://github.com/Pawankumar1432/CodeTech-Task2.git
cd housing-price-prediction

2.Ensure housingdata.csv is in the project directory.

3.Run the script:

python model_comparison.py


##Methodology
->Data Preprocessing

->Replace NaN values with the mean of their respective columns

->Model Training and Evaluation

->The following models were trained and evaluated:

  1)Linear Regression
  2)Lasso Regression
  3)Ridge Regression
  4)Elastic Net Regression
For each model, the following evaluation metrics were calculated:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R-squared (R²)

Results
Model	           RMSE	MAE	R²
Linear Regression	X	   Y	Z
Lasso Regression	A	   B	C
Ridge Regression	D	   E	F
Elastic Net Regression	G	H	I
(Note: Replace X, Y, Z, A, B, C, D, E, F, G, H, I with actual values.)

##Visualization
Scatter plots were created to visualize the actual vs. predicted values for each model. Additionally, bar plots were used to compare RMSE, MAE, and R² across models.

##Insights
Regularization techniques (Lasso, Ridge, Elastic Net) can significantly impact model performance, especially when dealing with multicollinearity.
Visualization tools are crucial for understanding model predictions versus actual values.

##Next Steps
Fine-tune hyperparameters for further optimization.
Explore additional regularization techniques and advanced models.

##Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

##License
This project is licensed under the MIT License.

##Contact
For any questions or suggestions, feel free to reach out to byallpawan@gmail.com.
