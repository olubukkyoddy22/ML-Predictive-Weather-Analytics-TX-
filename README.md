Source code for Predictive Analytics of Historical Weather in Texas from 2016-2020.py
#-------------------------------------------------------------
# Olubukola (Bukky) Odedairo
# Self-imposed project
#
# This project – “Predictive Analytics of historical Weather Pattern in Texas from 2016 – 2020, 
# using Python Programming Language” aims at developing a predictive model using some machine learning techniques # to analyze historical, raw weather data, to predict future weather tendencies. 
# By utilizing the dataset obtained from Kaggle, 
# this project analysis will identify some of the major weather trends from 2016 – 2020 weather report in Texas, # and employ predictive modeling to enhance the accuracy of the forecasting
# 
# Data was obtained from Kaggle.com   
# May 2025
#-------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

import pandas as pd
import numpy as np

#-------------------------------------------------------------
#Data Collection & Cleaning
#-------------------------------------------------------------
#Loading the Dataset
df = pd.read_csv("weather_2016_2020_daily.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

#Cleaning the dataset
df.set_index('Date', inplace=True)
full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(full_date_range)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

#Handle Mising values by using ffill(Forward Fill) and bfill(Backward Fill)
df.interpolate(method='linear', inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

#Convert date column to datetime format and set it as the index
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

#-------------------------------------------------------------
Identifying Patterns and Seasonality
#-------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

#Select the average temperature as target variable
target_column = 'Temp_avg'

#Visualizing average temperature trends
plt.figure(figsize=(15, 8))
df[target_column].plot(title='Avg Temperature Trend(2016-2020)', xlabel='Date', ylabel='Temp_avg')
plt.show()

#Correlation Analysis
df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


#-------------------------------------------------------------
Preprocessing & Feature Engineering
#-------------------------------------------------------------
#Scaling the features for better model performance
scaler = StandardScaler()
features = df.drop(columns=[target_column])
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=df.index)


#-------------------------------------------------------------
Training Multiple Models
#-------------------------------------------------------------
#Splitting dataset
X = features_scaled
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#Initializing models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
}

#Tain and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R_squared Score": r2_score(y_test, y_pred)
    }
    print(f"\n{name} Model Performance:")
    print(f"MAE: {results[name]['MAE']:.2f}")
    print(f"MSE: {results[name]['MSE']:.2f}")
    print(f"R_squared Score: {results[name]['R_squared Score']:.2f}"

import statsmodels.api as sm

#-------------------------------------------------------------
Training a Time-Series Model (OLS)
#-------------------------------------------------------------
#Train Time-Series Model (OLS)
X_train_const = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_const).fit()
y_pred_ols = ols_model.predict(sm.add_constant(X_test))

print("\nTime-Series Model Summary (OLS):\n", ols_model.summary())

#-------------------------------------------------------------
Comparing the models and selecting the best one
#-------------------------------------------------------------
#Finding the best model based on R_squared Score
best_model = max(results, key=lambda x: results[x]["R_squared Score"])
best_model_instance = models[best_model]
y_best_pred = best_model_instance.predict(X_test)

#Plotting the actual vs predicted values for the best model
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_best_pred, label=f"Predicted ({best_model})", linestyle="dashed")
plt.title(f"Actual vs Predicted Average Temperature ({best_model})")
plt.legend()
plt.show()

#-------------------------------------------------------------
Data forecasting using Prophet
#-------------------------------------------------------------
#reload the original dataset
df_or = pd.read_csv("weather_2016_2020_daily.csv")

#construct a new dataframe that contains the date and the target variable
df_prophet = df_or[["Date", "Temp_avg"]].copy()

df_prophet

#rename the date column as ds and the target variable as y
df_prophet.rename(columns={"Date":"ds", "Temp_avg":"y"}, inplace=True)
df_prophet

#specify the amount of the dataset to be used for training
train_size = int(len(df_prophet) * 0.8)
train, test = df_prophet[:train_size], df_prophet[train_size:]
print(f"Train Size: {len(train)}, Test Size: {len(test)}")

#load the Prophet model
model = Prophet()
model.fit(train)

#overfit and finetune the model
overfit_model = Prophet(changepoint_prior_scale=0.5)
overfit_model.fit(train)

tuned_model = Prophet(changepoint_prior_scale=0.05, seasonality_mode="additive")
tuned_model.fit(train)

#forecast future average tempearture for the period of 5 years, yhat contains the forecasted values
future = model.make_future_dataframe(periods=2190, freq="D")
forecast = tuned_model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

#visualize trends of the forecasted dataset
tuned_model.plot(forecast)
plt.show()

tuned_model.plot_components(forecast)
plt.show()

#display the forecasted average tempearture values of the 2024-12-09 to 2025-12-08
forecast_365 = forecast.iloc[-365:]
pd.set_option('display.max_rows',None)
print(forecast_365[['ds','yhat','yhat_lower','yhat_upper']])
#-------------------------------------------------------------

END OF CODE
