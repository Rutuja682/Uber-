# Uber Fare Prediction

## Project Overview  
This project develops a **predictive model for Uber fare estimation** using the `Uber.csv` dataset. 
The analysis includes **exploratory data analysis (EDA), feature engineering, and machine learning model evaluation** to improve fare predictions.

## Dataset Overview  
- **Source:** NYC Taxi Fare Data  
- **Records:** 50,000+ entries  
- **Features:** Pickup & dropoff locations, passenger count, fare amount, etc.  
- **Target:** `fare_amount` (Continuous variable)  

## Key Analyses Performed  

### 1. Exploratory Data Analysis (EDA)  
- **Univariate Analysis:** Distribution of fares, trip distances, passenger counts.  
- **Bivariate Analysis:** Relationships between fare and distance, pickup/drop-off locations.  
- **Multivariate Analysis:** Correlation heatmaps to detect feature relationships.  

### 2. Feature Engineering  
- **Distance Calculation:** Used the Haversine formula to compute trip distance.  
- **Time Features:** Extracted hour, weekday, and month from `pickup_datetime`.  
- **Outlier Detection:** Removed invalid fare values.  

### 3. Model Performance  
**Models Tested:**  
- **Linear Regression**  
- **Random Forest Regressor**  
- **XGBoost Regressor**  
- **Gradient Boosting Regressor**  
- **K-Nearest Neighbors (KNN)**  

**Evaluation Metrics:**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **Mean Absolute Error (MAE)**  
- **R-squared (R²) Score**  

### 4. Key Insights Discovered  
- **Distance & Fare:** Strong positive correlation.  
- **Time-based Trends:** Higher fares observed during peak hours.  
- **Best-Performing Models:** Random Forest and XGBoost yielded the most accurate fare predictions.  

## Model Performance Metrics  
### **Gradient Boosting Regressor**  
- **MSE:** 0.0154  
- **RMSE:** 0.1244  
- **MAE:** 0.0035  
- **R-squared:** 98.52%  

### **Decision Tree Regressor**  
- **MSE:** 0.0171  
- **RMSE:** 0.1310  
- **MAE:** 0.0011  
- **R-squared:** 98.36%  

### **XGBoost Regressor**  
- **MSE:** 0.0530  
- **RMSE:** 0.2303  
- **MAE:** 0.0118  
- **R-squared:** 94.93%  

### **K-Nearest Neighbors (KNN)**  
- **MSE:** 0.0154  
- **RMSE:** 0.1244  
- **MAE:** 0.0035  
- **R-squared:** 98.52%  

## Model Deployment  
- The best-performing model was deployed using **Flask API** for real-time fare prediction.  
- The deployment allows users to input trip details and get instant fare estimates.  
- The API is hosted on a cloud server for accessibility.  

## Hyperparameter Tuning  
- **RandomizedSearchCV** and **GridSearchCV** were used to optimize model parameters.  
- **XGBoost & Random Forest:** Tuned parameters such as learning rate, max depth, and n_estimators.  
- **Gradient Boosting:** Optimized learning rate and number of boosting stages.  

## Conclusion  
The **Random Forest and XGBoost Regressors** provided the best balance of performance and accuracy for Uber fare predictions. Distance and time-related features significantly influenced the fare amount, and feature engineering played a crucial role in enhancing model performance. Further improvements can be made by fine-tuning hyperparameters and incorporating additional real-world factors such as traffic conditions and weather data.


