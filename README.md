# Property Price Optimization: Regularized Regression Analysis

This project implements a predictive modeling pipeline to estimate property values using **Ridge (L2)** and **Lasso (L1) Regularized Regression**. By leveraging socioeconomic and physical property data, the model identifies key price drivers while preventing overfitting in a high-dimensional feature space.

---

## 1. Project Overview

Predicting house prices is a classic regression problem. However, real estate data often suffers from **multicollinearity** (highly correlated predictors) and redundant features. This project addresses these issues by:

- Normalizing distributions to meet linear regression assumptions.
- Applying Ridge Regression to manage multicollinearity.
- Utilizing Lasso Regression for automated feature selection and model parsimony.

---

## 2. Dataset Description

The analysis is based on `housing_ridge_lasso_dataset.csv`, which includes **17 variables**:

### Physical & Structural Features
- **bedrooms / bathrooms**: Number of rooms.
- **square_footage**: Total living area.
- **property_age**: Age of the structure.
- **total_rooms**: Aggregate count of all rooms.
- **living_area_index**: A composite metric of habitable space.

### Location & Socioeconomic Features
- **distance_to_city**: Proximity to the central business district.
- **crime_rate**: Safety metric for the neighborhood.
- **school_quality**: Rating of local educational institutions.
- **median_income / income_index**: Wealth metrics of the area residents.
- **employment_rate**: Local economic health indicator.

### External & Environmental
- **property_tax_rate**: Annual tax obligations.
- **environment_quality**: Rating of air quality and green spaces.

### Target Variable
- **house_price**: The market value to be predicted.

---

## 3. Workflow & Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Examining the skewness of `house_price`. Log transformation is applied if the target is non-normally distributed.
- **Multicollinearity Check**: Using Heatmaps to identify redundant features (e.g., `median_income` vs. `income_index`).
<Figure size 1200x800 with 2 Axes><img width="1048" height="821" alt="image" src="https://github.com/user-attachments/assets/1d401d21-c7da-4167-b752-271de5a1997d" />

### Phase 2: Preprocessing
- **Feature Scaling**: Since Regularized models (Ridge/Lasso) are sensitive to the magnitude of predictors, `StandardScaler` is used to bring all features to a common scale.
- **Missing Value Imputation**: Handling any null entries using median or mode strategies.

### Phase 3: Model Training & Hyperparameter Tuning
- **Ridge (L2)**: Shrinks coefficients of correlated features to reduce variance.
- **Lasso (L1)**: Performs feature selection by zeroing out coefficients of non-influential variables.
- **Cross-Validation**: Used to find the optimal penalty term (α).

### Phase 4: Evaluation
Models are evaluated based on:
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²) Score**
- **Residual Analysis**: Ensuring errors are randomly distributed around zero.
<Figure size 1500x600 with 1 Axes><img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/9a60d488-0eb5-4aa1-b885-5f7f71676b5a" />

---

## 4. Key Requirements
- Python 3.x
- NumPy & Pandas
- Scikit-Learn
- Matplotlib & Seaborn

---

**Note:** This project is part of a property price optimization study focusing on the impact of socioeconomic factors on urban real estate valuation.
