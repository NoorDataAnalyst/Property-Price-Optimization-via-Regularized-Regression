import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set page title and icon
st.set_page_config(page_title="Property Price Optimization", layout="wide", page_icon="🏡")

# --- DATA LOADING (Handles the long filename automatically) ---
@st.cache_data
def load_data():
    # Looking for your specific long filename
    target_file = "housing_ridge_lasso_dataset - housing_ridge_lasso_dataset.csv"
    if os.path.exists(target_file):
        return pd.read_csv(target_file)
    
    # Fallback: check for any CSV in the folder
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if files:
        return pd.read_csv(files[0])
    return None

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to:", ["1. Overview", "2. Data Visuals", "3. Model Results", "4. Analytical Q&A"])

if df is None:
    st.error("⚠️ Dataset not found in the folder!")
    st.info("Please ensure the CSV file is in the same folder as this script.")
    st.stop()

# --- PAGE 1: OVERVIEW ---
if page == "1. Overview":
    st.title("🏡 Property Price Prediction Overview")
    st.write("This application explores how Regularized Regression (Ridge and Lasso) improves house price predictions.")
    
    st.subheader("🛠️ Installation Troubleshooting")
    st.warning("If you see 'non-zero exit code' in your terminal:")
    st.markdown("""
    1. **Close all terminal windows.**
    2. Open a new Command Prompt as **Administrator**.
    3. Run: `pip install --upgrade pip`
    4. Then run: `pip install streamlit pandas numpy matplotlib seaborn scikit-learn`
    5. If using a virtual environment, ensure it is activated.
    """)

    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    st.subheader("Key Statistics")
    st.write(df.describe())

# --- PAGE 2: DATA VISUALS ---
elif page == "2. Data Visuals":
    st.title("📊 Exploratory Data Analysis")
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.info("Tip: Look for high correlation (dark red) between income features and house price.")

# --- PAGE 3: MODEL RESULTS ---
elif page == "3. Model Results":
    st.title("⚙️ Model Training & Evaluation")
    
    # Preprocessing
    X = df.drop('house_price', axis=1)
    y = df['house_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training Models
    lr = LinearRegression().fit(X_train_scaled, y_train)
    ridge = RidgeCV(alphas=np.logspace(-2, 2, 10)).fit(X_train_scaled, y_train)
    lasso = LassoCV(alphas=np.logspace(-2, 2, 10), max_iter=2000).fit(X_train_scaled, y_train)
    
    # Compare
    models = {"OLS": lr, "Ridge": ridge, "Lasso": lasso}
    results = []
    preds = {}
    
    for name, m in models.items():
        p = m.predict(X_test_scaled)
        preds[name] = p
        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, p),
            "RMSE": np.sqrt(mean_squared_error(y_test, p))
        })
    
    st.table(pd.DataFrame(results))
    
    # Residual Plot
    st.subheader("Residual Analysis Plot")
    fig_res, ax_res = plt.subplots(figsize=(10, 5))
    ax_res.scatter(preds["OLS"], y_test - preds["OLS"], color='blue', marker='o', label='OLS', alpha=0.5)
    ax_res.scatter(preds["Ridge"], y_test - preds["Ridge"], color='green', marker='x', label='Ridge', alpha=0.5)
    ax_res.scatter(preds["Lasso"], y_test - preds["Lasso"], color='red', marker='*', label='Lasso', alpha=0.5)
    ax_res.axhline(0, color='black', linestyle='--')
    ax_res.set_xlabel("Predicted House Price")
    ax_res.set_ylabel("Residuals (Error)")
    ax_res.legend()
    st.pyplot(fig_res)

# --- PAGE 4: Q&A ---
elif page == "4. Analytical Q&A":
    st.title("💡 Housing Analysis: Analytical Questions & Answers")
    st.markdown("""
### 1. What indicators in the dataset suggest the presence of multicollinearity?
Looking at the **Correlation Heatmap**, you can see very high correlation values (near 1.0) between variables like `median_income` and `income_index`, or `square_footage` and `living_area_index`. When two or more variables tell the "same story," it indicates multicollinearity.

### 2. Why might Ordinary Least Squares (OLS) regression perform poorly in this scenario?
OLS tries to find the best fit by looking at every variable equally. When you have multicollinearity (like the variables mentioned above), OLS becomes "unstable." It struggles to decide which variable is actually responsible for the price change, leading to wildly fluctuating coefficients and poor predictions on new data.

### 3. How would you detect overfitting in the baseline model?
You compare the model's performance on the **Training Data** vs. the **Test Data**. If your $R^2$ is very high (e.g., 0.95) on the training data but much lower (e.g., 0.70) on the test data, the model has "memorized" the noise in the training set rather than learning general patterns. This gap is a clear sign of overfitting.

### 4. What is the conceptual difference between L1 and L2 regularization?
* **L1 (Lasso):** Think of this as a "Slayer." It can shrink a coefficient all the way to **zero**, effectively removing that variable from the model.
* **L2 (Ridge):** Think of this as a "Shrinker." It reduces the size of coefficients but **never** makes them exactly zero. It keeps all variables but minimizes their impact.

### 5. How does Ridge regression affect coefficient magnitude compared to OLS?
Ridge regression adds a penalty based on the square of the coefficients. This forces the coefficients to be **smaller (closer to zero)** than they would be in OLS. It "mutes" the variables so that no single variable has an unfairly large influence on the price prediction.

### 6. Why does Lasso regression perform feature selection?
Because Lasso uses the absolute value of the coefficients in its penalty (L1), the mathematical "shape" of this penalty has sharp corners at zero. During optimization, it often hits these corners, forcing less important variables (like redundant indices) to have a coefficient of **exactly 0.0**, essentially "selecting" only the best features.

### 7. Which model would you prefer if interpretability is more important?
**Lasso Regression.** Because Lasso removes useless or redundant variables (by setting them to zero), you end up with a shorter, simpler list of factors that actually affect the house price. It is much easier to explain 5 key drivers to a client than 17 overlapping ones.

### 8. How does feature scaling influence Ridge and Lasso results?
Regularization penalizes the *size* of the coefficients. If `median_income` is in the tens of thousands and `bedrooms` is a single digit, the model will unfairly penalize the income variable just because its numbers are bigger. **Scaling** ensures all variables are on the same playing field (e.g., 0 to 1) so the penalty is applied fairly.

### 9. What happens when the regularization parameter (λ) is very large?
* As $\lambda$ (Alpha) increases, the penalty becomes heavier.
* The coefficients shrink closer and closer to zero.
* The model becomes **simpler** but potentially **underfits** (it becomes too "lazy" to learn the actual patterns in the data).

### 10. Can removing correlated variables manually be an alternative to regularization?
**Yes, but it's less efficient.** You could manually drop `income_index` if you already have `median_income`. However, in complex datasets with hundreds of variables, doing this manually is difficult and prone to human error. Regularization (especially Lasso) does this automatically and mathematically determines which variable is better to keep based on the actual target data.
    """)
