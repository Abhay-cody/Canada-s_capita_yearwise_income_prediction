import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Set up page configuration
st.set_page_config(page_title="Canada Income Predictor", layout="centered")

# App Header
st.title("🇨🇦 Canada's Capital Yearwise Income Prediction")
st.write("This app uses a simple **Linear Regression** model to predict Canada's per capita income based on historical data.")

# ==========================================
# 1. Load the Dataset
# ==========================================
try:
    df = pd.read_csv('canada_per_capita_income.csv')
except FileNotFoundError:
    st.error("Error: 'canada_per_capita_income.csv' file not found. Make sure it's in the same directory.")
    st.stop()

# Show raw data inside an expander
with st.expander("📊 View Historical Data"):
    st.dataframe(df)

# ==========================================
# 2. Prepare Data & Train Model
# ==========================================
X = df[['year']]
y = df['per capita income (US$)']

reg = linear_model.LinearRegression()
reg.fit(X, y)

# ==========================================
# 3. Interactive User Input & Prediction
# ==========================================
st.subheader("🔮 Make a Prediction")
# Add an input for the user to select the year
target_year = st.number_input("Enter or select a year to predict:", min_value=1970, max_value=2050, value=2020, step=1)

# Predict the income
predicted_income = reg.predict([[target_year]])

# Display the result prominently
st.success(f"💰 **Predicted Per Capita Income for {target_year}:** ${predicted_income[0]:,.2f} USD")

# ==========================================
# 4. Double-Checking the Math (Expander)
# ==========================================
with st.expander("🧮 View Model Math Equation (y = mx + b)"):
    m = reg.coef_[0]
    b = reg.intercept_
    st.write(f"**Slope (m):** `{m:.4f}`")
    st.write(f"**Intercept (b):** `{b:.4f}`")
    st.latex(f"\\text{{Income}} = ({m:.2f} \\times {target_year}) + ({b:.2f})")
    st.write(f"**Calculated Result:** ${m * target_year + b:,.2f}")

# ==========================================
# 5. Full Visualisation Plot
# ==========================================
st.subheader("📈 Visualization & Regression Line")

fig, ax = plt.subplots(figsize=(10, 6))
# Plot actual data points
ax.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+', label='Actual Data', s=60)
# Plot the trendline
ax.plot(df['year'], reg.predict(df[['year']]), color='blue', linewidth=2, label='Regression Line')
# Highlight the user's predicted point
ax.scatter(target_year, predicted_income, color='green', marker='o', s=150, label=f'Prediction ({target_year})', zorder=5)

# Graph styling
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Per Capita Income (US$)', fontsize=12)
ax.set_title("Canada's Per Capita Income Trend Over Time", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11)

# Pass the matplotlib figure to streamlit
st.pyplot(fig)

# ==============================
# Main Page Bottom Footer Branding
# ==============================
st.markdown("---")

# Using pure HTML inside markdown to ensure perfect centering and layout execution
st.markdown("""
<div style='text-align: center; margin-top: 30px; padding-bottom: 20px;'>
    <h3 style='margin-bottom: 15px;'>🚀 Developed by <span style="color:#4CAF50;">ABHAY KUMAR GUPTA</span></h3>
    <div style='display: flex; justify-content: center; gap: 15px; margin-bottom: 20px;'>
        <a href="https://github.com/Abhay-cody" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
    </div>
    <p style='color: #888888; font-size: 14px;'>⭐ Thank you for visiting this Machine Learning Project.</p>
</div>
""", unsafe_allow_html=True)
