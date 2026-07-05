#Canada's_capital_yearwise_income_prediction

#Import Libraries and Load Data

#import the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_modelimport streamlit as st
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
# Add a slider/input for the user to select the year
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

# ==========================================
# Sidebar Branding (Fixed & Cleaned)
# ==========================================
st.sidebar.markdown("""
<div style='text-align:center; padding-top: 20px;'>

### 🚀 Developed by <br><span style="color:#4CAF50; font-size:20px; font-weight:bold;">ABHAY KUMAR GUPTA</span>

<br>

<a href="https://github.com/Abhay-cody" target="_blank">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" width="130">
</a>

<br><br>

<a href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397" target="_blank">
<img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="130">
</a>

<br><br>
<hr>

⭐ Thank you for visiting this Machine Learning Project.

</div>
""", unsafe_allow_html=True)


# Load the dataset
df = pd.read_csv('canada_per_capita_income.csv')

# Let's look at the first few rows to understand the structure
print(df.head())

#Visualize the Data

plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+')
plt.show()

"""#Prepare the Data & Train the Model"""

# X must be a 2D array/dataframe
X = df[['year']]

# y can be a 1D series
y = df['per capita income (US$)']

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X, y)

"""#Make the Prediction"""

# Predict the income for the year 2020
predicted_income = reg.predict([[2020]])
print(f"Predicted per capita income for 2020: {predicted_income[0]}")

"""#Double-Checking the Math"""

# Show the slope (m)
print("Coefficient (m):", reg.coef_)

# Show the intercept (b)
print("Intercept (b):", reg.intercept_)

"""#Plotting the Regression line"""

# Plotting the regression line
plt.plot(df['year'], reg.predict(df[['year']]), color='blue', label='Regression Line')
plt.legend()
plt.show()

# ==============================
# Sidebar Branding
# ==============================

st.markdown("---")

st.markdown("""
<div style='text-align:center;'>

### 🚀 Developed by <span style="color:#4CAF50;">ABHAY KUMAR GUPTA</span>

<a href="https://github.com/Abhay-cody" target="_blank">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
</a>

&nbsp;

<a href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397" target="_blank">
<img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
</a>

<br><br>

⭐ Thank you for visiting this Machine Learning Project.

</div>
""", unsafe_allow_html=True)
