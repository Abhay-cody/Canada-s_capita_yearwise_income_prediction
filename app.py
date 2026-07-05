#Canada's_capital_yearwise_income_prediction

#Import Libraries and Load Data

#import the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


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
