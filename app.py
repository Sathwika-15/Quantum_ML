import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("C:\\Users\\Sathwika\\OneDrive\\Desktop\\BTP 2\\quantum_data.csv")

# Features
df['alpha_sq'] = df['alpha']**2
df['beta_sq'] = df['beta']**2
df['difference'] = abs(df['alpha'] - df['beta'])
df['product'] = df['alpha'] * df['beta']

X = df[['alpha', 'beta', 'alpha_sq', 'beta_sq', 'difference', 'product']]
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

st.title("Quantum State Classifier")

alpha = st.slider("Alpha", 0.0, 1.0, 0.5)
beta = np.sqrt(1 - alpha**2)

alpha_sq = alpha**2
beta_sq = beta**2
difference = abs(alpha - beta)
product = alpha * beta

input_data = [[alpha, beta, alpha_sq, beta_sq, difference, product]]

prediction = model.predict(input_data)

st.write("Beta:", beta)

if prediction[0] == 1:
    st.success("Entangled-like State")
else:
    st.error("Separable-like State")