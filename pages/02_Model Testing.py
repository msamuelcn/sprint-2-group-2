import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

with open("random_forests_hypertuned.pkl", "rb") as f:
    model = pickle.load(f)


#Caching the model for faster loading
@st.cache

# Define the prediction function
def predict(X_data, m):
    prediction = m.predict(X_data)
    return prediction


st.title('Impoverished Group Predictor')
#st.header('Enter the characteristics of the diamond:')
#carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
#cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
#color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
#clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
#depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0)
#table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)
#x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)
#y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)
#z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)

df = pd.read_csv('data/20240602_fies_cleaned_.csv')


option = st.selectbox(
    "Select a houshold number.",
    tuple(df['SEQUENCE_NO'])
    )

st.write("You selected:", option)

# if st.button('Predict Price'):
#     price = predict(X_data,model)
#     st.success(f'The predicted price of the diamond is ${price[0]:.2f} USD')


