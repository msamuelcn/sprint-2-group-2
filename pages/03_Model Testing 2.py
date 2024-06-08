import streamlit as st
import pickle
import pandas as pd


pickle_in = open("random_forests_hypertuned.pkl", "rb")
model = pickle.load(pickle_in)

st.title('Impoverished Group Predictor')

df = pd.read_csv('data/20240602_fies_cleaned_.csv')


option = st.selectbox(
    "Select a houshold number.",
    tuple(df['SEQUENCE_NO'])
    )

st.write("You selected:", option)
