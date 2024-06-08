import streamlit as st
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# pickle_in = open("random_forests_hypertuned.pkl", "rb")
# model = pickle.load(pickle_in)

def predict(X_data, m):
    prediction = m.predict(X_data)
    return prediction


st.title('Impoverished Group Predictor')

df = pd.read_csv('data/20240602_fies_cleaned_.csv')
to_drop = ['SEQUENCE_NO']
df_testing = df.drop(to_drop, axis=1)
for col in df_testing.columns:
  print(col)

# Define features (X) and target(y)
X = df_testing.drop(['IS_POVERTY'], axis=1) # feature, remove is_poverty
y = df_testing['IS_POVERTY'] # target
# Split the data
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(X, y, random_state=1337, test_size=0.25,
                                                                  stratify=y)

model = RandomForestClassifier(max_depth=8, max_features='log2', n_estimators=200,
                       random_state=1337, n_jobs=-1)
model.fit(X_trainval, y_trainval)


option = st.selectbox(
    "Select a houshold number.",
    tuple(df['SEQUENCE_NO'])
    )

st.write("You selected:", option)

if st.button('Predict the household'):
    predictIsPoverty = predict(X.loc[option],model)

    predictIsPovertyWord = 'positive' if predictIsPoverty ==1 else 'negative'
    st.success(f'The predicted household no.'+ option+' is '+predictIsPovertyWord+'.')

