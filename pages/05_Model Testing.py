import streamlit as st
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# pickle_in = open("random_forests_hypertuned.pkl", "rb")
# model = pickle.load(pickle_in)

def predict(X_data, m):
    prediction = m.predict(pd.DataFrame(X_data))
    return prediction


st.title('Impoverished Group Predictor')

df = pd.read_csv('data/20240602_fies_cleaned_.csv')
to_drop = ['SEQUENCE_NO']
df_testing = df.drop(to_drop, axis=1)

# Define features (X) and target(y)
X = df_testing.drop(['IS_POVERTY'], axis=1) # feature, remove is_poverty
y = df_testing['IS_POVERTY'] # target
# Split the data
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(X, y, random_state=1337, test_size=0.25,
                                                                  stratify=y)

model = RandomForestClassifier(max_depth=8, max_features='log2', n_estimators=200,
                       random_state=1337, n_jobs=-1)
model.fit(X_trainval, y_trainval)


# option = st.selectbox(
#     "Select a houshold number.",
#     tuple(df['SEQUENCE_NO'])
#     )
@st.cache
def predict1(option):
    predicted = model.predict(X)
    return predicted[option]


st.caption('Try the following household number below.')
sample_data = pd.DataFrame({
  'Household number':[82337,
96528,
98588,
107113,
157581,12188,15442,44205,26660,54774],
  'Class':['Below poverty line','Below poverty line','Below poverty line','Below poverty line','Below poverty line',
           'Above poverty line','Above poverty line','Above poverty line','Above poverty line','Above poverty line'
           ]
})
st.table(sample_data)

option  = st.number_input('Select a houshold number.', min_value=1, max_value=len(X), value=1)

if st.button('Predict the household'):
    predictIsPoverty = predict1(option)

    predictIsPovertyWord = 'below' if predictIsPoverty ==1 else 'above'
    st.success(f'The predicted household no. '+ str(option)+' is '+predictIsPovertyWord+' the poverty line.')


st.caption('Details of household #'+str(option)+':')
df_option= pd.DataFrame(X.loc[option]).reset_index()
df_option.columns=['Feature','value']
st.table(df_option)



# st.write("You selected:", option)



