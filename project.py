import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.write("""
# DSS Project

This app does the following:
1- Predicts if a customer comes with kids or not using **RandomForestClassifier**
2- 

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[CSV Input file](https://raw.githubusercontent.com/HusamBaobaid/dss-project/main/LAUNDRY_cleaned.csv)
""")

# Collects user input features into dataframe
def user_input_features():
        race = st.sidebar.selectbox('Race',('indian','malay','chinese','foreigner '))
        gender = st.sidebar.selectbox('Gender',('male','female'))
        age_range = st.sidebar.slider('Age', 29,53,43)
        basket_size = st.sidebar.selectbox('Basket Size',('big','small'))
        data = {'Race': race,
                'Gender': gender,
                'Age_Range': age_range,
                'Basket_Size': basket_size}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()
    

# Combines user input features with entire dataset
# This will be useful for the encoding phase
data_raw = pd.read_csv('LAUNDRY_cleaned.csv')
data = data_raw.drop(columns=['With_Kids'])
df = pd.concat([input_df,data],axis=0)

# Encoding of ordinal features
encode = ['Basket_Size','Gender','Race']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

st.write(df)

# Reads in saved classification model
load_clf_rfc = pickle.load(open('withkids_clf_rfc.pkl', 'rb'))
load_clf_knn = pickle.load(open('withkids_clf_knn.pkl', 'rb'))

# Apply model to make predictions
prediction_rfc = load_clf_rfc.predict(df)
prediction_proba_rfc = load_clf_rfc.predict_proba(df)

prediction_knn = load_clf_knn.predict(df)
prediction_proba_knn = load_clf_knn.predict_proba(df)

st.subheader('Prediction with RFC')
withkids = np.array(['Yes','No'])
st.write(withkids[prediction_rfc])

st.subheader('Prediction with KNN')
withkids = np.array(['Yes','No'])
st.write(withkids[prediction_knn])

