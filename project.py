import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# DSS Project

This app predicts if a customer comes **With Kids** or not!

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
    

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('LAUNDRY_cleaned.csv')
penguins = penguins_raw.drop(columns=['With_Kids'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
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
load_clf = pickle.load(open('withkids_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Yes','No'])
st.write(penguins_species[prediction])

