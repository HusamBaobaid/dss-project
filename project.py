import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.write("""
# DSS Project

This app does the following: \n
1- Predicts if a customer comes with kids or not using **RandomForestClassifier**  & **K-NN** \n
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

st.subheader('Prediction with RFC (With Kids?)')
withkids = np.array(['Yes','No'])
st.write(withkids[prediction_rfc])

st.subheader('Prediction with KNN (n=15) (With Kids?)')
withkids = np.array(['Yes','No'])
st.write(withkids[prediction_knn])

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"])
st.line_chart(chart_data)


from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

df = pd.read_csv("LAUNDRY.csv")

df = df.dropna()
X = df.drop('Membership', axis=1)
y = df['Membership']

X = pd.get_dummies(X, drop_first=True)
km = KMeans(n_clusters = 2, random_state=1)
km.fit(X)

distortions = []

for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
st.plotly_chart(plt)

