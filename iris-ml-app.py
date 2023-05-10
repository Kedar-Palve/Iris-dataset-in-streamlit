import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.write("""
    # Simple Iris Flower Prediction App

    ###This app predicts the Iris flower type! 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
    }

    features = pd.DataFrame(data, index=[0])
    return features

f = user_input_features()

st.subheader('User Input Parameters')
st.write(f)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(f)
prediction_proba = clf.predict_proba(f)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction Probability')
st.write(prediction_proba)

if prediction[0]==0:
    img = Image.open("iris_setosa.jpg")
    st.image(img, width=300)
    st.write('Iris Setosa')
elif prediction[0]==1:
    img = Image.open("iris_versicolor.jpg")
    st.image(img, width=300)
    st.write('Iris Versicolor')
else:
    img = Image.open("iris_virginica.jpg")
    st.image(img, width=300)
    st.write('Iris Virginica')

