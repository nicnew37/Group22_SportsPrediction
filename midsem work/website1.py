import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

sc=StandardScaler()
model='/Users/brian/Desktop/Python/midsem_best_model.pkl'
loaded_model = pickle.load(open(model, 'rb'))


st.title('My Streamlit Player Overall Prediction App')
potential = st.slider('Potential', 0, 100)
age = st.number_input('Age', min_value=16, max_value=50)
international_reputation= st.slider('International Reputation', 0, 5)
defending = st.slider('Defending', 0, 100)
physic= st.slider('Physicality', 0, 100)
movement_reaction = st.slider('Movement Reaction', 0, 100)
defending_marking_awareness= st.slider('Man Marking', 0, 100)
skills = st.slider('Skills', 0, 5)
shotlevel = st.slider('Shotlevel', 0, 100)
passing = st.slider('Passing', 0, 100)
movements = st.slider('Movements', 0, 100)
mentality = st.slider('Mentality', 0, 100)
power = st.slider('Power', 0, 100)



def predict(potential,age,international_reputation,defending ,physic,movement_reaction,defending_marking_awareness,skills,shotlevel ,passing ,movements,mentality,power):
    X_pred = np.array([potential,age,international_reputation,defending ,physic,movement_reaction,defending_marking_awareness,skills,shotlevel ,passing ,movements,mentality,power])
    X_pred_scaled = sc.fit_transform(X_pred.reshape(1, -1))
    rating = loaded_model.predict(X_pred_scaled)
    return rating



if st.button('Predict'):
    # Perform prediction using the loaded model
    prediction = loaded_model.predict([[potential,age,international_reputation,defending ,physic,movement_reaction,defending_marking_awareness,skills,
                                        shotlevel ,passing ,movements,mentality,power]])  
    st.write(f'Prediction: ', prediction)