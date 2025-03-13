import streamlit as st 
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import dill

#load Trained model
model = tf.keras.models.load_model('model.keras')

#load encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('label_encoder_default.pkl','rb') as file:
    label_encoder_default = pickle.load(file)
    
with open('onehot_encoder_intent.pkl','rb') as file:
    one_encoder_intent = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
with open('custom_encoding_house.pkl','rb') as file:
    custom_house = dill.load(file)
    
with open('custom_encoding_education.pkl','rb') as file:
    custom_education = dill.load(file)    
    
#Stremelit app
st.title('Loan Approval Prediction')

with st.form("loan_form"):
    person_age = st.slider('Age', 18, 92)
    person_gender = st.selectbox('Gender', label_encoder_gender.classes_)
    person_education = st.selectbox('Education', ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    person_income = st.number_input('Income')
    person_emp_exp = st.number_input('Work Experience')
    person_home_ownership = st.selectbox('House Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    loan_amnt = st.number_input('Loan Amount')
    loan_intent = st.selectbox('Loan Intent', one_encoder_intent.categories_[0])
    loan_int_rate = st.number_input('Interest Rate')
    loan_percent_income = st.number_input('Loan percent of Income')
    cb_person_cred_hist_length = st.number_input('Credit History Length')
    credit_score = st.number_input('Credit Score')
    previous_loan_defaults_on_file = st.selectbox('Previous defaults records', label_encoder_default.classes_)

    submit_button = st.form_submit_button("Predict Loan Outcome")

if submit_button:
  


    input_df = pd.DataFrame([{'person_age':person_age,
            'person_gender':person_gender,
            'person_education':person_education,
            'person_income':person_income,
            'person_emp_exp':person_emp_exp,
            'person_home_ownership':person_home_ownership,
            'loan_amnt':loan_amnt,
            'loan_intent':loan_intent,
            'loan_int_rate':loan_int_rate,
            'loan_percent_income':loan_percent_income,
            'cb_person_cred_hist_length':cb_person_cred_hist_length,
            'credit_score':credit_score,
            'previous_loan_defaults_on_file':previous_loan_defaults_on_file}])

    encoded = one_encoder_intent.transform(input_df[['loan_intent']]).toarray()
    encoded_df = pd.DataFrame(encoded,columns=one_encoder_intent.get_feature_names_out(['loan_intent']))
    input_df = pd.concat([input_df.drop(['loan_intent'],axis=1), encoded_df],axis=1)
    input_df['person_gender'] = label_encoder_gender.transform(input_df['person_gender'])
    input_df['previous_loan_defaults_on_file'] = label_encoder_default.transform(input_df['previous_loan_defaults_on_file'])
    input_df['person_education'] = custom_education(input_df)
    input_df['person_home_ownership'] = custom_house(input_df)
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)
    prob = pred[0][0]

    st.write(f'Probability: {prob:.2f}')
    if prob > 0.5:
        st.write("Loan is  likely to be repaid")
    else:
        st.write("Loan will not be repaid")
