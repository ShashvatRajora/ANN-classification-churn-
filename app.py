# import streamlit as st
# import pandas as pd 
# import numpy as np 
# import tensorflow as tf
# import pickle
# from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# # Load the trined model :
# model = tf.keras.models.load_model('model.h5')

# with open ('label_encoder_gender.pkl','rb') as file:
#     label_encoder_gender = pickle.load(file)
    
# with open ('onehot_encoder_geo.pkl','rb') as file:
#     label_encoder_geo = pickle.load(file)
    
# with open ('scaler.pkl','rb') as file:
#     scaler = pickle.load(file)
    
    
# # Steamlit app 
# st.title("Customer churn prediction")

# # user input 
# geography = st.selectbox('Geography',label_encoder_geo.categories_[0])
# gender = st.selectbox('Gender',label_encoder_gender.classes_)
# age = st.selectbox('Age',18,92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
# tenure = st.slider('Tenure',0,10)
# num_of_products = st.slider('Number of Products',1,4)
# has_cr_card = st.selectbox('Has Credit Card',[0,1])
# is_active_member = st.selectbox('Is Active Member',[0,1])

# input_data = pd.DataFrame({
#     'CreditScore' : [credit_score],
#     'Gender' : [label_encoder_gender.transform([gender])[0]],
#     'Age' : [age],
#     'Tenure' : [tenure],
#     'Balance' : [balance],
#     'NumOfProducts' : [num_of_products],
#     'HasCrCard' : [has_cr_card],
#     'IsActiveMember' : [is_active_member],
#     'EstimatedSalary' : [estimated_salary], 
# })

# # One hot encode 'Geography' 

# geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

# # Combine the one hot encoded coumns with input data

# input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# # Scale the input data 
# input_data_scaled = scaler.transform(input_data)

# # Predict churn 
# prediction = model.predict(input_data_scaled)
# prediction_probab = prediction[0][0]

# if prediction_probab>0.5:
#     print("The customer is likely to churn.")
# else:
#     print("The customer is not likely to churn.")





import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Encode categorical inputs
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(['Geography'])
)

# Create input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine with one-hot encoded geography
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Match order of training features
input_data = input_data[scaler.feature_names_in_]

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

# Output
if prediction_probab > 0.5:
    st.warning(f"⚠️ The customer is likely to churn. (Probability: {prediction_probab:.2f})")
else:
    st.success(f"✅ The customer is not likely to churn. (Probability: {prediction_probab:.2f})")
