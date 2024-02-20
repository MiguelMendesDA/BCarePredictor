#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from ssl import Options
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import time
from streamlit_modal import Modal


##Set the page
st.set_page_config(page_title='BCare Predictor',  layout='wide', page_icon="images/BCare_Logo.png")

#Pop-up disclaimer

modal = Modal(key="BCarePredictorDisclaimer", title="Disclaimer - BCare Predictor", padding=50, max_width=900)

if 'popup_closed' not in st.session_state:
    st.session_state.popup_closed = False

if not st.session_state.popup_closed:
    with modal.container():
        st.markdown(
        "This application only uses the input data provided for predictive purposes and does not store or collect any personal information. Your input data is processed solely for generating predictions and is not retained by the system. We respect your privacy and do not track or store any user-specific information. If you have any questions or concerns about data usage, please feel free to contact us.")
        st.markdown('')
        st. markdown("Thank you for using our service.")    
        st.markdown('')
        value = st.checkbox("By checking this box, you agree to the data usage policies.")
        if value:
            st.button('Close')
            st.session_state.popup_closed = True


# Sidebar
st.sidebar.image("images/BCare_Logo.png", width=200) 
st.sidebar.title("BCare Predictor")
st.sidebar.markdown("Confidence in Every Prediction.")
st.sidebar.markdown ("**For an immersive experience, explore the menu.**")

# Selectbox in the sidebar
selected_option = st.sidebar.selectbox("Choose an option:", ["Home","Prediction Breast Cancer","What is Breast Cancer?", "Breast Cancer Symptoms", "Breast Self-Exam", "General Information"])
home = False
prediction_breast_cancer = False
explain_breast_cancer = False
breast_cancer_symptoms = False
explain_self_exam = False
general_information =False

if selected_option == "Home":
    home = True
elif selected_option == "Prediction Breast Cancer":
    prediction_breast_cancer = True
elif selected_option == "What is Breast Cancer?":
    explain_breast_cancer = True
elif selected_option == "Breast Cancer Symptoms":
    breast_cancer_symptoms = True
elif selected_option == "Breast Self-Exam":
    explain_self_exam = True
elif selected_option == "General Information":
    general_information = True


# Define the validation function
def validate_input(data):
    valid_values_gender = ['Female', 'Male']
    
    required_features = ['Age_Group', 'Gender', 'Height(m)', 'Weight(Kg)', 'Alcohol_habits', 'Oral_contraception', 'Breastfeeding']

    for feature in required_features:
        if feature not in data:
            return False, f"Missing required feature: {feature}"
        
        if feature == 'Age_Group' and (data[feature] not in ['18-49', '50+']):
            return False, "Invalid Age Group"

        if feature == 'Gender' and (data[feature] not in ['Female', 'Male']):
            return False, "Invalid Gender value. Use 'Female' or 'Male.'"

        if feature == 'Height(m)' and (not isinstance(data['Height(m)'], (float)) or not (1 <= data['Height(m)'] <= 3)):
            return False, f"Invalid {feature} value. Use values between 1 an 3 meters."

        if feature == 'Weight(Kg)' and (not isinstance(data['Weight(Kg)'], (int, float)) or data['Weight(Kg)'] < 20):
            return False, f"Invalid {feature} value.  Use values in upper than 20kg."
        
        if feature == 'Alcohol_habits' and data[feature] not in ['Yes', 'No']:
            return False, "Invalid Alcohol consumption value. Use 'Yes' or 'No'"
        
        if feature == 'Oral_contraception' and data[feature] not in ['Yes', 'No']:
            return False, "Invalid Oral Contraception value. Use 'Yes' or 'No'"
        
        if feature == 'Oral_contraception' and data['Gender'] == 'Male' and data[feature] == 'Yes':
            return False, "Men do not take oral contraception."

        if feature == 'Breastfeeding' and data[feature] not in ['Yes', 'No']:
            return False, "Invalid Breastfeeding value. Use 'Yes' or 'No'"
        
        if feature == 'Breastfeeding' and data['Gender'] == 'Male' and data[feature] == 'Yes':
            return False, "Men do not breastfeed."

    return True, None

# Define the encoding functions
def encode_gender(gender):
    gender_mapping = {'Female': 1, 'Male': 0}
    return gender_mapping.get(gender, gender)

def encode_categorical_features(data):
    mappings = {
        'Alcohol_habits': {'Yes': 1, 'No': 0},
        'Breastfeeding': {'Yes': 1, 'No': 0},
        'Oral_contraception': {'Yes': 1, 'No': 0},
        'Age_Group': {'18-49': 0, '50+': 1}
    }
    for feature, mapping in mappings.items():
        data[feature] = data[feature].map(mapping)
    return data

# Load the entire pipeline
loaded_pipeline = joblib.load(os.path.join('models', 'full_pipeline.pkl'))



if home:
    st.header("Welcome to BCare Predictor: Your Breast Cancer Risk Prediction Tool")
    time.sleep(0.1)
    st.markdown("""
    BCare Predictor is a powerful tool designed to assess the risk of breast cancer based on your input data. 
    Our advanced predictive model analyzes various factors to provide you with valuable insights into your breast health.""")
    time.sleep(0.1)
    st.markdown("""
    ### Why Use BCare Predictor?
    - **Early Detection:** Timely identification of potential risks for breast cancer.
    - **Personalized Results:** Tailored predictions based on your unique characteristics.
    - **Informative Recommendations:** Receive actionable information for proactive health management.""")
   
    time.sleep(0.1)
    st.markdown("Let's get started on your breast health journey!")
    time.sleep(0.1)
    
   
if prediction_breast_cancer:
    st.header("BCare Predictor")
    time.sleep(0.1)
    st.markdown ("""
    ### How it Works:
    1. Fill in the requested information.
    2. Click the "Predict" button to get your personalized breast cancer risk prediction.
    3. Explore additional information and recommendations based on the prediction.""")
    # collect user input
    age = st.radio("Please select your age group:", options = ['18-49', '50+'] )
    time.sleep(0.1)
    gender = st.radio("Please select your gender:", options=["Female", "Male"])
    time.sleep(0.1)
    height = st.number_input("Please enter your height (in meters):", value=0.00, min_value=0.0, max_value=2.5)
    time.sleep(0.1)
    weight = st.number_input("Please enter your weight (in Kg):", value=0.0, min_value=0.0)
    time.sleep(0.1)
    alcohol_consumption = st.radio("Do you consume or have you consumed alcoholic beverages, even socially?", options=["Yes", "No"])
    time.sleep(0.1)
    oral_contraception = st.radio("Are you taking oral contraception or have you taken it?", options=["Yes", "No"])
    time.sleep(0.1)
    breastfeeding = st.radio("Are you breastfeeding or have you breastfed?", options=["Yes", "No"])

    # Handle user input
    if st.button("Predict"):
        new_data = {
            "Age_Group": age,
            "Gender": gender, 
            "Height(m)": height,
            "Weight(Kg)": weight,
            "Alcohol_habits": alcohol_consumption, 
            "Oral_contraception": oral_contraception,
            "Breastfeeding": breastfeeding,
        }
        

        # Validate the input data
        is_valid, validation_message = validate_input(new_data)
        if not is_valid:
            st.warning(f"Invalid input data: {validation_message}")
        else:
            # Encode gender and categorical features
            new_data['Gender'] = encode_gender(new_data['Gender'])
            new_data = encode_categorical_features(pd.DataFrame([new_data]))
            # Calculate BMI
            new_data['BMI'] = new_data['Weight(Kg)'] / (new_data['Height(m)'] ** 2)

            # Create a DataFrame from the input data
            input_data_transformed = pd.DataFrame(new_data, index=[0])

            # Make predictions using the loaded pipeline
            probabilities = loaded_pipeline.predict_proba(input_data_transformed)

            if len(probabilities[0]) == 2:
                prediction = probabilities[0][1]
                threshold = 0.7

                if not isinstance(prediction, (int, float)):
                    st.error("Invalid prediction format. Please check the input data.")
                else:
                    if prediction >= threshold:
                        st.error("Prediction: Risk of Breast Cancer")
                        # Additional information 
                        st.markdown("### Aditional Information for Breast Cancer Risk:")
                        time.sleep(0.1)
                        st.markdown("- **Prioritize Regular Screenings:** Regular screenings are a cornerstone of breast cancer prevention. Mammograms and other early detection methods can identify potential issues at an early, more treatable stage.")
                        time.sleep(0.1)
                        st.markdown("- **Comprehensive Breast Examination:** Consider scheduling more comprehensive examinations, such as mammograms or even breast MRIs. These advanced techniques can provide a more detailed view of breast tissue, aiding in early detection.")
                        time.sleep(0.1)
                        st.markdown("- **Genetic Counseling Exploration:** If there is a family history of breast cancer, consider exploring genetic counseling. This involves assessing your genetic risk factors and can help guide personalized prevention and screening strategies.")
                        time.sleep(0.1)
                        st.markdown("- **Lifestyle Changes for Risk Reduction:** Implementing lifestyle changes is crucial. Focus on maintaining a healthy diet rich in fruits, vegetables, and whole grains. Engage in regular physical activity, aiming for at least 150 minutes of moderate-intensity exercise per week.")
                        time.sleep(0.1)
                        st.markdown("- **Monitor Breast Health Changes:** Stay vigilant about changes in your breast health. Perform regular self-exams and be aware of any lumps, pain, or nipple discharge. Promptly report any changes to your healthcare provider for further evaluation.")
                        time.sleep(0.1)
                        st.warning(" Breast cancer risk varies among individuals, and early detection remains crucial for effective treatment.")
                        time.sleep(0.1)
                        st.error("Note: This information serves as awareness and education. It is not a substitute for professional medical advice. Consultation with healthcare professionals is paramount for personalized risk assessment and guidance.")
                    # If the prediction indicates low risk
                    else:
                        st.success("Prediction: Low Risk of Breast Cancer")

                        # Additional information for low risk
                        st.markdown("### Additional Information for Low Risk:")
                        time.sleep(0.1)
                        st.markdown("- While your current risk is low, it's essential to continue with regular breast health check-ups.")
                        time.sleep(0.1)
                        st.markdown("- Schedule routine mammograms and screenings as recommended by your healthcare provider.")
                        time.sleep(0.1)
                        st.markdown("- Practice a healthy lifestyle, including a balanced diet, regular exercise, and maintaining a healthy weight.")
                        time.sleep(0.1)
                        st.markdown("- Be aware of any changes in your breast health and promptly report them to your healthcare provider.")
                        time.sleep(0.1)
                        st.markdown("- Stay informed about the latest guidelines for breast cancer prevention.")
                        time.sleep(0.1)
                        st.error("Note: This information serves as awareness and education. It is not a substitute for professional medical advice. Consultation with healthcare professionals is paramount for personalized risk assessment and guidance.")
    
            else:
                st.error("Unexpected number of probabilities. Unable to make a prediction.")

# Information about breast cancer
if explain_breast_cancer:
    st.markdown("### What is Breast Cancer?")
    time.sleep(0.1)
    st.markdown ("Breast cancer is cancer that forms in the cells of the breasts.") 
    time.sleep(0.1)
    st.markdown ("Breast cancer is one of the most common cancer diagnosed in women. Breast cancer can occur in both men and women, but it's far more common in women.")
    time.sleep(0.1)
    st.markdown ("""Substantial support for breast cancer awareness and research funding has helped create advances in the diagnosis and treatment of breast cancer.
                 Breast cancer survival rates have increased, and the number of deaths associated with this disease is steadily declining, largely due to factors such as earlier detection, a new personalized approach to treatment and a better understanding of the disease.""")
    st.image("images/breast.png", width=500)
    time.sleep(0.1)
    st.markdown ("**Breast Anatomy**")
    time.sleep(0.1)
    st.markdown ("""Each breast contains 15 to 20 lobes of glandular tissue, arranged like the petals of a daisy. 
                 The lobes are further divided into smaller lobules that produce milk for breastfeeding. Small tubes (ducts) conduct the milk to a reservoir that lies just beneath your nipple.""")

#Information about breast cancer symptoms
if breast_cancer_symptoms:
    st.markdown("### Breast Cancer Symptoms")
    time.sleep(0.1)
    st.markdown ("""Breast cancer symptoms can vary, and not everyone with breast cancer experiences the same signs. 
                 It's important to note that many breast changes are benign (non-cancerous), but any changes should be promptly evaluated by a healthcare professional. 
                 Common symptoms of breast cancer may include:""") 
    time.sleep(0.1)
    st.markdown ("**-Lump or Thickening:** A lump in the breast or underarm is one of the most common symptoms. It may feel different from the surrounding tissue.")
    time.sleep(0.1)
    st.markdown("**- Changes in Breast Size or Shape:** Any unexplained changes in the size or shape of the breast, such as asymmetry or swelling, should be evaluated.")
    time.sleep(0.1)
    st.markdown("**- Changes in the Skin:** Look for changes in the skin on or around the breast, such as redness, dimpling, or puckering. Changes in the texture of the skin, resembling an orange peel, may also occur.")
    time.sleep(0.1)
    st.markdown("**- Nipple Changes:** Changes in the nipples, such as inversion (turning inward), discharge (other than breast milk), or pain, can be signs of concern.")
    time.sleep(0.1)
    st.markdown("**- Breast or Nipple Pain:** While breast cancer is not often associated with pain, persistent pain or discomfort in the breast or nipple should be evaluated.")
    time.sleep(0.1)
    st.markdown("**- Nipple Discharge:** Any unexplained discharge from the nipple, especially if it is bloody, should be checked.")
    time.sleep(0.1)
    st.markdown("""
        It's essential to perform regular breast self-exams and be familiar with the normal look and feel of your breasts. 
        Additionally, women are encouraged to undergo regular mammograms as part of breast cancer screening, particularly as they age.

        If you notice any unusual changes or experience symptoms, it's crucial to consult with a healthcare professional for a thorough examination and, if necessary, further diagnostic tests. 
        Early detection and diagnosis improve the chances of successful treatment for breast cancer.""")


# Information about breast self-exams
if explain_self_exam:
    st.markdown("### Breast Self-Examination:")
    time.sleep(0.1)
    st.markdown("Performing a monthly breast self-exam can help you become familiar with the normal look and feel of your breasts.")
    time.sleep(0.1)
    st.markdown("Here's a step-by-step guide:")
    time.sleep(0.1)
    st.markdown("1. **Visual Inspection:** Stand in front of a mirror with your arms at your sides. Look for any changes in the size, shape, or contour of your breasts. Note any skin changes, redness, or dimpling.")
    time.sleep(0.1)
    st.markdown("2. **Raise Your Arms:** Raise your arms and look for the same changes with your arms overhead.")
    time.sleep(0.1)
    st.markdown("3. **Check Your Nipples:** Look for any changes in the nipples, such as inversion, discharge, or changes in direction.")
    time.sleep(0.1)
    st.markdown("4. **Manual Examination (Lying Down):** Lie down with a pillow under your right shoulder. Use your left hand to feel your right breast in a circular motion. Move from the outer part of the breast toward the center, checking for any lumps or changes. Repeat on the other side.")
    time.sleep(0.1)
    st.markdown("5. **Manual Examination (Sitting or Standing):** Use the same circular motion to check your breasts while sitting or standing.")
    time.sleep(0.1)
    st.markdown("6. **Underarm Examination:** Don't forget to check the area under your arms for any lumps or swelling.")
    time.sleep(0.1)
    st.markdown("If you notice any changes during the self-exam, report them to your healthcare provider promptly.")
    # Add an image to illustrate the breast self-exam steps
    st.image("https://www.check4cancer.com/images/breast-cancer/6_steps_to_performing_a_breast_self_-examination.jpg", caption="Illustration of Breast Self-Exam", use_column_width=True)        

#general information
if general_information:
    st.markdown("### Sources")
    time.sleep(0.1)
    st.markdown("www.mayoclinic.org")
    time.sleep(0.1)
    st.markdown("www.check4cancer.com")
    
   