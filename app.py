import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from streamlit_option_menu import option_menu

# Loading the saved models
diabetes_disease_model = pickle.load(
    open('diabetes_disease_model_RandomForestClassifier.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model_ANN.sav', 'rb'))
cataract_model = tf.keras.models.load_model('cataract_disease_model_CNN.h5')


with st.sidebar:
    # Create a sidebar menu for disease selection
    selected = option_menu(menu_title='Multiple Disease Prediction System',
                           options=['Diabetes Disease Prediction',
                                    'Heart Disease Prediction', 'Cataract Disease Prediction'],
                           icons=['activity', 'heart', 'eye'],
                           default_index=0)


# Diabetes Disease Prediction Page
if selected == 'Diabetes Disease Prediction':
    # Page title
    st.title('Diabetes Prediction using RandomForestClassifier')
    st.write('Exclusively intended for women with diabetes.')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    # Input fields for user data
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input(
            'Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Test Result'):
        # Convert input variables to the appropriate data types
        Pregnancies = int(Pregnancies)
        Glucose = int(Glucose)
        BloodPressure = int(BloodPressure)
        SkinThickness = int(SkinThickness)
        Insulin = float(Insulin)
        BMI = float(BMI)
        DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
        Age = int(Age)

        # Make the prediction using the loaded model
        diab_prediction = diabetes_disease_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'Person is diabetic'
        else:
            diab_diagnosis = 'Person is not diabetic'

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Page title
    st.title('Heart Disease Prediction using ANN')

    # Input fields for user data
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain type')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')

    with col1:
        thal = st.text_input(
            'thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    # Code for Prediction
    heart_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Test Result'):
        # Convert input variables to the appropriate data types
        age = int(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = int(trestbps)
        chol = int(chol)
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = int(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)

        # Make the prediction using the loaded model
        heart_prediction = heart_disease_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'Person is having heart disease'
        else:
            heart_diagnosis = 'Person does not have any heart disease'

    st.success(heart_diagnosis)


# Cataract Prediction Page
if selected == "Cataract Disease Prediction":
    # Function to predict cataract from the test image
    def predict_cataract(test_image):
        # Load a Test Image and Perform Prediction
        test_image = image.load_img(test_image, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cataract_model.predict(test_image)

        # Get the Prediction Label
        if result[0][0] == 1:
            prediction = 'No Cataract Found!'
        else:
            prediction = 'Cataract Confirmed!'

        return prediction

    def main():
        st.title("Cataract Disease Detection using CNN")

        # Upload a test image
        uploaded_file = st.file_uploader(
            "Upload an eye image", type=["jpg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.',
                     use_column_width=True)

            # Perform prediction on the uploaded image
            prediction = predict_cataract(uploaded_file)
            st.write(f"Prediction: {prediction}")

    if __name__ == '__main__':
        main()
