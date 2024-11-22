import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import streamlit as st

# Set page config at the very beginning
st.set_page_config(page_title="Healthcare Chatbot", page_icon=":hospital:")

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Text to Speech (modified to work with Streamlit)
def readn(nstr):
    try:
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")  # You can change the voice by setting this
        engine.setProperty('rate', 130)  # Adjust speech rate (speed)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Load datasets
@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    return training, testing

# Prepare model
@st.cache_resource
def prepare_model():
    # Load datasets
    training, testing = load_data()
    
    # Feature and target separation
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    # Map strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    # Decision Tree Classifier
    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    
    # Support Vector Machine Classifier
    model = SVC()
    model.fit(x_train, y_train)
    
    return clf, model, le, cols, x, y

# Load additional dictionaries
@st.cache_data
def load_dictionaries():
    # Get description from file
    description_list = {}
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]
    
    # Get severity data from file
    severity_dictionary = {}
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    severity_dictionary[row[0]] = int(row[1])
                except ValueError:
                    continue
    
    # Get precaution data from file
    precaution_dictionary = {}
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 5:
                precaution_dictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    
    return description_list, severity_dictionary, precaution_dictionary

# Streamlit App
def main():
    # Load model and data
    clf, model, le, cols, X, y = prepare_model()
    description_list, severity_dictionary, precaution_dictionary = load_dictionaries()
    
    # Symptom list for multiselect
    symptom_list = list(cols)

    # Main content
    st.title("🩺 Healthcare Diagnostic Chatbot")
    
    # Sidebar
    st.sidebar.header("Patient Information")
    patient_name = st.sidebar.text_input("Enter Your Name")
    
    # Symptom Selection
    st.subheader("Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Choose symptoms you are experiencing:", 
        symptom_list
    )
    
    # Days of Symptoms
    days_of_symptoms = st.slider("For how many days have you been experiencing these symptoms?", 1, 30, 7)
    
    # Prediction Button
    if st.button("Predict Possible Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            readn("Please select at least one symptom.")
        else:
            # Prediction Logic
            input_vector = np.zeros(len(symptom_list))
            for symptom in selected_symptoms:
                input_vector[symptom_list.index(symptom)] = 1
            
            # Predict using Decision Tree
            prediction = clf.predict([input_vector])
            disease = le.inverse_transform(prediction)[0]
            
            # Display Results
            st.success(f"Predicted Disease: {disease}")
            readn(f"Predicted Disease: {disease}")
            
            # Description
            st.subheader("Disease Description")
            description = description_list.get(disease, "Description not available.")
            st.write(description)
            readn(description)
            
            # Precautions
            st.subheader("Recommended Precautions")
            precautions = precaution_dictionary.get(disease, ["No specific precautions found."])
            for i, precaution in enumerate(precautions, 1):
                st.write(f"{i}. {precaution}")
                readn(f"Precaution {i}: {precaution}")
            
            # Severity Assessment
            severity_score = sum(severity_dictionary.get(symptom, 0) for symptom in selected_symptoms)
            severity_status = "High" if (severity_score * days_of_symptoms / (len(selected_symptoms) + 1)) > 13 else "Moderate"
            
            st.warning(f"Severity Assessment: {severity_status}")
            readn(f"Severity Assessment: {severity_status}")
            
            if severity_status == "High":
                st.warning("You should consult a doctor immediately.")
                readn("You should consult a doctor immediately.")
            else:
                st.info("Monitor your symptoms and take necessary precautions.")
                readn("Monitor your symptoms and take necessary precautions.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Disclaimer: This is a diagnostic aid, not a substitute for professional medical advice.")
    st.sidebar.markdown("AIML PROJECT MADE BY: ")
    st.sidebar.markdown("    SHEEN PANDITA 22CSU162")
    st.sidebar.markdown("    SHERIN BAIJU 22CSU163")
    st.sidebar.markdown("    SWATI KAR 22CSU174")

# Run the main function
if __name__ == "__main__":
    main()
