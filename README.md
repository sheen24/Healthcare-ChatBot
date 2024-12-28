# Healthcare-ChatBot
  Contents 
  1. [Introduction](https://github.com/sheen24/Healthcare-ChatBot/edit/main/README.md#1-introduction)
  2. [Python Libraries and Packages Used](https://github.com/sheen24/Healthcare-ChatBot/edit/main/README.md#2-python-libraries-and-packages-used)
  3. [Model Training and Optimisation](https://github.com/sheen24/Healthcare-ChatBot/edit/main/README.md#3-model-training-and-optimisation)
# 1. Introduction 
The Healthcare Diagnostic Chatbot is an AI-driven solution designed to assist users in identifying potential diseases based on their symptoms. It uses machine learning models, including Decision Trees and Support Vector Machines, to predict diseases, assess symptom severity, and provide precautionary recommendations.

The chatbot features an intuitive Streamlit interface, making healthcare insights easily accessible. 
This project aims to bridge gaps in healthcare accessibility, promote early diagnosis, and empower users with actionable health information, especially in areas with limited medical resources.

# 2. Python Libraries and Packages Used
- pandas: For data manipulation and analysis.

- numpy: For numerical computations.

- scikit-learn: For machine learning models (Decision Tree, SVC).

- pyttsx3: For text-to-speech functionality.

- streamlit: For building the web-based chatbot interface.

 # 3. Model Training and Optimisation

Decision Tree Classifier: Trained to provide explainable predictions by identifying patterns in the data.

Support Vector Classifier (SVC): Used for precise disease classification by capturing complex relationships in the data.

Label Encoding: The LabelEncoder from scikit-learn is used to convert categorical disease labels (prognosis) into numerical values for machine learning models.
