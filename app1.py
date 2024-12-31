import streamlit as st
import pickle
import numpy as np

# Load saved models and vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("logistic_regression_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("naive_bayes_model.pkl", "rb") as f:
    naive_bayes_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

with open("decision_tree_model.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

with open("neural_network_model.pkl", "rb") as f:
    neural_network_model = pickle.load(f)

# Function to predict and return probabilities
def predict_news(model, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    
    # Handle predict_proba availability
    try:
        probabilities = model.predict_proba(text_vectorized)[0]
    except AttributeError:
        probabilities = [0, 0]  # Default probabilities if the method is not available
        probabilities[prediction] = 1  # Assign 100% to the predicted class
    return prediction, probabilities

# Streamlit app
st.title("Fake News Detection App")
st.write("This app predicts whether a news article is real or fake using machine learning models.")

# User input
user_input = st.text_area("Enter the news article text here:", "")

# Select model
model_choice = st.selectbox(
    "Choose a model:",
    ["Logistic Regression", "Naive Bayes", "Random Forest", "Decision Tree", "Neural Network"]
)

if st.button("Predict"):
    if user_input.strip():
        # Select the appropriate model
        if model_choice == "Logistic Regression":
            model = logistic_model
        elif model_choice == "Naive Bayes":
            model = naive_bayes_model
        elif model_choice == "Random Forest":
            model = random_forest_model
        elif model_choice == "Decision Tree":
            model = decision_tree_model
        elif model_choice == "Neural Network":
            model = neural_network_model
        
        # Get prediction and probabilities
        prediction, probabilities = predict_news(model, tfidf_vectorizer, user_input)

        # Display prediction and probabilities
        label = "Real News" if prediction == 1 else "Fake News"
        st.write(f"Prediction: **{label}**")
        st.write("Confidence Levels:")
        st.write(f"- Real News: {probabilities[1]:.2%}")
        st.write(f"- Fake News: {probabilities[0]:.2%}")
    else:
        st.write("Please enter some text to predict.")



# Add some information about the app
st.write("---")
st.write("This app uses Logistic Regression, Naive Bayes, and Random Forest models to classify news articles as real or fake based on the text content.")
st.write("Deployment with💓 using streamlit")
