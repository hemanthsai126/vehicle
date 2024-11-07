import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
df = pd.read_csv('vehicle_details.csv')
df['combined_features'] = df['How was the issue observed'] + ' ' + df['What is the issue'] + ' ' + df['Why is this an issue']

# Create TF-IDF vectorizer and transform existing data
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])

def predict_issue(observation, issue, issue_reason):
    # Combine input features
    input_text = f"{observation} {issue} {issue_reason}"
    
    # Transform input text
    input_vector = vectorizer.transform([input_text])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(input_vector, feature_matrix)
    
    # Find the most similar issue
    most_similar_index = similarities.argmax()
    similarity_score = similarities[0][most_similar_index]
    
    # Get the prediction
    prediction = df.iloc[most_similar_index]
    
    return prediction, similarity_score

# Streamlit interface
st.title("Issue Prediction Tool")

# Get user input
observation = st.text_input("How was the issue observed:")
issue = st.text_input("What is the issue:")
issue_reason = st.text_input("Why is this an issue:")

if st.button("Predict"):
    if observation and issue and issue_reason:
        # Get prediction based on input
        prediction, similarity_score = predict_issue(observation, issue, issue_reason)
        
        # Display the results
        st.subheader("Predicted Issue Details")
        st.write("Ticket Number:", prediction["Ticket Number"])
        st.write("Root cause:", prediction["Root cause"])
        st.write("Root cause category:", prediction["Root cause category"])
        st.write("Solution implemented:", prediction["Solution implemented"])
        st.write("Team solved:", prediction["Team solved"])
        st.write("Similarity score:", f"{similarity_score:.2f}")

        if similarity_score < 0.5:
            st.warning("Low similarity score. The prediction may not be accurate.")
    else:
        st.error("Please fill out all fields.")
