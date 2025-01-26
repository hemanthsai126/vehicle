import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

# Load the dataset (replace 'vehicle_details.csv' with your actual dataset file)
df = pd.read_csv('Vehiclebdata.csv')

# Combine relevant features for text processing
df['combined_features'] = df['How was the issue observed'] + ' ' + df['What is the issue'] + ' ' + df[
    'Why is this an issue']

# Create TF-IDF vectorizer and transform existing data
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])


# Function to predict the output columns based on input
def predict_issue(observation, issue, issue_reason):
    input_text = f"{observation} {issue} {issue_reason}"
    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, feature_matrix)
    most_similar_index = similarities.argmax()  # Get the most similar record
    similarity_score = similarities[0][most_similar_index]

    # Retrieve the details from the most similar record
    prediction = df.iloc[most_similar_index]

    return prediction, similarity_score


# Streamlit interface
def home_page():
    st.title("Vehicle Issue Prediction Tool")
    st.write("This is just a sample page and the data used here is generated using AI")

    # Get user input for the issue description
    observation = st.text_input("How was the issue observed:")
    issue = st.text_input("What is the issue:")
    issue_reason = st.text_input("Why is this an issue:")

    if st.button("Predict"):
        if observation and issue and issue_reason:
            # Get prediction based on input
            prediction, similarity_score = predict_issue(observation, issue, issue_reason)

            # Display predicted results
            st.subheader("Predicted Issue Details")
            st.write("Ticket Number:", prediction["Ticket Number"])
            st.write("Root cause:", prediction["Root cause"])
            st.write("Root cause category:", prediction["Root cause category"])
            st.write("Solution implemented:", prediction["Solution implemented"])
            st.write("Team solved:", prediction["Team solved"])
            st.write("Time Required (hrs):", prediction["Time Required (hrs)"])
            st.write("Total Cost (USD):", prediction["Total Cost (USD)"])
            st.write("Cosine similarity score:", f"{similarity_score:.2f}")

            # Show warning if the similarity score is low
            if similarity_score < 0.5:
                st.warning("Low similarity score. The prediction may not be accurate.")
        else:
            st.error("Please fill out all fields.")

def service_details_page():
    st.title("Vehicle Service History")
    vehicle_id = st.text_input("Enter Vehicle ID (e.g., V01, V02, etc.):")

    if st.button('Get Service Details'):
        # Get service details based on Vehicle ID
        service_data = df[df['Vehicle ID'] == vehicle_id]

        if not service_data.empty:
            # Sort by 'date' column in descending order to get the latest service record
            service_data_sorted = service_data.sort_values(by='date', ascending=False)

            # Display the most recent service details as a list
            st.write("Last Service Details:")
            last_service = service_data_sorted.iloc[0]  # Get the first row (latest service)

            # Print each detail one by one
            st.write(f"Ticket Number: {last_service['Ticket Number']}")
            st.write(f"How was the issue observed: {last_service['How was the issue observed']}")
            st.write(f"What is the issue: {last_service['What is the issue']}")
            st.write(f"Root cause: {last_service['Root cause']}")
            st.write(f"Root cause category: {last_service['Root cause category']}")
            st.write(f"Solution implemented: {last_service['Solution implemented']}")
            st.write(f"Team solved: {last_service['Team solved']}")
            st.write(f"Time Required (hrs): {last_service['Time Required (hrs)']}")
            st.write(f"Total Cost (USD): {last_service['Total Cost (USD)']}")
            st.write(f"Service Rating: {last_service['Service Rating']}")
            st.write(f"Follow-Up Date: {last_service['Follow-Up Date']}")
            st.write(f"Vehicle ID: {last_service['Vehicle ID']}")
            st.write(f"Service Date: {last_service['date']}")

        else:
            st.write("No service record found for this vehicle.")

def technicians_page():
    st.title("Technician Ratings Page")

    # Aggregating technician ratings
    technician_ratings = df.groupby('Technician ID')['Service Rating'].mean().reset_index()

    # Plot technician ratings
    fig, ax = plt.subplots()
    sns.barplot(x='Technician ID', y='Service Rating', data=technician_ratings, palette='Set2', ax=ax)
    ax.set_title('Technician Ratings')
    st.pyplot(fig)

def ticket_info_page():
    st.title("Ticket Info Page")

    # Dummy data for open and closed tickets (Replace with actual logic if needed)
    ticket_status = ['Open', 'Closed', 'Open', 'Closed', 'Open']
    ticket_counter = Counter(ticket_status)

    # Plot a pie chart for open/closed tickets
    fig, ax = plt.subplots()
    ax.pie(ticket_counter.values(), labels=ticket_counter.keys(), autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    ax.set_title('Open vs Closed Tickets')
    st.pyplot(fig)
# Main function to navigate to home page
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "Service Details", "Technicians", "Ticket Info"])

    if page == "Home":
        home_page()
    elif page == "Service Details":
        service_details_page()
    elif page == "Technicians":
        technicians_page()
    elif page == "Ticket Info":
        ticket_info_page()

if __name__ == '__main__':
    main()
