import streamlit as st
import requests
from user_utils import *
import os
from dotenv import load_dotenv
load_dotenv()

if "HR_tickets" not in st.session_state:
    st.session_state["HR_tickets"] = []
if "IT_tickets" not in st.session_state:
    st.session_state["IT_tickets"] = []
if "Transport_tickets" not in st.session_state:
    st.session_state["Transport_tickets"] = []

def main():
    st.header("Automatic Ticket Classification Tool")
    st.write("Please ask the question")
    user_input = st.text_input("🔍")

    if user_input:

        embeddings = create_embedding()

        index = pull_pinecone(embeddings)

        similar_docs = get_similar_docs(index, user_input)

        response = get_answer(similar_docs, user_input)
        st.write(response)
        button = st.button("Raise Ticket", key="ticket")
        if button:
            st.write("Ticket Raised")
            embeddings = create_embedding()
            query_result = embeddings.embed_query(user_input)

            department_value = predict(query_result)
            st.write("The ticket has been submitted to : "+department_value)

            if department_value=="HR":
                st.session_state["HR_tickets"].append(user_input)
            if department_value=="IT":
                st.session_state["IT_tickets"].append(user_input)
            if department_value=="Transportation":
                st.session_state["Transport_tickets"].append(user_input)
            


if __name__ == "__main__":
    main()
