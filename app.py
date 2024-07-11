import streamlit as st
from main import get_qa_chain, create_vector_db

st.title("AUTOMATED Q&A for ED-TECH 🌱")

# Button to create the vector database
if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

question = st.text_input("Question: ")

if question:
    try:
        chain = get_qa_chain()
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    except FileNotFoundError as e:
        st.error(str(e))
