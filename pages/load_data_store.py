import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from pages.admin_utils import *

def main():
    st.set_page_config(page_title="PDF to Pinecone Vector Store")
    st.title("Please upload your files 📂")

    pdf = st.file_uploader("Only PDF", type=["pdf"])

    if pdf is not None:
        with st.spinner("Wait for it..."):

            text = read_pdf_data(pdf)
            st.write("Reading the pdf")


            docs_chunks=split_data(text)
            st.write("Splitting data into chunks done")


            embeddings = create_embedding()
            st.write("Embeddings done")

            push_pinecone(embeddings, docs_chunks)

        st.success("Pushed to Pinecone")

if __name__ == "__main__":
    main()
