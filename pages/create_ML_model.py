import streamlit as st
from pages.admin_utils import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


if "cleaned_data" not in st.session_state:
    st.session_state["cleaned_data"] = ""
if "sentences_train" not in st.session_state:
    st.session_state["sentences_train"] = ""
if "sentences_test" not in st.session_state:
    st.session_state["sentences_test"] = ""
if "labels_train" not in st.session_state:
    st.session_state["labels_train"] = ""
if "labels_test" not in st.session_state:
    st.session_state["labels_test"] = ""
if "svm_classifier" not in st.session_state:
    st.session_state["svm_classifier"] = ""


st.title("Building Our Model")

tab_titles = ["Data Preprocessing", "Model Training", "Model Evaluation", "Save Model"]
tabs=st.tabs(tab_titles)


with tabs[0]:
    st.header("Data Preprocessing")

    data = st.file_uploader("Upload CSV File", type="csv")
    button = st.button("Load data", key="data")

    if button:
        with st.spinner("Wait for it"):
            our_data=read_data(data)
            embeddings = create_embedding()
            st.session_state["cleaned_data"] = create_embeddingdf(our_data, embeddings)
        st.success("Done")


with tabs[1]:
    st.header("Model Training")
    button=st.button("Train Model", key="model")

    if button:
        with st.spinner("Wait for it"):
            st.session_state["sentences_train"], st.session_state["sentences_test"], st.session_state["labels_train"], st.session_state["labels_test"] = split_train_test_data(st.session_state["cleaned_data"])
            # tfidf_vectorizer = TfidfVectorizer()
            # st.session_state["svm_classifier"] = make_pipeline(tfidf_vectorizer, SVC(class_weight="balanced"))
            st.session_state["svm_classifier"] = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
            st.session_state["svm_classifier"].fit(st.session_state["sentences_train"], st.session_state["labels_train"])

        st.success("Done")


with tabs[2]:
    st.header("Model Evaluation")
    button = st.button("Evaluate Model", key="Evaluation")

    if button:
        with st.spinner("Wait for it"):
            accuracy_score = get_score(st.session_state["svm_classifier"], st.session_state["sentences_test"], st.session_state["labels_test"])
            st.success(f"Validation Score : {100*accuracy_score} % ")

            st.write("A Sample Run : ")

            text = "Rude Driver with Scary Driving"
            st.write("Our Issue : " + text)

            embeddings = create_embedding()
            query_result = embeddings.embed_query(text)

            result = st.session_state["svm_classifier"].predict([query_result])
            st.write("Department it belongs to : " + result[0])
        st.success("Done")

with tabs[3]:
    st.header("Save Model")
    button = st.button("Save Model", key="Save")
    if button:
        with st.spinner("Wait for it"):
            joblib.dump(st.session_state["svm_classifier"], "modelsvm.pk1")
        st.success("Done")

