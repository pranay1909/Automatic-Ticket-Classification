from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_cohere import ChatCohere
from langchain.chains.question_answering import load_qa_chain
import joblib
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import os
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def create_embedding():
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:22m")
    return embeddings

def pull_pinecone(embeddings):
    pc=Pinecone(api_key=PINECONE_API_KEY)
    index=pc.Index("ticket-classification")
    vector_store = PineconeVectorStore(index=index, embedding= embeddings)
    return vector_store

def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k = k)
    return similar_docs

def get_answer(docs, user_input):
    chain = load_qa_chain(ChatCohere(model="command-r-plus"), chain_type="stuff")
    response = chain.run(input_documents = docs, question = user_input)
    return response

def predict(query_result):
    query_result = np.array(query_result)
    Fitmodel = joblib.load("modelsvm.pk1")
    query_result = query_result.reshape(1, -1)
    result = Fitmodel.predict(query_result)
    return result[0]