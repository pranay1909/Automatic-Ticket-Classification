from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

from sklearn.model_selection import train_test_split
import pandas as pd



def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text+= page.extract_text()
    return text

def split_data(text):
    text_splitting = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    docs = text_splitting.split_text(text)
    # docs_chunks = text_splitting.create_documents(docs)
    # return docs_chunks
    return docs

def create_embedding():
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:22m")
    return embeddings

def push_pinecone(embeddings, docs_chunks):
    pc=Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("ticket-classification")
    vector_store = PineconeVectorStore(index=index, embedding= embeddings)
    vector_store.add_texts(docs_chunks)



def read_data(data):
    df = pd.read_csv(data)
    return df

# def create_embeddingdf(df, embeddings):
#     df.iloc[:, 2] = df.iloc[:, 0].apply(lambda x: embeddings.embed_query(x))
#     return df

def split_train_test_data(df_sample):
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(list(df_sample['embeddings']), list(df_sample['label']), test_size=0.25, random_state= 0)
    return sentences_train, sentences_test, labels_train, labels_test

def get_score(svm_classifier, sentences_test, labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score

# Generating embeddings for our input dataset
def create_embeddingdf(df,embeddings):
    #df['embeddings'] = df['text'].apply(lambda x: embeddings.embed_query(x))    
    #return df
    # Check if the DataFrame has the correct columns
    if df.columns[0] != 'text' or df.columns[1] != 'label':
        # If not, set the column names explicitly (assuming the CSV has no header)
        df.columns = ['text', 'label']
    df['embeddings'] = df['text'].apply(lambda x: embeddings.embed_query(x))    
    return df
    #embeddings_list = embeddings.embed_documents(df['text'].tolist())  # Convert the 'text' column to a list
    #df['embeddings'] = embeddings_list  # Assign the embeddings to the 'embeddings' column
    #return df

