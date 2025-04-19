# app.py

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set the Hugging Face token for LangChain
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Load the FAISS vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_store", embeddings)

# Load vectorstore and setup retrieval-based QA chain
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Use a compatible Hugging Face model
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # you can also try mistralai/Mistral-7B-Instruct-v0.1
    model_kwargs={"temperature": 0.5, "max_new_tokens": 256}
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="Amazon Prime Chatbot", page_icon="🎬")
st.title("🎬 Amazon Prime Chatbot")
st.write("Ask me anything about Amazon Prime content!")

user_query = st.text_input("🧑 Ask a question")

if user_query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_query)
    st.success("🤖 Answer:")
    st.write(answer)
