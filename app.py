import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv
import base64
st.set_page_config(page_title="Amazon Prime Chatbot", page_icon="🎬")
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: centre;
            background-position: fit;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"amazon.png")
# Load local .env (for development)
load_dotenv()

# Read Hugging Face token from environment (works for .env, GitHub Secrets, or Streamlit Secrets)
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
# Set it explicitly for HuggingFaceHub to access
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Load the FAISS vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_store", embeddings,allow_dangerous_deserialization=True)

# Load vectorstore and setup retrieval-based QA chain
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Compatible Hugging Face LLM (must support text generation)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 250},
    task="text-generation"
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("🎬 Amazon Prime Chatbot")
st.write("Ask me anything about Amazon Prime content!")

user_query = st.text_input("🧑 Ask a question")

if user_query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_query)
    st.success("🤖 Answer:")
    st.write(answer)
