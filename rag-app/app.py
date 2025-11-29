import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


nvidiaapi=os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key=nvidiaapi, 
  temperature=0.2,
  top_p=0.7,
  max_tokens=256,
)
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("../data")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=90)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)


st.title("NVIDIA NIM KA ISTEMAL")
prompt = """
Answer the question based on the provided context only.
Please provide an accurate answer.
Context: {context}

Question: {input}
"""
real_prompt=ChatPromptTemplate.from_template(
    template=prompt
)

question = st.text_input("Enter your question from the documents")

if st.button("document embed"):
    vector_embedding()
    st.write("vector store db is ready using nvidiaEmbedding")

if question:
    document_chain = create_stuff_documents_chain(llm,real_prompt)
    retriever=st.session_state.vectors.as_retriever()
    chain=create_retrieval_chain(retriever,document_chain)

    response=chain.invoke({"input":question})
    st.write(response['answer'])

    with st.expander("Doc similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------------")