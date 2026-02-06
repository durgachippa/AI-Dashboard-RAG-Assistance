import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.set_page_config(page_title="BI RAG PDF Assistant", page_icon="📄")

st.title("📄 BI RAG PDF Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    collection_name = "documents"
    vectorstore = Chroma.from_texts(chunks, embeddings, collection_name=collection_name)

    # Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

    # Ask questions
    query = st.text_input("Ask a question about your PDF:")
    if query:
        response = qa_chain.run(query)
        st.write(response)

