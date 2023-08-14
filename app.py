import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings   import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings   import HuggingFaceInstructEmbeddings
import torch
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css, user_template, bot_template
# Set PYTORCH_CUDA_ALLOC_CONF environment variable

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.empty_cache()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def main():

    load_dotenv()

    st.set_page_config(page_title="LibraryGPT", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with LibraryGPT :books: [WIP] :construction:")
    user_question = st.text_input("Ask a question about your documentation: ")
    
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your Documentation")
        pdf_docs = st.file_uploader(
            "Upload Files to chat with LibraryGPT", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                #Get PDF Text
                raw_text = get_pdf_text(pdf_docs)
                # Get Text from PDF
                text_chunks = get_text_chunks(raw_text)
                # Create VectorStore
                vectorstore = get_vectorstore(text_chunks)
                
                st.session_state.conversation = get_conversation_chain(vectorstore)

    st.session_state.conversation



if __name__ == "__main__":
    main()