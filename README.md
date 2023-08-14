# LibraryGPT Document Chat Interface

This project provides a conversational interface that allows users to interact with PDF documents using GPT-4. It is implemented using Streamlit and integrates various features such as PDF reading, text splitting, vector storage, and conversational chains. The agent will remember context of your conversation and your document.

## Embeddings

In the file app.py there is a method 

``` py
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
```
Users can uncomment this line and comment out the OpenAIEmbeddings function if they wish to process the embeddings using their own compute resources. 

If you wish to do this, change this line from 

``` py
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
```
to 

``` py
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
```
This will allow torch to access the GPU and block the CPU. Setting this value to 1 will block torch from processing with the GPU and prefer the CPU.


## Features

- **PDF Reading**: Extracts text from uploaded PDF files.
- **Text Chunking**: Splits text into chunks for efficient processing.
- **Vector Storage**: Utilizes FAISS for vector storage and retrieval.
- **Conversational Chains**: Implements a retrieval chain to handle user questions.

## Libraries and Components

- **Streamlit**: Used for the web interface.
- **dotenv**: To manage environment variables.
- **PyPDF2**: For PDF reading.
- **langchain**: Contains several modules for text processing, embeddings, vector storage, and conversation handling.
- **Torch**: For handling GPU memory and other CUDA-related tasks.
- **htmlTemplate**: Contains templates for custom HTML styling.

## How It Works

1. **Initialization**: Sets up the page layout and initializes session variables.
2. **User Input Handling**: Accepts user questions and handles responses.
3. **PDF Processing**: Allows users to upload PDFs and extracts text.
4. **Text Chunking**: The extracted text is split into chunks using `CharacterTextSplitter`.
5. **Vector Store Creation**: The text chunks are embedded using `OpenAIEmbeddings` or `HuggingFaceInstructEmbeddings`, and a vector store is created using FAISS.
6. **Conversation Chain Setup**: A conversational chain is set up using `ConversationalRetrievalChain` and the vector store.
7. **Chat Handling**: Processes user input and provides responses via the chat interface.

## Usage

``` bash
git clone https://github.com/tdolan21/libraryGPT
cd libraryGPT
pip install -r requirements.txt
```
Once you have the requirements installed, you can run the command 
``` bash
streamlit run app.py
```

This will run the script, and the Streamlit application will start. Users can upload PDF files, ask questions related to the documentation, and receive answers from the LibraryGPT model.
