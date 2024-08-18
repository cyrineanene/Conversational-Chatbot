# Chatbot with Pinecone, LangChain, and Azure OpenAI

This repository contains a chatbot project that stores data in Pinecone, a vector database, and uses LangChain and Azure OpenAI to generate responses. The project includes a Streamlit interface to interact with the chatbot and test the code.

## Getting Started

#### 1. Set Up Your Virtual Environment

First, create a virtual environment to isolate your project dependencies. Run the following command:

    python -m venv .venv

Activate the virtual environment:

- On Windows:

        .venv\Scripts\activate

- On macOS/Linux:
        
        source .venv/bin/activate

Install the required dependencies:

    pip install -r requirements.txt

#### 2. Set Up Your Environment Variables:

- Create a `.env` file in the `embedding/`and `LLM/` directories. 
- Store your `API keys and some variables` (AzureOpenAI API Key, AzureOpenAI Endpoint, AzureOpenAI Version, Pinecone API Key and Pinecone Index Name) in these files.

#### 3. Prepare Your Data

    Place the file you want to use as input data into the `data_input/` directory.

#### 4. Process and Store Data in Pinecone

To process and store your data in Pinecone, run the following scripts in sequence:

##### Run the Chunking Script:

    python embedding/chunking.py

##### Run the Embedding Script:

    python embedding/embedding.py

##### Run the Vector Uploader Script:

    python embedding/vector_uploader.py

#### 5. Run the Streamlit Interface

To interact with the chatbot via the Streamlit interface, run the main.py script using the following command:

    streamlit run main.py