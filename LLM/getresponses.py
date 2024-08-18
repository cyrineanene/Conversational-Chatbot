import os
from dotenv import load_dotenv

load_dotenv()

azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_version = os.getenv("azure_version")
azure_endpoint = os.getenv("azure_endpoint")

pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("pinecone_index_name")

llm_temperature = 0.0
llm_model_name = "gpt-4o"

#initialize the pinecone and azureopenai
from openai import AzureOpenAI
from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore
client = AzureOpenAI(
    api_key = azure_openai_key,  
    api_version = azure_version,
    azure_endpoint = azure_endpoint
)

pc = Pinecone(api_key=pinecone_key)
index_name = index_name
index = pc.Index(index_name)

#initialize the azureopenai embedding model
from langchain_openai import AzureOpenAIEmbeddings
azure_emb_api_key = os.getenv("azure_emb_api_key")
azure_emb_endpoint = os.getenv("azure_emb_endpoint")
api_emb_version = os.getenv("api_emb_version")
embeddings_model_name = "text-embedding-ada-002"  

# Initialize the Azure OpenAI Embeddings service
azure_embedding_service = AzureOpenAIEmbeddings(
    api_key=azure_emb_api_key,
    azure_endpoint=azure_emb_endpoint,
    api_version = api_emb_version, 
    model=embeddings_model_name
)

pinecone_vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=azure_embedding_service)

#initializing the chains
#the ConversationalRetrievalChain handles: semantic search using pinecone_vector_store.as_retriever(), Generating refined questions using question_generator and Combining the retrieved documents and generating a response using doc_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

_template = """  Given the following extracted parts of a long document and a question,
         create a final answer with references ("SOURCES") unless identified below.
         
        But if you are asked something similar to what your purpose is as an AI Assistant, then answer with the following:
         I'm a helpful assistant for {username} answering his questions based on the informations from the website.
        Also, ALWAYS return a "SOURCES" part in your answer.

        QUESTION: {question}
         =========
         {summaries}
         =========
        FINAL ANSWER:
"""


QA_PROMPT = PromptTemplate.from_template(_template)
chat_llm = AzureChatOpenAI(api_key = azure_openai_key, api_version = azure_version, azure_endpoint = azure_endpoint, temperature=llm_temperature, model=llm_model_name)
question_generator = LLMChain(llm=chat_llm, prompt=CONDENSE_QUESTION_PROMPT,  verbose=False)
doc_chain = load_qa_with_sources_chain(chat_llm, chain_type="stuff", verbose=False, prompt=QA_PROMPT)
qa_chain = ConversationalRetrievalChain(
            retriever=pinecone_vector_store.as_retriever(),
            question_generator=question_generator,
           combine_docs_chain=doc_chain,
            verbose=True
        )

#This is a chat_history exemple
chat_history = [("context1", "Hello, how can I help you today?"),
("context2", "I'm looking for information on movies."),
("context3", "Sure! What specific movie gender are you interested in?"),
("context4", "I want to know best movies for action and thriller."),
]

#Testing the chain
user = input('enter your name: ') 
question = input('What is your question?')

result = qa_chain({"username": user, "question": question, "chat_history": chat_history})

answer = result["answer"]
answer = answer.replace('\n', '').replace("'", "\\'")
print("Answer:", answer)