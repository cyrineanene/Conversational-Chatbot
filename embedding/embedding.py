#Goal is to create vector text embeddings from the split texts 
import json
from langchain_openai import AzureOpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

class VectorEmbedding:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name=model_name

        self.azure_emb_api_key= os.getenv("azure_emb_api_key")
        self.azure_emb_endpoint= os.getenv("azure_emb_endpoint")
        self.api_emb_version=os.getenv("api_emb_version")
        
        self.azure_embedding_service = AzureOpenAIEmbeddings(
            api_key=self.azure_emb_api_key,
            azure_endpoint=self.azure_emb_endpoint,
            api_version = self.api_emb_version, 
            model=model_name
            )
        
    def embedd (self, text):
        response = self.azure_embedding_service.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding
    
    def create_output_files(self, folder_path="data_output/"):
        embeddings = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                document = json.loads(content)
                
                text = document['content']
                embedding = self.embedd(text)
                if embedding:
                    embeddings.append({"embedding": embedding})
        
        return embeddings

#Testing the embedding
embedder = VectorEmbedding()
embedder.create_output_files()