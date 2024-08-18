#Goal is to convert an .md file to chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
#from langchain_community.document_loaders import MarkdownLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import datetime
import json
import time

class Chunker:
    #Step 1: Initializing the variables
    def __init__(self, chunk_size=200, chuck_overlap = 20, directory_path = "data_input/", output_folder= "data_output/" ):
        self.chunk_size = chunk_size
        self.chuck_overlap = chuck_overlap
        self.directory_path = directory_path
        self.output_folder = output_folder


    #Step 2: The chunking using a Recursive Method
    def chunking (self, doc):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chuck_overlap
            )
        chunked_docs = text_splitter.split_documents(doc)
        return chunked_docs

    #Step 3: Converting the data to json
    def convert_json(self):
        chunk_docs_json = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".md"):
                    file_path = os.path.join(self.directory_path, filename)
                    loader = UnstructuredMarkdownLoader(file_path)
                    doc = loader.load()
                
                    chunked_docs = self.chunking(doc)

                    for doc in chunked_docs:
                        created_str = time.strftime("%Y%m%dT%H%M%SZ", doc.metadata.get('created', time.gmtime()))
                        updated_str = time.strftime("%Y%m%dT%H%M%SZ", doc.metadata.get('updated', time.gmtime()))

                        create_dt = datetime.datetime.strptime(created_str, "%Y%m%dT%H%M%SZ")
                        update_dt = datetime.datetime.strptime(updated_str, "%Y%m%dT%H%M%SZ")

                        doc.metadata['created'] = int(create_dt.timestamp())
                        doc.metadata['updated'] = int(update_dt.timestamp())

                        doc_json = doc.json()
                        chunk_docs_json.append(doc_json)
            
            return chunk_docs_json
        
    #Step 4: adding the json files to an output folder
    def create_output(self):
        chunk_docs_json = self.convert_json()
        filename = f"data.json"
        file_path = os.path.join(self.output_folder, filename)
        if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
        with open(file_path, 'w') as json_file:
            json.dump(chunk_docs_json, json_file, indent=4)

#Testing the Class
chunker = Chunker()
chunker.create_output()