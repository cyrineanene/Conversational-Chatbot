#put the vectors into pinecone
import json
import uuid
import os
from pinecone import Pinecone

class VectorEmbeddingUploader:
    def __init__(self, batch_size=100):
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("pinecone_index_name")

        self.batch_size = batch_size
        self.initialize_services()

    def initialize_services(self):
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.pinecone_index = self.pc.Index(self.index_name)

    def upsert_embeddings(self, chunk_docs_json_list):
        texts = []
        embeddings = []
        metadatas = []

        for doc_dict in chunk_docs_json_list:
            texts.append(doc_dict["text"])
            embeddings.append(doc_dict['embedding'])
            metadatas.append(doc_dict['metadata'])

        vector_ids = []
        for i in range(0, len(texts), self.batch_size):
            i_end = min(i + self.batch_size, len(texts))
            lines_batch = texts[i:i_end]
            ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
            vector_ids.extend(ids_batch)
            embeddings_batch = embeddings[i:i_end]
            metadata_batch = metadatas[i:i_end]

            for j, line in enumerate(lines_batch):
                metadata_batch[j]["text"] = line
            to_upsert = zip(ids_batch, embeddings_batch, metadata_batch)

            self.pinecone_index.upsert(vectors=list(to_upsert), namespace=self.namespace)

        vector_ids_json_string = json.dumps(vector_ids)
        return vector_ids_json_string
 
#Testing the uploader
batch_size = 100  
uploader =  VectorEmbeddingUploader(batch_size)
chunk_docs_json_list = [
    {
        "text": "example text",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"key": "value"}
    },
]
uploader.upsert_embeddings(chunk_docs_json_list)
