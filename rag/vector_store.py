import chromadb
from sentence_transformers import SentenceTransformer

from rag.documents import MEDICAL_KNOWLEDGE_BASE


class MedicalVectorStore:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path="./data/chromadb"
        )

        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge"
        )

        self._initialize_documents()

    def _initialize_documents(self):

        # Avoid inserting documents multiple times
        existing = self.collection.count()

        if existing > 0:
            return

        documents = []
        ids = []
        metadatas = []

        for document in MEDICAL_KNOWLEDGE_BASE:

            documents.append(document.content)

            ids.append(document.document_id)

            metadatas.append({
                "title": document.title,
                "disease": document.disease,
                "keywords": ",".join(document.keywords)
            })

        embeddings = self.embedding_model.encode(
            documents
        ).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ):

        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        return results