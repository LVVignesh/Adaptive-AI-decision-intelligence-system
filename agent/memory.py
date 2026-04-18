import chromadb
import os

class Memory:
    def __init__(self):
        # Use a persistent client so memory isn't lost between runs
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("reflections")

    def add(self, state: str, reflection: str):
        self.collection.add(
            documents=[reflection],
            metadatas=[{"state": state}],
            ids=[str(hash(state + reflection))]
        )

    def query(self, state: str, k: int = 3):
        results = self.collection.query(
            query_texts=[state],
            n_results=k
        )
        return results["documents"][0] if results["documents"] else []
