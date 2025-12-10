import os
import shutil
import tiktoken
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import Config

class DataBase:
    def __init__(self, data_path=Config.DATA_PATH, chroma_path=Config.CHROMA_PATH):
        self.DATA_PATH = data_path
        self.CHROMA_PATH = chroma_path
        self.embedding_model = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL_NAME)

    def get_db(self):
        """Returns the Chroma DB instance. Creates/Loads it automatically."""
        return Chroma(
            persist_directory=self.CHROMA_PATH,
            embedding_function=self.embedding_model
        )

    def sync_data(self):
        """
        Orchestrator: Loads -> Splits -> Adds new documents only.
        """
        print("🔄 Starting database synchronization...")
        
        # 1. Load
        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH)
            print(f"⚠️ Created data directory at {self.DATA_PATH}")
            return

        print(f"📄 Loading documents from: {self.DATA_PATH}")
        loader = PyPDFDirectoryLoader(self.DATA_PATH)
        docs = loader.load()
        
        if not docs:
            print("⚠️ No documents found to load.")
            return

        # 2. Split
        print("✂️ Splitting documents...")
        encoding = tiktoken.get_encoding("cl100k_base")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=lambda text: len(encoding.encode(text)),
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Add (with ID check)
        self.add_docs_to_db(chunks)
        print("✅ Database sync complete.")

    def calculate_chunk_ids(self, chunks: list[Document]):
        """Generates unique IDs based on Source:Page:Index."""
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks

    def add_docs_to_db(self, chunks: list[Document]):
        """Adds new documents to the DB, skipping duplicates."""
        db = self.get_db()
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Check existing items
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"📊 Existing documents in DB: {len(existing_ids)}")

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if new_chunks:
            print(f"👉 Adding {len(new_chunks)} new documents...")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add.")

    def delete_docs_from_db(self, source_path: str):
        """Removes all chunks associated with a specific file path."""
        db = self.get_db()
        results = db.get(where={"source": source_path})
        ids_to_delete = results["ids"]
        
        if not ids_to_delete:
            print(f"⚠️ No documents found for source: {source_path}")
            return

        print(f"🗑️ Deleting {len(ids_to_delete)} chunks for '{source_path}'...")
        db.delete(ids=ids_to_delete)
        print("✅ Deletion complete.")

    def clear_db(self):
        """Wipes the entire database directory."""
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)
            print(f"🔥 Database at {self.CHROMA_PATH} cleared.")
        else:
            print("Database does not exist.")