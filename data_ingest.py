from main import ChromaRepository
from langchain.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid


class ChromaDataIngestorCsv:
    def __init__(self, chroma_db: ChromaRepository, file_path):
        loader = CSVLoader(file_path)
        self.docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.split_docs = text_splitter.split_documents(self.docs)
        self.chromaDb = chroma_db

    def ingest(self):
        for doc in self.split_docs:
            self.chromaDb.get_collection().add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )
