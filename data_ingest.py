from main import ChromaRepository
from langchain.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
dotenv.load_dotenv()

import uuid


class ChromaDataIngestorCsv:
    def __init__(self, chroma_db: ChromaRepository, file_path):
        loader = CSVLoader(file_path)
        self.docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.split_docs = text_splitter.split_documents(self.docs)
        self.chroma_db = chroma_db

    def ingest(self):
        for doc in self.split_docs:
            self.chroma_db.get_collection().add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )


def main():
    chroma_db = ChromaRepository()
    data_ingestor = ChromaDataIngestorCsv(chroma_db, "responses.csv")
    data_ingestor.ingest()


if __name__ == '__main__':
    main()