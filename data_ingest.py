from langchain.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_community.vectorstores import Chroma
import os
import dotenv
from langchain_openai import OpenAIEmbeddings
import warnings

dotenv.load_dotenv()
warnings.filterwarnings("ignore")
openai_api_key = os.getenv('OPENAI_API_KEY')
embedding_model = "text-embedding-ada-002"


class ChromaDataManager:
    def __init__(self, collection_name="chatbot"):
        self.client = chromadb.HttpClient(host="localhost", port=8000)
        self.collection_name = collection_name
        self.embedding_function = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def ingest_from_text(self, text_path="doc.txt"):
        loader = TextLoader(text_path)
        text_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(text_docs)
        Chroma.from_documents(documents=split_docs, embedding=self.embedding_function,
                              collection_name=self.collection_name,
                              client=self.client)
        return print(f"Data ingested successfully, total collection count: {self.collection.count()}")

    def ingest_csv(self, file_path="responses.csv"):
        loader = CSVLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        Chroma.from_documents(documents=split_docs, embedding=self.embedding_function,
                              collection_name=self.collection_name,
                              client=self.client)
        return print(f"Data ingested successfully, total collection count: {self.collection.count()}")

    def reset_all(self):
        self.client.reset()


def ingest_data_from_csv():
    try:
        data_ingestor = ChromaDataManager()
        data_ingestor.ingest_csv()
    except Exception as e:
        print(f"Error ingesting data: {e}")


def ingest_data_from_text():
    try:
        data_ingestor = ChromaDataManager()
        data_ingestor.ingest_from_text()
    except Exception as e:
        print(f"Error ingesting data: {e}")


def delete_all_data():
    try:
        data_ingestor = ChromaDataManager()
        data_ingestor.reset_all()
    except Exception as e:
        print(f"Error deleting data: {e}")


ingest_data_from_text()
