from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv
import warnings
from typing import Union
from fastapi import FastAPI, Header
import uvicorn
from pydantic import BaseModel
from typing import Annotated

warnings.filterwarnings("ignore")

dotenv.load_dotenv()
chromadb_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chat_hist_msg_count = int(os.environ.get('CHAT_HISTORY_MESSAGE_COUNT', '24').strip())
model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.5, model=model, max_tokens=4096)


def format_docs(documents):
    return "\n\n".join(d.page_content for d in documents)


class ChatHistory:
    def __init__(self, session_id, collection_name, mongo_uri, mongo_db_name):
        self.client = MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=mongo_uri,
            database_name=mongo_db_name,
            collection_name=collection_name,
        )
        pass

    def get_chat_history_client(self):
        return self.client

    def add_user_message(self, message):
        self.client.add_user_message(message)
        pass

    def add_ai_message(self, message):
        self.client.add_ai_message(message)
        pass

    def get_messages(self):
        return self.client.messages


class ChromaRepository:
    def __init__(self, collection_name="chatbot"):
        self.collection_name = collection_name
        self.collection = chromadb_client.get_or_create_collection(self.collection_name)
        self.langchain_chroma = Chroma(
            client=chromadb_client,
            collection_name=self.collection_name,
            embedding_function=embedding_function,
        )
        pass

    def get_retriever(self):
        return self.langchain_chroma.as_retriever()

    def get_collection(self):
        return self.collection

    def get_chroma(self):
        return self.langchain_chroma

    def query(self, query, limit=3) -> dict:
        result = self.langchain_chroma.similarity_search_with_score(query, k=limit)
        return {
            "result": result[0][0].page_content,
            "score": result[0][1]
        }


chroma_repository = ChromaRepository()


class ChatBot:
    def __init__(self, chat_history: ChatHistory, prompt_template,
                 context_template_prompt):

        self.chroma_repository = chroma_repository
        self.chat_history = chat_history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", context_template_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.context_chain = contextualize_q_prompt | llm | StrOutputParser()
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.rag_chain = (
                RunnablePassthrough.assign(
                    contexto=self.__contextualized_question | chroma_repository.get_retriever() | format_docs
                )
                | qa_prompt
                | llm
        )

        pass

    def __contextualized_question(self, input_question: dict):
        if input_question.get("chat_history"):
            return self.context_chain
        else:
            return input_question["question"]

    def query(self, texto):
        chat_history = self.chat_history.get_chat_history_client()
        chat_history_messages = chat_history.messages
        if len(chat_history_messages) <= chat_hist_msg_count:
            msgs = chat_history_messages
        else:
            msgs = chat_history_messages[-chat_hist_msg_count:]

        response = self.rag_chain.invoke({"question": texto, "chat_history": msgs})
        content = response.content
        if "## Orden de Pedido" in content:
            print("pedido realizado!!")

        chat_history.add_user_message(texto)
        chat_history.add_ai_message(content)
        return content


def main(session_id):
    chat_history = ChatHistory(
        session_id=session_id,
        collection_name="histories",
        mongo_uri="mongodb://admin:password@localhost:27017",
        mongo_db_name="chat"
    )
    chat_bot = ChatBot(
        chat_history=chat_history,
        prompt_template="""
            Hola, soy tu asistente virtual de pedidos. Estoy aquí para ayudarte a realizar tu pedido de manera rápida y eficiente. Por favor, proporcióname los siguientes detalles para poder procesar tu pedido correctamente:

            Nombre del Producto o Servicio: (Por ejemplo, "Pizza Margarita grande", "Reservación para dos personas", etc.)
            
            Cantidad: (Indica cuántas unidades del producto o servicio deseas.)
            
            Opciones Específicas: (Si el producto o servicio tiene opciones adicionales, como tamaño, color, ingredientes extra, etc., inclúyelas aquí.)
            
            Fecha y Hora de Entrega o Reservación: (Especifica cuándo necesitas que se entregue tu pedido o para cuándo deseas hacer la reservación.)
            
            Dirección de Entrega: (Si tu pedido requiere entrega, proporciona la dirección completa y cualquier instrucción específica para el repartidor.)
            
            Información de Contacto: (Incluye un número de teléfono o correo electrónico donde podamos contactarte para confirmar el pedido o en caso de necesitar más detalles.)
            
            Una vez que tengas toda esta información, puedes decírmela o escribirla aquí. Yo me encargaré de revisar los detalles y generar tu pedido. Si hay algo que necesito aclarar o confirmar, te lo haré saber.
            
            En caso de obtener la informacion del cliente y de la orden, retornarla formateada y lista para ser procesada, con una cabezera que diga "Orden de Pedido", generando un numero aleatorio para la orden y los detalles de la orden.
            
            Cuando el cliente confirme la orden se debe enviar un mensaje agradeciendo la orden, con el mismo numero del pedido y dando un tiempo estimado de entrega.
            
            {contexto}
            """,
        context_template_prompt="""Dado un historial de chat y la última pregunta del usuario,
que podría hacer referencia al contexto en el historial de chat, formula una pregunta independiente
que se pueda entender sin el historial de chat. NO respondas a la pregunta,
solo reformúlala si es necesario y, de lo contrario, devuélvela tal cual."""
    )

    return chat_bot


class Chat(BaseModel):
    content: str


app = FastAPI()
if __name__ == "__main__":
    @app.post("/chat/")
    async def chat(chat_json: Chat, session_id: Annotated[str | None, Header(convert_underscores=False)] = None):
        chat_bot = main(session_id)
        return {"Response": chat_bot.query(chat_json.content)}


    uvicorn.run(app, host="0.0.0.0", port=8082)
