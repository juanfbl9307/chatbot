from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import warnings
from langchain_core.output_parsers import StrOutputParser
import chromadb
from http.server import BaseHTTPRequestHandler, HTTPServer

warnings.filterwarnings("ignore")
model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.5, model=model, max_tokens=4096)
chat_hist_msg_count = 24
chat_session_id = "1"
mongo_uri = "mongodb://admin:password@localhost:27017"
mongo_db_name = "chat"
mongo_collection_name = "histories"
embeddings = OpenAIEmbeddings()
chromadb_client = chromadb.Client()


class VectorStore:
    def __init__(self):
        self.collection_name = "chatbot"
        chromadb_client.get_or_create_collection(self.collection_name)
        self.langchain_chroma = Chroma(
            client=chromadb_client,
            collection_name=self.collection_name,
            embedding_function=embeddings,
        )

        pass

    def get_retriever(self):
        return self.langchain_chroma.as_retriever()


class ChatHistoryDb:
    def __init__(self):
        self.client = MongoDBChatMessageHistory(
            session_id=chat_session_id,
            connection_string=mongo_uri,
            database_name=mongo_db_name,
            collection_name=mongo_collection_name,
        )
        pass

    def add_user_message(self, message):
        self.client.add_user_message(message)
        pass

    def add_ai_message(self, message):
        self.client.add_ai_message(message)
        pass

    def get_messages(self):
        return self.client.messages


chat_history = ChatHistoryDb()
vector_store = VectorStore()
prompt = """
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
            """
contextualize_q_system_prompt = """Dado un historial de chat y la última pregunta del usuario,
que podría hacer referencia al contexto en el historial de chat, formula una pregunta independiente
que se pueda entender sin el historial de chat. NO respondas a la pregunta,
solo reformúlala si es necesario y, de lo contrario, devuélvela tal cual."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


def contextualized_question(input_element: dict):
    if input_element.get("chat_history"):
        return contextualize_q_chain
    else:
        return input_element["question"]


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        RunnablePassthrough.assign(
            contexto=contextualized_question | vector_store.get_retriever() | format_docs
        )
        | qa_prompt
        | llm
)


def query(texto):
    if len(chat_history.get_messages()) <= chat_hist_msg_count:
        msgs = chat_history.get_messages()
    else:
        msgs = chat_history.get_messages()[-chat_hist_msg_count:]

    response = rag_chain.invoke({"question": texto, "chat_history": msgs})
    content = response.content
    if "## Orden de Pedido" in content:
        print("pedido realizado!!")

    chat_history.add_user_message(texto)
    chat_history.add_ai_message(content)
    return content


# Define the request handler class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # Handler for GET requests
    def do_GET(self):
        if self.path == '/chat':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Welcome to the chat room!")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"404 - Not Found")


# Set up the HTTP server
def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


# Run the HTTP server
if __name__ == '__main__':
    print(query('que servicios ofrecen?'))

