import chromadb

chromadb_client = chromadb.HttpClient(host="localhost", port=8000)
col_name = "chatbot"
col = chromadb_client.get_or_create_collection(col_name)
# col.add(
#     ids=["1"],
#     documents=["This is a 1 document."],
# )
print(chromadb_client.heartbeat())
print(col.peek())
