import chromadb

phrases= ["The journey of a thousand miles begins with a single step.",
          "A mind that is stretched by a new experience can never go back to its old dimensions.",
          "The best and most beautiful things in the world cannot be seen or even touched - they must be felt with the heart.",
          "Curiosity is the key to creativity",
          "The only person you are destined to become is the person you decide to be.",
          "John's Cookies were only half baked but he still carries them for Mary.",
          "Amanada baked Cookies for Mary."]

phrase2= ['apple iPhone is expensive']

ids=["001","002", "003", "004", "005","006","007","008"]
metadatas=[
    {"source": "pdf-1"},
    {"source": "doc-1"},
    {"source": "web-1"},
    {"source": "pdf-2"},
    {"source": "txt-1"},
    {"source": "txt-1"},
    {"source": "txt-1"},
    {"source": "pdf-1"},
    ]

chroma_client =chromadb.Client()
collection = chroma_client.create_collection(name="embeddings_demo")

collection.add(
    documents= phrases + phrase2,
    metadatas= metadatas,
    ids=ids)

collection.peek()
results = collection.query(
    query_texts=["Mary"],
    n_results=1
)

results_src= collection.query(
    query_texts=["cookies"],
    where={"source":"pdf-1"},
    n_results=1
)

print(results['documents'])