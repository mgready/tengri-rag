import chromadb
from sentence_transformers import SentenceTransformer

# ---- Настройки ----
DB_DIR = "chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-base"
model = SentenceTransformer(EMBED_MODEL)

# ---- подключаем Chroma ----
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("tengri_news")

# ---- запрос ----
query = "чемпион по боксу"
query_emb = model.encode(query).tolist()  # <-- сами делаем эмбеддинг (768-dim)

results = collection.query(
    query_embeddings=[query_emb],  # <-- сюда передаём эмбеддинг
    n_results=3
)

# ---- вывод ----
print(f"\n🔎 Результаты по запросу: {query}\n")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"📌 {meta['title']}")
    print(f"🔗 {meta['url']}")
    print(f"📝 Фрагмент: {doc[:200]}...\n")
