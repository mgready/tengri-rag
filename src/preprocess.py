import json
import re
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb

# ---- Настройки ----
JSONL_PATH = "/Users/maga/Documents/Parcer/data_tengrinews/tengrinews_ru.jsonl"
DB_DIR = "chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-base"

# ---- Очистка текста ----
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"Наши новости.*WhatsApp.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---- Чанкование ----
def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# ---- Чтение и подготовка ----
docs = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if not obj.get("text"):
            continue

        text = clean_text(obj["text"])
        title = clean_text(obj.get("title", ""))
        url = obj.get("url", "")
        date = obj.get("published", "")

        # нормализуем дату
        try:
            date = datetime.fromisoformat(date).date().isoformat()
        except:
            date = ""

        # делаем чанки
        for idx, chunk in enumerate(chunk_text(text, max_words=300)):
            docs.append({
                "id": f"{url}#{idx}",
                "title": title,
                "chunk": chunk,
                "url": url,
                "date": date
            })

print(f"Подготовлено {len(docs)} документов (с учётом чанков)")

# ---- Embeddings ----
model = SentenceTransformer(EMBED_MODEL)

# ---- ChromaDB ----
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="tengri_news")

for d in tqdm(docs):
    emb = model.encode(d["title"] + " " + d["chunk"]).tolist()
    collection.add(
        ids=[d["id"]],
        embeddings=[emb],
        documents=[d["chunk"]],
        metadatas=[{
            "title": d["title"],
            "url": d["url"],
            "date": d["date"]
        }]
    )

print("✅ Индексация завершена")
