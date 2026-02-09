"""
CW/03/main.py
混合檢索 pipeline:
1) 讀 day6/*.txt 並切塊
2) 呼叫 embed API 批次產生 embeddings
3) 建立 Qdrant collection (dense)
4) 建立 rank_bm25 sparse index (local)
5) Query_ReWrite: 呼叫 LLM API 把 user query rewrite 成更好檢索的 query
6) Retrieval: 使用 rewrite 去 dense + sparse 各自檢索 top_k，融合結果
7) LLM answer: 用檢索到的上下文呼叫 LLM 生成最終答案
8) 把 Re_Write_questions.csv 全部處理並輸出 rewritten_questions.csv
"""

import os
import glob
import uuid
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# ---------------------------
# Config 
# ---------------------------
EMBED_API = "https://ws-04.wade0426.me/embed"     # user 提供的 embed endpoint
LLM_API = "https://ws-03.wade0426.me/v1/chat/completions"  # user 提供的 LLM endpoint
LLM_MODEL = "/models/gpt-oss-120b"   # 如需在請求裡指定 model

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "day6_hybrid_demo"
VECTOR_DIM_EXPECTED = 4096  # 若你的 embed API 給不同維度，請改成真實值

DAY6_PATH = "day6/cw-data"
CHUNK_SIZE = 1200    # 以字元（chars）為單位切塊，視需要修改
CHUNK_OVERLAP = 200  # 重疊
TOP_K = 5

OUTPUT_DIR = "day6/03-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Helpers: chunking
# ---------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    以字元為單位進行簡單切塊（不使用 tokenizer），保留 overlap.
    更進階可換成 sentence-break 或 tokenizer-based chunking。
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

# ---------------------------
# Load files and chunk
# ---------------------------
def load_and_chunk_all(txt_folder: str) -> Tuple[List[str], List[Dict]]:
    txt_paths = sorted(glob.glob(os.path.join(txt_folder, "*.txt")))
    documents = []
    metadatas = []
    for p in txt_paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        chunks = chunk_text(raw)
        for i, c in enumerate(chunks):
            doc_id = uuid.uuid4().hex
            documents.append(c)
            metadatas.append({
                "id": doc_id,
                "source_file": os.path.basename(p),
                "chunk_index": i,
            })
    print(f"Loaded {len(txt_paths)} files → {len(documents)} chunks")
    return documents, metadatas

# ---------------------------
# Embedding (batch)
# ---------------------------
def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    呼叫 user 提供的 embed API。
    範例 payload 和回傳結構參考題目說明。
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        body = {
            "texts": batch,
            "task_description": "檢索技術文件",
            "normalize": True
        }
        resp = requests.post(EMBED_API, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch_embeds = data["embeddings"]
        embeddings.extend(batch_embeds)
    # Optionally assert dimension
    if len(embeddings) > 0:
        dim = len(embeddings[0])
        print("embedding dim:", dim)
        if dim != VECTOR_DIM_EXPECTED:
            print(f"Warning: expected dim {VECTOR_DIM_EXPECTED} but got {dim}. If it differs intentionally, update VECTOR_DIM_EXPECTED.")
    return embeddings

# ---------------------------
# Qdrant dense index upsert
# ---------------------------
def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    # 如果 collection 已存在，可以先刪除或重建（視情況）
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection {collection_name} already exists. Recreating...")
        client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"ERROR when trying to delete qdrant collection: {e}")
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created Qdrant collection {collection_name} (dim={vector_size})")

def upsert_qdrant_points(client: QdrantClient, collection_name: str, metadatas: List[Dict], embeddings: List[List[float]]):
    # Qdrant expects list of dicts with "id", "vector", "payload"
    points = []
    for meta, emb in zip(metadatas, embeddings):
        points.append({
            "id": meta["id"],
            "vector": emb,
            "payload": {
                "text": meta.get("text", ""),  # we'll attach the text later
                "source_file": meta.get("source_file"),
                "chunk_index": meta.get("chunk_index")
            }
        })
    # Use client.upsert
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Upserted {len(points)} points into {collection_name}")

# ---------------------------
# BM25 sparse index (local)
# ---------------------------
class BM25Index:
    def __init__(self, docs: List[str], metadatas: List[Dict]):
        # tokenize per doc (simple whitespace tokenizer)
        tokenized = [doc.split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.docs = docs
        self.metadatas = metadatas

    def query(self, q: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        token_q = q.split()
        scores = self.bm25.get_scores(token_q)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            results.append((self.metadatas[idx], float(scores[idx])))
        return results

# ---------------------------
# Query rewrite using LLM
# ---------------------------
def query_rewrite_llm(original_query: str) -> str:
    """
    使用 LLM 把 query rewrite 成更適合檢索的 query。
    你可以修改 system/user prompt 以符合作業需求。
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": """# Role
你是一個 RAG (Retrieval-Augmented Generation) 系統的查詢重寫專家 (Query Rewriter)。
你的任務是將使用者的「最新問題」，結合「對話歷史」，重寫成一個適合讓向量資料庫或搜尋引擎理解的「獨立搜尋語句」。

# Context & Data Knowledge
我們的知識庫包含以下領域的資訊，請在重寫時參考這些背景：
1. **Google Cloud 硬體**：包含 Ironwood (第7代 TPU)、Axion 處理器、C4A (裸機)、N4A (虛擬機器)、GKE 支援等。
2. **AI 開發工具**：Google LiteRT、TensorFlow Lite、NPU/GPU 加速、裝置端推論。
3. **氣象與生活**：台中天氣預報、台灣氣候特徵 (季風/地形雨)、日本旅遊與流感疫情 (H3N2/B型)。

# Rules
1. **指代消解 (Coreference Resolution)**：將「它」、「那個」、「第二個」、「那邊」等代名詞，替換為對話歷史中提到的具體實體 (如：N4A, 台中, LiteRT)。
2. **補全上下文 (Contextualization)**：如果問題簡短（例如「效能如何？」），請補上主詞（例如「Google N4A 虛擬機器的效能如何？」）。
3. **保留原意**：不要回答問題，只要「重寫問題」。不要自行捏造不存在的資訊。
4. **關鍵字增強**：如果使用者的詞彙模糊，請嘗試加入上述 Knowledge 中的專有名詞（例如將「Google 新出的那個 CPU」重寫為包含 `Axion` 或 `Ironwood` 的語句），但不要過度發散。
5. **語言一致性**：輸出必須是繁體中文。

# Output Format
請直接輸出重寫後的搜尋語句，不要包含任何解釋或標點符號以外的文字，絕對禁止輸出任何思考過程、前言、解釋。"""},
            {"role": "user", "content": f"請將下列使用者查詢改寫成更適合做文件檢索的查詢（保留原意、補足關鍵字、移除模糊或無效詞語）：\n\n原查詢：{original_query}\n\n請直接回傳改寫後的查詢，不需要任何額外說明。"}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    resp = requests.post(LLM_API, json=payload, timeout=60)
    resp.raise_for_status()
    out = resp.json()

    # 依 API 回傳格式有差異，這裡嘗試通用擷取
    # 典型回傳可能是 out["choices"][0]["message"]["content"]
    text = None
    if "choices" in out and len(out["choices"]) > 0:
        ch = out["choices"][0]
        if "message" in ch and "content" in ch["message"]:
            text = ch["message"]["content"]
        elif "text" in ch:
            text = ch["text"]
    if text is None:
        raise RuntimeError("LLM 回傳格式不符合預期: " + str(out))
    return text.strip()

# ---------------------------
# Hybrid retrieval (dense + sparse fusion)
# ---------------------------
def hybrid_retrieval(client: QdrantClient, collection_name: str, bm25_index: BM25Index, rewrite_query: str, top_k=TOP_K):
    # 1) dense: embed rewrite and query qdrant
    emb = embed_texts([rewrite_query], batch_size=1)[0]
    # qdrant search
    search_res = client.search(
        collection_name=collection_name,
        query_vector=emb,
        limit=top_k,
        with_payload=True
    )
    dense_results = []
    for hit in search_res:
        dense_results.append({
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload
        })

    # 2) sparse: BM25
    sparse_hits = bm25_index.query(rewrite_query, top_k=top_k)
    sparse_results = []
    for meta, score in sparse_hits:
        sparse_results.append({
            "id": meta["id"],
            "score": score,
            "metadata": meta
        })

    # 3) fusion: union with simple scoring (normalized)
    # convert to dict keyed by id
    combined = {}
    # dense scores: smaller is better? qdrant returns score depending on distance; assume higher is better for similarity
    for d in dense_results:
        combined.setdefault(d["id"], {"dense_score": float(d["score"]), "sparse_score": 0.0, "payload": d.get("payload")})
    for s in sparse_results:
        if s["id"] in combined:
            combined[s["id"]]["sparse_score"] = float(s["score"])
        else:
            combined.setdefault(s["id"], {"dense_score": 0.0, "sparse_score": float(s["score"]), "payload": None})

    # compute combined_score = alpha * dense_norm + (1-alpha) * sparse_norm
    alpha = 0.6
    dense_vals = [v["dense_score"] for v in combined.values()]
    sparse_vals = [v["sparse_score"] for v in combined.values()]
    # normalize (min-max) but handle constant arrays
    def norm(vals):
        if len(vals) == 0:
            return []
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return [1.0 for _ in vals]
        return [(v - mn) / (mx - mn) for v in vals]

    dense_norm = norm(dense_vals)
    sparse_norm = norm(sparse_vals)
    keys = list(combined.keys())
    fused = []
    for i, k in enumerate(keys):
        s = alpha * dense_norm[i] + (1 - alpha) * sparse_norm[i]
        item = combined[k]
        fused.append({
            "id": k,
            "score": s,
            "dense_score": item["dense_score"],
            "sparse_score": item["sparse_score"],
            "payload": item["payload"]
        })
    fused_sorted = sorted(fused, key=lambda x: x["score"], reverse=True)[:top_k]
    return fused_sorted

# ---------------------------
# LLM answer using retrieved docs
# ---------------------------
def answer_with_llm(user_question: str, retrieved: List[Dict]) -> str:
    # 組裝 context
    context_blocks = []
    for r in retrieved:
        payload = r.get("payload") or {}
        text = payload.get("text", "(no text in payload)")
        src = payload.get("source_file", "")
        ci = payload.get("chunk_index", "")
        context_blocks.append(f"【來源】{src} chunk {ci}\n{text}\n")

    system_prompt = "你是能利用檢索到的文件內容來回答問題的助手。若文件無法直接回答，請誠實說不確定並指出可能的資訊缺口。回答務求精準且扼要。"
    user_prompt = f"User question: {user_question}\n\nUse the following retrieved documents as context to answer. If you quote text, indicate the source next to it. Documents:\n\n" + "\n---\n".join(context_blocks)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    resp = requests.post(LLM_API, json=payload, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    text = None
    if "choices" in out and len(out["choices"]) > 0:
        ch = out["choices"][0]
        if "message" in ch and "content" in ch["message"]:
            text = ch["message"]["content"]
        elif "text" in ch:
            text = ch["text"]
    if text is None:
        raise RuntimeError("LLM 回傳格式不符合預期: " + str(out))
    return text.strip()

# ---------------------------
# Full pipeline runner & CSV rewrite processing
# ---------------------------
def main():
    # 0) load and chunk
    print("Loading and chunking...")
    documents, metadatas = load_and_chunk_all(DAY6_PATH)
    # attach text into metadata for later upsert payload
    for m, doc in zip(metadatas, documents):
        m["text"] = doc

    # 1) embed
    print("Embedding all chunks...")
    embeddings = embed_texts(documents, batch_size=8)

    # 2) setup Qdrant and upsert
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    create_qdrant_collection(client, COLLECTION_NAME, vector_size=len(embeddings[0]))
    # upsert in batches (qdrant-client will accept list)
    print("Upserting to Qdrant...")
    # Build points expected by qdrant-client upsert (id, vector, payload)
    qdrant_points = []
    for m, emb in zip(metadatas, embeddings):
        qdrant_points.append({
            "id": m["id"],
            "vector": emb,
            "payload": {
                "text": m["text"],
                "source_file": m["source_file"],
                "chunk_index": m["chunk_index"]
            }
        })
    # chunked upsert to avoid too large payload
    BATCH = 256
    for i in range(0, len(qdrant_points), BATCH):
        client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points[i:i+BATCH])

    # 3) BM25 index
    print("Building BM25 index...")
    bm25_index = BM25Index(documents, metadatas)

    # 4) Process Re_Write_questions.csv
    csv_in = "day6/Re_Write_questions.csv"
    if not os.path.exists(csv_in):
        print(f"Input CSV {csv_in} not found. Please place it in repo root. Exiting.")
        return
    df = pd.read_csv(csv_in)
    # 假設 CSV 內有一欄 "questions"（請根據實際欄位更改）
    if "questions" not in df.columns:
        raise ValueError("CSV must contain 'questions' column.")
    rewritten = []
    for q in tqdm(df["questions"].tolist(), desc="Rewriting queries with LLM"):
        try:
            rw = query_rewrite_llm(q)
        except Exception as e:
            print("Rewrite error:", e)
            rw = q  # fallback: 原 query
        rewritten.append(rw)
    df["rewritten"] = rewritten
    out_csv = os.path.join(OUTPUT_DIR, "rewritten_questions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved rewritten CSV to", out_csv)

    # 5) For each rewritten query, do retrieval + LLM answer and save result
    results = []
    for orig, rw in tqdm(zip(df["questions"], df["rewritten"]), total=len(df), desc="Retrieval+Answer"):
        retrieved = hybrid_retrieval(client, COLLECTION_NAME, bm25_index, rw, top_k=TOP_K)
        answer = answer_with_llm(orig, retrieved)
        results.append({
            "questions": orig,
            "rewritten": rw,
            "retrieved": retrieved,
            "answer": answer
        })
    with open(os.path.join(OUTPUT_DIR, "retrieval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved retrieval+answers to outputs/retrieval_results.json")

if __name__ == "__main__":
    main()
