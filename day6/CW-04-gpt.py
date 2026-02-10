"""
CW/04/main.py
課堂作業-04 完整 pipeline 範例

請先準備：
- day6/CW/*.txt (原始文件)
- questions.csv (題目ID,題目,標準答案,來源文件)
- 本機 reranker model 目錄 models/Qwen3-Reranker-0.6B (可改路徑)
- Qdrant running (預設 http://localhost:6333)

執行: python main.py
產出: outputs/answers.csv, outputs/retrieval_debug.json
"""

import os
import glob
import uuid
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

# text processing
# import jieba

# BM25
from rank_bm25 import BM25Okapi

# qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# reranker (torch)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config (請依你的環境調整)
# -------------------------
EMBED_API = "https://ws-04.wade0426.me/embed"   # 你先前提供的 embed endpoint
LLM_API = "https://ws-03.wade0426.me/v1/chat/completions"
LLM_MODEL = "/models/gpt-oss-120b"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "day6_hybrid_search"
VECTOR_DIM_EXPECTED = 4096  # 如果 embed 回傳維度不同，程式會自動調整

DAY6_PATH = "day6/CW"
QUESTIONS_CSV = "questions.csv"

RERANKER_MODEL_PATH = "models/Qwen3-Reranker-0.6B"  # local
TOP_K_INITIAL = 50   # dense/sparse 各取 top50 作候選
TOP_K_FINAL = 5      # rerank 後取 top5 給 LLM
BATCH_EMBED = 16

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utilities: chunking
# -------------------------
CHUNK_SIZE = 1200    # 可改
CHUNK_OVERLAP = 200

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

# -------------------------
# Load and chunk files
# -------------------------
def load_and_chunk(folder: str) -> Tuple[List[str], List[Dict]]:
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    docs = []
    metas = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        chunks = chunk_text(txt)
        for i, c in enumerate(chunks):
            doc_id = uuid.uuid4().hex
            docs.append(c)
            metas.append({
                "id": doc_id,
                "source_file": os.path.basename(p),
                "chunk_index": i,
                "text": c
            })
    print(f"Loaded {len(paths)} files -> {len(docs)} chunks")
    return docs, metas

# -------------------------
# Embedding API (batch)
# -------------------------
def embed_texts(texts: List[str], batch_size: int = BATCH_EMBED) -> List[List[float]]:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
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
    if len(embeddings) > 0:
        print("embedding dim:", len(embeddings[0]))
    return embeddings

# -------------------------
# Qdrant helpers
# -------------------------
def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection {collection_name} exists -> deleting and recreating")
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print("Created Qdrant collection:", collection_name)

def upsert_qdrant(client: QdrantClient, collection_name: str, metas: List[Dict], embeddings: List[List[float]], batch=256):
    points = []
    for meta, emb in zip(metas, embeddings):
        points.append({
            "id": meta["id"],
            "vector": emb,
            "payload": {
                "text": meta["text"],
                "source_file": meta["source_file"],
                "chunk_index": meta["chunk_index"]
            }
        })
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection_name, points=points[i:i+batch])
    print(f"Upserted {len(points)} points into {collection_name}")

# -------------------------
# BM25 (jieba tokenization for Chinese)
# -------------------------
def build_bm25_index(docs: List[str], metas: List[Dict]):
    tokenized = []
    for d in docs:
        # jieba cut for Chinese; fallback to whitespace split
        toks = [t for t in jieba.lcut(d) if t.strip()]
        tokenized.append(toks)
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized, metas

# -------------------------
# Reranker (local model)
# -------------------------
class LocalReranker:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading reranker on", device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(device).eval()
        self.device = device
        # get ids for "yes"/"no"
        # NOTE: depending on reranker vocabulary, "yes"/"no" might be different tokens
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        # prepare prefix/suffix as in example
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.max_len = 8192

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def process_inputs(self, pairs: List[str]):
        inputs = self.tokenizer(pairs, padding=False, truncation='longest_first', return_tensors=None,
                                max_length=self.max_len - len(self.prefix_tokens) - len(self.suffix_tokens))
        # the tokenizer returns lists; ensure input_ids present
        input_ids = inputs["input_ids"]
        new_ids = []
        for ids in input_ids:
            new_ids.append(self.prefix_tokens + ids + self.suffix_tokens)
        # pad
        padded = self.tokenizer.pad({"input_ids": new_ids}, padding=True, return_tensors="pt", max_length=self.max_len)
        for k in padded:
            padded[k] = padded[k].to(self.device)
        return padded

    @torch.no_grad()
    def compute_logits(self, inputs):
        out = self.model(**inputs).logits  # shape [B, seq_len, vocab]
        # take last position logits
        batch_scores = out[:, -1, :]
        # get yes/no logits
        if self.token_true_id is None or self.token_false_id is None:
            raise RuntimeError("Could not find tokens for 'yes'/'no' in reranker tokenizer.")
        true_vec = batch_scores[:, self.token_true_id]
        false_vec = batch_scores[:, self.token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        logsoft = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = logsoft[:, 1].exp().tolist()
        return scores

    def rerank(self, query: str, documents: List[str], instruction=None, batch_size=8) -> List[Tuple[str, float]]:
        pairs = [self.format_instruction(instruction, query, d) for d in documents]
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            inputs = self.process_inputs(batch_pairs)
            batch_scores = self.compute_logits(inputs)
            scores.extend(batch_scores)
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores

# -------------------------
# Hybrid retrieval
# -------------------------
def hybrid_retrieval(qdrant_client: QdrantClient, collection_name: str, bm25: BM25Okapi,
                     tokenized_docs: List[List[str]], metas: List[Dict], query: str,
                     top_k_initial: int = TOP_K_INITIAL) -> List[Dict]:
    # dense: embed query
    q_emb = embed_texts([query], batch_size=1)[0]
    dense_hits = qdrant_client.search(collection_name=collection_name, query_vector=q_emb, limit=top_k_initial, with_payload=True)
    dense_ids = [h.id for h in dense_hits]
    dense_map = {h.id: {"score": float(h.score), "payload": h.payload} for h in dense_hits}

    # sparse: BM25 -- tokenize query with jieba
    q_tokens = [t for t in jieba.lcut(query) if t.strip()]
    sparse_scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(sparse_scores)[::-1][:top_k_initial]
    sparse_list = []
    for idx in top_idx:
        sparse_list.append((metas[idx]["id"], float(sparse_scores[idx])))

    # union IDs
    candidate_ids = list(dict.fromkeys(dense_ids + [sid for sid, _ in sparse_list]))
    # build candidate documents list
    candidates = []
    id_to_meta = {m["id"]: m for m in metas}
    for cid in candidate_ids:
        meta = id_to_meta.get(cid)
        if meta is None:
            # maybe came from qdrant payload but missing id mapping; skip
            continue
        candidates.append({
            "id": cid,
            "text": meta["text"],
            "source_file": meta["source_file"],
            "chunk_index": meta["chunk_index"],
            "dense_score": dense_map.get(cid, {}).get("score", 0.0),
            "sparse_score": 0.0
        })
    # attach sparse scores
    sparse_dict = {sid: sc for sid, sc in sparse_list}
    for c in candidates:
        c["sparse_score"] = sparse_dict.get(c["id"], 0.0)

    # optionally do a simple fusion score to pre-rank before rerank
    # normalize dense and sparse then combine
    dense_vals = [c["dense_score"] for c in candidates]
    sparse_vals = [c["sparse_score"] for c in candidates]
    def norm(vals):
        if len(vals) == 0: return []
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return [1.0 for _ in vals]
        return [(v - mn) / (mx - mn) for v in vals]
    dn = norm(dense_vals)
    sn = norm(sparse_vals)
    alpha = 0.6
    for i, c in enumerate(candidates):
        c["fused_score"] = alpha * dn[i] + (1 - alpha) * sn[i]
    # sort by fused_score descending and return top_k_initial (keep more for reranker)
    candidates_sorted = sorted(candidates, key=lambda x: x["fused_score"], reverse=True)[:top_k_initial]
    return candidates_sorted

# -------------------------
# Answering with LLM
# -------------------------
def answer_with_llm(question: str, top_docs: List[Dict]) -> str:
    # build context blocks
    blocks = []
    for d in top_docs:
        blocks.append(f"【來源】{d['source_file']} chunk:{d['chunk_index']}\n{d['text']}\n")
    system_prompt = "你是能利用檢索到的文件內容來回答問題的助手。若文件無法直接回答，請誠實說不確定並指出資訊缺口。回答務求精準且扼要。"
    user_prompt = f"問題：{question}\n\n使用下列檢索到的文件內容作為上下文來回答。若引用，請標注來源。\n\n" + "\n---\n".join(blocks)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }
    resp = requests.post(LLM_API, json=payload, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    text = None
    if "choices" in out and len(out["choices"])>0:
        ch = out["choices"][0]
        if "message" in ch and "content" in ch["message"]:
            text = ch["message"]["content"]
        elif "text" in ch:
            text = ch["text"]
    if text is None:
        text = "LLM 回傳格式不符合預期或無內容。請檢查 LLM endpoint 回傳。"
    return text.strip()

# -------------------------
# Main pipeline
# -------------------------
def main():
    # 1) load & chunk
    docs, metas = load_and_chunk(DAY6_PATH)
    if len(docs) == 0:
        print("No docs found in", DAY6_PATH)
        return

    # 2) embed all docs
    embeddings = embed_texts(docs, batch_size=BATCH_EMBED)

    # auto adjust vector dim
    vect_dim = len(embeddings[0])
    print("Detected vector dim:", vect_dim)

    # 3) setup qdrant and upsert
    client = QdrantClient(url=QDRANT_URL)
    create_qdrant_collection(client, COLLECTION_NAME, vector_size=vect_dim)
    upsert_qdrant(client, COLLECTION_NAME, metas, embeddings, batch=256)

    # 4) build bm25
    bm25, tokenized_docs, metas_ref = build_bm25_index(docs, metas)

    # 5) load reranker
    reranker = LocalReranker(RERANKER_MODEL_PATH)

    # 6) read questions.csv
    if not os.path.exists(QUESTIONS_CSV):
        print("questions.csv not found in repo root. Exiting.")
        return
    qdf = pd.read_csv(QUESTIONS_CSV)
    if "題目" not in qdf.columns and "題目" not in qdf.columns:
        # try english headers
        if "題目" not in qdf.columns:
            print("CSV must contain '題目' column (the question). Found columns:", qdf.columns.tolist())
            return

    results = []
    debug = []
    for idx, row in tqdm(qdf.iterrows(), total=len(qdf), desc="Processing questions"):
        qid = row.get("題目_ID", idx)
        question = row.get("題目")
        # 7) hybrid retrieval (get initial candidates)
        candidates = hybrid_retrieval(client, COLLECTION_NAME, bm25, tokenized_docs, metas, question, top_k_initial=TOP_K_INITIAL)
        # 8) prepare candidate texts for reranker (we will pass the text bodies)
        candidate_texts = [c["text"] for c in candidates]
        if len(candidate_texts) == 0:
            answer = "找不到相關文件。"
            results.append({"題目_ID": qid, "題目": question, "answer": answer, "sources": []})
            continue

        # 9) rerank
        reranked = reranker.rerank(question, candidate_texts, instruction="Given a web search query, retrieve relevant passages that answer the query")
        # collect top_k_final
        top_reranked = reranked[:TOP_K_FINAL]
        top_docs = []
        for doc_text, score in top_reranked:
            # find meta by exact text (note: if duplicates exist, match first)
            meta = next((m for m in metas if m["text"] == doc_text), None)
            if meta is None:
                meta = {"source_file": "unknown", "chunk_index": -1}
            top_docs.append({
                "text": doc_text,
                "score": float(score),
                "source_file": meta.get("source_file"),
                "chunk_index": meta.get("chunk_index")
            })

        # 10) LLM answer
        answer = answer_with_llm(question, top_docs)
        sources = [{"source_file": d["source_file"], "chunk_index": d["chunk_index"], "score": d["score"]} for d in top_docs]
        results.append({"題目_ID": qid, "題目": question, "answer": answer, "sources": sources})

        # debug record
        debug.append({
            "題目_ID": qid,
            "question": question,
            "initial_candidates": candidates[:10],
            "top_reranked": top_docs,
            "answer": answer
        })
        # optionally small sleep to avoid rate limiting
        time.sleep(0.2)

    # save outputs
    out_df = pd.DataFrame([{"題目_ID": r["題目_ID"], "題目": r["題目"], "answer": r["answer"], "sources": json.dumps(r["sources"], ensure_ascii=False)} for r in results])
    out_df.to_csv(os.path.join(OUTPUT_DIR, "answers.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, "retrieval_debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    print("Saved answers ->", os.path.join(OUTPUT_DIR, "answers.csv"))
    print("Saved debug ->", os.path.join(OUTPUT_DIR, "retrieval_debug.json"))

if __name__ == "__main__":
    main()
