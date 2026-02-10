import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct



collection_name = "cw04"

client = QdrantClient(host="localhost", port=6333)

if not client.collection_exists(collection_name=collection_name):
    # If not, create the collection with specified parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=100, distance=Distance.COSINE), # Adjust size and distance as needed
    )
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")




# 載入模型
model_path = os.path.expanduser("models/Qwen3-Reranker-0.6B")

reranker_tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    padding_side='left'
)

reranker_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
).eval()

# 配置參數
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_reranker_length = 8192

# 設定前綴與後綴
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

# 格式化指令函數
def format_instruction(instruction, query, doc):
    """格式化 reranker 輸入"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    return output

# 處理輸入函數
def process_inputs(pairs):
    """處理輸入以供 reranker 使用"""
    inputs = reranker_tokenizer(
        pairs, 
        padding=False, 
        truncation='longest_first',
        return_attention_mask=False, 
        max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)
    )
    
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    
    inputs = reranker_tokenizer.pad(
        inputs, 
        padding=True, 
        return_tensors="pt", 
        max_length=max_reranker_length
    )
    
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    
    return inputs

# 計算相關度分數
@torch.no_grad()
def compute_logits(inputs):
    """計算相關度分數"""
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# Rerank 文件函數
def rerank_documents(query, documents, task_instruction=None):
    """
    根據查詢相關性重新排序文件
    
    Args:
        query: 搜尋查詢
        documents: 要重新排序的文件列表
        task_instruction: 任務指令
    
    Returns:
        按相關度分數排序的 (文件, 分數) 元組列表
    """
    if task_instruction is None:
        task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    
    # 格式化輸入
    pairs = [format_instruction(task_instruction, query, doc) for doc in documents]
    
    # 處理輸入並計算分數
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    
    # 合併文件與分數並按分數排序（降序）
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores

# ===== 測試範例 =====
if __name__ == "__main__":
    # 定義查詢和文件
    query = "What is the capital of China?"
    
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Paris is the capital city of France.",
        "Beijing is a major city in China with a rich history.",
    ]
    
    print(f"查詢: {query}\n")
    print("=" * 60)
    
    # 執行 rerank
    reranked_results = rerank_documents(query, documents)
    
    # 顯示結果
    print("\nRerank 結果：\n")
    for i, (doc, score) in enumerate(reranked_results, 1):
        print(f"{i}. 分數: {score:.4f}")
        print(f"   文件: {doc}\n")

