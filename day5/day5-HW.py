import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter as smTextSplitter

# import tiktoken



API_URL = "https://ws-04.wade0426.me/embed"
collection_name = "day5_hw_collection"
GID = 1


client = QdrantClient(host="localhost", port=6333)


client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

# === FOR NOT RECREATING COLLECTION EACH TIME ===
#
# if not client.collection_exists(collection_name=collection_name):
#     # If not, create the collection with specified parameters
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=100, distance=Distance.COSINE), # Adjust size and distance as needed
#     )
#     print(f"Collection '{collection_name}' created.")
# else:
#     print(f"Collection '{collection_name}' already exists. Skipping creation.")



def read_file_to_var(filename):
    """Read from file."""
    with open(filename, 'r', encoding="utf-8") as file:
        file_content_all = file.read()

    return file_content_all



def ts_character(text):
    """固定分割"""
    print("[DEBUG] USING FIXED TEXT SPLIT")
    text_splitter = CharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 0,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    tsplit_results = []

    # print(f"Chunks: {len(chunks)}")
    # for i, chunk in enumerate(chunks, 1):
    #     tsplit_results.append(chunk.split())
    #     # print(f"===Chunk {i} ===")
    #     # print(f"Len: {len(chunk)}")
    #     # print(f"Content: {chunk.split()}")
    #     # print()

    # print(f"Final results: {tsplit_results}")
    return tsplit_results



def ts_semantic(text):
    """語句切塊"""
    print("[DEBUG] USING SEMANTIC TEXT SPLIT")
    max_characters = (200, 1000)
    splitter = smTextSplitter(max_characters)

    chunks = splitter.chunks(text)
    
    return chunks




def ts_recur(text):
    """滑動分割"""
    print("[DEBUG] USING RECURSIVE TEXT SPLIT")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 80,
        chunk_overlap = 10,
        model_name = "gpt-4",
        separators=[""]
    )

    # encoding = tiktoken.encoding_for_model("gpt-4")

    chunks = text_splitter.split_text(text)


    # print(f"Original content len {len(text)}")
    # print(f"Chunks: {len(chunks)}")

    # for i, chunk in enumerate(chunks, 1):
    #     # token_count = len(encoding.encode(chunk))
    #     # print(f"===Chunk {i} ===")
    #     # print(f"Len: {token_count}")
    #     # print(f"Content: {chunk.split()}")
    #     print()




def get_embeddings(text_list):
    print(f"[DEBUG] CALLING EMBEDDING APIs")
    file_embeddings = []
    for i in text_list:
        response = requests.post(
            API_URL,
            json={
                "texts": list(i), 
                "task_description": "Call embedding",
                "normalize": True
            }
        )
        embeddings = response.json()["embeddings"]
        print(f"[Embedding API] 維度: {len(embeddings[0])}")
        file_embeddings.append(embeddings)

    return file_embeddings



def save_embedding_to_db(embedding, original_text, method):
    """將embedding與原始文字儲存到向量資料庫內"""
    global GID
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=GID,
                vector=embedding[0],
                payload={"original_text": original_text, "method": method}
            )
        ]
    )
    print(f"[DEBUG] Saved embedding to GID: {GID}")
    GID += 1
    return 'PASS'




files_to_process = ["./day5/data_01.txt","./day5/data_02.txt","./day5/data_03.txt","./day5/data_04.txt","./day5/data_05.txt",]

if __name__ == "__main__":
    for i in files_to_process:
        print(f"===========PROCESSING FILE {i}=============")
        print(f"=== Current GID: {GID} ===")
        file_raw_content = read_file_to_var(i)

        print(f"[DEBUG] {i} Content: \n{file_raw_content}")

        file_ts = ts_character(file_raw_content)

        file_embeddings = get_embeddings(file_ts)

        print(f"=== START OF EMBEDDINGS FOR {i}")
        print(file_embeddings)
        print(f"=== END OF EMBEDDINGS FOR {i}")

        try:
            print(f"===== SAVING EMBEDDINGS TO DB =====")
            for i in range(0, len(file_embeddings)):
                save_embedding_to_db(file_embeddings[i], file_ts[i], "Fixed Text Split")

        except Exception as e:
            print(f"[ERROR]: {e}")

        print(f"[END] GGs for {i}")