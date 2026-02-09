import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter as smTextSplitter

# import tiktoken



API_URL = "https://ws-04.wade0426.me/embed"
collection_name = "day5_hw_collection"



client = QdrantClient(host="localhost", port=6333)


client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
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
    print(" USING FIXED TEXT SPLIT")
    text_splitter = CharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap = 0,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    tsplit_results = []

    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        tsplit_results.append(chunk.split())
        # print(f"===Chunk {i} ===")
        # print(f"Len: {len(chunk)}")
        # print(f"Content: {chunk.split()}")
        print()

    print(f"Final results: {tsplit_results}")
    return tsplit_results



def ts_recur(text):
    """滑動分割"""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 80,
        chunk_overlap = 10,
        model_name = "gpt-4",
        separators=[""]
    )

    # encoding = tiktoken.encoding_for_model("gpt-4")

    chunks = text_splitter.split_text(text)


    print(f"Original content len {len(text)}")
    print(f"Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        # token_count = len(encoding.encode(chunk))
        # print(f"===Chunk {i} ===")
        # print(f"Len: {token_count}")
        # print(f"Content: {chunk.split()}")
        print()





def get_embeddings(text_list):
    file_embeddings = []
    for i in text_list:
        response = requests.post(
            API_URL,
            json={
                "texts": i, 
                "task_description": "Call embedding",
                "normalize": True
            }
        )
        embeddings = response.json()["embeddings"]
        file_embeddings.append(embeddings)

    embeddings = response.json()["embeddings"]
    print(f"維度: {len(embeddings[0])}")

    return file_embeddings




file1 = read_file_to_var("./day5/data_01.txt")
file1_ts = ts_character(file1)

file1_embeddings = get_embeddings(file1_ts)

print(file1_embeddings)

