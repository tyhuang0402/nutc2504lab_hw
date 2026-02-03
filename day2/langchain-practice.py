import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
import json

print('-- init --')
llm = ChatOpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    model="Qwen3-VL-8B-Instruct-BF16.gguf",
    api_key="",
    temperature=0,
    max_tokens=10
)
print('-- langchain --')
prompt = ChatPromptTemplate.from_messages([
    ("system", "請依照使用者的輸入給予回應。"),
    ("human", "請依照{ident}的身分，對「天氣很熱」寫出一篇文章。")
])

parser = StrOutputParser()

input_data = [{"ident": "IG小編"},
              {"ident": "Linkedin主管"}]

chain = prompt | llm | parser

print('-- result --')


T = time.time()
for chunk in chain.stream(input_data):
    print(chunk, end="", flush=True)
E = time.time()
print('=== run time:', E - T)


T = time.time()
result = chain.batch(input_data)
E = time.time()

for i, res in enumerate(result):
    print(f"{res}")

print('=== run time:', E - T)