from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
import json

print('-- init --')
llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    model="google/gemma-3-27b-it",
    api_key="1311432044"
)
print('-- langchain --')
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個專業的文章編輯。請將使用者提供的文章內容，歸納出 3 個重點，並以繁體中文條列式輸出。"),
    ("human", "{article_content}")
])

parser = JsonOutputParser()

system_prompt = """你是一個資料提取助手。
{format_instructions}
需要的欄位: name, phone, product, quantity, address"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{text}")
])

chain = prompt | llm | parser

user_input = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

try:
    result = chain.invoke({
        "text": user_input,
        "format_instructions": parser.get_format_instructions()
    })

    print(json.dumps(result, ensure_ascii=False, indent=2))

except Exception as e:
    print(f"ERROR: {e}")