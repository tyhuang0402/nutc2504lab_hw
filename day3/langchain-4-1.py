
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import json
"""
Docstring:
資料提取專用工具。
專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。

system prompt:
你是一個精準的訂單管理員，請從對話中提取訂單資訊。

user_prompt:
你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。
"""

print('-- init --')

@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """

    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }

llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    model="google/gemma-3-27b-it",
    api_key="1311432044"
)
llm_with_tools = llm.bind_tools([extract_order_data])

print('-- langchain --')

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("human", "{user_input}")
])

def extract_tools_args(ai_messages):
    if ai_messages.tool_calls:
        return ai_messages.tool_calls[0]['args']
    return None

chain = prompt | llm_with_tools | extract_tools_args

user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

result = chain.invoke({"user_input": user_text})

if result:
    print("Success: ")
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("FAILED.")

"""
-- init --
-- langchain --
Success: 
{
  "name": "陳大明",
  "phone": "0912-345-678",
  "product": "筆記型電腦",
  "quantity": 3,
  "address": "台中市北區"
}
"""