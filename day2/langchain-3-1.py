from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputKeyToolsParser
from langchain_core.output_parsers import StrOutputParser
import json
print('-- init --')
llm = ChatOpenAI(
    base_url = "https://ws-05.huannago.com/v1",
    model="Qwen3-VL-8B-Instruct-BF16.gguf",
    api_key="1311432044"
)
print('-- laingchain --')
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個專業的文章編輯。請將使用者提供的文章內容，歸納出 3 個重點，並以繁體中文條列式輸出。"),
    ("human", "{article_content}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

tech_article =  """
以《Liebe~回憶的禮物~》出道，擔綱不少galgame原畫師與人物設定的資深繪師池上茜，近日在X卻意外掀起不小風波。
負責《我的眼睛能看光光！不可知的未來與透視的命運》（俺の瞳で丸裸！不可知な未來と視透かす運命）角色設計與原畫的池上茜，公開宣布作品「確定動畫化」，沒想到不到數小時，版權方居然火速發布聲明否認。讓事件瞬間演變成一場由原畫師帶起的「動畫化假消息」，池上茜本人也在後續發文道歉坦言「都是自己自導自演的鬧劇」。
事件的起因是1 月 31 日，池上茜在個人 X 發文，自己擔任角色設計與原畫的《我的眼睛能看光光！不可知的未來與透視的命運》「終於決定動畫化」，並同步公開全新繪製的插畫，甚至附上看似官方的動畫 X 帳號，引發粉絲一片驚喜。
隨後她更接連表示，該插畫是「製作委員會委託繪製」，未來也會全力替動畫應援，並呼籲粉絲在後續消息公開前，暫時不要向相關單位詢問。
然而，當晚作品版權方 HULOTTE便發布「重要公告」，明確指出社群平台上流傳的動畫化消息屬於不實資訊，否認有任何動畫化計畫，正式為這波傳言踩下煞車。
"""

print('-- starting --')
result = chain.invoke({"article_content": tech_article})
print(result)