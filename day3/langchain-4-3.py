from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
import json

print('-- init --')

llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    model="google/gemma-3-27b-it",
    api_key="1311432044"
)

print('-- langchain --')

@tool
def generate_tech_summary(article_content: str):
    """
    科技文章專用摘要生成工具。
    【判斷邏輯】：
    1. 只有當輸入內容屬於「科技」、「程式設計」、「AI」、「軟體工程」或「IT 技術」領域時，才使用此工具。
    2. 如果內容是「閒聊」、「食譜」、「天氣」、「日常日記」等非技術內容，請勿使用此工具。

    功能：將輸入的技術文章歸納出 3 個重點。
    """
    prompt = ChatPromptTemplate.from_messages({
        ("system", "你是一個資深的科技主編。請將輸入的技術文章內容，精簡地歸納出 3 個關鍵重點 (Key Takeaways)。請用繁體中文輸出。"),
        ("user", "{text}")
    })

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": article_content})
    return result

llm_with_tools = llm.bind_tools([generate_tech_summary])

router_prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

while True:
    user_input = input("user: ")
    if user_input.lower() in ['exit', 'q']:
        print("goodbye.")
        break
    
    chain = router_prompt | llm_with_tools
    ai_msg = chain.invoke({"input": user_input})
    if ai_msg.tool_calls:
        print(f"[DECISION] This is tech article")
        tools_args = ai_msg.tool_calls[0]['args']
        final_result = generate_tech_summary.invoke(tools_args)
        print(f"[RESULT]:\n{final_result}")

    else:
        print(f"[DECISION] This is chatting")
        print(f"[AGENT RSP]: {ai_msg.content}")

tech_article =  """
以《Liebe~回憶的禮物~》出道，擔綱不少galgame原畫師與人物設定的資深繪師池上茜，近日在X卻意外掀起不小風波。

負責《我的眼睛能看光光！不可知的未來與透視的命運》（俺の瞳で丸裸！不可知な未來と視透かす運命）角色設計與原畫的池上茜，公開宣布作品「確定動畫化」，沒想到不到數小時，版權方居然火速發布聲明否認。讓事件瞬間演變成一場由原畫師帶起的「動畫化假消息」，池上茜本人也在後續發文道歉坦言「都是自己自導自演的鬧劇」。

《我的眼睛能看光光！不可知的未來與透視的命運》是由HULOTTE製作的一款GALGAME，於2023年7月28日發售。

以主角久瀨森羅現在正作為「森之魔女」的奴隸生活著，但是完全沒有被教魔法，取而代之的是每天都被當作傭人使喚著。他能做到的事就是，用他唯一學到的「透視魔法」觀察身邊女孩子們的裸體！對著青梅竹馬使用透視，對著那日漸成長的身體目光發直成為了森羅的日常。

事件的起因是1 月 31 日，池上茜在個人 X 發文，自己擔任角色設計與原畫的《我的眼睛能看光光！不可知的未來與透視的命運》「終於決定動畫化」，並同步公開全新繪製的插畫，甚至附上看似官方的動畫 X 帳號，引發粉絲一片驚喜。


隨後她更接連表示，該插畫是「製作委員會委託繪製」，未來也會全力替動畫應援，並呼籲粉絲在後續消息公開前，暫時不要向相關單位詢問。

然而，當晚作品版權方 HULOTTE便發布「重要公告」，明確指出社群平台上流傳的動畫化消息屬於不實資訊，否認有任何動畫化計畫，正式為這波傳言踩下煞車。

官方否認後，網路輿論迅速炸鍋。有網友開始質疑這是否只是「同人動畫計畫」遭誤會，也有人擔心原畫師是否遭到詐騙，認為整起事件「包裝得太完整，反而很詭異」。

直到今日（2/2），池上茜本人發布道歉聲明，坦承此次動畫化公告並非正式決定，而是基於自己「想要讓作品動畫化」的個人心願而做出的行動，所謂的「動畫官方帳號」也是他自己建立的，曾發布「受到製作委員會委託」等內容，但該組織並不存在，該表述與事實不符，並為造成混亂向各界致歉。


隨後，有網友整理出指出，池上茜雖然是遊戲角色設計與原畫，但作品著作權屬於 HULOTTE，並非由原畫師持有；此次動畫化消息，並未經版權方同意。也有不少網友對她「自行認定有動畫化企劃」的行為感到難以理解，直言情況「相當不妙」。
"""

print('-- starting --')
result = chain.invoke({"article_content": tech_article})
print(result)