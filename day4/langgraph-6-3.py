import random
import json
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage 
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

print('-- init --')
llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    model="google/gemma-3-27b-it",
    api_key="1311432044",
    temperature=0
)


VIP_LIST = ["AI哥", "一龍哥"]

@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料擷取專用工具。
    專門用來從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """

    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }


llm_with_tools = llm.bind_tools([extract_order_data])
tool_node = ToolNode([(extract_order_data)])

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: AgentState):
    """思考節點"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
    
def human_review_node(state: AgentState):
    """
    人工審核節點
    當檢測到 VIP 時，流程會在這裡等待人工輸入。
    """
    print("\n" + "="*30)
    print("獨屬人工審核機制：檢測到 VIP 客戶！")
    print("="*30)

    last_msg = state["messages"][-1]
    print(f"【待審核資料】：{last_msg.content}")

    review = input(">>> 管理員請批示 (輸入 'ok' 通過，其他則拒絕)：")

    if review.lower() == "ok":
        return {
            "messages": [
            AIMessage(content="已收到訂單資料，因檢測到 VIP 客戶，系統將轉交人工審核。"),
            HumanMessage(content="[system] 管理員已人工審核通過此 vip 訂單，請確認完成後動作。")
            ]
        }
    else:
        return {
            "messages": [
            AIMessage(content="已收到訂單資料，等待人工審核結果..."),
            HumanMessage(content="[system] 管理員拒絕了此訂單，請取消交易並告知用戶。")
            ]
        }

def entry_router(state: AgentState):
    """判斷 Agent 是否要呼叫工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def post_tool_router(state: AgentState) -> Literal["human_review", "agent"]:
    """
    Tool router
    """

    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        try:
            data = json.loads(last_message.content)
            user_name = data.get("name", "")

            if user_name in VIP_LIST:
                print(f"[debug] VIP found -> human review required")
                return "human_review"

        except Exception as e:
            print(f"JSON ANALYST FAILED: {e}")

    return "agent"


wf = StateGraph(AgentState)

wf.add_node("agent", agent_node)
wf.add_node("tools", tool_node)
wf.add_node("human_review", human_review_node)

wf.set_entry_point("agent")

wf.add_conditional_edges(
    "agent",
    entry_router,
    {"tools": "tools", END: END}
)

wf.add_conditional_edges(
    "tools",
    post_tool_router,
    {
        "human_review": "human_review",
        "agent": "agent"
    }
)

wf.add_edge("human_review", "agent")

app = wf.compile()
print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    print(f"VIP list: {VIP_LIST}")
    while True:
            user_input = input("\nuser: ")
            if user_input.lower() in ['exit', 'q']: break

            for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
                for key, value in event.items():
                    if key == "agent":
                        msg = value["messages"][-1]
                        if not msg.tool_calls:
                            print(f" -> [Agent]: {msg.content}")
                        elif key == "human_review":
                            print(f" - > [Human]: Review completed.")


"""
-- init --
                  +-----------+
                  | __start__ |
                  +-----------+
                        *
                        *
                        *
                    +-------+
                    | agent |*
                 ...+-------+ ***
              ...       .        ****        
           ...          .            ***     
         ..             .               ***  
+---------+         +-------+              **
| __end__ |         | tools |             ** 
+---------+         +-------+            *   
                             .         **    
                              ..     **      
                                .   *        
                          +--------------+   
                          | human_review |   
                          +--------------+   
VIP list: ['AI哥', '一龍哥']

user: 你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。
 -> [Agent]: 好的，陳大明先生，您訂購了 3 台筆記型電腦， 
將於下週五送到台中市北區。請確認您的電話號碼是 0912-345-678。有其他需要嗎？

user: 你好，我是AI哥，電話是 0988-999-999，我想要訂購 3 
台RTX 5090，下週五送到台中市北區。
[debug] VIP found -> human review required

==============================
獨屬人工審核機制：檢測到 VIP 客戶！
==============================
【待審核資料】：{"name": "AI哥", "phone": "0988-999-999", "product": "RTX 5090", "quantity": 3, "address": "台中市
北區"}
>>> 管理員請批示 (輸入 'ok' 通過，其他則拒絕)：ok
 -> [Agent]: 訂單已確認，將於下週五送至台中市北區。感謝您
的訂購！

user: 你好，我是一龍哥，電話是 +1-9-845-961，我想要訂購 1 台Falcon 9火箭，下週五送到近地軌道。
[debug] VIP found -> human review required

==============================
獨屬人工審核機制：檢測到 VIP 客戶！
==============================
【待審核資料】：{"name": "一龍哥", "phone": "+1-9-845-961", "product": "Falcon 9火箭", "quantity": 1, "address": "近地軌道"}
>>> 管理員請批示 (輸入 'ok' 通過，其他則拒絕)：reject
 -> [Agent]: 您好一龍哥，非常抱歉，您的訂單已被管理員拒絕
。我們已取消交易，請您留意相關退款資訊。如有任何疑問，請 
隨時與我們聯繫。
"""