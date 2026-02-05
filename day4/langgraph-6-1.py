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


@tool
def get_weather(city: str):
    """查詢指定城市的天氣。輸入參數 city 必須是城市名稱。"""
    if random.random() < 0.5:
        return "SYSTEM ERROR: 天氣資料庫連結失敗。請再試一次。 (PLEASE RETRY AGAIN)"
    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    if "台中" in city:
        return "台中晴天，氣溫 23 度"
    if "高雄" in city:
        return "高雄多雲，氣溫 30 度"

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot_node(state: AgentState):
    """思考節點"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node_executor = ToolNode(tools)

def fallback_node(state: AgentState):
    """
    備援節點：重試過多次後傳回一個空ToolMessage。
    """
    last_message = state["messages"][-1]
    tool_call_id = last_message.tool_calls[0]["id"]

    error_message = ToolMessage(
        content="系統警告: 已達最大重試次數 (Max retries reached)。請停止蟲是。並告知使用者服務暫時無法使用。",
        tool_call_id = tool_call_id
    )
    return {"messages": [error_message]}

def router(state: AgentState) -> Literal["tools", "fallback", "end"]:
    """
    路由邏輯：
    1. 檢查是否有tool_calls
    2. 檢查是否已經連續失敗太多次
    """

    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"

    retry_count = 0
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            if "SYSTEM ERROR" in msg.content:
                retry_count += 1
            else:
                break
        elif isinstance(msg, HumanMessage):
            break

    print(f"[debug] current retry count: {retry_count}")

    if retry_count >= 3:
        return "fallback"

    return "tools"

workflow = StateGraph(AgentState)

workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "fallback" : "fallback",
        "end": END
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("fallback", "agent")

app = workflow.compile() 
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    while True:
        user_input = input("\nuser: ")
        if user_input.lower() in ['exit', 'q']: break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if msg.tool_calls:
                        print(f" -> [Agent]: Calling tools (retrying.....)")
                    else:
                        print(f" -> [Agent]: {msg.content}")
                elif key == "tools":
                    print(f" -> [Tools] SYSTEM ERROR (failure)")
                elif key == 'fallback':
                    print(f" -> [Fallback]: Triggering fallback: retry stopped.")

"""
-- init --
                   +-----------+
                   | __start__ |
                   +-----------+
                          *
                          *
                          *
                     +-------+
                     | agent |.
                  ...+-------+ ...
              ....        .       ....
           ...            .           ...
         ..               .              ..
+---------+         +----------+         +-------+       
| __end__ |         | fallback |         | tools |       
+---------+         +----------+         +-------+       

user: 你好
 -> [Agent]: 你好！請問有什麼我可以幫忙的嗎？

user: 台北天氣
[debug] current retry count: 0
 -> [Agent]: Calling tools (retrying.....)
 -> [Tools] SYSTEM ERROR (failure)
 -> [Agent]: 台北下大雨，氣溫 18 度。

user: 我想要查台北天氣 
[debug] current retry count: 0
 -> [Agent]: Calling tools (retrying.....)
 -> [Tools] SYSTEM ERROR (failure)
[debug] current retry count: 1
 -> [Agent]: Calling tools (retrying.....)
 -> [Tools] SYSTEM ERROR (failure)
[debug] current retry count: 2
 -> [Agent]: Calling tools (retrying.....)
 -> [Tools] SYSTEM ERROR (failure)
 -> [Agent]: 台北下大雨，氣溫 18 度。
"""