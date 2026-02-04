
import json
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

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
    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    if "台中" in city:
        return "台中晴天，氣溫 23 度"
    if "高雄" in city:
        return "高雄多雲，氣溫 30 度"

llm_with_tools = llm.bind_tools([get_weather])


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode([get_weather])

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

workflow.add_edge("tools", "agent")

app = workflow.compile()
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    while True:
        user_input = input("user: ")
        if user_input.lower() in ['exit', 'q']: break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                print(f"--- Node: {key} --")
                print(value["messages"][-1].content or value["messages"][-1].tool_calls)

"""
-- init --
        +-----------+
        | __start__ |
        +-----------+
               *
               *
               *
          +-------+
          | agent |
          +-------+.
          .         .
        ..           ..
       .               .
+---------+         +-------+
| __end__ |         | tools |
+---------+         +-------+
user: 你好
--- Node: agent --
你好！請問有什麼我可以幫忙的嗎？
user: 我要查詢台北的天氣
--- Node: agent --
[{'name': 'get_weather', 'args': {'city': '台北'}, 'id': 'pxsvKXhqlsyI3YjheIjQxlLscn5Einup', 'type': 'tool_call'}]
--- Node: tools --
台北下大雨，氣溫 18 度
--- Node: agent --
台北下大雨，氣溫 18 度。
user: 高雄天氣
--- Node: agent --
[{'name': 'get_weather', 'args': {'city': '高雄'}, 'id': 'DeurbDAkz8MbTNfWpwh39lytjky81cj4', 'type': 'tool_call'}]
--- Node: tools --
高雄多雲，氣溫 30 度
--- Node: agent --
高雄多雲，氣溫 30 度。
user:
"""