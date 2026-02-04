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

llm_with_tools = llm.bind_tools([extract_order_data])


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode([extract_order_data])

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
          +-------+*
          .         *
        ..           **
       .               *
+---------+         +-------+
| __end__ |         | tools |
+---------+         +-------+
user: hi
--- Node: agent --
Hi there! How can I help you today? Do you have an order you'd like me to process, or information you'd like me to extract?
user: 你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市
北區。
--- Node: agent --
[{'name': 'extract_order_data', 'args': {'name': '陳大明', 'phone': '0912-345-678', 'product': '筆記型電腦', 'quantity': 3, 'address': '台中市北區'}, 'id': 'pYya4RK1K8LUrsmG2N8IdFOb0bwGe2ig', 'type': 'tool_call'}]
--- Node: tools --
{"name": "陳大明", "phone": "0912-345-678", "product": "筆記型電腦", "quantity": 3, "address": "台中市北區"}
--- Node: agent --
好的，陳大明先生，您訂購了3台筆記型電腦，將於下週五送到台中市北區，您的電話是0912-345-678。請確認是否正確？
user:
"""