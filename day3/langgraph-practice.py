import json
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

from HW_asr import transcribe_audio

print('-- init --')
llm = ChatOpenAI(
    base_url = "https://ws-02.wade0426.me/v1",
    model="google/gemma-3-27b-it",
    api_key="1311432044",
    temperature=0
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# @tool
def call_asr_from_script(state: AgentState):
    """將會議內容音檔轉成逐字稿"""
    messages = state['messages']
    transcribed = transcribe_audio()
    print(type(transcribed))
    return {"messages": [transcribed]} 

def call_model(state: AgentState):
    """將接收到的會議內容逐字稿寫出一篇重點摘要整理"""
    messages = state['messages']
    # response = llm_with_tools.invoke(messages)
    response = llm.invoke(messages)
    return {"messages": [response]}

def call_minute(state: AgentState):
    """將接收到的會議內容逐字稿按時間軸與對應台詞逐一列出"""
    messages = state['messages']
    # response = llm_with_tools.invoke(messages)
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def writerf():
    pass

toolasr = ToolNode([call_asr_from_script])

workflow = StateGraph(AgentState)
workflow.add_node("asr", call_asr_from_script)
workflow.add_node("summarizer", call_model)
workflow.add_node("minutes_taker", call_minute)
workflow.add_node("writer", writerf)

workflow.add_edge("asr", "summarizer")
workflow.add_edge("asr", "minutes_taker")


workflow.add_edge("summarizer", "writer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("writer", END)
workflow.set_entry_point("asr")

app = workflow.compile()
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    user_input = "請將語音或記錄整理成逐字稿。"
    for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
        for key, value in event.items():
            print(f"--- Node: {key} --")
            print(value["messages"][-1].content or value["messages"][-1].tool_calls)