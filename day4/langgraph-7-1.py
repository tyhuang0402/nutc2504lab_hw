import random
import time
import json
import os
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

CACHE_FILE = "translation_cache.json"



def load_cache():
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_cache(original: str, translated: str):
    data = load_cache()
    data[original] = translated
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

class State(TypedDict):
    original_text: str
    translated_text: str
    critique: str
    attempts: int
    is_cache_hit: bool 



def check_cache_node(state: State):
    """檢查快取節點"""
    print("\n-- 檢查快取 (Check Cache) ---")
    data = load_cache()
    original = state["original_text"]

    if original in data:
        print("Cache hit, returning results.")
        return {
            "translated_text": data[original],
            "is_cache_hit": True
        }
    else:
        print("Cache did not hit, starting translating sequence...")
        return {"is_cache_hit": False}

def translator_node(state: State):
    """翻譯節點"""
    print(f"\n-- 翻譯 (第 {state['attempts'] + 1} 次) ---")
    prompt =  f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋： {state['original_text']}"

    if state["critique"]:
        prompt += f"\n\n上一輪的審查意見是： {state['critique']}。請根據意見修正翻譯。"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "translated_text": response.content,
        "attempts": state["attempts"] + 1
    }

def reflector_node(state: State):
    """審查節點"""
    print("--- Reflection ---")
    prompt = f"原文： {state['original_text']}\n翻譯： {state['translated_text']}\n請檢察翻譯是否正確。若完整正確則輸出 'PASS'。"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": response.content}

def cache_router(state: State) -> Literal["end", "translator"]:
    if state["is_cache_hit"]:
        return "end"
    return "translator"

def critique_router(state: State) -> Literal["translator", "end"]:
    if "PASS" in state['critique'].upper():
        print("--- Passed! ---")
        return "end"
    elif state['attempts'] >= 3:
        print("--- Maximum tries reached. ---")
        return "end"
    else:
        print(f"--- Return and rewrite: {state['critique']}")
        return "translator"

workflow = StateGraph(State)

# 加入節點
workflow.add_node("check_cache", check_cache_node) # 新節點
workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

# 一律先走 check_cache
workflow.set_entry_point("check_cache")

# 設定快取後的路徑 (Cache Hit -> END; Cache Miss -> Translator)
workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "end": END,
        "translator": "translator"
    }
)

# 正常的翻譯迴圈路徑
workflow.add_edge("translator", "reflector")
workflow.add_conditional_edges(
    "reflector",
    critique_router,
    {
        "translator": "translator",
        "end": END
    }
)

app = workflow.compile()
print(app.get_graph().draw_ascii())



if __name__ == "__main__":
    print(f"快取檔案：{CACHE_FILE}")

    while True:
        user_input = input("\n請輸入要翻譯的中文 (exit/q 離開)：")
        if user_input.lower() in ["exit", "q"]: break

        inputs = {
            "original_text": user_input,
            "attempts": 0,
            "critique": "",
            "is_cache_hit": False, 
            "translated_text": ""
        }

        # 執行 Graph
        result = app.invoke(inputs)

        # 如果不是從快取來的 (代表是新算出來的)，就寫入快取
        if not result["is_cache_hit"]:
            save_cache(result["original_text"], result["translated_text"])

        print(f"===============FINAL RESULT===================")
        print(f"原文：{result['original_text']}")
        print(f"翻譯：{result['translated_text']}")
        print(f"來源：{'快取 (Cache)' if result['is_cache_hit'] else '生成 (LLM)'}")
