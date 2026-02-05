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


class State(TypedDict):
    original_text : str
    translated_text: str
    critique: str
    attempts: int


def translator_node(state: State):
    """負責翻譯的節點"""
    print(f"\n--- TRANSLATING (Attempts: {state['attempts'] + 1})")

    prompt = f"你是一名翻譯員，請將以下中文翻譯成英文，不需任何解釋： '{state['original_text']}"

    if state['critique']:
        prompt += f"\n\n上一輪的審查意見為： {state['critique']}。請根據意見修正翻譯。"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "translated_text": response.content,
        "attempts": state["attempts"] + 1
    }


def reflector_node(state: State):
    """Critique node"""
    print("--- Reflecting ---")
    print(f"Translation: {state['translated_text']}")
    prompt = f"""
    你是一個嚴格的翻譯審查員。
    原文： {state['original_text']}
    翻譯： {state['translated_text']}

    請檢察翻譯是否準確且通順。
    - 如果翻譯很完美，請只回覆 "PASS"
    - 如果需要修改，請給出簡短的具體建議。
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"critique": response.content}


def should_continue(state: State) -> Literal["translator", "end"]:
    critique = state['critique'].strip().upper()

    if "PASS" in critique:
        print("--- PASSED ---")
        return "end"
    elif state['attempts'] >= 3:
        print("--- MAX TRIES REACHED, ENDING ---")
        return "end"
    else:
        print(f"--- NOT PASSED: {state['critique']} ---")
        print("--- REJECTING... ---")
        return "translator"



workflow = StateGraph(State)

workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("translator")

workflow.add_edge("translator", "reflector")

workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "translator": "translator",
        "end": END
    }
)

app = workflow.compile() 
print(app.get_graph().draw_ascii())



if __name__ == "__main__":
    while True:
        user_input = input("\nuser: ")
        if user_input.lower() in ['exit', 'q']: break

        inputs = {
            "original_text": user_input,
            "attempts": 0,
            "critique": ""
        }

        result = app.invoke(inputs)
        print("\n===== FINAL RESULT =====")
        print(f"Original: {result['original_text']}")
        print(f"Final translationL: {result['translated_text']}")
        print(f"Final attempts: {result['attempts']}")

"""
-- init --
+-----------+  
| __start__ |
+-----------+
       *
       *
       *
+------------+
| translator |
+------------+
       .
       .
       .
+-----------+
| reflector |
+-----------+
       .
       .
       .
  +---------+
  | __end__ |
  +---------+

user: 認為義大利麵就應該拌42號混凝土

--- TRANSLATING (Attempts: 1)
--- Reflecting ---
Translation: Believes spaghetti should be mixed with 42-grade concrete.
<|im_end|>
--- NOT PASSED: The translation is inaccurate and nonsensical. "42號混凝土" doesn't translate to "42-grade concrete." It's a humorous reference to a specific type of concrete, but the translation loses the cultural context and sounds bizarre.

Suggestion: "Believes spaghetti should be mixed with type 42 concrete." or, better yet, a more contextual translation like "Believes spaghetti should be served with concrete (specifically, type 42)."
<|im_end|> ---
--- REJECTING... ---

--- TRANSLATING (Attempts: 2)
--- Reflecting ---
Translation: Believes spaghetti should be served with concrete (specifically, type 42).
<|im_end|>
--- NOT PASSED: Needs improvement.

Suggestion: "Believes spaghetti should be *mixed with* concrete (specifically, type 42)."  "Served with" implies it's a side dish, while "mixed with" conveys the absurdity of incorporating concrete *into* the pasta.
<|im_end|> ---
--- REJECTING... ---

--- TRANSLATING (Attempts: 3)
--- Reflecting ---
Translation: Believes spaghetti should be *mixed with* concrete (specifically, type 42).
<|im_end|>
--- MAX TRIES REACHED, ENDING ---

===== FINAL RESULT =====
Original: 認為義大利麵就應該拌42號混凝土
Final translationL: Believes spaghetti should be *mixed with* concrete (specifically, type 42).
<|im_end|>
Final attempts: 3


user: 你好世界我愛你

--- TRANSLATING (Attempts: 1)
--- Reflecting ---
Translation: Hello world, I love you.
<|im_end|>
--- PASSED ---

===== FINAL RESULT =====
Original: 你好世界我愛你
Final translationL: Hello world, I love you.
<|im_end|>
Final attempts: 1

"""