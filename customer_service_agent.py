import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Define Agent Memory
class State(TypedDict):
    text: str
    intent: str
    entities: List[str]
    summary: str
    reply: str

# 2. Intent Classification
def classify_intent(state: State):
    prompt = PromptTemplate.from_template("""
    Classify the customer's message into one of these intents:
    - Complaint
    - Inquiry
    - Feedback
    - Booking
    - Other

    Message: {text}

    Intent:
    """)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    result = llm.invoke([message]).content.strip()
    return {"intent": result}

# 3. Entity Extraction
def extract_entities(state: State):
    prompt = PromptTemplate.from_template("""
    Extract key details from the message such as:
    - Service types (e.g., screen repair, battery replacement)
    - Dates or times
    - Device issues
    - Any named products or locations

    Respond as a comma-separated list.

    Message: {text}

    Entities:
    """)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    result = llm.invoke([message]).content.strip().split(", ")
    return {"entities": result}

# 4. Summarization
def summarize(state: State):
    prompt = PromptTemplate.from_template("""
    Summarize the customer's message in one short sentence.

    Message: {text}

    Summary:
    """)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    result = llm.invoke([message]).content.strip()
    return {"summary": result}

# 5. Reply Suggestion
def generate_reply(state: State):
    prompt = PromptTemplate.from_template("""
    You're a support agent for The Mobile Techs.

    Based on the customer message below, generate a polite and helpful reply.
    Use a friendly and professional tone.

    Message: {text}

    Reply:
    """)
    message = HumanMessage(content=prompt.format(text=state["text"]))
    result = llm.invoke([message]).content.strip()
    return {"reply": result}

# 6. Build Workflow
workflow = StateGraph(State)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("extract_entities", extract_entities)
workflow.add_node("summarize", summarize)
workflow.add_node("generate_reply", generate_reply)

workflow.set_entry_point("classify_intent")
workflow.add_edge("classify_intent", "extract_entities")
workflow.add_edge("extract_entities", "summarize")
workflow.add_edge("summarize", "generate_reply")
workflow.add_edge("generate_reply", END)

app = workflow.compile()

# 7. Run Example
if __name__ == "__main__":
    input_text = """
    Hi, I booked a screen replacement service last Tuesday but my phone is still not fixed.
    I haven’t heard back yet. Please check and let me know what’s going on.
    """

    input_state = {
        "text": input_text,
        "intent": "",
        "entities": [],
        "summary": "",
        "reply": ""
    }

    result = app.invoke(input_state)

    print("\n🔎 Intent:", result["intent"])
    print("📌 Entities:", result["entities"])
    print("📝 Summary:", result["summary"])
    print("💬 Suggested Reply:\n", result["reply"])
