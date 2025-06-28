# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from customer_service_agent import app as agent_app

app = FastAPI(title="Customer Service AI Agent")

# Input data model
class InputText(BaseModel):
    text: str

# API endpoint
@app.post("/analyze/")
async def analyze_message(input_data: InputText):
    input_state = {
        "text": input_data.text,
        "intent": "",
        "entities": [],
        "summary": "",
        "reply": ""
    }

    result = agent_app.invoke(input_state)

    return {
        "intent": result["intent"],
        "entities": result["entities"],
        "summary": result["summary"],
        "reply": result["reply"]
    }
