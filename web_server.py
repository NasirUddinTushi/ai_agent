from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import aiosqlite
import uuid
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from datetime import datetime
import csv

# Load .env
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load business FAQ and service pricing data
with open("data/faq_and_policies.txt", "r", encoding="utf-8") as f:
    faq_text = f.read()

with open("data/services_and_pricing.csv", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    services = list(reader)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
sessions = {}

@app.on_event("startup")
async def startup():
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_message TEXT,
                ai_reply TEXT,
                timestamp TEXT
            )
        """)
        await db.commit()

@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse("/chat")

@app.get("/chat", response_class=HTMLResponse)
async def get_chat(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        sessions[session_id] = []
    history = sessions.get(session_id, [])
    response = templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session_id,
        "history": history
    })
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_message: str = Form(...), session_id: str = Form(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history.append({"type": "human", "message": user_message, "timestamp": timestamp})

    # 🔍 Check if the user's question is relevant
    keywords = ["repair", "price", "cost", "warranty", "policy", "service", "fix", "replace"]
    if not any(kw in user_message.lower() for kw in keywords):
        fallback = "Hello! How can I assist you today? If you have any questions about our repair services, pricing, or policies, feel free to ask!"
        history.append({"type": "ai", "message": fallback, "timestamp": timestamp})

        async with aiosqlite.connect("chat_history.db") as db:
            await db.execute("""
                INSERT INTO chats (session_id, user_message, ai_reply, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_message, fallback, timestamp))
            await db.commit()

        return templates.TemplateResponse("chat.html", {
            "request": request,
            "session_id": session_id,
            "history": history
        })

    # Prepare company context for model
    services_summary = "\n".join([f"{row['Device Type']} - {row['Service']} = {row['Price']}" for row in services])
    context = f"""
    The Mobile Techs Company Info:
    -----------------------------
    FAQs and Policies:
    {faq_text}

    Services and Pricing:
    {services_summary}
    """

    full_prompt = f"{context}\n\nCustomer: {user_message}\nAgent:"
    full_history = [HumanMessage(content=full_prompt)]

    response = llm.invoke(full_history)
    history.append({"type": "ai", "message": response.content, "timestamp": timestamp})

    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("""
            INSERT INTO chats (session_id, user_message, ai_reply, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_message, response.content, timestamp))
        await db.commit()

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session_id,
        "history": history
    })

@app.get("/admin", response_class=HTMLResponse)
async def view_chats(request: Request):
    async with aiosqlite.connect("chat_history.db") as db:
        cursor = await db.execute("SELECT session_id, user_message, ai_reply, timestamp FROM chats ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
    return templates.TemplateResponse("admin.html", {"request": request, "chats": rows})
