import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env file
load_dotenv()

# Debug: Print API key 
print("API KEY loaded:", os.getenv("OPENAI_API_KEY"))

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# test message
response = llm.invoke("Hello! Are you working?")
print("Response from model:", response.content)
