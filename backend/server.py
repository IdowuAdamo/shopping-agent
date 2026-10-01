from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv

from agent import run_agent

# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (reads OPENAI_API_KEY from the environment)
client = OpenAI()

# Memory settings. Flip MEMORY_ENABLED in backend/.env to change the behaviour.
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
MEMORY_DIR = Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)


def load_conversation(session_id: str) -> List[Dict]:
    """Load past user/assistant messages for this session."""
    file_path = MEMORY_DIR / f"{session_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save the conversation for this session."""
    file_path = MEMORY_DIR / f"{session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def valid_session_id(raw: Optional[str]) -> str:
    """The session id becomes part of a file name, so only accept real UUIDs."""
    if raw is None:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/")
def root():
    return {"message": "Shopping Agent API", "memory_enabled": MEMORY_ENABLED}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = valid_session_id(request.session_id)
    try:
        history = load_conversation(session_id) if MEMORY_ENABLED else []

        reply = run_agent(client, history, request.message)

        if MEMORY_ENABLED:
            # Save only the visible conversation (not the tool calls) to keep it small
            history.append({"role": "user", "content": request.message})
            history.append({"role": "assistant", "content": reply})
            save_conversation(session_id, history)

        return ChatResponse(response=reply, session_id=session_id)

    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong. Check the backend terminal.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)