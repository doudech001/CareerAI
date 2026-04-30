from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agenticrag import run_agent, get_conversation_history, reset_conversation_history
import uvicorn

# 1. Create the App
app = FastAPI()

# 2. OPEN THE CANAL (CORS)
# This tells the backend: "It's okay to trust requests coming from Port 5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Your Vite port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define what a "Message" looks like
class UserQuery(BaseModel):
    message: str


def is_token_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "token" in text
        and any(keyword in text for keyword in [
            "token limit",
            "maximum context length",
            "max tokens",
            "exceeded",
            "input too long",
            "length limit",
            "context length",
        ])
    )


def token_limit_reply() -> dict:
    return {
        "reply": (
            "I couldn't complete the request because the token limit was exceeded. "
            "Please try again with a shorter message or come back later."
        )
    }


# 4. The Chat Endpoint
@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"Received from Frontend: {query.message}")
    
    try:
        response_text = run_agent(query.message)
        return {"reply": response_text}
    
    except Exception as e:
        if is_token_limit_error(e):
            return token_limit_reply()
        raise HTTPException(status_code=500, detail=str(e))

# 5. Conversation history endpoints
@app.get("/history")
def get_history():
    return {"history": get_conversation_history()}

@app.post("/history/reset")
def reset_history():
    reset_conversation_history()
    return {"status": "history reset"}

# 6. Root check (just to see if it's alive)
@app.get("/")
def home():
    return {"status": "Backend is running on Port 8000"}