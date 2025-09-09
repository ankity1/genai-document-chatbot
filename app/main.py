from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from app.chatbot import start_session, agent_call
from app.storage import save_upload_file_bytes, cleanup_file
from app.utils.extractor import extract_text
from app.config import settings
import uuid
from fastapi.templating import Jinja2Templates
from typing import Optional

app = FastAPI(title="AI Document Chatbot")

templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start_chat/")
async def start_chat(file: Optional[UploadFile] = None):
    session_id = str(uuid.uuid4())
    text = ""

    if file:
        content = await file.read()
        path = save_upload_file_bytes(file.filename, content)
        try:
            text = extract_text(path)
        finally:
            cleanup_file(path)

    start_session(session_id, text)

    return {"session_id": session_id, "message": "Chat session started"}



@app.post("/chat/")
async def chat_endpoint(request: Request):
    form_data = await request.form()
    session_id = form_data.get("session_id")
    message = form_data.get("message")
    from app.chatbot import select_tool_based_on_semantics
    tool_name = select_tool_based_on_semantics(message)
    print(f"Tool name: {tool_name}")
    if not message:
        return {"reply": "Please send a valid message."}

    reply = agent_call(tool_name=tool_name, query=message, session_id=session_id)
    return {"reply": reply}
