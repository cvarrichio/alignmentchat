# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

from models import chat_model, question_model

load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
app.secret_key = 'NBcY8hc0aZ15DHJ'
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

class MessageInput(BaseModel):
    message: str


class UpdateMemoryInput(BaseModel):
    question: str
    answer: str


@app.post("/update_memory")
async def update_memory_endpoint(data: UpdateMemoryInput):
    global chat_model
    chat_model.update_memory(data.question, data.answer)
    return {"status": "ok"}


@app.post("/get_questions")
async def get_questions(data: MessageInput):
    
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit_message")
async def submit_message(message_data: MessageInput):
    global chat_model
    question = message_data.message
    message = '<strong>' + question + '</strong>'
    
    MODEL = 'gpt-3.5-turbo'
    return StreamingResponse(
        chat_model.stream_messages(MODEL,question),
        media_type="text/plain",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)