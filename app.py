# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import json

load_dotenv()

from models import chat_model, question_model

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            data = await websocket.receive_json()

            if data['type'] == 'submit_message':
                question = data['message']
                async for message in chat_model.stream_messages('gpt-3.5-turbo', question):
                    await websocket.send_json({
                        'type': 'message',
                        'message': message
                    })

                #updateMemory(question, message)
                await websocket.send_json({
                    'type': 'memory_updated'
                })

            elif data['type'] == 'get_questions':
                question = data['message']
                response = question_model.create(question)
                await websocket.send_json({
                    'type': 'questions_received'
                })

            elif data['type'] == 'update_memory':
                question = data['question']
                answer = data['answer']
                chat_model.update_memory(question, answer)
                await websocket.send_json({
                    'type': 'memory_updated'
                })

        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)