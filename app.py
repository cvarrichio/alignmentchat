# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
load_dotenv(override=True)



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


@app.post("/update_memory")
async def update_memory_endpoint(data: UpdateMemoryInput):
    chat_model.update_memory(data.question, data.answer)
    logging.debug('Updating memory!')
    return {"status": "ok"}


@app.post("/get_questions")
def get_questions(message_data: MessageInput):
    question = message_data.message
    response = question_model.create(question)
    logging.debug(response)
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit_message")
async def submit_message(message_data: MessageInput):
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