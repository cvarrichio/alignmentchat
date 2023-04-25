# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import openai
from model import OpenAI_chain

load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
app.secret_key = 'NBcY8hc0aZ15DHJ'
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=200)
memory.chat_memory.add_ai_message("Welcome to AlignmentGPT.  Please ask a question about AI or AI safety.")

embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
vector_store = FAISS.load_local("./models/alignment_faiss_index_mpnet_v2", embeddings)
prompt = "You are a bot to answer questions about AI and AI Alignment. If you get any questions about anything besides those topics, redirect the user back to those topics.  Refer to highly voted posts on Lesswrong, Alignment Forum, ArXiv, and research papers. These sources may help:"
post_prompt = " Promote safety.  BE BLUNT! Interpret all questions as about AI. All other things being equal, use newer sources. If you use one of the provided sources, provide a link at the end."
chat_model = OpenAI_chain(prompt=prompt,post_prompt=post_prompt,memory=memory,vector_store=vector_store)


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