# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from flask import jsonify
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

class MessageInput(BaseModel):
    message: str


@app.post("/get_questions")
async def get_questions(message_data: MessageInput):
    import asyncio
    asyncio.sleep(10)
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def random_numbers():
    import random
    import asyncio
    while True:
        await asyncio.sleep(.1)  # Wait for 1 second
        yield random.randint(1, 10)

@app.post("/submit_message")
async def submit_message(message_data: MessageInput):
    from fastapi.encoders import jsonable_encoder
    async for number in random_numbers():
        return jsonable_encoder({'message': number})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)