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

load_dotenv()


app = FastAPI()
app.secret_key = 'NBcY8hc0aZ15DHJ'
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

chat_model = ChatOpenAI(temperature=0)
embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
search_index = FAISS.load_local("./models/alignment_faiss_index_mpnet_v2", embeddings)
prompt = "You are a bot to answer questions about AI and AI Alignment. If you get any questions about anything besides those topics, redirect the user back to those topics.  Refer to highly voted posts on Lesswrong, Alignment Forum, ArXiv, and research papers. These sources may help:"
post_prompt = " Promote safety.  BE BLUNT! Interpret all questions as about AI. All other things being equal, use newer sources. If you use one of the provided sources, provide a link at the end."

from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=200)
memory.chat_memory.add_ai_message("Welcome to AlignmentGPT.  Please ask a question about AI or AI safety.")


def update_memory(input, output):
    global memory
    memory.save_context({'input': input}, {'output': output})
    return

def langchain_to_openai(messages):
    from langchain.schema import messages_to_dict
    mess_dict = messages_to_dict(messages)
    new_messages = [{'role':'user' if message['type']=='human' else 'assistant' if message['type']=='ai' else message['type'],'content':message['data']['content']} for message in mess_dict]
    return new_messages

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class MessageInput(BaseModel):
    message: str

from fastapi.responses import StreamingResponse
import json
import openai

async def stream_messages(model, messages):
    async for chunk in await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content:
            yield json.dumps({"message": content})

@app.post("/submit_message")
async def submit_message(message_data: MessageInput):
    question = message_data.message
    global search_index
    global prompt
    global post_prompt
    global memory
    global chat_model
    question = message_data.message
    message = '<strong>' + question + '</strong>'

    #Sources portion of system prompt
    docs = search_index.similarity_search(question, k=4)
    prompt_sources = "\n".join([str(doc) for doc in docs])
    #Conversation history of system prompt
    if memory.moving_summary_buffer:
        prompt_history = 'Conversation history:\n'
        prompt_history += memory.moving_summary_buffer
    else:
        prompt_history = ''

    prompt_combined = "\n".join([prompt, prompt_sources, prompt_history, post_prompt])

    #Create message history
    messages = []
    from langchain.schema import SystemMessage
    messages += [SystemMessage(content=prompt_combined)]

    #Add message history
    messages += memory.buffer

    #Add question
    from langchain.schema import HumanMessage
    messages += [HumanMessage(content="In the context of AI safety, " + question)]

    #response = chat_model(messages)
    #app.logger.error(response)
    #answer = response.content
    #message = "<br>" + answer
    #update_memory(question, answer)
    MODEL = 'gpt-3.5-turbo'
    return StreamingResponse(
        stream_messages(model=MODEL, messages=langchain_to_openai(messages)),
        media_type="text/plain",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)