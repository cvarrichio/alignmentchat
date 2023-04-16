# -*- coding: utf-8 -*-
from flask import Flask, jsonify, render_template, request, g, session

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
app.secret_key = 'NBcY8hc0aZ15DHJ'

@app.before_first_request
def init_app():
    global chat_model    
    from langchain.chat_models import ChatOpenAI
    import os
    chat_model = ChatOpenAI(temperature=0)
    global search_index
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
    search_index = FAISS.load_local("./models/alignment_faiss_index_mpnet_v2",embeddings)
    global prompt
    global post_prompt
    prompt = "You are a bot to answer questions about AI and AI Alignment. If you get any questions about anything besides those topics, redirect the user back to those topics.  Refer to highly voted posts on Lesswrong, Alignment Forum, ArXiv, and research papers. These sources may help:"
    post_prompt = " Promote safety.  BE BLUNT! Interpret all questions as about AI. All other things being equal, use newer sources. If you use one of the provided sources, provide a link at the end."
    from langchain.memory import ConversationSummaryBufferMemory
    global memory
    memory = ConversationSummaryBufferMemory(llm=chat_model,max_token_limit=200)
    memory.chat_memory.add_ai_message("Welcome to AlignmentGPT.  Please ask a question about AI or AI safety.")


def URL(url, vars=None) : 
	url =Flask.url_for(url,**vars)
	return url 


def LOAD(url,data=None,div=None, vars=None,loading_text='Loading…'):
    'Load a page via a j ax into the current page.' 
    data = data or vars #Take non-null value for data argument
    if data is not None: 
        url=URL(url,vars=data)
    output = ' '
    if div in [None, 'None']: 
        div = Flask.get_uuid() #Generate a DIV with a unique ID if none specied
        output += f"<div id='{div}'>{loading_text}</div>"
    output += """
    <script>     
    $('#{div}').load("{url}")
    </script>
    """.format(div=div,url=url)
    return output 

def update_memory(input,output):
    global memory
    memory.save_context({'input':input},{'output':output})
    return

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_message', methods=['POST'])
def submit_message():
    question = request.form.get('message')
    global search_index
    global prompt
    global post_prompt
    global memory
    global chat_model
    question = request.form.get('message')
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

    prompt_combined = "\n".join([prompt,prompt_sources,prompt_history,post_prompt])

    #Create message history
    messages = []
    from langchain.schema import SystemMessage
    messages += [SystemMessage(content = prompt_combined)]

    #Add message history
    messages += memory.buffer

    #Add question
    from langchain.schema import HumanMessage
    messages += [HumanMessage(content = "In the context of AI safety, " + question)]


    response = chat_model(messages)
    app.logger.error(response)
    answer = response.content
    message = "<br>"+answer
    update_memory(question,answer)

    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
