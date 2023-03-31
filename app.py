# -*- coding: utf-8 -*-

from flask import Flask, jsonify, render_template, request, g, session

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings



app = Flask(__name__)
app.secret_key = 'NBcY8hc0aZ15DHJ'

@app.before_first_request
def init_app():
    global search_index
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
    search_index = FAISS.load_local("./models/alignment_faiss_index_mpnet_v2",embeddings)
    global prompt
    prompt = "You are a bot to answer questions about AI and AI Alignment.  If you get any questions about anything besides those topics, redirect the user back to those topics.  Refer to highly voted posts on Lesswrong, Alignment Forum, ArXiv, and research papers.  Promote safety.  These sources may help:"


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



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_message', methods=['POST'])
def submit_message():
    question = request.form.get('message')
    global search_index
    global prompt
    question = request.form.get('message')
    message = '<strong>' + question + '</strong>'
    docs = search_index.similarity_search(question, k=4)
    previous_question = session.get('previous_question')
    previous_answer = session.get('previous_answer')
    import openai
    openai.api_key = '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'
    enhanced_prompt = prompt + "\n".join([str(doc) for doc in docs])
    messages = [{"role": "system", "content":enhanced_prompt}]
    if previous_question is not None:
        messages += [{"role":"user","content":previous_question}]
        messages += [{"role":"assistant","content":previous_answer}]
    messages += [{"role":"user","content":question}]
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    response['prompt'] = enhanced_prompt
    app.logger.error(response)
    answer = response.choices[0].message.content
    session['previous_question'] = question
    session['previous_answer'] = answer
    message = "<br>"+answer
    # do something with the message, like store it in a database
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
