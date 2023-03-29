from flask import Flask, jsonify, render_template, request, g

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.embeddings import HuggingFaceEmbeddings

#embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
#search_index8 = FAISS.from_documents(text_chunks, embeddings)
#search_index8 = FAISS.from_documents(summary_chunks, OpenAIEmbeddings())
#search_index8.save_local("alignment_faiss_index_mpnet_v2")


app = Flask(__name__)

@app.before_request
def before_request():
    #embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
    #g.faiss = FAISS.load_local("alignment_faiss_index_mpnet_v2",embeddings)
    pass

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
    message = request.form.get('message')
    # do something with the message, like store it in a database
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)