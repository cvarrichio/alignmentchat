from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

import jsonlines

line=0
labels  = {}
text_chunks = []
summary_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
with jsonlines.open("models/alignment_texts.jsonl", "r") as reader:
    for entry in reader:
        line+=1
        print('Reading line ' + str(line),end='\r')
        try:
            if entry.get('score') and entry.get('score').isdigit() and float(entry.get('score')) < 10:
                #print('Omitting low score ' + entry.get('score',10))
                continue
            for chunk in splitter.split_text(entry['text']):
                text_chunks.append(Document(page_content=chunk, metadata={'source':entry['url'],'title':entry['title'],'authors':entry['authors'],'date':entry['date_published'],'score':entry.get('score','10')}))
        except Exception as e:
            print(e)
            pass

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
search_index = FAISS.from_documents(text_chunks, embeddings)
search_index.save_local("models/alignment_faiss_index_mpnet_v2")
