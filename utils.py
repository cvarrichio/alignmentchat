from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

import jsonlines

line=0

text_chunks = []
summary_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=2048, chunk_overlap=64)
with jsonlines.open("alignment_texts.jsonl", "r") as reader:
    for entry in reader:
        line+=1
        print('Reading line ' + str(line))
        try:
            #from bs4 import BeautifulSoup
            #soup = BeautifulSoup(entry['summary'], "html.parser")
            #text = soup.get_text()
            #summary_chunks.append(Document(page_content=text, metadata={'source':entry['source']}))
            for chunk in splitter.split_text(entry['text']):
                text_chunks.append(Document(page_content=chunk, metadata={'source':entry['url'],'date':entry['date_published']}))
        except Exception as e:
            print(e)
            pass

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
search_index = FAISS.from_documents(text_chunks, embeddings)
search_index.save_local("alignment_faiss_index_mpnet_v2")
