
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from classes import OpenAI_chain, WorkingOutputFixingParser

response_schemas = [
    ResponseSchema(name="question1", description="First of three followup questions the user might want to ask next."),
    ResponseSchema(name="question2", description="Second of three followup questions the user might want to ask next."),
    ResponseSchema(name="question3", description="Third of three followup questions the user might want to ask next.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

parser = WorkingOutputFixingParser.from_llm(ChatOpenAI(model_name='gpt-4'),output_parser)

memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=200)
prompt = "The user is asking questions about AI alignment and AI safety.  Propose 3 more questions closely related to their question that will deepen their investigation and lead them to the conclusion about safety they are seeking. Make sure no question is more than 12 words."
question_model = OpenAI_chain(model='gpt-4',prompt=prompt,memory=memory,parser=parser)


memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=200)
memory.chat_memory.add_ai_message("Welcome to AlignmentGPT.  Please ask a question about AI or AI safety.")

embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
vector_store = FAISS.load_local("./models/alignment_faiss_index_mpnet_v2", embeddings)
prompt = "You are a bot to answer questions about AI and AI Alignment. If you get any questions about anything besides those topics, redirect the user back to those topics.  Refer to highly voted posts on Lesswrong, Alignment Forum, ArXiv, and research papers. These sources may help:"
post_prompt = " Promote safety.  BE BLUNT! Interpret all questions as about AI. All other things being equal, use newer sources. If you use one of the provided sources, provide a link at the end."
chat_model = OpenAI_chain(model='gpt-3.5-turbo',prompt=prompt,post_prompt=post_prompt,memory=memory,vector_store=vector_store)
