from langchain.schema import BaseOutputParser
from langchain.vectorstores.base import VectorStore
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Optional
import openai
import json

class OpenAI_chain(BaseModel):
    model: str = 'gpt-3.5-turbo'
    prompt: str = ''
    post_prompt: str = ''
    parser: Optional[BaseOutputParser]
    memory: Optional[BaseChatMemory]
    vector_store: Optional[VectorStore]


    class Config:
        arbitrary_types_allowed = True

    def prepare_messages(self,question):
        
        #Sources portion of system prompt
        docs = self.vector_store.similarity_search(question, k=4)
        prompt_sources = "\n".join([str(doc) for doc in docs])
        #Conversation history of system prompt
        if self.memory.moving_summary_buffer:
            prompt_history = 'Conversation history:\n'
            prompt_history += self.memory.moving_summary_buffer
        else:
            prompt_history = ''

        prompt_combined = "\n".join([self.prompt, prompt_sources, prompt_history, self.post_prompt])

        #Create message history
        messages = []
        messages += [SystemMessage(content=prompt_combined)]

        #Add message history
        messages += self.memory.buffer

        #Add question
        messages += [HumanMessage(content="In the context of AI safety, " + question)]

        return(messages)

    def langchain_to_openai(self,messages):
        from langchain.schema import messages_to_dict
        mess_dict = messages_to_dict(messages)
        new_messages = [{'role':'user' if message['type']=='human' else 'assistant' if message['type']=='ai' else message['type'],'content':message['data']['content']} for message in mess_dict]
        return new_messages

    async def update_memory(self, input, output):
            self.memory.save_context({'input': input}, {'output': output})
            return

    async def stream_messages(self, model, question):
        messages = self.langchain_to_openai(messages=self.prepare_messages(question))
        async for chunk in await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            stream=True,
        ):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content:
                #logging.debug(content)
                yield json.dumps({"message": content})