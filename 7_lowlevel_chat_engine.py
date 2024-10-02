import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model=openai_model
)

documents = SimpleDirectoryReader(
    input_files=["./whitepapers/bitcoin.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
    documents
)

# Chat_engine generally adds new functionalities and is built on the foundation of query_engine. 
# As mentioned in the previous module, a significant difference is the addition of the conversation history throughout the entire conversation. 
# So far, mostly high-level API concepts have been used. For utilizing a custom conversation history, low-level API composition must be used.
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.condense_plus_context import (
    CondensePlusContextChatEngine,
)

# The conversation history is defined; first, the user's message is defined, and then the assistant's.
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="What is the best crypto project?",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Bitcoin."),
]

# Define query_engine
query_engine = index.as_query_engine(llm=llm)

# Define retriver
# Retrievers are responsible for fetching the most relevant context based on the user's query from the index.
retreiver = index.as_retriever()

# Define CondensePlusContextChatEngine
# This is equivalent to "chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")" 
# but with low-level API composition, where each aspect is defined "manually."
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=retreiver,
    query_engine=query_engine,
    chat_history=custom_chat_history,
    verbose=True # Show context and prompt
)
  
# Define prompt
streaming_response = chat_engine.stream_chat("Can you tell me more about it?")
streaming_response.print_response_stream()
