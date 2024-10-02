import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq

llama_llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)

# Chat_engine is used in cases where communication involves multiple iterations, during which the conversation history is tracked. 
# This is in contrast to query_engine, which is used for single questions.

documents = SimpleDirectoryReader(
    input_files=["./whitepapers/bitcoin.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
  documents,
  llm=llama_llm
)

# # Chat_engine has several operating modes. During this tutorial, chat_mode="context" and "condense_plus_context" will be used. 
# # "Context" finds nodes that are most similar to the query. "Condense_question" reviews the conversation history 
# # and rephrases the user's message to be a query for the index. "Condense_plus_context" is a combination of "context" and "condense_question."

# # A simple chat_engine that provides results based on the found context.
# chat_engine = index.as_chat_engine(chat_mode="context", similarity_top_k=4)
# response = chat_engine.chat("What is it about?")
# print(response)

# # # Sources
# # for node in response.source_nodes:
# #     print("\n----------------------------------------------------")
# #     print("node.text -> ", node.text)

## ------------------------------------------------------------------------------------------------------------------------------

# # Printing all actions before obtaining the result.
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # A simple memory is added as a buffer.
# from llama_index.core.memory import ChatMemoryBuffer
# memory = ChatMemoryBuffer.from_defaults()

# chat_engine = index.as_chat_engine(
#   chat_mode="condense_plus_context",
#   memory=memory,
# )
# streaming_response = chat_engine.stream_chat("What is it about?")
# streaming_response.print_response_stream()

# # It can be inferred that the conversation history is taken into account, and the question is reformulated as "Which token does Bitcoin use?"
# streaming_response = chat_engine.stream_chat("Which tokes does it use?") 
# streaming_response.print_response_stream()

# # Proof it has history
# streaming_response = chat_engine.stream_chat("What was my first question?")
# streaming_response.print_response_stream()

## -------------------------------------------------------------------------------------------------------------------------
    
# # It is also possible to start a conversation session; this will be used for easier testing and evaluations.
# chat_engine.chat_repl()