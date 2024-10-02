import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

# Sequence of events leading to the result:
# Loading -> Indexing -> Query (Result)

# During the loading phase, documents, resources, and LLM are loaded.

# Before starting a query, it is possible to customize the LLM used,
# typically adjusting parameters like temperature (controls randomness and creativity, 0 - 1, where 1 is the highest value),
# system query (initial query), and response or query size.

# Set query size
context_window = 4096
# Set max result size
num_output = 256

# Define llm
llm = OpenAI(
    temperature=0.1,  # temperature
    model=openai_model,  # or any other version like gpt-4
    max_tokens=num_output,  # max answer
    system_prompt="Respond in Croatian language",  # system prompt
)

# Settings object is a global settings, with parameters that are lazily instantiated.
Settings.llm = llm
Settings.num_output = num_output
Settings.context_window = context_window

# To load local documents, the SimpleDirectoryReader is used, which supports popular formats
# such as .pdf, .docx, .csv, .md, .jpg, .jpeg, and others.
documents = SimpleDirectoryReader(
    input_files=["./godisnje-izvjesce-2022-CA.pdf"]
).load_data()

# The next step is to create an index used for extracting context and/or knowledge from your own documents and/or sources.
# For this example, the annual report of Croatia Airlines for the year 2022 will be used.

# An index is a data structure that enables fast retrieval of relevant context for user queries.
# Throughout this tutorial, a VectorStoreIndex will be used. The VectorStoreIndex stores each node as a numerical representation within a vector database.
# During queries, the query itself is transformed into a numerical representation,
# then the most similar nodes are retrieved and passed to the LLM as part of the query.
index = VectorStoreIndex.from_documents(documents)

# To obtain results, a query engine is created using the created index, and a query is set.
query_engine = index.as_query_engine()
response = query_engine.query("Branch offices abroad")
print(response)

# Streaming
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Branch offices abroad")
streaming_response.print_response_stream()

## ----------------------------------------------------------------------------------

# If it is necessary to customize the stream, it can be done in the following way:
# for text in streaming_response.response_gen:
#     # do something with text while streaming
#     pass
