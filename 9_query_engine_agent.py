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

# Load documents
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

# Load first index for year 2021
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/2021"
    )
    twentyone_index = load_index_from_storage(storage_context)

    index_one_loaded = True
except:
    index_one_loaded = False

if not index_one_loaded:
    twentyone_docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    twentyone_index = VectorStoreIndex.from_documents(twentyone_docs)

    twentyone_index.storage_context.persist(persist_dir="./storage/2021")

# Load second index for year 2022
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/2022"
    )
    twentytwo_index = load_index_from_storage(storage_context)

    index_two_loaded = True
except:
    index_two_loaded = False

if not index_two_loaded:
    twentytwo_docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2022-CA.pdf"]
    ).load_data()

    twentytwo_index = VectorStoreIndex.from_documents(twentytwo_docs)

    twentytwo_index.storage_context.persist(persist_dir="./storage/2022")

# Define query_engine
twentyone_engine = twentyone_index.as_query_engine(similarity_top_k=3, llm=llm)
twentytwo_engine = twentytwo_index.as_query_engine(similarity_top_k=3, llm=llm)

# Define query_engine tools for agent usage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=twentyone_engine,
        metadata=ToolMetadata(
            name="croatia_airlines_report_2021",
            description=(
                "Provides information about Croatia Airlines bussines report for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=twentytwo_engine,
        metadata=ToolMetadata(
            name="croatia_airlines_report_2022",
            description=(
                "Provides information about Croatia Airlines bussines report for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

# Define agent and pass the tools 
from llama_index.agent.openai import OpenAIAgent
agent = OpenAIAgent.from_tools(
    query_engine_tools, 
    verbose=True,
    system_prompt="""
        - The questions are for Croatia Airlines report for years 2021 and 2022. 
        - Remember to always use available tools
        - explain how did you get the final answer 
    """
)

# Ask a single prompt
res = agent.query("Which year had more employes?")
print(res)

## --------------------------------------------------------------------------------------------------

# # Chat with agent
# agent.chat_repl()
