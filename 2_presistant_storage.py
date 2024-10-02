import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model=openai_model,
    system_prompt="Always respond in Croatian language"
)

Settings.llm = llm

#To avoid creating a new vector record every time the script is run, 
#it is possible to save the record locally and load it each time the script is executed.

# If a created record already exists, load that record.
from llama_index.core import StorageContext, load_index_from_storage
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/2021"
    )
    index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

# If not, create new one
if not index_loaded:
    # Load data
    docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    # Create index
    index = VectorStoreIndex.from_documents(docs)

    # Save
    index.storage_context.persist(persist_dir="./storage/2021")

query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Poslovnice u inozemstvu")
streaming_response.print_response_stream()
