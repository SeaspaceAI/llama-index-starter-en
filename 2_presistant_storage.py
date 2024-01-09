import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    set_global_service_context
)
from llama_index.llms import OpenAI

llm = OpenAI(
  system_prompt="Always respond in croatian language"
)

service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context) # Define global_service_context, it is always used for indexes.

#To avoid creating a new vector record every time the script is run, 
#it is possible to save the record locally and load it each time the script is executed.

# If a created record already exists, load that record.
from llama_index import (
    StorageContext,
    load_index_from_storage,
)
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
