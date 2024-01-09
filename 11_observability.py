import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# In this module, it will be shown how queries can be tracked. 
# Llama-index offers many integrations for tracking, and here, Traceloop is used.

# Traceloop is initialized with an API key obtained after registration on their website (https://app.traceloop.com/). 
# For the purposes of this tutorial, Traceloop is free. 
# After sending a query, all steps are recorded and can be reviewed on the Traceloop website.
from traceloop.sdk import Traceloop
traceloop_key = os.getenv("TRACELOOP_API_KEY")

Traceloop.init(disable_batch=True, api_key=traceloop_key)

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
set_global_service_context(service_context)

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

if not index_loaded:
    docs = SimpleDirectoryReader(
        input_files=["./godisnje-izvjesce-2021-CA.pdf"]
    ).load_data()

    index = VectorStoreIndex.from_documents(docs)

    index.storage_context.persist(persist_dir="./storage/2021")

query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Branches abroad.")
streaming_response.print_response_stream()
