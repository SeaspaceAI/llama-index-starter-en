import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# In this module, the way to evaluate the relevance of results and retrieved nodes will be shown. 
# There are more complex evaluation methods, but for the purposes of this tutorial, a basic approach will be sufficient.

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

llm = OpenAI(
  system_prompt="Always respond in croatian language"
)

from llama_index.core import (
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

# Define evaluator
from llama_index.core.evaluation import FaithfulnessEvaluator
evaluator = FaithfulnessEvaluator()

# Send prompt
query_engine = index.as_query_engine()
response = query_engine.query("Branch offices abroad")
print(response)

# Evaluate the relevance of answers based on the results.
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))

# Evaluate the relevance of retrieved nodes.
response_str = response.response
for source_node in response.source_nodes:
    eval_result = evaluator.evaluate(
        response=response_str, contexts=[source_node.get_content()]
    )
    print(str(eval_result.passing))
