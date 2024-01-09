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

llm = OpenAI(temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context)

# In this module, working with documents, metadata, and the vector index will be explained in more detail. 
#  The goal is to better tailor the nodes for more successful and accurate queries, 
# ultimately resulting in more precise answers.

# # During loading, documents are broken down into smaller parts. To load individual documents, input_files are used. 
# # Since it is a list, it is possible to add a larger number of documents in this way.
# documents = SimpleDirectoryReader(
#     input_files=["./godisnje-izvjesce-2022-CA.pdf"]
# ).load_data()

# For documents within files, input_dir is used. input_dir will load all recognized documents, 
# and it is possible to set restrictions on the types of documents to load. 
# If recursive=True is set, documents in subdirectories are also loaded.
documents = SimpleDirectoryReader(
    input_dir="./whitepapers", 
    required_exts=[".pdf"], 
    recursive=True
).load_data()

# Default metadata is set, including the document's title, source location, document type, and more.
print([x.metadata for x in documents])

# Define index. 
# During this process, the VectorStoreIndex transforms documents into numerical records with lengths of 1024 tokens (default setting) 
# (1 token is approximately 4 characters), and then it creates a VectorStoreIndex object to work with further. 
# Generally, in practice, there will be dedicated vector databases like Chroma or Pinecone from which the index will be loaded. 
# As shown in previous modules, during this tutorial, the created numerical records will be saved locally. 
# By default, OpenAI is used for conversion into numerical records.
index = VectorStoreIndex.from_documents(
    documents
)

# As mentioned earlier, during the query phase, the query is transformed into a numerical representation, 
# and based on that, the most similar nodes are sought. By default, the top 2 most similar nodes are taken, 
# and they are set as context, along with the set query, and sent to the LLM. 
# Sometimes, it's beneficial to increase the number of most similar nodes passed as context. 
# The `similarity_top_k` attribute is used to define the number of results passed.
query_engine = index.as_query_engine(similarity_top_k=4) # Take top 4 most similar sources
response = query_engine.query("In comparison to Solana, which proof systems does Bitcoin use??")
print(response)

# izvori
for node in response.source_nodes:
  print("----------------------------------------------------")
  print("node.score -> ", node.score) # Relevance of sources ranges from 0 to 1, where 1 represents the highest relevance.
  print("node.text -> ", node.text) # Source text
  print("node.metadata -> ", node.metadata) # Source metadata

# # It is also possible to filter sources based on metadata, which is very useful when metadata is tailored to the documents being used. 
# # Metadata is set based on the document names. When setting custom metadata, it should be noted that only the file_path argument is available.
# def get_meta(file_path):
#     if "bitcoin" in file_path:
#         return {"tech": "blockchain", "coin": "btc"}
#     elif "Ethereum" in file_path:
#         return {"tech": "blockchain", "coin": "eth"}
#     elif "solana" in file_path:
#         return {"tech": "blockchain", "coin": "sol"}
#     else:
#         return {"tech": "blockchain"}

# documents = SimpleDirectoryReader(
#     input_dir="./whitepapers", 
#     required_exts=[".pdf"], 
#     recursive=True,
#     file_metadata=get_meta
# ).load_data()

# index = VectorStoreIndex.from_documents(
#     documents
# )

# # A filter is defined to specify the criteria for searching.
# from llama_index.vector_stores import MetadataFilters, ExactMatchFilter
# filters = MetadataFilters(
#     filters=[ExactMatchFilter(key="coin", value="btc")]
# )

# query_engine = index.as_query_engine(streaming=True, filters=filters)
# streaming_response = query_engine.query("Which proof system is being used?")
# streaming_response.print_response_stream()

# # Sources
# for node in streaming_response.source_nodes:
#   print("\n----------------------------------------------------")
#   print("node.score -> ", node.score)
#   print("node.text -> ", node.text)
#   print("node.metadata -> ", node.metadata)
