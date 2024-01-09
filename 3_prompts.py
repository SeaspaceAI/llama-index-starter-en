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
from llama_index.prompts import PromptTemplate

llm = OpenAI(
  system_prompt="Always respond in croatian language"
)
service_context = ServiceContext.from_defaults(
    llm=llm
)
set_global_service_context(service_context)

documents = SimpleDirectoryReader(
    input_files=["./whitepapers/bitcoin.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
  documents
)


# To find relevant nodes and obtain accurate answers, having a good prompt is essential. 
# The ideal situation is when precise steps for the LLM to execute are provided, 
# and this is done with customized prompts.

# By default, the query_engine uses "text_qa_template" for each query, 
# and if the retrieved context is too long for a single LLM call, then "refine_template" is also employed
query_engine = index.as_query_engine()
prompts_dict = query_engine.get_prompts()
for k, v in prompts_dict.items():
    print("Prompt key -> ", k)
    print(v.get_template())
    print(f"\n\n")

# # To change the query templates, it is always necessary to have variables "context_str" for the retrieved context and 
# # "query_str" for the query.
# custom_qa_prompt = PromptTemplate(
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the query in the style of a Shakespeare play.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

# query_engine = index.as_query_engine(
#     text_qa_template=custom_qa_prompt
# )
# prompts_dict = query_engine.get_prompts()
# for k, v in prompts_dict.items():
#     print("Prompt key -> ", k)
#     print(v.get_template())
#     print(f"\n\n")

# # Sometimes it is necessary to dynamically change the query.
# qa_prompt_tmpl_str = """\
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Please write the answer in the style of {tone_name}
# Query: {query_str}
# Answer: \
# """

# prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
# partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="Shakespeare")
# query_engine = index.as_query_engine(
#     text_qa_template=partial_prompt_tmpl
# )
# # prompts_dict = query_engine.get_prompts()
# # for k, v in prompts_dict.items():
# #     print("Prompt key -> ", k)
# #     print(v.get_template())
# #     print(f"\n\n")
# response = query_engine.query("o cemu se radi")
# print(response)
