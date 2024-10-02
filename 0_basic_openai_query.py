import os
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

# Llama-index provides the capability of querying OpenAI directly.
# For this case, it is generally recommended to make a direct API call to OpenAI.

# Basic example
res = OpenAI().complete("What is the biggest port in Croatia?")
print(res)

# Basic example - streaming
llm = OpenAI(
    model=openai_model
)
resp = llm.stream_complete("What is the biggest port in Croatia?")
for res in resp:
    print(res)
