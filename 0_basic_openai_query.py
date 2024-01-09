import os
from llama_index.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Llama-index provides the capability of querying OpenAI directly.
# For this case, it is generally recommended to make a direct API call to OpenAI.

# Basic example
res = OpenAI().complete("Koja je najveća luka u Hrvatskoj?")
print(res)

# Basic example - streaming
llm = OpenAI()
resp = llm.stream_complete("Koja je najveća luka u Hrvatskoj?")
for res in resp:
    print(res)
