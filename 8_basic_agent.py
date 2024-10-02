import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model=openai_model
)

# Agents in LlamaIndex are powered by the LLM. They are intelligent chatbots with knowledge, capable of 
# performing tasks on given reading documents and even writing data. 
# More precisely, they have access to customized tools. These tools can be functions, other LLMs, query_engine, etc.

# For this example, functions (tools) for multiplication and addition are defined.
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two numbers and return the result"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# An agent is defined, and tools are passed to it.
from llama_index.agent.openai import OpenAIAgent
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], 
    llm=llm, 
    verbose=True
)

# Call to the agent and sources.
response = agent.chat("What is (121 * 3) + 42?")
print(str(response))
print("------------------------------")
print(response.sources)

## ---------------------------------------------------------------------------------------------------

# # stream
# response = agent.stream_chat(
#     "What is 121 * 2? Once you have the answer, use that number to write a"
#     " story about a group of mice."
# )
# response_gen = response.response_gen
# for token in response_gen:
#     print(token, end="")
