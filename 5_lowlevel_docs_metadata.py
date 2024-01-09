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

# In this module, low-level API concepts for managing documents and creating metadata will be demonstrated, 
# including the creation of automatic metadata. There are various extractors available, 
# such as SummaryExtractor (node summarization), QuestionsAnsweredExtractor (sets a set of questions for the context of the node), 
# TitleExtractor (sets the title of the node), and EntityExtractor (extracts entities).
from llama_index.schema import MetadataMode
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor
)

# Tool for automatic metadata setting. Caution should be exercised with these tools as they utilize LLM calls, 
# incurring a cost with their usage. One or more tools can be set.
extractors_1 = [
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]

extractors_2 = [
    SummaryExtractor(summaries=["self"], llm=llm),
    QuestionsAnsweredExtractor(
        questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    ),
]

# Below is an example of independently defining a document instance and node length. 
# Reminder: SimpleDirectoryReader uses a node size of 1024 tokens, but here, creating nodes of 512 tokens will be demonstrated.
from pypdf import PdfReader
from llama_index import Document

# An instance of a document is created for each page of text, and arbitrary metadata is added.
reader = PdfReader("./whitepapers/bitcoin.pdf")
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    documents.append(
        Document(
            text=text,
            metadata={
                "filename": "bitcoin.pdf", 
                "page_label": i + 1,
                "coin": "btc"
            },
        )
    )

# TokenTextSplitter is used for creating nodes. Documents that need to be broken down into individual nodes are provided.
from llama_index.node_parser import TokenTextSplitter
node_parser = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=64
)

# Nodes of size 512 tokens are created. It should be noted that 16 nodes have been created from a total of 9 previously created documents, 
# and each node has corresponding metadata from the source page.
custom_nodes = node_parser.get_nodes_from_documents(documents)

for node in custom_nodes:
    print("--------------")
    print(node.text)
    print(node.metadata)

#  Processing nodes with automatic metadata.
from llama_index.ingestion import IngestionPipeline
pipeline = IngestionPipeline(transformations=[node_parser, *extractors_1])
nodes_1 = pipeline.run(nodes=custom_nodes, in_place=False, show_progress=True)
print(nodes_1[3].get_content(metadata_mode="all"))

pipeline = IngestionPipeline(transformations=[node_parser, *extractors_2])
nodes_2 = pipeline.run(nodes=custom_nodes, in_place=False, show_progress=True)
print(nodes_2[3].get_content(metadata_mode="all"))

index1 = VectorStoreIndex(
    nodes_1
)

index2 = VectorStoreIndex(
    nodes_2
)

query_engine1 = index1.as_query_engine(streaming=True, similarity_top_k=1)
query_engine2 = index2.as_query_engine(streaming=True, similarity_top_k=1)
response1 = query_engine1.query("koji mehanizam dokazivanja se koristi")
response2 = query_engine2.query("koji mehanizam dokazivanja se koristi")
print("response1 -> ", response1)
print("response2 -> ", response2)

# Sources
for node in response1.source_nodes:
    print("\n----------------------------------------------------")
    print("node.score -> ", node.score)
    print("node.text -> ", node.text)
    print("node.metadata -> ", node.metadata)

# Sources
for node in response2.source_nodes:
    print("\n----------------------------------------------------")
    print("node.score -> ", node.score)
    print("node.text -> ", node.text)
    print("node.metadata -> ", node.metadata)

# Conclusion: If it is necessary to independently define metadata, the QuestionsAnsweredExtractor will often be sufficient for precisely retrieving relevant nodes. 
# Other tools for automatic metadata setting will be used in specific cases. 
# One such case might be when setting long nodes is desired due to the need for context during queries, where SummaryExtractor could be helpful.