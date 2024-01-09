import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# In this module, it will be shown how the obtained information can be structured in a meaningful way. 
# The example includes several nodes from which the most important information is extracted and placed within a DataFrame.

# Load documents
from pathlib import Path
from llama_index import download_loader
import re

PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()

pdf_files = Path("./resumes/").glob("*.pdf")

# Load OpenAIPydanticProgram, exclusively using gpt-4 as older models are not sufficiently good for such tasks.
from llama_index.program import (
    OpenAIPydanticProgram,
    DataFrameRowsOnly,
)
from llama_index.llms import OpenAI

program = OpenAIPydanticProgram.from_defaults(
    output_cls=DataFrameRowsOnly,
    llm=OpenAI(temperature=0, model="gpt-4-1106-preview"), # gpt-4-trubo
    prompt_template_str=(
        "Please extract the following text into a structured data:"
        " {input_str}. The column names are the following: ['Name', 'Birth date',"
        " 'City', 'Current company', 'Proffesion', 'Years of experience', 'Technologies', 'E-mail', 'Phone']. "
        " For tecnologies column, extract only programming languages, if there are none set value to null. For proffesion column, write the industry in which the person is operating. "
        " Do not specify additional parameters that"
        " are not in the function schema. "
    ),
    verbose=True,
)

# Analyze documents.
rows_list =[]
for i, pdf_file in enumerate(pdf_files):
    document = loader.load_data(file_path=str(pdf_file), metadata=True)
    doc_text = ""
    file = ""

    for doc in document:
        # If the text contains more than one newline, set only one newline.
        file = os.path.basename(doc.metadata['file_path'])
        doc_text = doc_text + re.sub(r'\n\s*\n', '\n', doc.text)

    # if i == 0: # Limitation for analyzing only one document.
    res = program(
        input_str=doc_text
    )
    row = res.rows[0].row_values
    row.append(file)
    rows_list.append(row)

# Structuring the results into a table.
import pandas as pd

columns = ['Name', 'Birth date', 'City', 'Current company', 'Profession', 'Year of experience', 'Technologies', 'E-mail', 'Phone', 'File']
df = pd.DataFrame(rows_list, columns=columns)
print(df)
