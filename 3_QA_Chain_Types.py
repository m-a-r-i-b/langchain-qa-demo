# Longer text example, with stuff, fails, then show following example
# map_reduce


# Simple stuff chain with short text
from dotenv import load_dotenv
load_dotenv(".env")

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("./assets/qc-book.pdf")
documents = loader.load()
llm = OpenAI()

print("-"*20)
print(len(documents))
# print(documents[0].page_content)

## Does not work because of context limit
chain = load_qa_chain(llm, chain_type="stuff")
query = "Who are the front-runners in experimental quantum computing?"

## Works, but makes multiple calls => expensive, better to use Retrieval chain
# chain = load_qa_chain(llm, chain_type="map_reduce",verbose=True)
# query = "Who are the front-runners in experimental quantum computing?"

ans = chain.run(input_documents=documents, question=query)
print(ans)