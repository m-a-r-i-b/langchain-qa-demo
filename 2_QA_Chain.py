from dotenv import load_dotenv
load_dotenv(".env")

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("./assets/quantum_computing.pdf")
documents = loader.load()

print(len(documents))

llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")
# query = "How much power does a 20 Q-bits quantum computer use?"
# query = "How much power does a 300 Q-bits quantum computer use?"
query = "How many states does a classical bit represent?"
ans = chain.run(input_documents=documents, question=query)
print(ans)