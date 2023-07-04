from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv(".env")


loader = PyPDFLoader("./assets/qc-book.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# select which embeddings we want to use
embeddings = OpenAIEmbeddings()

# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

# create a chain to answer questions 
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
query = "Who are the front-runners in experimental quantum computing?"

ans = qa({"query": query})
print("Result : ",ans['result'])

print("-"*20)
print("Source : ",ans['source_documents'])
