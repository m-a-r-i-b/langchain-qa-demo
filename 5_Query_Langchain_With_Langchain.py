import os
from dotenv import load_dotenv
load_dotenv(".env")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

embeddings = OpenAIEmbeddings(disallowed_special=())


root_dir = '/mnt/nvme0n1p3/Work/langchain/langchain'
db = None

try :
    print("Local DB found")
    db = FAISS.load_local(root_dir, embeddings)
except:
    print("Local DB not found")

    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(dirnames)
        for file in filenames:
            if file.endswith(".py"):
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    pass
    print(f"Total docs found : {len(docs)}")


    print("Starting chunking...")
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    print("Creating db...")
    db = FAISS.from_documents(texts, embeddings)

    print("Saving db...")
    db.save_local(root_dir)



retriever = db.as_retriever()
model = ChatOpenAI(model_name="gpt-3.5-turbo") 
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


question1 = "When we call the from_llm method of ConversationalRetrievalChain class, which methods are executed and in what order?"
result = qa({"question": question1, "chat_history": []})
print(result["answer"])
print("-"*30)
