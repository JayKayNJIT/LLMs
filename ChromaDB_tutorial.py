from langchain.document_loaders import DirectoryLoader

directory = '/Users/jatinkashyap/Documents/Work/innovationsLLC/July_20_2023/read'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

query = "What are the different kinds of criteria people uses to make a decision?"
matching_docs = db.similarity_search(query)

matching_docs[0]

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()

import os
os.environ["OPENAI_API_KEY"] = "XXXXXXXXXXXXXXXXXX"

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)


from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "What are the different coding libraries to make decision?"
matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)
answer
