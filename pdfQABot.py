from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

import faiss
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
##Loader
loader=PyPDFLoader("quantStudy.pdf")
docs=loader.load()

##Splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)

chunks=splitter.split_documents(docs)

##Embeddings
embeddings=OllamaEmbeddings(model="nomic-embed-text")

##Vector Store
db=FAISS.from_documents(chunks, embeddings)

##LLM
llm=OllamaLLM(model="llama3",temperature=0.5)

##retriever
retriever=db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant.
    Answer the question using ONLY the provided context.

    Context:
    {context}

    Question:
    {input}
    """
)
str=StrOutputParser()

chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | str
)

print(chain.invoke("Explain the probability theory behind quantitative finance in simple terms."))

