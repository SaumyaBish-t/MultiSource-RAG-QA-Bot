from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

##Loader
loader=WebBaseLoader("https://docs.langchain.com/oss/javascript/integrations/text_embedding/index#embedding-models")

docs = loader.load()

##Splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks=splitter.split_documents(docs)

##Embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)


##Vector Store
db=FAISS.from_documents(chunks,embeddings)

##LLM
llm_primary = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
)

llm_fallback = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
)

##retriever
retriever=db.as_retriever(search_kwargs={"k":3})


prompt=PromptTemplate.from_template(

"""
You are a helpful assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain_primary = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm_primary
    | StrOutputParser()
)

chain_fallback = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm_fallback
    | StrOutputParser()
)


def ask_with_fallback(question: str):
    try:
        return {
            "answer": chain_primary.invoke(question),
            "model_used": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        print(f"Primary model failed, switching to fallback: {e}")
        return {
            "answer": chain_fallback.invoke(question),
            "model_used": "llama-3.1-8b-instant"
        }

response = ask_with_fallback("What are the embedding models supported by LangChain?")
print(response["answer"])