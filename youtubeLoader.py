import os
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. LOADER: Fix by disabling add_video_info to avoid HTTP 400
try:
    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=wjZofJX0v4M", 
        add_video_info=False
    )
    docs = loader.load()
except Exception as e:
    print(f"Error loading transcript: {e}")
    docs = []

# 2. SPLITTER
if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3. EMBEDDINGS & VECTOR STORE
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 4. LLM CONFIGURATION
    llm_primary = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    llm_fallback = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

    # 5. PROMPT & CHAIN
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the question ONLY using the video transcript context.
    If the answer is not in the context, say "The video does not mention this."

    Context: {context}
    Question: {question}
    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

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

response = ask_with_fallback("What are transformers uses?")
print(response["answer"])
