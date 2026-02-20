import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import tempfile

load_dotenv()

st.set_page_config(page_title="Multi-Source Q&A Bot", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Source Q&A Bot")
st.markdown("Ask questions from YouTube videos, websites, or PDF documents!")

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Source selection
source_type = st.selectbox("Select Source Type", ["YouTube", "Web", "PDF"])

# Function to create retriever
def create_retriever(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

# Function to get answer
def ask_question(question, retriever):
    llm_primary = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    llm_fallback = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
    
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the question ONLY using the provided context.
    If the answer is not in the context, say "I don't know based on the provided content."

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
    
    try:
        return chain_primary.invoke(question)
    except Exception as e:
        st.warning(f"Primary model failed, using fallback model")
        return chain_fallback.invoke(question)

# Source-specific UI
if source_type == "YouTube":
    youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Load YouTube Video"):
        if youtube_url:
            with st.spinner("Loading YouTube transcript..."):
                try:
                    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
                    docs = loader.load()
                    st.session_state.retriever = create_retriever(docs)
                    st.session_state.chat_history = []
                    st.success("✅ YouTube video loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading video: {e}")
        else:
            st.warning("Please enter a YouTube URL")

elif source_type == "Web":
    web_url = st.text_input("Enter Website URL:", placeholder="https://example.com")
    
    if st.button("Load Website"):
        if web_url:
            with st.spinner("Loading website content..."):
                try:
                    loader = WebBaseLoader(web_url)
                    docs = loader.load()
                    st.session_state.retriever = create_retriever(docs)
                    st.session_state.chat_history = []
                    st.success("✅ Website loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading website: {e}")
        else:
            st.warning("Please enter a website URL")

elif source_type == "PDF":
    uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
    
    if uploaded_file and st.button("Load PDF"):
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                st.session_state.retriever = create_retriever(docs)
                st.session_state.chat_history = []
                os.unlink(tmp_path)
                st.success("✅ PDF loaded successfully!")
            except Exception as e:
                st.error(f"Error loading PDF: {e}")

# Q&A Section
st.markdown("---")
st.subheader("💬 Ask Questions")

if st.session_state.retriever:
    question = st.text_input("Your question:", key="question_input")
    
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                answer = ask_question(question, st.session_state.retriever)
                st.session_state.chat_history.append({"question": question, "answer": answer})
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")
else:
    st.info("👆 Please load a source first (YouTube, Web, or PDF) to start asking questions.")
