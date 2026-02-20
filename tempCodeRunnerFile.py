docs = [Document(page_content=text)]

# ##Splitter
# Splitter=RecursiveCharacterTextSplitter(
#     chunk_size=600,
#     chunk_overlap=100,
# )

# chunks=Splitter.split_documents(docs)

# ##Embeddings
# embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ##Vector Store
# db=FAISS.from_documents(chunks,embeddings)

# ##LLM
# llm_primary = ChatGroq(
#     model_name="llama-3.3-70b-versatile",
#     temperature=0.2,
# )

# llm_fallback = ChatGroq(
#     model_name="llama-3.1-8b-instant",
#     temperature=0.2,
# )

# ##retriever
# retriever=db.as_retriever(search_kwargs={"k":3})

# prompt=PromptTemplate.from_template(

# """
# You are a helpful assistant.
# Answer the question ONLY using the video transcript context.
# If the answer is not in the video, say "The video does not mention this."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# chain_primary = (
#     {
#         "context": retriever | format_docs,
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm_primary
#     | StrOutputParser()
# )

# chain_fallback = (
#     {
#         "context": retriever | format_docs,
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm_fallback
#     | StrOutputParser()
# )

# def ask_with_fallback(question: str):
#     try:
#         return {
#             "answer": chain_primary.invoke(question),
#             "model_used": "llama-3.3-70b-versatile"
#         }
#     except Exception as e:
#         print(f"Primary model failed, switching to fallback: {e}")
#         return {
#             "answer": chain_fallback.invoke(question),
#             "model_used": "llama-3.1-8b-instant"
#         }
    
# response=ask_with_fallback("What are transformers?")
# print(response["answer"])
