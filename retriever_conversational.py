import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load vector database
loaded_vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Initialize LLM
llm = ChatGroq(
    temperature=0.3, 
    model_name="mixtral-8x7b-32768", 
    max_tokens=2000,
    streaming=True
)

# Initialize retriever
retriever = loaded_vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        'k': 3,
        'score_threshold': 0.7,
        'fetch_k': 8
    }
)

# Initialize conversation store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Set up the contextualize prompt
contextualize_q_system_prompt = """Given the chat history and latest question, 
create a standalone question. Be brief and direct. Do not add unnecessary context."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Set up the explanation prompt
system_prompt_explain = (
    "You are an assistant for explaining. "
    "Your users are blind, so generate text that will be read out loud to them."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "your answer generated should be in speaking tone"
    "answer concise."
    "\n\n"
    "{context}"
)

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_explain),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

# Create the explanation chain
explain_answer_chain = create_stuff_documents_chain(
    llm, 
    explain_prompt,
    document_variable_name="context"
)

# Create the retrieval chain
rag_chain_explain = create_retrieval_chain(
    history_aware_retriever, 
    explain_answer_chain
)

# Create the conversational chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_explain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def def_explain(query: str, session_id: str = "default") -> str:
    """Function to get explanatory answers using the conversational RAG chain."""
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    return response["answer"] 