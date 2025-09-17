# ---------------------------
# Imports & API Key Setup
# ---------------------------
import os
import streamlit as st
from typing import Tuple, List, Any

# Set Groq API key from Streamlit secrets or environment
try:
    # Try Streamlit secrets first (for Streamlit Cloud)
    api_key = st.secrets.get("GROQ_API_KEY")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        # Fallback to environment variable
        api_key = os.environ.get("GROQ_API_KEY")
except (AttributeError, FileNotFoundError):
    # If no secrets.toml, use environment variable
    api_key = os.environ.get("GROQ_API_KEY")

st.write("✅ API key loaded:", bool(api_key))

# ---------------------------
# LangChain / Community imports
# ---------------------------
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

# Groq LLM + agent utilities
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# Optional message types for chat history
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------
# Step 2: Retriever + Agent
# ---------------------------

@st.cache_resource
def create_pdf_retriever_tool(
    pdf_path: str,
    tool_name: str = "pdf_search_tool",
    tool_description: str = "Search PDFs",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
) -> Tuple[Any, Any]:
    """
    Load PDF, chunk it, build embeddings + FAISS vectorstore,
    and return (retriever_tool, vector_store).
    """
    # 1) Load PDF
    loader = UnstructuredPDFLoader(pdf_path, strategy="auto")
    docs = loader.load()

    # 2) Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(docs)

    # 3) Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # 4) Build FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)

    # 5) Make a retriever tool
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retriever_tool = create_retriever_tool(
        retriever, name=tool_name, description=tool_description
    )

    return retriever_tool, vector_store


@st.cache_resource
def build_agent_executor(
    _tools: List[Any],
    groq_model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    extra_rules: str = "",
) -> AgentExecutor:
    """
    Build the ChatGroq LLM and a ReAct agent executor.
    """
    # 1) LLM
    llm = ChatGroq(model_name=groq_model_name, temperature=temperature)

    # 2) Try to pull the ReAct prompt
    try:
        prompt = hub.pull("hwchase17/react-chat")
        if extra_rules:
            try:
                prompt.template += "\n" + extra_rules
            except Exception:
                pass
        agent = create_react_agent(llm, _tools, prompt)
    except Exception as e:
        st.warning(f"Could not pull hub prompt (fallback). hub.pull error: {e}")
        agent = create_react_agent(llm, _tools)

    agent_executor = AgentExecutor(agent=agent, tools=_tools, verbose=False)
    return agent_executor


def run_agent_query(
    agent_executor: AgentExecutor, query: str, chat_history: List[Any] = None
) -> Tuple[str, List[Any]]:
    """
    Run the agent on the query with optional chat history.
    """
    if chat_history is None:
        chat_history = []

    result = agent_executor.invoke({"input": query, "chat_history": chat_history})
    answer = result.get("output") if isinstance(result, dict) else str(result)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    return answer, chat_history

# ---------------------------
# Step 3: Streamlit UI
# ---------------------------

st.title("📄 RAG Q&A with Groq + LangChain")

# Sidebar for PDF upload and settings
st.sidebar.header("Upload & Settings")
pdf_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
top_k = st.sidebar.slider("Retriever top-k", 1, 5, 3)
embedding_model = st.sidebar.text_input(
    "Embedding model", value="sentence-transformers/all-MiniLM-L6-v2"
)
groq_model = st.sidebar.text_input("Groq LLM", value="llama-3.3-70b-versatile")
temp = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.0)

# Keep chat history across reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if pdf_file is not None:
    tmp_path = "uploaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(pdf_file.read())

    st.info("Building retriever... (first time may take a while)")

    try:
        # 1) Build retriever tool
        tool, vector_store = create_pdf_retriever_tool(
            pdf_path=tmp_path,
            tool_name="pdf_search",
            tool_description="Search within uploaded PDF",
            embedding_model=embedding_model,
            top_k=top_k,
        )

        # 2) Build agent executor
        agent_executor = build_agent_executor(
            _tools=[tool],
            groq_model_name=groq_model,
            temperature=temp,
            extra_rules="Always answer using retrieved context. If unsure, say 'I don’t know'.",
        )

        st.success("✅ Model & retriever ready!")

        # Question input
        st.header("Ask a question")
        user_q = st.text_area("Your question:", height=100)

        if st.button("Get Answer"):
            with st.spinner("Running agent..."):
                try:
                    answer, new_history = run_agent_query(
                        agent_executor, user_q, st.session_state.chat_history
                    )
                    st.session_state.chat_history = new_history

                    st.subheader("Answer")
                    st.write(answer)

                    # Show past conversation
                    st.subheader("Chat History")
                    for msg in st.session_state.chat_history:
                        if isinstance(msg, HumanMessage):
                            st.markdown(f"👤 **You:** {msg.content}")
                        elif isinstance(msg, AIMessage):
                            st.markdown(f"🤖 **AI:** {msg.content}")

                except Exception as e:
                    st.error(f"Agent query failed: {e}")

    except Exception as e:
        st.error(f"Retriever build failed: {e}")
