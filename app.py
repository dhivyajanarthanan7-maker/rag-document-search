import streamlit as st
import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot • DeepSeek-R1",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Custom CSS — dark glassmorphism theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep:    #0a0e1a;
    --bg-card:    rgba(255,255,255,0.04);
    --border:     rgba(255,255,255,0.08);
    --accent:     #6c63ff;
    --accent2:    #00d4aa;
    --text-main:  #e8eaf6;
    --text-muted: #8890a4;
    --user-bub:   linear-gradient(135deg,#6c63ff,#a78bfa);
    --bot-bub:    rgba(255,255,255,0.06);
    --radius:     16px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--accent) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label { color: var(--accent) !important; font-weight:600; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #a78bfa) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    cursor: pointer !important;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

.chat-wrapper { display:flex; flex-direction:column; gap:1rem; padding:0.5rem 0; }

.bubble-row { display:flex; align-items:flex-end; gap:0.75rem; }
.bubble-row.user  { flex-direction:row-reverse; }
.bubble-row.bot   { flex-direction:row; }

.avatar {
    width:36px; height:36px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; flex-shrink:0;
}
.avatar.user { background: linear-gradient(135deg,#6c63ff,#a78bfa); }
.avatar.bot  { background: linear-gradient(135deg,#00d4aa,#0099cc); }

.bubble {
    max-width:72%; padding:0.85rem 1.1rem;
    border-radius: 18px; line-height:1.6; font-size:0.93rem;
    word-break:break-word; white-space: pre-wrap;
}
.bubble.user {
    background: var(--user-bub);
    border-bottom-right-radius: 4px;
    color: #fff;
}
.bubble.bot {
    background: var(--bot-bub);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text-main);
    backdrop-filter: blur(8px);
}

.pill {
    display:inline-flex; align-items:center; gap:0.4rem;
    background: var(--bg-card); border:1px solid var(--border);
    border-radius:999px; padding:0.3rem 0.8rem;
    font-size:0.78rem; color:var(--text-muted);
}
.pill .dot { width:7px; height:7px; border-radius:50%; background:var(--accent2); }

[data-testid="stMetric"] {
    background: var(--bg-card); border:1px solid var(--border);
    border-radius:12px; padding:0.6rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--accent2) !important; font-weight:700; }

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────
OLLAMA_LLM_MODEL   = "deepseek-r1:7b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
TEMP_DIR           = Path("./tmp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────
#  Session-state defaults
# ─────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history":  [],   # list of {"role": "user"|"assistant", "content": str}
        "lc_messages":   [],   # LangChain HumanMessage / AIMessage list
        "vectorstore":   None,
        "retriever":     None,
        "chain":         None,
        "doc_loaded":    False,
        "doc_name":      "",
        "chunk_size":    800,
        "chunk_overlap": 100,
        "num_chunks":    0,
        "k_docs":        4,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────
#  Helper — strip DeepSeek <think> blocks
# ─────────────────────────────────────────
def clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ─────────────────────────────────────────
#  Core pipeline
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


def build_vectorstore(pdf_path: str, chunk_size: int, chunk_overlap: int):
    with st.spinner("📄 Loading & splitting PDF…"):
        loader   = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)

    with st.spinner(f"🔢 Embedding {len(chunks)} chunks via `{OLLAMA_EMBED_MODEL}`…"):
        embeddings  = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def build_chain(vectorstore, k_docs: int):
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.2)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_docs},
    )

    # Contextualise question prompt (rewrites follow-ups using chat history)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and a follow-up question, "
         "rewrite it as a standalone question. "
         "Return ONLY the rewritten question, nothing else."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question "
         "using ONLY the context below. If the answer is not in the context, "
         "say you don't know.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def contextualize_question(inputs):
        if inputs.get("chat_history"):
            chain = contextualize_q_prompt | llm | StrOutputParser()
            return chain.invoke(inputs)
        return inputs["input"]

    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(contextualize_question)
                    | retriever
                    | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem'>
        <div style='font-size:2.5rem'>🤖</div>
        <h2 style='margin:0.2rem 0 0; color:#6c63ff; font-size:1.25rem; font-weight:700;'>RAG Chatbot</h2>
        <p style='color:#8890a4; font-size:0.78rem; margin:0;'>DeepSeek-R1 · FAISS · Ollama</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**🛠 Model Stack**")
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;gap:0.4rem;margin-bottom:1rem;'>
        <span class='pill'><span class='dot'></span>LLM: {OLLAMA_LLM_MODEL}</span>
        <span class='pill'><span class='dot'></span>Embed: {OLLAMA_EMBED_MODEL}</span>
        <span class='pill'><span class='dot'></span>VectorDB: FAISS-CPU</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**📂 Upload Document**")
    uploaded_file = st.file_uploader(
        label="Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**⚙️ Chunking Settings**")
    chunk_size    = st.slider("Chunk size (chars)",    200,  2000, st.session_state.chunk_size,    step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)",   0,   500, st.session_state.chunk_overlap, step=25)
    k_docs        = st.slider("Docs to retrieve (k)",   1,    10, st.session_state.k_docs,        step=1)

    st.session_state.chunk_size    = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    st.session_state.k_docs        = k_docs

    st.divider()

    process_clicked = st.button("⚡ Process PDF", use_container_width=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.lc_messages  = []
        st.rerun()

    if st.button("🔄 Reset All", use_container_width=True):
        for key in ["chat_history","lc_messages","vectorstore","retriever",
                    "chain","doc_loaded","doc_name","num_chunks"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    if st.session_state.doc_loaded:
        st.markdown("**📊 Index Stats**")
        c1, c2 = st.columns(2)
        c1.metric("Chunks", st.session_state.num_chunks)
        c2.metric("k", st.session_state.k_docs)
        st.success(f"✅ {st.session_state.doc_name}")
    else:
        st.markdown("""
        <div style='color:#8890a4;font-size:0.8rem;text-align:center;'>
            Upload a PDF and click <strong>Process PDF</strong> to begin.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Process PDF
# ─────────────────────────────────────────
if process_clicked:
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF first.")
    else:
        save_path = TEMP_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            vs, n_chunks = build_vectorstore(
                str(save_path),
                st.session_state.chunk_size,
                st.session_state.chunk_overlap,
            )
            chain, retriever = build_chain(vs, st.session_state.k_docs)

            st.session_state.vectorstore  = vs
            st.session_state.retriever    = retriever
            st.session_state.chain        = chain
            st.session_state.doc_loaded   = True
            st.session_state.doc_name     = uploaded_file.name
            st.session_state.num_chunks   = n_chunks
            st.session_state.chat_history = []
            st.session_state.lc_messages  = []

            st.sidebar.success(f"✅ Ready! {n_chunks} chunks indexed.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# ─────────────────────────────────────────
#  MAIN — Header
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem;'>
    <h1 style='
        font-size:2.2rem; font-weight:700; margin:0;
        background:linear-gradient(135deg,#6c63ff,#00d4aa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    '>📚 RAG Chatbot</h1>
    <p style='color:#8890a4; margin:0.4rem 0 0; font-size:0.9rem;'>
        Ask anything about your uploaded document
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────
#  Chat display
# ─────────────────────────────────────────
with st.container():
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='
            text-align:center; margin:3rem auto; max-width:480px;
            padding:2rem; background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.07); border-radius:20px;
        '>
            <div style='font-size:3rem; margin-bottom:1rem;'>💬</div>
            <h3 style='color:#e8eaf6; margin:0 0 0.5rem;'>Ready to chat!</h3>
            <p style='color:#8890a4; font-size:0.875rem; margin:0;'>
                Upload a PDF via the sidebar, click <strong>Process PDF</strong>,
                then ask your questions below.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        bubble_html = "<div class='chat-wrapper'>"
        for msg in st.session_state.chat_history:
            role    = msg["role"]
            content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            if role == "user":
                bubble_html += f"""
                <div class='bubble-row user'>
                    <div class='avatar user'>👤</div>
                    <div class='bubble user'>{content}</div>
                </div>"""
            else:
                bubble_html += f"""
                <div class='bubble-row bot'>
                    <div class='avatar bot'>🤖</div>
                    <div class='bubble bot'>{content}</div>
                </div>"""
        bubble_html += "</div>"
        st.markdown(bubble_html, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────
#  Chat input
# ─────────────────────────────────────────
if prompt := st.chat_input(
    placeholder="Ask a question about your document…",
    disabled=not st.session_state.doc_loaded,
):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("🧠 Thinking…"):
        try:
            raw = st.session_state.chain.invoke({
                "input":        prompt,
                "chat_history": st.session_state.lc_messages,
            })
            answer = clean_response(raw)

            # Retrieve source pages for attribution
            retriever = st.session_state.retriever
            src_docs  = retriever.invoke(prompt)
            if src_docs:
                pages = sorted({
                    (doc.metadata.get("page", 0) + 1)
                    for doc in src_docs
                    if isinstance(doc.metadata.get("page"), int)
                })
                if pages:
                    answer += f"\n\n📄 Sources: page(s) {', '.join(str(p) for p in pages)}"

        except Exception as e:
            answer = f"⚠️ Error: {e}"

    # Update LangChain message history
    st.session_state.lc_messages.append(HumanMessage(content=prompt))
    st.session_state.lc_messages.append(AIMessage(content=answer))
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

# ─────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-top:2rem; color:#8890a4; font-size:0.75rem;'>
    Powered by <strong style='color:#6c63ff;'>LangChain</strong> ·
    <strong style='color:#00d4aa;'>Ollama</strong> ·
    <strong style='color:#a78bfa;'>FAISS</strong>
</div>
""", unsafe_allow_html=True)
