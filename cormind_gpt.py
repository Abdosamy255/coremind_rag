import streamlit as st
from google import genai
from google.genai import types
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG & SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="CoreMind | RAG Neural Core",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_ID    = "gemini-3.1-flash-lite-preview"
EMBED_MODEL = "gemini-embedding-001"
API_KEY = st.secrets["API_KEY"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CYBERPUNK CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Fira+Code:wght@300;400;500&display=swap');

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 0%, #1a0b2e 0%, #050505 80%) !important;
    background-attachment: fixed !important;
    font-family: 'Fira Code', monospace !important;
    color: #e0e0e0;
}
header { background: transparent !important; }
[data-testid="stHeaderActionElements"] { display: none; }
footer { visibility: hidden; }

.cyber-title {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #fff;
    text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #00e6ff,
                 0 0 40px #00e6ff, 0 0 80px #00e6ff;
    margin-bottom: 0.2rem;
    letter-spacing: 2px;
}
.cyber-subtitle {
    text-align: center;
    font-family: 'Fira Code', monospace;
    color: #b026ff;
    font-size: 0.9rem;
    letter-spacing: 4px;
    margin-bottom: 3rem;
    text-transform: uppercase;
}

/* ── RAG status badge ── */
.rag-active {
    text-align: center;
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem;
    color: #00ff66;
    background: rgba(0,255,102,0.08);
    border: 1px solid rgba(0,255,102,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    display: inline-block;
    margin-bottom: 1.5rem;
    letter-spacing: 2px;
}
.rag-inactive {
    text-align: center;
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem;
    color: #888;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 4px 14px;
    display: inline-block;
    margin-bottom: 1.5rem;
    letter-spacing: 2px;
}

[data-testid="stChatMessage"] {
    background: rgba(20, 20, 30, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    padding: 1rem 1.5rem !important;
    margin-bottom: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease-in-out;
}
[data-testid="stChatMessage"]:hover { transform: translateY(-2px); }

[data-testid="stChatMessage"]:has([data-testid="stIconMaterial"][aria-label="user icon"]) {
    border-left: 4px solid #00e6ff !important;
    box-shadow: -5px 0 15px rgba(0, 230, 255, 0.1) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stIconMaterial"][aria-label="assistant icon"]) {
    border-left: 4px solid #b026ff !important;
    box-shadow: -5px 0 15px rgba(176, 38, 255, 0.1) !important;
}

[data-testid="stChatInput"] { padding-bottom: 2rem !important; }
[data-testid="stChatInput"] textarea {
    background: rgba(10, 10, 15, 0.8) !important;
    color: #00e6ff !important;
    font-family: 'Fira Code', monospace !important;
    border: 1px solid #b026ff !important;
    border-radius: 8px !important;
    box-shadow: 0 0 15px rgba(176, 38, 255, 0.2) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 25px rgba(0, 230, 255, 0.4), inset 0 0 10px rgba(0, 230, 255, 0.1) !important;
    border-color: #00e6ff !important;
}
[data-testid="stChatInput"] button       { color: #00e6ff !important; }
[data-testid="stChatInput"] button:hover {
    color: #fff !important;
    background: #b026ff !important;
    box-shadow: 0 0 15px #b026ff !important;
    border-radius: 4px;
}

[data-testid="stSidebar"] {
    background: rgba(10, 10, 15, 0.9) !important;
    border-right: 1px solid #b026ff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(0, 230, 255, 0.3); }

pre  { background: #0d1117 !important; border: 1px solid rgba(0, 230, 255, 0.2) !important; border-radius: 8px !important; }
code { color: #00ff66 !important; font-family: 'Fira Code', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLIENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def get_client() -> genai.Client:
    return genai.Client(api_key=API_KEY)

client = get_client()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SYSTEM PROMPT  (من CoreMind)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM_PROMPT = """أنت 'CoreMind'، نظام ذكاء اصطناعي متطور.
تم تصميمي بواسطة بشمهندس عبدالرحمن سامي لأكون مساعداً ذكياً في مجال البرمجة، تحليل البيانات، وبناء النماذج (ML/DL/NLP).
مهمتي هي مساعدة مهندس البرمجيات والذكاء الاصطناعي في كتابة الأكواد، تحليل البيانات، وبناء النماذج.
تحدث بأسلوب تقني، دقيق، واستخدم تنسيق الـ Markdown دائماً للأكواد والشرح.
إذا أُعطيت سياقاً من ملف المستخدم، استخدمه أساساً لإجابتك."""

bot_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.7,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAG HELPERS  (من NEXUS-V)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_file(uploaded_file):
    """قراءة PDF أو TXT وتقطيعه لـ chunks."""
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    else:
        text = uploaded_file.read().decode("utf-8")

    chunk_size = 1000
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_embeddings(texts: list[str]) -> list:
    """تحويل النصوص لـ vectors باستخدام Gemini Embeddings."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return [e.values for e in response.embeddings]


def retrieve_context(prompt: str, top_k: int = 3) -> str:
    """جلب أقرب chunks للسؤال."""
    query_vec = get_embeddings([prompt])[0]
    sims = cosine_similarity([query_vec], st.session_state.doc_embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return "\n\n".join([st.session_state.doc_chunks[i] for i in top_idx])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for key, default in [
    ("messages",       []),
    ("doc_chunks",     []),
    ("doc_embeddings", []),
    ("doc_name",       None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def build_history(messages: list[dict]) -> list[types.Content]:
    return [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in messages
    ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR  –  DATA UPLINK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        "<h3 style='color:#00e6ff;text-align:center;font-family:Orbitron;'>⚡ DATA UPLINK</h3>",
        unsafe_allow_html=True,
    )
    st.write("---")

    uploaded_file = st.file_uploader("Upload Knowledge Base (PDF / TXT)", type=["pdf", "txt"])

    if uploaded_file and st.button("⚙️ PROCESS DATA", use_container_width=True):
        with st.spinner("Extracting & Vectorizing..."):
            chunks = process_file(uploaded_file)
            st.session_state.doc_chunks     = chunks
            st.session_state.doc_embeddings = get_embeddings(chunks)
            st.session_state.doc_name       = uploaded_file.name
        st.success(f"✅ '{uploaded_file.name}' processed! ({len(chunks)} chunks)")

    # حالة الـ RAG
    st.write("---")
    if st.session_state.doc_name:
        st.markdown(
            f"<div style='color:#00ff66;font-size:0.8rem;'>🟢 RAG ACTIVE<br>"
            f"<span style='color:#aaa;font-size:0.7rem;'>📄 {st.session_state.doc_name}<br>"
            f"{len(st.session_state.doc_chunks)} chunks loaded</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#888;font-size:0.8rem;'>⚫ RAG OFFLINE<br>"
            "<span style='font-size:0.7rem;'>No file loaded</span></div>",
            unsafe_allow_html=True,
        )

    st.write("---")
    if st.button("🗑️ PURGE ALL MEMORY", use_container_width=True):
        for key in ["messages", "doc_chunks", "doc_embeddings", "doc_name"]:
            st.session_state[key] = [] if key != "doc_name" else None
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN UI  –  HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="cyber-title">CoreMind</div>', unsafe_allow_html=True)

if st.session_state.doc_name:
    st.markdown(
        f'<div style="text-align:center"><span class="rag-active">⚡ RAG ACTIVE // {st.session_state.doc_name}</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="text-align:center"><span class="rag-inactive">Neural Core // Online · No File Loaded</span></div>',
        unsafe_allow_html=True,
    )

# زر مسح المحادثة فقط (بدون مسح الـ RAG)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("⚡ PURGE CHAT", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.write("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHAT INTERFACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# عرض الرسائل القديمة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])

# استقبال السؤال
placeholder = (
    "Ask a question about the uploaded file..."
    if st.session_state.doc_name
    else "Enter command or query..."
)

if prompt := st.chat_input(placeholder):
    # ── عرض رسالة المستخدم ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # ── بناء الـ prompt (مع RAG لو فيه ملف) ──
    augmented_prompt = prompt
    if st.session_state.doc_embeddings:
        with st.spinner("🔍 Searching knowledge base..."):
            context = retrieve_context(prompt)
        augmented_prompt = (
            f"هذه معلومات من ملف المستخدم:\n{context}\n\n"
            f"بناءً على المعلومات السابقة فقط، أجب على سؤال المستخدم التالي:\n{prompt}"
        )

    # ── إرسال لـ Gemini والرد Streaming ──
    with st.chat_message("assistant", avatar="⚡"):
        try:
            # بنبعت كل التاريخ ما عدا آخر رسالة، وبعدين نضيف الـ augmented_prompt
            api_history = build_history(st.session_state.messages[:-1])
            api_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=augmented_prompt)],
                )
            )

            response_stream = client.models.generate_content_stream(
                model=MODEL_ID,
                contents=api_history,
                config=bot_config,   # ← System Prompt من CoreMind
            )

            def stream_parser():
                for chunk in response_stream:
                    yield chunk.text

            full_reply = st.write_stream(stream_parser)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})

        except Exception as exc:
            st.error(f"System Error: {exc}")
