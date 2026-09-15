import io
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_OVERLAP = 120
DEFAULT_TOP_K = 5
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False


# -----------------------------
# Cached embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    """Return one record per PDF page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            records.append(
                {
                    "filename": filename,
                    "page": page_number,
                    "text": text,
                }
            )

    return records


def extract_docx(file_bytes, filename):
    """DOCX does not reliably expose page numbers, so page is None."""
    document = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = clean_text("\n".join(paragraphs))

    return [
        {
            "filename": filename,
            "page": None,
            "text": text,
        }
    ] if text else []


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")
    text = clean_text(text)

    return [
        {
            "filename": filename,
            "page": None,
            "text": text,
        }
    ] if text else []


def extract_md(file_bytes, filename):
    # Markdown is treated as text so the original readable content is preserved.
    text = file_bytes.decode("utf-8", errors="ignore")
    text = clean_text(text)

    return [
        {
            "filename": filename,
            "page": None,
            "text": text,
        }
    ] if text else []


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(records, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Split each extracted record into overlapping chunks."""
    all_chunks = []
    step = max(1, chunk_size - overlap)

    for record in records:
        text = record["text"]

        for start in range(0, len(text), step):
            chunk = text[start:start + chunk_size].strip()

            if not chunk:
                continue

            all_chunks.append(
                {
                    "filename": record["filename"],
                    "page": record["page"],
                    "text": chunk,
                }
            )

            if start + chunk_size >= len(text):
                break

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_index(chunks):
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "what", "why", "how", "when", "where", "who", "which", "to", "of",
    "in", "on", "for", "with", "from", "this", "that", "these", "those",
    "can", "could", "should", "would", "do", "does", "did", "about",
    "tell", "me", "please", "give", "explain", "describe"
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower())
    return [word for word in words if word not in STOP_WORDS]


def keyword_score(question, text):
    words = important_words(question)
    if not words:
        return 0.0

    text_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower()))
    matches = sum(1 for word in words if word in text_words)

    return matches / len(set(words))


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=DEFAULT_TOP_K):
    if not st.session_state.chunks or st.session_state.index is None:
        return []

    model = load_embedding_model()
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(len(st.session_state.chunks), max(top_k * 3, 10))
    semantic_scores, indices = st.session_state.index.search(query_embedding, search_k)

    candidates = []
    semantic_values = semantic_scores[0]
    candidate_indices = indices[0]

    for semantic_score, index in zip(semantic_values, candidate_indices):
        if index < 0:
            continue

        chunk = st.session_state.chunks[index]
        keyword = keyword_score(question, chunk["text"])

        # 70% semantic relevance + 30% keyword overlap.
        hybrid_score = (0.70 * float(semantic_score)) + (0.30 * keyword)

        candidates.append(
            {
                **chunk,
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword),
                "hybrid_score": float(hybrid_score),
            }
        )

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Groq answer generation
# -----------------------------
def get_groq_model(client):
    """Use a current active text model without hardcoding an obsolete model."""
    models = client.models.list().data

    preferred = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    available = {model.id for model in models}

    for model_id in preferred:
        if model_id in available:
            return model_id

    # Fallback to the first active model that looks like a text-generation model.
    excluded = ("whisper", "guard", "safeguard")
    for model in models:
        if getattr(model, "active", True) and not model.id.startswith(excluded):
            return model.id

    raise RuntimeError("No suitable active Groq text model was found.")


def answer_question(question, retrieved_chunks):
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit Secrets."
        )

    client = Groq(api_key=api_key)
    model_name = get_groq_model(client)

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_text = f"Page {chunk['page']}" if chunk["page"] else "Page not available"
        context_parts.append(
            f"[Source {i}] {chunk['filename']} | {page_text}\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are an AI document assistant.
Answer the user's question ONLY from the provided document context.
Do not use outside knowledge.
If the answer is not present in the context, say:
"I could not find that information in the provided documents."
Be concise and clear. Do not invent facts or sources."""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=700,
    )

    return response.choices[0].message.content, model_name


# -----------------------------
# Google Drive
# -----------------------------
def download_from_google_drive(url):
    """Download a public/shared Drive file or folder into a temporary directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="drive_docs_"))

    if "folders/" in url:
        output_dir = temp_dir / "folder"
        output_dir.mkdir(parents=True, exist_ok=True)

        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=True,
            use_cookies=False,
            remaining_ok=True,
        )

        files = [
            path for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return [(path.name, path.read_bytes()) for path in files]

    output_file = temp_dir / "drive_file"
    downloaded = gdown.download(
        url=url,
        output=str(output_file),
        quiet=True,
        fuzzy=True,
    )

    if not downloaded or not output_file.exists():
        raise RuntimeError(
            "Could not download the Drive file. Make sure the link is publicly accessible."
        )

    # Google Drive direct downloads can lose the original extension.
    # Try to get the filename from the URL when possible.
    name_match = re.search(r"/d/([^/]+)", url)
    filename = name_match.group(1) if name_match else "drive_document"

    # If gdown created a filename with a supported extension, keep it.
    if output_file.suffix.lower() in SUPPORTED_EXTENSIONS:
        filename = output_file.name

    return [(filename, output_file.read_bytes())]


# -----------------------------
# Processing pipeline
# -----------------------------
def process_files(file_items, chunk_size, overlap):
    all_records = []
    processed_names = []

    for filename, file_bytes in file_items:
        try:
            records = extract_document(file_bytes, filename)
            if records:
                all_records.extend(records)
                processed_names.append(filename)
        except Exception as error:
            st.warning(f"Could not read {filename}: {error}")

    if not all_records:
        raise ValueError("No readable text was found in the supplied documents.")

    chunks = chunk_text(
        all_records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    embeddings, index = build_vector_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.index = index
    st.session_state.processed_files = processed_names
    st.session_state.documents_ready = True


# -----------------------------
# Modern Streamlit UI
# -----------------------------
CSS = """
<style>
    .stApp { background: #f7f8fc; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .hero {
        padding: 28px 32px; border-radius: 24px; margin-bottom: 22px;
        background: linear-gradient(135deg, #111827 0%, #263b72 100%);
        color: white; box-shadow: 0 12px 35px rgba(17,24,39,.16);
    }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -.04em; }
    .hero p { margin: 8px 0 0; color: #dbe4ff; font-size: 1rem; }
    .card { background: white; border: 1px solid #e8eaf1; border-radius: 18px; padding: 20px; box-shadow: 0 6px 22px rgba(31,41,55,.06); margin-bottom: 18px; }
    .section-title { font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 5px; }
    .section-subtitle { color: #6b7280; font-size: .9rem; margin-bottom: 15px; }
    .stat { background: #f8f9fc; border: 1px solid #eceef4; border-radius: 14px; padding: 15px; text-align: center; }
    .stat-number { font-size: 1.45rem; font-weight: 750; color: #111827; }
    .stat-label { color: #6b7280; font-size: .78rem; margin-top: 3px; }
    .source-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px; margin: 10px 0; }
    .source-head { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:10px; }
    .source-name { font-weight: 700; color:#111827; }
    .source-meta { color:#6b7280; font-size:.8rem; }
    .score { background:#eef2ff; color:#3730a3; padding:4px 8px; border-radius:999px; font-size:.75rem; font-weight:700; white-space:nowrap; }
    .chat-label { font-size:.78rem; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:.08em; margin: 5px 0 8px; }
    div[data-testid="stFileUploader"] { border-radius: 16px; }
    div[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #b8c0d4; border-radius: 16px; background:#fafbff; padding: 18px; }
    .stButton > button { border-radius: 12px; font-weight: 650; min-height: 42px; }
    div[data-testid="stMetric"] { background:#f8f9fc; border:1px solid #eceef4; padding:12px; border-radius:14px; }
    .stChatMessage { border-radius: 16px; }
    footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📄 AI Document Assistant</h1>
  <p>Ask questions across your documents with semantic search, keyword matching, and grounded AI answers.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("Tune retrieval without changing the document pipeline.")
    chunk_size = st.slider("Chunk size", 300, 1500, DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("Chunk overlap", 0, 300, DEFAULT_OVERLAP, 20)
    top_k = st.slider("Sources per answer", 2, 10, DEFAULT_TOP_K)
    st.divider()
    if st.session_state.documents_ready:
        st.success("● Knowledge base ready")
        st.caption(f"{len(st.session_state.processed_files)} document(s) • {len(st.session_state.chunks)} chunks")
    else:
        st.info("Upload documents to create your knowledge base.")

st.markdown('<div class="card"><div class="section-title">1. Add documents</div><div class="section-subtitle">Upload files or connect a public/shared Google Drive source.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop PDF, DOCX, TXT or Markdown files here",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    help="Supported: PDF, DOCX, TXT and MD",
)

drive_url = st.text_input(
    "Google Drive link",
    placeholder="Paste a public/shared Drive file or folder link…",
    label_visibility="visible",
)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button("✨ Process documents", type="primary", use_container_width=True)
with col2:
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} local file(s) selected")

if process_clicked:
    if overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        file_items = [(uploaded.name, uploaded.getvalue()) for uploaded in (uploaded_files or [])]
        if drive_url.strip():
            with st.spinner("Loading Google Drive files…"):
                try:
                    file_items.extend(download_from_google_drive(drive_url.strip()))
                except Exception as error:
                    st.error(f"Google Drive error: {error}")
        if not file_items:
            st.warning("Add at least one document or a Google Drive link.")
        else:
            with st.spinner("Building your document knowledge base…"):
                try:
                    process_files(file_items, chunk_size, overlap)
                    st.success(f"Knowledge base ready — {len(st.session_state.chunks)} chunks created.")
                except Exception as error:
                    st.error(f"Processing failed: {error}")

if st.session_state.documents_ready:
    st.markdown('<div class="card"><div class="section-title">Knowledge base</div><div class="section-subtitle">Your processed documents and reusable vector index.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    stats = [
        (c1, len(st.session_state.processed_files), "Documents"),
        (c2, len(st.session_state.chunks), "Text chunks"),
        (c3, st.session_state.embeddings.shape[1] if st.session_state.embeddings is not None else 0, "Embedding dimensions"),
    ]
    for col, value, label in stats:
        with col:
            st.markdown(f'<div class="stat"><div class="stat-number">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    for filename in st.session_state.processed_files:
        st.markdown(f"📎 **{filename}**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">2. Ask your documents</div><div class="section-subtitle">Answers are grounded only in the retrieved document context.</div>', unsafe_allow_html=True)
    question = st.chat_input("Ask a question about your documents…")
    st.markdown('</div>', unsafe_allow_html=True)

    if question:
        with st.chat_message("user", avatar="👤"):
            st.write(question)
        with st.spinner("Searching your documents…"):
            retrieved = hybrid_search(question, top_k=top_k)
        if not retrieved:
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("No relevant document chunks were found.")
        else:
            with st.spinner("Generating a grounded answer…"):
                try:
                    answer, model_name = answer_question(question, retrieved)
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(answer)
                        st.caption(f"Generated from retrieved context • {model_name}")
                except Exception as error:
                    st.error(f"Groq error: {error}")
                    answer = None

            if answer:
                st.markdown('<div class="card"><div class="section-title">🔎 Retrieved sources</div><div class="section-subtitle">Evidence used to generate this answer.</div>', unsafe_allow_html=True)
                for number, source in enumerate(retrieved, start=1):
                    page_label = f"Page {source['page']}" if source["page"] is not None else "Page unavailable"
                    with st.expander(f"Source {number}  ·  {source['filename']}  ·  {page_label}"):
                        st.markdown(
                            f'<div class="source-card"><div class="source-head"><div><div class="source-name">📄 {source["filename"]}</div><div class="source-meta">{page_label}</div></div><div class="score">Relevance {source["hybrid_score"]:.3f}</div></div><div>{source["text"]}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Semantic {source['semantic_score']:.3f}  •  Keyword {source['keyword_score']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card" style="text-align:center; padding:38px;"><div style="font-size:2rem">💬</div><div class="section-title">Your document chat will appear here</div><div class="section-subtitle">Upload documents above and process them to start asking questions.</div></div>', unsafe_allow_html=True)
