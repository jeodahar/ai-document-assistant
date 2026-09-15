# 📄 AI Document Assistant

A simple Streamlit RAG-style document assistant.

It supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Public/shared Google Drive files and folders
- Text extraction
- Overlapping text chunks
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid retrieval
- Groq-powered answers
- Retrieved source display
- Streamlit session-state reuse of embeddings

## Project files

```text
document-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add your Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never put the real API key inside `app.py` or commit `secrets.toml` to GitHub.

For Streamlit Community Cloud, open your app's **Settings → Secrets** and add the same value.

## 3. Run locally

```bash
streamlit run app.py
```

## 4. How the pipeline works

```text
PDF/DOCX/TXT/MD or Google Drive
             ↓
        Text extraction
             ↓
      Overlapping chunks
             ↓
 Sentence Transformers embeddings
             ↓
          FAISS index
             ↓
        User question
             ↓
 Semantic search + keyword search
             ↓
        Hybrid ranking
             ↓
       Relevant chunks
             ↓
           Groq
             ↓
      Answer + sources
```

Embeddings are created when the documents are processed, not every time a question is asked. The embeddings, FAISS index, and chunk metadata are kept in Streamlit session state.

## 5. Google Drive

Paste a Google Drive file or folder link.

The Drive item must be accessible to the app (for example, a public/shared link that can be downloaded without interactive Google login).

Supported downloaded files:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

Drive files go through the same extraction, chunking, embedding and hybrid-search pipeline as local uploads.

## 6. GitHub

Create a new GitHub repository and upload:

- `app.py`
- `requirements.txt`
- `README.md`

Do **not** upload:

```text
.streamlit/secrets.toml
```

## 7. Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Create a new app.
3. Select your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy.
6. Open the app's Settings → Secrets.
7. Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

8. Save and reboot the app if requested.

## Notes

- PDF page numbers are preserved because PDFs expose individual pages.
- DOCX, TXT and MD files normally do not provide reliable page numbers, so their page value is shown as unavailable.
- The embedding model is `all-MiniLM-L6-v2`.
- The app uses cosine-style similarity through normalized embeddings and FAISS inner-product search.
- Hybrid ranking uses 70% semantic similarity and 30% keyword overlap.
- The Groq model is selected from currently active Groq models, preferring a current production text model instead of depending on one hard-coded model ID.
