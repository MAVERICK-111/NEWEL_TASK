# Swiggy Annual Report — RAG Q&A System

A question-answering system built on Swiggy's Annual Report FY 2023-24. You upload the PDF, it indexes the document, and you can ask anything about it in plain English. Answers are pulled directly from the report — nothing is made up.

Built as part of an ML intern assignment. The goal was a working RAG pipeline, not just theory.

---

## What it does

You ask a question like *"What was Swiggy's revenue in FY24?"* and it:

1. Searches through the indexed report using a combination of semantic similarity (FAISS) and keyword matching (BM25)
2. Pulls the most relevant paragraphs
3. Runs those paragraphs through an extractive QA model that finds and returns the exact answer span
4. Shows you the answer along with the source chunks it used, so you can verify

It will not guess. If the answer is not in the document, it says so.

---

## How to run it

There are two ways depending on what you have available.

### Option 1 — Hugging Face Spaces (deployed, no setup)

Hit the url
[Deployed]()

---

### Option 2 — Google Colab

If you want to try it quickly without installing anything locally, use the notebook.

1. Open `swiggy_rag_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** — Runtime → Change runtime type
3. Run all cells top to bottom
4. When prompted, upload the PDF
5. Use the interactive widget at the bottom to ask questions

---

## The technical side

Here is what actually happens when you upload a PDF and ask a question.

**Ingestion**

PyMuPDF reads the PDF page by page and extracts raw text. Before chunking, the text goes through a cleaning pass — ligature characters get fixed (`ﬁ` → `fi`), hyphenated line breaks get rejoined, and stray single characters like page numbers and running headers get removed. The cleaned text is then split into 350-word chunks with a 70-word overlap between adjacent chunks. The overlap exists so that sentences near chunk boundaries are not lost.

**Indexing**

Each chunk is embedded using `all-MiniLM-L6-v2`, a sentence transformer that is about 90 MB and runs at a reasonable speed on CPU. The embeddings are normalized and loaded into a FAISS flat inner-product index, which is equivalent to cosine similarity search. Separately, a BM25 scorer is built over the same chunks for keyword-based retrieval.

**Retrieval**

When you ask a question, it gets embedded the same way. FAISS returns the top candidates by cosine similarity. BM25 scores are computed in parallel. The final ranking blends both: 65% dense score and 35% BM25 score.

The reason for the hybrid is that dense retrieval understands semantic meaning well but can miss exact figures, dates, and proper nouns. BM25 catches those. Annual reports are full of specific numbers that you want to find precisely, so both signals matter.

**Answer extraction**

The top 3 ranked chunks are joined together as context and passed to `deepset/roberta-base-squad2` along with your question. This model does extractive QA — it reads the context and highlights the exact text span that answers the question. It does not generate new sentences. The answer is always a direct quote from the retrieved chunks.

If the model's confidence score falls below 5%, the system treats that as "no clear answer found" and tells you to check the source chunks manually.

---

## Project structure

```
swiggy-rag/
├── app.py                     FastAPI server — upload and query endpoints
├── rag_engine.py              RAG logic — extraction, chunking, FAISS, BM25, QA
├── static/
│   └── index.html             Web UI (single self-contained file, no build step)
├── setup.bat                  Windows one-click setup
├── setup.sh                   Mac/Linux one-click setup
├── requirements.txt           Python dependencies
└── swiggy_rag_colab.ipynb     Colab fallback notebook

swiggy-rag-hf-spaces/
├── app.py                     Gradio app for Hugging Face Spaces
├── requirements.txt
└── README.md                  HF Spaces config + description
```

---

## Why these choices

**PyMuPDF over pdfplumber or pypdf** — fastest extraction, and handles the multi-column layouts that annual reports typically use better than the alternatives.

**all-MiniLM-L6-v2** — good balance of quality and size for this use case. Larger models like `all-mpnet-base-v2` give marginally better recall but are four times the size and noticeably slower on CPU. For a domain-specific single document, MiniLM is more than enough.

**FAISS flat index over approximate indexes** — the chunk count for a single PDF is in the hundreds, not millions. An exact flat index is faster and more accurate at this scale than HNSW or IVF, which are designed for much larger corpora.

**Hybrid BM25 + dense retrieval** — dense retrieval alone sometimes matches the right topic but the wrong paragraph. BM25 keeps retrieval grounded in the actual terms from your query, which is important for financial documents where specific numbers and entity names matter.

**RoBERTa extractive QA instead of a generative LLM** — extractive QA cannot hallucinate because it can only return text that literally exists in the context. The tradeoff is that answers to open-ended or multi-part questions may be incomplete, but for a financial document where precision matters, this is the right call. It also means no Ollama, no API keys, and no separate server process — just `python app.py` and you are running.

---

## Known limitations

- Extractive QA works well for specific factual lookups but is not great for questions that need synthesis across sections, like "summarize the company's growth strategy." The answer will be a snippet rather than a coherent summary.
- Indexing a 300+ page PDF takes 30–90 seconds on CPU. This is a one-time cost per session; subsequent queries are fast.
- On Hugging Face free tier, the Space goes to sleep after 30 minutes of inactivity. The first request after waking up takes around 20–30 seconds.
- The system requires a native PDF with selectable text. Scanned PDFs will extract garbage or nothing. The Swiggy Annual Report from the official IR page is a native PDF and works correctly.

---
