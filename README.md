# 🌾 Tamil Nadu Agriculture Scheme AI Assistant

An AI-powered chatbot that helps farmers get clear, accurate answers about Tamil Nadu Government Agriculture & Farmers Welfare Department schemes — built using **Retrieval-Augmented Generation (RAG)**.

> Ask a question in plain language → the app retrieves the most relevant official scheme(s) → an AI model answers using that retrieved information.

🔗 **Official data source:** [tn.gov.in – Agriculture Schemes](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==)

---

## 1. What This Project Does

A farmer can ask things like:
- *"What schemes help with seed subsidy?"*
- *"How do I apply for crop loan support?"*
- *"Tell me about training programs for farmers"*

The chatbot:
1. Searches Tamil Nadu's official Agriculture scheme data for the most relevant scheme(s).
2. Generates a simple, human-friendly answer using **only that official data** for scheme-specific facts (funding, eligibility, how to apply).
3. Optionally explains general farming knowledge (e.g., "what is a quintal?") — but **clearly labels** it as general info, separate from official scheme data.
4. Shows **which scheme(s)** were used as the source for every answer, for transparency and trust.

---

## 2. Why RAG (and not just a regular chatbot)?

A plain LLM (like ChatGPT) doesn't know the specific details of Tamil Nadu's schemes — and worse, it might **confidently guess wrong** numbers (a serious risk for something like subsidy amounts).

**RAG (Retrieval-Augmented Generation)** solves this with a simple idea:

> Don't ask the AI to *recall* facts from memory. *Give it the facts*, fresh, every time — then ask it to answer using only what you gave it.

```
Farmer's Question
      ↓
Convert question into a vector (embedding)
      ↓
Search FAISS vector store → find top 3 most relevant schemes
      ↓
Insert those schemes' text into the AI's prompt as "context"
      ↓
LLM generates an answer using ONLY that context
      ↓
Answer + source scheme names shown to the farmer
```

This is why the chatbot can correctly answer "What is the funding for maize farmer training?" with the exact figure (`Rs.300 per farmer for 2 days`) instead of guessing.

---

## 3. Tech Stack

| Tool | Role in this project |
|---|---|
| **Python** | Core language for scraping, data processing, and the RAG pipeline |
| **BeautifulSoup + Requests** | Scrape scheme data from the official TN Government website |
| **LangChain (LCEL)** | Framework to build the retrieval → prompt → LLM pipeline as a "chain" |
| **OpenAI Embeddings** (`text-embedding-3-small`) | Converts scheme text & questions into vectors for semantic search |
| **FAISS** | Local vector database — stores embeddings, performs similarity search |
| **OpenAI LLM** (`gpt-4o-mini`) | Lightweight, low-cost chat model that generates the final answer |
| **LangSmith** | Traces every run (what was retrieved, what prompt was sent, token usage) — used for debugging and analytics |
| **Streamlit** | Builds the web-based chat UI, dashboard stats, and session handling |

---

## 4. Project Architecture

```
tn-agri-scheme-chatbot/
│
├── scraper/
│   ├── fetch_page.py          # Step 1: Scrape the scheme LIST page
│   ├── scrape_details.py      # Step 2: Visit each scheme's detail page & extract data
│   ├── scheme_list.json       # Output: 54 scheme names + URLs
│   └── schemes_data.json      # Output: full scraped data for all 54 schemes
│
├── rag_pipeline/
│   ├── prepare_documents.py   # Step 3: Clean & format scraped data into text documents
│   ├── build_vectorstore.py   # Step 4: Generate embeddings & build FAISS index
│   ├── qa_chain.py            # Step 5: Core RAG chain (retriever + prompt + LLM)
│   ├── dashboard_stats.py     # Computes quick stats shown on the UI
│   ├── cleaned_documents.json # Output: cleaned text + metadata, ready for embedding
│   └── faiss_index/           # Output: saved FAISS vector store (local vector DB)
│
├── app.py                     # Streamlit web app (UI, chat, dashboard, source citations)
├── .streamlit/config.toml     # App theme (Tamil Nadu agriculture-inspired color palette)
├── .env                       # API keys (OpenAI, LangSmith) — not committed to GitHub
└── requirements.txt           # Python dependencies
```

---

## 5. Step-by-Step Build Process

### Step 1 — Environment Setup
Created a Python virtual environment (`venv`) and installed all required libraries via `requirements.txt`. API keys (OpenAI, LangSmith) are stored securely in a `.env` file, never hardcoded or committed to GitHub.

### Step 2 — Data Collection (Web Scraping)
The official scheme page lists schemes as simple links (`<ul id="content"><li><a href="...">`). The scraper:
1. Fetches the list page → extracts all scheme names + detail-page URLs (`fetch_page.py`).
2. Visits each of the 54 detail pages → extracts structured fields (Funding Pattern, Eligibility, Description, How to Avail, etc.) using BeautifulSoup, with a polite delay between requests (`scrape_details.py`).

### Step 3 — Data Cleaning
Raw scraped text contained messy line breaks (`\r\n`) and placeholder junk values (`"-"`, `"na"`). `prepare_documents.py`:
- Cleans and normalizes text.
- Filters out empty/junk fields.
- Formats each scheme into one consistent, readable paragraph.
- Attaches metadata (`scheme_name`, `source_url`) to each document — required for source citations later.

### Step 4 — Embeddings & Vector Store
`build_vectorstore.py` converts each cleaned scheme document into a vector using OpenAI's lightweight `text-embedding-3-small` model, then stores all vectors in a **FAISS** index — saved locally so embeddings are generated only once, not on every run.

### Step 5 — The RAG Chain (LangChain LCEL)
`qa_chain.py` builds the core retrieval-and-answer pipeline using LangChain Expression Language (`|` pipe syntax):

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

- **Retriever** → fetches the top 3 most relevant scheme documents for a question.
- **Prompt** → a **Hybrid RAG** prompt:
  - Scheme-specific facts → must come *only* from retrieved context (no guessing).
  - General knowledge (e.g., unit conversions) → allowed, but clearly labeled `📘 General info:` so it's never confused with official data.
- **LLM** (`gpt-4o-mini`, `temperature=0`) → generates a factual, consistent answer.

### Step 6 — Observability with LangSmith
Enabled via environment variables (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`). Every chat interaction is automatically traced, showing exactly which documents were retrieved and what prompt was sent to the LLM — essential for debugging RAG quality issues.

### Step 7 — Streamlit Web App
`app.py` brings everything together:
- **Caching** (`@st.cache_resource`) — loads the FAISS index, embeddings, and LLM once, not on every message.
- **Chat memory** (`st.session_state`) — keeps the conversation visible across messages.
- **Custom theme** — an agriculture-inspired color palette (soil brown, wheat gold, leaf green) defined in `.streamlit/config.toml` and custom CSS.
- **Quick stats strip** — shows total schemes covered, benefit categories, and beneficiary categories using data already scraped (no extra API calls).
- **Source citations** — for every scheme-based answer, a collapsible section shows exactly which scheme(s) were used, so the farmer can verify the source.
- **Sidebar quick-questions** — example prompts for first-time users.

---

## 6. Key RAG Concepts Demonstrated

- **Retrieval before generation** — the model never answers from memory for factual scheme details; it always retrieves first.
- **Hybrid RAG with explicit labeling** — balances helpfulness (answering general questions) with safety (never silently guessing official facts).
- **Source transparency** — every answer can be traced back to the exact scheme(s) used.
- **Observability** — LangSmith tracing turns the RAG pipeline from a "black box" into something fully inspectable.
- **Cost-conscious design** — lightweight embedding model (`text-embedding-3-small`) + lightweight chat model (`gpt-4o-mini`) + local FAISS (no per-query vector DB costs) + caching to avoid redundant API calls.

---

## 7. How to Run Locally

```bash
# 1. Clone the repo and enter the project folder
git clone <your-repo-url>
cd tn-agri-scheme-chatbot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys in a .env file
OPENAI_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=tn-agri-scheme-chatbot

# 5. (One-time) Build the data pipeline
python scraper/fetch_page.py
python scraper/scrape_details.py
python rag_pipeline/prepare_documents.py
python rag_pipeline/build_vectorstore.py

# 6. Run the app
streamlit run app.py
```

---

## 8. Scope & Limitations

- Covers only the **Agriculture – Farmers Welfare Department** schemes (54 schemes) from the official TN Government site, identified by `dep_id=Mg==`.
- Does not currently support multi-turn conversational memory (each question is treated independently by the LLM, though the UI displays full chat history).
- Data is scraped once and stored locally — it does not auto-refresh if the government website is updated. Re-running the scraper + vector store build steps refreshes the data.

---

## 9. Possible Future Enhancements

- Add conversational memory so follow-up questions don't need to repeat the scheme name.
- Extend scraping to cover additional departments (the site structure supports this via different `dep_id` values).
- Add a Tamil-language toggle for wider farmer accessibility.
- Deploy publicly via Streamlit Community Cloud.

---

*Built as a hands-on learning project to understand the full RAG application lifecycle — from data scraping to a deployed, user-facing AI assistant.*
