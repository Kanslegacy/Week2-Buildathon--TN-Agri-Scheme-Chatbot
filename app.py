import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag_pipeline.dashboard_stats import get_dashboard_stats

load_dotenv()
VECTORSTORE_DIR = "rag_pipeline/faiss_index"

@st.cache_resource
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for Tamil Nadu farmers. You answer questions about
government Agriculture schemes, and you may also explain general farming/agriculture
knowledge (e.g., units like quintal/acre, common terms, basic farming concepts) to help
the farmer understand the scheme information better.

Follow these rules strictly:

1. For any SCHEME-SPECIFIC facts (funding amount, eligibility, subsidy %, how to apply,
   beneficiaries, deadlines) — use ONLY the "Scheme Context" below. NEVER guess or
   invent scheme details. If the scheme context doesn't contain the answer, clearly say
   "I don't have that specific information in the scheme records" and suggest contacting
   the local Agriculture Officer.

2. For GENERAL knowledge questions not specific to any scheme (e.g., unit conversions,
   common farming terms, basic definitions) — you MAY answer using your own general
   knowledge, but you MUST prefix that part of the answer with "📘 General info:" so the
   farmer knows it's not from the official scheme records.

3. Never blend the two without labeling — always be clear about what is official scheme
   data versus general knowledge.

Scheme Context:
{context}

Question: {question}

Answer simply and clearly, following the rules above:
""")

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"{d.page_content}\n(Source: {d.metadata.get('scheme_name')})" for d in docs
        )

    # NEW: a small sub-chain that fetches and formats context only
    retrieve_context = retriever | format_docs

    # NEW: the main answer-generating chain
    answer_chain = (
        {"context": retrieve_context, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Return BOTH the retriever and the answer chain, so the UI can show sources separately
    return retriever, answer_chain

st.set_page_config(page_title="TN Agri AI Assistant", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .header-banner {
        background: linear-gradient(135deg, #5D4037 0%, #C9A227 100%);
        padding: 30px 28px; border-radius: 16px; margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(93, 64, 55, 0.3);
    }
    .header-banner h1 { color: white; font-size: 30px; margin: 0; }
    .header-banner p { color: #F5EEDC; margin-top: 6px; font-size: 15px; }
    div[data-testid="stMetric"] {
        background-color: #EFE7D8; border-left: 5px solid #66BB6A;
        border-radius: 10px; padding: 12px 16px;
    }
    .stButton button {
        background-color: #EFE7D8; border: 1px solid #C9A227;
        border-radius: 10px; color: #3E2C1C; text-align: left;
    }
    .stButton button:hover { background-color: #C9A227; color: white; }
    [data-testid="stChatMessage"] { border-radius: 14px; padding: 4px 8px; }
</style>
<div class="header-banner">
    <h1>🌾 Tamil Nadu Agri AI Assistant</h1>
    <p>Your AI guide to Agriculture &amp; Farmers Welfare Department schemes — ask in plain language, get clear answers.</p>
</div>
""", unsafe_allow_html=True)

retriever, chain = load_chain()

# ---------- Session state must be initialized BEFORE sidebar uses it ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------- Sidebar: navigation + quick questions ----------
with st.sidebar:
    st.markdown("### 💡 Try asking")
    examples = [
        "What schemes help with seed subsidy?",
        "Tell me about training programs for farmers",
        "How do I apply for crop loan support?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

# ---------- Quick stats strip (always visible) ----------
stats = get_dashboard_stats()
col1, col2, col3 = st.columns(3)
col1.metric("🌱 Total Schemes Covered", stats["total_schemes"])
col2.metric("🏷️ Benefit Categories", len(stats["benefit_types"]))
col3.metric("👨‍🌾 Beneficiary Categories", len(stats["beneficiary_types"]))

st.markdown("---")

# ---------- Chat ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_question = st.chat_input("Ask about a farming scheme...")
user_question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if user_question and user_question.strip():
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching scheme database..."):
            answer = chain.invoke(user_question)
            st.markdown(answer)

            retrieved_docs = retriever.invoke(user_question)
            scheme_names = sorted(set(
                d.metadata.get("scheme_name", "Unknown") for d in retrieved_docs
            ))
            if scheme_names:
                with st.expander("📄 Source schemes referenced"):
                    for name in scheme_names:
                        st.markdown(f"- {name}")

    st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown("""
<hr style="margin-top: 40px; border-color: #C9A227;">
<p style="text-align: center; color: #8A7A5C; font-size: 13px;">
🌾 Powered by RAG · LangChain · OpenAI · FAISS — Built for Tamil Nadu Farmers Welfare
</p>
""", unsafe_allow_html=True)