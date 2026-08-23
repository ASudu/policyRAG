"""
In this file, we implement the Streamlit app interface for the RAG (Retrieval-Augmented Generation) system. The app has two tabs:

Tab One: Chat Interface (Question --> Retrieval --> Answer)
- User asks custom questions about Northstar Analytics policies.
- The behaves like a RAG enabled chatbot similar to ChatGPT, but with scope restricted to the company's policies and procedures.

Tab Two: RAG evaluation (Select certified question --> Retrieval --> Answer --> Evaluation).
- In this, user selects a question from a list of certified questions and the system retrieves relevant policy information and generates an answer.
- Once the AI provides the answer, the internal evaluation pipeline evaluates the answer based on the certified answer and provides a score covering various aspects of evaluation along with feedback to the user.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.rag.embeddings import EmbeddingModel
from src.rag.vectorstore import ChromaVectorStore

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", ROOT / "data" / "chroma")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL", "llama3.2:3b")


# ---------- Configuration ----------

st.set_page_config(
    page_title="Policy RAG",
    page_icon="🤖",
    layout="centered",
)

# ---------- RAG initialization ----------

@st.cache_resource
def load_rag():
    embedder = EmbeddingModel(EMBEDDING_MODEL_NAME)

    vectorstore = ChromaVectorStore(persist_directory=CHROMA_DIR)

    retriever = Retriever(embedder=embedder, vector_store=vectorstore,)

    generator = Generator(GENERATION_MODEL_NAME)

    return retriever, generator


retriever, generator = load_rag()


# ---------- Session state ----------

if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False


# ---------- Header ----------

st.title("🤖 Policy RAG")


# ---------- Tabs ----------

chat_tab, eval_tab = st.tabs(
    ["💬 Chat", "📊 Evaluation"]
)


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    # ---------- First-question state ----------

    if not st.session_state.question_submitted:
        st.markdown(
            """
            <div style="
                text-align: center;
                margin-top: 20vh;
                margin-bottom: 20px;
            ">
                <h2>What can I help you find?</h2>
                <p style="color: #888;">
                    Ask a question about company policies and procedures.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Chat input ----------

    question = st.chat_input(
        "Ask about company policies..."
    )

    if question:
        st.session_state.question_submitted = True

        # User message
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        # Retrieval
        with st.status(
            "Surfing through the documents...",
            expanded=False,
        ) as status:

            chunks = retriever.retrieve(
                query=question,
                top_k=5,
            )

            status.update(
                label="Documents retrieved",
                state="complete",
            )

        # Generation
        with st.status(
            "Loading response...",
            expanded=False,
        ) as status:

            answer = generator.generate(
                query=question,
                chunks=chunks,
            )

            status.update(
                label="Response generated",
                state="complete",
            )

        # AI response
        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):
            st.write(answer)

        # Retrieved evidence
        with st.expander("🔎 Retrieved evidence"):
            for i, chunk in enumerate(chunks, 1):

                st.markdown(
                    f"**{i}. {chunk.policy_id} / "
                    f"{chunk.section_id}**  \n"
                    f"Similarity: `{1 - chunk.distance:.4f}`"
                )

                st.write(chunk.text)
                st.divider()


# ============================================================
# EVALUATION TAB
# ============================================================

with eval_tab:

    st.header("Evaluation Dashboard")

    st.info(
        "🚧 Evaluation UI is under construction."
    )