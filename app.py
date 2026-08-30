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
import json
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.rag.embeddings import EmbeddingModel
from src.rag.vectorstore import ChromaVectorStore

from src.evaluation.eval_orchestrate import evaluate
from src.evaluation.schemas import RetrievedEvidence
from src.evaluation.failure_analysis import summarize_failures

load_dotenv()

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", ROOT / "data" / "chroma")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL", "llama3.2:3b")

# ---------- Load QA pairs ----------
QA_PATH = ROOT / "data" / "golden_qa.jsonl"

@st.cache_data
def load_qa_dataset():
    return [
        json.loads(line)
        for line in QA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

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

            start = time.perf_counter()

            chunks = retriever.retrieve(
                query=question,
                top_k=5,
            )

            elapsed = time.perf_counter() - start

            status.update(
                label=f"Documents retrieved · {elapsed:.1f}s",
                state="complete",
            )

        # Generation
        with st.status(
            "Loading response...",
            expanded=False,
        ) as status:

            start = time.perf_counter()

            answer = generator.generate(
                query=question,
                chunks=chunks,
            )

            elapsed = time.perf_counter() - start

            status.update(
                label=f"Response generated · {elapsed:.1f}s",
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

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []

    qa_data = load_qa_dataset()

    questions = {
        qa["id"]: qa["question"]
        for qa in qa_data
    }

    selected_id = st.selectbox(
        "Select a certified question",
        options=list(questions.keys()),
        format_func=lambda x: questions[x],
    )

    selected_qa = next(
        qa for qa in qa_data
        if qa["id"] == selected_id
    )

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    if st.button("Run Evaluation", type="primary"):

        question = selected_qa["question"]

        # -------------------------
        # Retrieval
        # -------------------------

        with st.status(
            "Surfing through the documents...",
            expanded=False,
        ) as status:

            start = time.perf_counter()

            chunks = retriever.retrieve(
                query=question,
                top_k=5,
            )

            elapsed = time.perf_counter() - start

            status.update(
                label=f"Documents retrieved · {elapsed:.1f}s",
                state="complete",
            )

        retrieved_evidence = [
            RetrievedEvidence(
                chunk_id=str(chunk.chunk_id),
                document_id=chunk.document_id,
                section_id=chunk.section_id,
                text=chunk.text,
                distance=chunk.distance,
            )
            for chunk in chunks
        ]

        # -------------------------
        # Generation
        # -------------------------

        with st.status(
            "Loading response...",
            expanded=False,
        ) as status:

            start = time.perf_counter()

            generated_answer = generator.generate(
                query=question,
                chunks=chunks,
            )

            elapsed = time.perf_counter() - start

            status.update(
                label=f"Response generated · {elapsed:.1f}s",
                state="complete",
            )

        # -------------------------
        # Evaluation
        # -------------------------

        with st.spinner("Evaluating response..."):

            result = evaluate(
                qa=selected_qa,
                generated_answer=generated_answer,
                retrieved_evidence=retrieved_evidence,
            )

        # Store latest result
        st.session_state["evaluation_result"] = result
        st.session_state["evaluation_question"] = question
        st.session_state["evaluation_answer"] = generated_answer
        st.session_state["evaluation_evidence"] = retrieved_evidence

        # Store result for aggregate failure analysis
        st.session_state.evaluation_results.append(result)

    # ========================================================
    # CURRENT EVALUATION
    # ========================================================

    result = st.session_state.get("evaluation_result")

    if result:

        st.divider()

        st.subheader("Question")
        st.write(
            st.session_state["evaluation_question"]
        )

        st.subheader("Generated Answer")
        st.info(
            st.session_state["evaluation_answer"]
        )

        # -------------------------
        # Scores
        # -------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Groundedness",
            f"{result.groundedness:.2f}",
        )

        col2.metric(
            "Completeness",
            f"{result.completeness:.2f}",
        )

        col3.metric(
            "Correctness",
            f"{result.correctness:.2f}",
        )

        col4.metric(
            "Overall",
            f"{result.weighted_score:.2f}",
        )

        # -------------------------
        # Decision
        # -------------------------

        st.subheader("Governance Decision")

        if result.decision == "PASS":
            st.success("✅ PASS")
        elif result.decision == "REVIEW":
            st.warning("⚠️ REVIEW")
        else:
            st.error("❌ FAIL")

        # -------------------------
        # Hard checks
        # -------------------------

        with st.expander("Hard Checks"):

            for check in result.hard_checks:

                if check.passed:
                    st.success(f"✅ {check.name}")
                else:
                    st.error(f"❌ {check.name}")

                if check.details:
                    st.caption(check.details)

        # -------------------------
        # Retrieval metrics
        # -------------------------

        with st.expander("Retrieval Metrics"):

            for metric in result.retrieval_metrics:
                st.write(
                    f"**{metric.name}:** "
                    f"{metric.score:.3f}"
                )

        # -------------------------
        # Current evaluation failures
        # -------------------------

        st.subheader("Failures in Current Evaluation")

        if not result.failures:

            st.success("No failures detected.")

        else:

            for failure in result.failures:

                with st.expander(
                    f"{failure.severity} — {failure.type}"
                ):
                    st.markdown(failure.details)

        # -------------------------
        # Retrieved evidence
        # -------------------------

        with st.expander("Retrieved Evidence"):

            for i, evidence in enumerate(
                st.session_state["evaluation_evidence"],
                1,
            ):

                st.markdown(
                    f"**{i}. {evidence.section_id}**  \n"
                    f"Similarity: `{(1 - evidence.distance):.4f}`"
                )

                st.write(evidence.text)
                st.divider()

    # ========================================================
    # AGGREGATE FAILURE ANALYSIS
    # ========================================================

    st.divider()

    st.subheader("Failure Analysis — Across Evaluations")

    results = st.session_state.evaluation_results

    if not results:

        st.info(
            "Run evaluations to generate failure statistics."
        )

    else:

        summary = summarize_failures(results)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Evaluations",
            len(results),
        )

        col2.metric(
            "Total Failures",
            summary["total"],
        )

        col3.metric(
            "Failure Types",
            len(summary["by_type"]),
        )

        # -------------------------
        # Failures by type
        # -------------------------

        st.write("#### Failures by Type")

        if summary["by_type"]:
            st.bar_chart(summary["by_type"])
        else:
            st.success("No failures detected.")

        # -------------------------
        # Failures by severity
        # -------------------------

        st.write("#### Failures by Severity")

        if summary["by_severity"]:
            st.bar_chart(summary["by_severity"])
        else:
            st.success("No failures detected.")