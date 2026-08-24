# Policy RAG Evaluation Framework

A lightweight, multi-layer evaluation framework for Retrieval-Augmented Generation (RAG) systems operating over HR policies, compliance rules, and standard operating procedures.

The project uses a synthetic policy corpus and SME-reviewed QA dataset to evaluate whether a RAG system retrieves the right evidence and produces answers that are grounded, complete, and correct.

---

## Overview

A typical RAG evaluation pipeline often reduces quality to a single LLM-as-a-judge score. That is risky for policy-oriented applications. 

An answer can sound semantically correct while:

- citing the wrong policy
- changing a numerical requirement
- changing a date
- reversing the meaning of a policy
- making unsupported claims
- omitting required information
- retrieving irrelevant evidence
- contradicting authoritative policy information

This project therefore combines deterministic checks, retrieval metrics, and semantic evaluation into a single evaluation and governance pipeline.

```text
Question
    |
    v
Retriever
    |
    +--------------------> Retrieved Evidence
    |
    v
Generator
    |
    v
Generated Answer
    |
    +----> Hard Checks
    |        - Policy references
    |        - Numerical values
    |        - Dates
    |        - Polarity
    |
    +----> Retrieval Metrics
    |        - Evidence recall
    |        - Context relevance
    |
    +----> Groundedness
    |        - Claim extraction
    |        - Claim-to-evidence evaluation
    |
    +----> Completeness
    |        - Question obligations
    |        - Evidence-bounded evaluation
    |
    +----> Correctness
    |        - Certified-vs-generated claims
    |        - Contradiction detection
    |
    v
Weighted Score
    |
    v
Failure Classification
    |
    v
PASS / REVIEW / FAIL
````

---

## Key Features

### RAG Pipeline

The project implements a complete local RAG pipeline:

```text
Documents
   ↓
Document Loader
   ↓
Chunker
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Generator
   ↓
Answer
```

Components include:

* Markdown policy document loading
* Configurable text chunking with overlap
* Ollama embeddings
* ChromaDB vector storage
* Semantic retrieval
* Ollama-based answer generation

---

# Evaluation Framework

## 1. Hard Checks

Deterministic checks identify high-risk factual mismatches.

Current checks include:

* Policy/reference checks
* Numerical checks
* Date checks
* Polarity checks

These checks are deliberately deterministic rather than relying entirely
on an LLM judge.

For example, if the certified policy says:

```text
Employees must submit vacation requests at least 10 business days before the first requested day.
```

and the generated answer says:

```text
Employees must submit vacation requests at least 5 business days before the first requested day.
```

the numerical check can identify the mismatch directly.

A failed hard check can force a `FAIL` decision regardless of semantic score.

---

## 2. Retrieval Evaluation

Two retrieval-level metrics are implemented.

### Evidence Recall

Measures whether authoritative/certified evidence was retrieved.

```text
Evidence Recall =
|certified evidence ∩ retrieved evidence|
------------------------------------------
        |certified evidence|
```

### Context Relevance

Uses the vector-store distances to estimate how relevant the retrieved
context is to the query.

---

## 3. Groundedness

Generated answers are decomposed into atomic factual claims.

Each claim is evaluated against retrieved evidence:

| Score | Meaning             |
| ----: | ------------------- |
|   1.0 | Fully supported     |
|   0.5 | Partially supported |
|   0.0 | Unsupported         |

This identifies hallucinated or insufficiently supported claims.

---

## 4. Completeness

The framework models the information required to answer a question using
answer obligations.

Obligations are filtered using authoritative evidence so that the evaluator
does not penalize the system for information that was not actually
answerable from the available evidence.

Each answerable obligation receives:

| Score | Meaning            |
| ----: | ------------------ |
|   1.0 | Answered           |
|   0.5 | Partially answered |
|   0.0 | Unanswered         |

---

## 5. Correctness

Generated claims are compared against certified claims using a semantic
judge.

Each generated claim is classified as:

```text
SUPPORTED_BY_CERTIFIED
NOT_COVERED
CONTRADICTS_CERTIFIED
```

Scoring:

```text
SUPPORTED_BY_CERTIFIED → 1.0
NOT_COVERED             → 0.5
CONTRADICTS_CERTIFIED   → 0.0
```

The evaluator explicitly checks for factual contradictions and polarity
reversals.

---

# Weighted Scoring

The current evaluation weights are:

| Dimension    | Weight |
| ------------ | -----: |
| Groundedness |    40% |
| Completeness |    30% |
| Correctness  |    30% |

Overall score:

```text
Overall =
    0.40 × Groundedness
  + 0.30 × Completeness
  + 0.30 × Correctness
```

The weights are centralized in:

```text
src/evaluation/scoring.py
```

and can be calibrated as the evaluation dataset grows.

---

# Failure Taxonomy

Evaluation failures are mapped into controlled categories.

Current categories include:

```text
POLICY_VIOLATION
NUMERICAL_MISMATCH
DATE_MISMATCH
POLARITY_MISMATCH

RETRIEVAL_FAILURE
LOW_CONTEXT_RELEVANCE

UNGROUNDED_CLAIM
PARTIALLY_GROUNDED_CLAIM

INCOMPLETE_ANSWER
PARTIALLY_COMPLETE_ANSWER

CONTRADICTION
UNCOVERED_CLAIM
```

Failures also carry severity levels such as:

```text
CRITICAL
MAJOR
MINOR
```

This allows the system to provide more useful feedback than a single
numeric score.

---

# Governance Decision

The current governance engine uses the following thresholds:

```text
PASS_THRESHOLD   = 0.85
REVIEW_THRESHOLD = 0.60
```

Decision logic:

```text
IF any hard check fails
    → FAIL

ELSE IF score >= 0.85
    → PASS

ELSE IF score >= 0.60
    → REVIEW

ELSE
    → FAIL
```

A semantic score therefore cannot override a deterministic policy violation.

---

# Dataset

The project currently uses a synthetic HR policy corpus designed to
simulate an enterprise policy environment.

Dataset characteristics:

* 12 policy documents
* 92 policy sections
* Synthetic policy content
* Certified QA pairs
* SME-style review metadata
* Certified answers
* Certified claims
* Answer obligations
* Authoritative evidence
* Positive and negative evaluation examples

A QA record contains information such as:

```json
{
  "id": "QA-001",
  "question": "How much advance notice is required for vacation?",
  "ai_response": "Vacation requests should be submitted 10 business days ahead.",
  "retrieved_chunks": [
    {
      "chunk_id": "LV-003",
      "document_id": "leave_policy",
      "section_id": "LV-003",
      "text": "...",
      "relevance": "high"
    }
  ],
  "sme_review": {
    "decision": "THUMBS_UP",
    "primary_failure": null,
    "feedback": null
  },
  "certified_answer": "...",
  "certified_claims": [...],
  "answer_obligations": [...],
  "authoritative_evidence": ["LV-003"]
}
```

The dataset is intentionally structured to support evaluation beyond
simple answer-vs-reference string matching.

---

# Models

The project uses Ollama for local inference.

Example configuration:

```text
Embedding model:  nomic-embed-text
Generation model: llama3.2:3b
Evaluation model:  gemma3:12b
```

Models are configurable through environment variables.

Example:

```env
EMBEDDING_MODEL=nomic-embed-text
GENERATION_MODEL=llama3.2:3b
EVALUATION_MODEL=gemma3:12b
CHROMA_DIR=data/chroma
```

---

# Project Structure

```text
policyRAG/
│
├── data/
│   ├── documents/
│   │   └── *.md
│   │
│   ├── qa/
│   │   └── qa_pairs.jsonl
│   │
│   └── chroma/
│
├── scripts/
│   ├── run_evaluation.py
│   └── validation/
│       ├── validate_chunking.py
│       └── validate_embeddings.py
│
├── src/
│   ├── rag/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   └── evaluation/
│       ├── schemas.py
│       ├── hard_checks.py
│       ├── claims.py
│       ├── groundedness.py
│       ├── completeness.py
│       ├── correctness.py
│       ├── retrieval_metrics.py
│       ├── scoring.py
│       ├── failures.py
│       ├── failure_analysis.py
│       ├── decision.py
│       └── eval_orchestrate.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   ├── test_completeness.py
│   ├── test_correctness.py
│   └── ...
│
├── app.py
├── pyproject.toml
├── .env
└── README.md
```

---

# Installation

The project uses `uv` for Python environment and dependency management.

Clone the repository and install dependencies:

```bash
uv sync
```

Make sure Ollama is installed and running.

Pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama pull gemma3:12b
```

Configure the environment:

```env
EMBEDDING_MODEL=nomic-embed-text
GENERATION_MODEL=llama3.2:3b
EVALUATION_MODEL=gemma3:12b
CHROMA_DIR=data/chroma
```

---

# Running the Application

Start the Streamlit application:

```bash
uv run streamlit run app.py
```

The application contains two tabs.

## Chat

The Chat tab provides a simple ChatGPT-style interface:

```text
User Question
     ↓
Surfing through the documents
     ↓
Retrieved Evidence
     ↓
Loading response
     ↓
Generated Answer
```

Retrieved chunks can also be inspected directly in the interface.

## Evaluation

The Evaluation tab allows a certified QA question to be selected and
evaluated end-to-end:

```text
Certified Question
      ↓
Retrieval
      ↓
Generation
      ↓
Hard Checks
      ↓
Retrieval Metrics
      ↓
Groundedness
      ↓
Completeness
      ↓
Correctness
      ↓
Weighted Score
      ↓
PASS / REVIEW / FAIL
```

The dashboard displays:

* Groundedness
* Completeness
* Correctness
* Overall score
* Governance decision
* Hard-check results
* Retrieval metrics
* Retrieved evidence
* Failure analysis

Failure statistics can also be aggregated across evaluations performed
during the current Streamlit session.

---

# Running a Single Evaluation

An end-to-end evaluation can be run using:

```bash
uv run python scripts/run_evaluation.py
```

Example output:

```text
RAG pipeline initialized.

Question:
How much advance notice is required for vacation?

Generated answer:
The required advance notice for vacation is at least 10 business days
before the first requested day...

==================================================
EVALUATION RESULT
==================================================
Question:      How much advance notice is required for vacation?
Groundedness:  1.000
Completeness:  1.000
Correctness:   0.667
Overall score: 0.900
Decision:      PASS

Failures:
- [MINOR] UNCOVERED_CLAIM:
  2 generated claim(s) are not covered by certified claims.
```

This illustrates an important property of the framework: an answer can
receive a `PASS` while still exposing lower-level evaluation findings.

---

# Running Tests

Run the test suite:

```bash
uv run pytest
```

Individual test modules can also be run:

```bash
uv run pytest tests/test_chunker.py
uv run pytest tests/test_embeddings.py
uv run pytest tests/test_vectorstore.py
uv run pytest tests/test_completeness.py
uv run pytest tests/test_correctness.py
```

---

# Validation

Validation scripts are separate from unit tests.

Unit tests verify implementation behavior and edge cases.

Validation scripts verify that the complete corpus can pass through a
pipeline stage and satisfy expected corpus-level invariants.

For example:

```bash
uv run python scripts/validation/validate_chunking.py
```

and:

```bash
uv run python scripts/validation/validate_embeddings.py
```

The current corpus validation checks include:

* Expected policy count
* Expected section count
* Chunk generation
* Embedding generation
* Embedding dimensionality
* Metadata integrity
* Numeric embedding sanity
* Deterministic embedding behavior

---

# Caching

The evaluation framework separates relatively static certified information
from dynamically generated information.

Certified claims and question obligations can be cached because they are
properties of the evaluation dataset rather than properties of an individual
RAG response.

This avoids repeatedly invoking the evaluation model for information that
does not change between runs.

Conceptually:

```text
                    ┌──────────────────────┐
                    │ Certified QA Dataset │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ↓                      ↓
             Certified Claims       Question Obligations
                    │                      │
                    └──────────┬───────────┘
                               │
                             Cache
                               │
                               ↓
Question → RAG → Generated Answer → Evaluation
```

---

# Design Principles

## Deterministic checks before semantic judgment

Policy-critical information such as dates, numbers, and polarity should
not rely exclusively on an LLM judge.

## Evidence-bounded evaluation

The evaluator should distinguish between:

* information that should have been answered
* information that could actually be answered from authoritative evidence

This reduces false penalties for unavailable information.

## Separate retrieval from generation evaluation

A poor answer may be caused by:

1. retrieval failure
2. generation failure
3. both

The framework therefore evaluates retrieval separately from the generated
answer.

## Failure taxonomy over a single score

A score such as `0.72` does not explain why an answer failed.

The failure taxonomy provides actionable categories such as:

```text
NUMERICAL_MISMATCH
RETRIEVAL_FAILURE
UNGROUNDED_CLAIM
INCOMPLETE_ANSWER
CONTRADICTION
```

---

# Limitations

This project is a **case-study evaluation framework**, not a production
compliance or governance platform.

### Synthetic data

The current policy corpus and QA dataset are synthetic.

Real enterprise policies are likely to contain substantially more ambiguity,
cross-references, exceptions, versioning, and conflicting documents.

### LLM-based evaluation

Groundedness and correctness rely partly on an LLM-based semantic judge.
The evaluator can therefore make mistakes.

### Certified-answer coverage

Correctness currently compares generated claims against certified claims.
A generated claim can therefore be technically correct but classified as
`NOT_COVERED` if the SME-certified answer did not include that information.

### Initial thresholds

PASS/REVIEW/FAIL thresholds are currently manually selected and require
empirical calibration against human/SME judgments.

### Local inference

The current implementation uses local Ollama models. Results and latency
can vary substantially depending on model size and hardware.

### Single-turn evaluation

The framework currently focuses on individual question-answer pairs.
Multi-turn conversations and conversational memory are outside the current
scope.

### No production monitoring

The project does not currently provide production telemetry, alerting,
model drift monitoring, access control, or audit-log infrastructure.

---

# Current Scope

### In scope

* Policy document ingestion
* Chunking
* Embedding generation
* Vector retrieval
* RAG generation
* Deterministic policy checks
* Retrieval evaluation
* Claim extraction
* Groundedness
* Evidence-bounded completeness
* Correctness
* Contradiction detection
* Weighted scoring
* Failure taxonomy
* PASS/REVIEW/FAIL governance
* Streamlit demonstration interface
* Evaluation dashboard
* Failure analysis

### Out of scope

* Production deployment
* Multi-user authentication
* Multi-turn conversational memory
* Real enterprise policy data
* Automated SME review
* Production monitoring
* Model fine-tuning
* Distributed vector infrastructure
* Formal regulatory certification

---

# Future Work

## 1. SME Alignment

Compare automated evaluation decisions against SME judgments.

Potential metrics:

```text
Agreement rate
Precision / Recall
F1
Cohen's Kappa
```

This would allow the evaluator itself to be evaluated.

## 2. Threshold Calibration

Instead of manually selecting:

```text
PASS  >= 0.85
REVIEW >= 0.60
```

learn thresholds from SME-reviewed examples and optimize for the desired
tradeoff between false passes and false failures.

## 3. Better Claim-to-Evidence Matching

Improve semantic matching between generated claims and authoritative
evidence using:

* entailment models
* structured claim representations
* evidence ranking
* domain-specific NLI

## 4. Policy Versioning

Support policies that change over time:

```text
Policy v1 → effective Jan 2025
Policy v2 → effective Jun 2025
Policy v3 → effective Jan 2026
```

Evaluation should use the policy version applicable to the question.

## 5. Cross-Document Reasoning

Many real policy questions require combining multiple policies.

Future evaluation should explicitly test whether the system:

* retrieves all required documents
* resolves conflicting policies
* correctly prioritizes authoritative sources

## 6. Production Evaluation

Extend the framework toward continuous evaluation:

```text
Production Queries
       ↓
Sample
       ↓
RAG Evaluation
       ↓
Failure Classification
       ↓
Monitoring Dashboard
       ↓
Alerts / Human Review
```

## 7. Human-in-the-Loop Review

Introduce a review workflow where `REVIEW` cases can be inspected and
approved or rejected by an SME.

---

# Project Status

The core RAG and evaluation pipeline is implemented.

```text
RAG Pipeline
[x] Document loading
[x] Chunking
[x] Embeddings
[x] ChromaDB
[x] Retrieval
[x] Generation

Core Evaluation
[x] Claim extraction
[x] Claim-to-evidence evaluation
[x] Groundedness
[x] Question obligation extraction
[x] Evidence-bounded completeness
[x] Authoritative evidence
[x] Correctness
[x] Contradiction detection
[x] Weighted scoring
[x] Failure classification

Governance + UI
[x] PASS / REVIEW / FAIL
[x] Threshold calibration
[x] Streamlit Chat
[x] Evaluation Dashboard
[x] Failure Analysis
[x] Basic tests
[x] Full evaluation
[ ] SME alignment metrics
```

The remaining SME-alignment work is intentionally deferred until the evaluation framework has been validated against the current synthetic dataset.