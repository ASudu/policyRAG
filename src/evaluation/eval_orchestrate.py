# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
In this module, we combine all the evaluation components into a single orchestrated evaluation process. This includes hard checks, groundedness, completeness and correctness evaluation given a QA pair.The orchestrator manages the flow of data between these components and ensures that the evaluation is performed in a structured manner.
"""

from concurrent.futures import ThreadPoolExecutor
from src.evaluation.schemas import EvaluationResult
from src.evaluation.hard_checks import evaluate_hard_checks
from src.evaluation.claims import extract_claims
from src.evaluation.correctness import evaluate_correctness, correctness_score
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.groundedness import evaluate_claim_grounding, groundedness_score
from src.evaluation.completeness import get_answerable_obligations, evaluate_obligations, completeness_score
from src.evaluation.scoring import weighted_score
from src.evaluation.failures import classify_failures
from src.evaluation.decision import make_decision


def evaluate(qa, generated_answer, retrieved_evidence,) -> EvaluationResult:

    question = qa["question"]
    # --------------------------------------------------
    # Hard checks
    # --------------------------------------------------

    hard_checks = evaluate_hard_checks(
        qa["certified_answer"],
        generated_answer,
    )

    # --------------------------------------------------
    # Ground truth artifacts — PRECOMPUTED, never regenerate
    # --------------------------------------------------

    certified_claims = qa["certified_claims"]
    answer_obligations = qa["answer_obligations"]
    authoritative_evidence = qa["authoritative_evidence"]

    # --------------------------------------------------
    # Runtime artifacts
    # --------------------------------------------------

    generated_claims = extract_claims(question, generated_answer)

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retrieval_metrics = evaluate_retrieval(
        authoritative_evidence,
        retrieved_evidence,
    )

    eligible_obligations = get_answerable_obligations(
        answer_obligations,
        authoritative_evidence,
        certified_claims,
    )

    # These evaluations are independent LLM calls once their inputs are ready.
    with ThreadPoolExecutor(max_workers=3) as executor:
        correctness_future = executor.submit(
            evaluate_correctness,
            question,
            certified_claims,
            generated_claims,
        )
        groundedness_future = executor.submit(
            evaluate_claim_grounding,
            generated_claims,
            retrieved_evidence,
        )
        completeness_future = executor.submit(
            evaluate_obligations,
            generated_answer,
            eligible_obligations,
        )

        correctness_results = correctness_future.result()
        claim_evaluations = groundedness_future.result()
        completeness_results = completeness_future.result()

    correctness = correctness_score(correctness_results)
    groundedness = groundedness_score(claim_evaluations)
    completeness = completeness_score(completeness_results)

    # --------------------------------------------------
    # Final scoring
    # --------------------------------------------------

    score = weighted_score(
        groundedness,
        completeness,
        correctness,
    )

    failures = classify_failures(
        hard_checks=hard_checks,
        retrieval_metrics=retrieval_metrics,
        groundedness_score=groundedness,
        completeness_score=completeness,
        correctness_evaluations=correctness_results,
        groundedness_results=claim_evaluations,
        completeness_results=completeness_results,
    )

    # hard_pass = all(
    #     check.passed for check in hard_checks
    # )
    contradictions = any(check.label=="CONTRADICTS_CERTIFIED" for check in correctness_results)

    decision = make_decision(score, contradictions)

    return EvaluationResult(
        question=qa["question"],
        certified_answer=qa["certified_answer"],
        generated_answer=generated_answer,

        retrieval_metrics=retrieval_metrics,

        groundedness=groundedness,
        completeness=completeness,
        correctness=correctness,

        hard_checks=hard_checks,
        overall_hard_pass=not contradictions,

        claim_evaluations=claim_evaluations,
        obligation_evaluations=completeness_results,
        correctness_evaluations=correctness_results,

        weighted_score=score,
        failures=failures,
        decision=decision,
    )