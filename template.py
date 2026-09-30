"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

# Shared thresholds (lecture: pass >= 0.5, severe < 0.3, regression drop > 0.05).
PASS_THRESHOLD: float = 0.5
SEVERE_THRESHOLD: float = 0.3
REGRESSION_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _coverage(reference: set[str], candidate: set[str]) -> float:
    """Share of ``reference`` tokens found in ``candidate``, clamped to [0, 1].

    An empty reference has nothing to cover, so it scores 1.0.
    """
    if not reference:
        return 1.0
    return max(0.0, min(1.0, len(reference & candidate) / len(reference)))


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        return _coverage(_tokenize(answer), _tokenize(context))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _coverage(_tokenize(question), _tokenize(answer))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _coverage(_tokenize(expected), _tokenize(answer))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        return _coverage(_tokenize(expected), union_tokens)

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevant_so_far = 0
        precision_sum = 0.0
        for rank, chunk in enumerate(contexts, start=1):
            if _coverage(expected_tokens, _tokenize(chunk)) >= relevance_threshold:
                relevant_so_far += 1
                precision_sum += relevant_so_far / rank  # Precision@k at a hit
        if relevant_so_far == 0:
            return 0.0
        return max(0.0, min(1.0, precision_sum / relevant_so_far))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)
        passed = all(
            score >= PASS_THRESHOLD
            for score in (faithfulness, relevance, completeness)
        )

        failure_type: str | None = None
        if not passed:
            if faithfulness < SEVERE_THRESHOLD:
                failure_type = "hallucination"
            elif relevance < SEVERE_THRESHOLD:
                failure_type = "irrelevant"
            elif completeness < SEVERE_THRESHOLD:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(
                question=question,
                expected_answer=expected,
                context=context,
                retrieved_contexts=list(contexts or []),
            ),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    query_tokens = _tokenize(query)
    # sorted() is stable, so equally-scored chunks keep the retriever's order.
    return sorted(
        contexts,
        key=lambda chunk: len(_tokenize(chunk) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    DEFAULT_SCORE: float = 0.5
    # First-position item must beat the rest by this margin to flag positional bias.
    POSITIONAL_MARGIN: float = 0.1
    LENIENCY_THRESHOLD: float = 0.8
    SEVERITY_THRESHOLD: float = 0.3

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    @staticmethod
    def _build_prompt(question: str, answer: str, rubric: dict[str, Any]) -> str:
        criteria = "\n".join(
            f"- {name}: {description}" for name, description in rubric.items()
        )
        keys = ", ".join(f'"{name}": <score>' for name in rubric)
        return (
            "You are an impartial evaluator for a customer-support assistant.\n"
            "Score the answer on each criterion below. Judge correctness and "
            "coverage, not length or style: a longer answer must not score "
            "higher unless it adds required information.\n\n"
            f"Criteria:\n{criteria}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            "Return ONLY a JSON object with a score from 0.0 (worst) to 1.0 "
            "(best) for every criterion plus a short reasoning string, e.g.\n"
            f'{{{keys}, "reasoning": "<one or two sentences>"}}'
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Parse the first JSON object in ``text``; None if there is none."""
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalize_score(value: Any) -> float | None:
        """Map a raw judge score onto [0, 1]; None if it is not numeric.

        Scores in (1, 5] are treated as the lecture's 1–5 rubric scale.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        score = float(value)
        if 1.0 < score <= 5.0:
            score = (score - 1.0) / 4.0
        return max(0.0, min(1.0, score))

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        prompt = self._build_prompt(question, answer, rubric)
        try:
            raw_response = str(self.judge_llm_fn(prompt))
        except Exception as exc:  # judge outage must not crash the benchmark
            return {
                "scores": {name: self.DEFAULT_SCORE for name in rubric},
                "reasoning": f"Judge call failed: {exc}",
            }

        parsed = self._extract_json(raw_response) or {}
        # Accept both {"accuracy": 0.8} and {"scores": {"accuracy": 0.8}}.
        nested = parsed.get("scores")
        raw_scores = nested if isinstance(nested, dict) else parsed

        scores: dict[str, float] = {}
        for name in rubric:
            normalized = self._normalize_score(raw_scores.get(name))
            scores[name] = self.DEFAULT_SCORE if normalized is None else normalized

        reasoning = parsed.get("reasoning")
        return {
            "scores": scores,
            "reasoning": reasoning if isinstance(reasoning, str) else raw_response,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }

        The batch order is treated as presentation order: positional bias is
        flagged when the first response's mean score exceeds the mean of the
        remaining responses by more than ``POSITIONAL_MARGIN``.
        """
        item_means: list[float] = []
        all_scores: list[float] = []
        for item in scores_batch:
            scores = item.get("scores") if isinstance(item, dict) else None
            if not isinstance(scores, dict):
                continue
            values = [float(v) for v in scores.values()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if values:
                item_means.append(sum(values) / len(values))
                all_scores.extend(values)

        positional_bias = False
        if len(item_means) >= 2:
            rest_mean = sum(item_means[1:]) / len(item_means[1:])
            positional_bias = item_means[0] - rest_mean > self.POSITIONAL_MARGIN

        overall_mean = sum(all_scores) / len(all_scores) if all_scores else None
        return {
            "positional_bias": positional_bias,
            "leniency_bias": overall_mean is not None
            and overall_mean > self.LENIENCY_THRESHOLD,
            "severity_bias": overall_mean is not None
            and overall_mean < self.SEVERITY_THRESHOLD,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts,
            )
            # Keep the caller's pair so metadata (id, difficulty) survives.
            result.qa_pair = pair
            results.append(result)
        return results

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @classmethod
    def _optional_mean(cls, values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return cls._mean(present) if present else None

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        failed = [result for result in results if not result.passed]
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_faithfulness": self._mean([r.faithfulness for r in results]),
            "avg_relevance": self._mean([r.relevance for r in results]),
            "avg_completeness": self._mean([r.completeness for r in results]),
            "avg_context_recall": self._optional_mean(
                [r.context_recall for r in results]
            ),
            "avg_context_precision": self._optional_mean(
                [r.context_precision for r in results]
            ),
            "failure_types": FailureAnalyzer().categorize_failures(failed),
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions
        """
        report: dict[str, Any] = {}
        regressions: list[str] = []
        for metric in ("faithfulness", "relevance", "completeness"):
            new_avg = self._mean([getattr(r, metric) for r in new_results])
            baseline_avg = self._mean([getattr(r, metric) for r in baseline_results])
            report[f"new_avg_{metric}"] = new_avg
            report[f"baseline_avg_{metric}"] = baseline_avg
            # Round so float noise (0.9 - 0.85 = 0.05000000000000004) is not
            # mistaken for a drop larger than the threshold.
            if round(baseline_avg - new_avg, 10) > REGRESSION_THRESHOLD:
                regressions.append(metric)
        report["regressions"] = regressions
        report["passed"] = not regressions
        return report

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            result
            for result in results
            if min(result.faithfulness, result.relevance, result.completeness)
            < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    ROOT_CAUSES: dict[str, str] = {
        "faithfulness": "Context is missing or irrelevant — improve retrieval",
        "relevance": "Answer does not address the question — improve prompt clarity",
        "completeness": (
            "Answer is missing key information — increase context window "
            "or improve generation"
        ),
    }
    MULTIPLE_ISSUES: str = "Multiple issues detected — review full pipeline"

    # One concrete fix per failure cluster, keyed by failure_type.
    FIXES_BY_TYPE: dict[str, str] = {
        "hallucination": (
            "Add a claim-level grounding check that rejects sentences not "
            "supported by retrieved chunks, and require the generator to cite "
            "the source document for every policy number or date"
        ),
        "irrelevant": (
            "Rewrite the prompt to restate the customer's question and answer "
            "each sub-question explicitly before adding extra policy detail"
        ),
        "incomplete": (
            "Increase retrieval recall (query expansion for dates/amounts, "
            "top_k 5 → 8, follow cross-document references) and add few-shot "
            "examples that list every condition and exception"
        ),
        "off_topic": (
            "Add intent detection that routes out-of-scope and injection "
            "requests to a fixed scope response grounded in 00_system_scope.md"
        ),
        "refusal": (
            "Relax over-strict refusal rules: answer the supported policy part "
            "and only decline the unsupported action (refund, approval, unlock)"
        ),
    }
    GENERAL_FIXES: tuple[str, ...] = (
        "Add every failing case to the golden dataset regression suite and "
        "block deploys when any metric average drops by more than 0.05",
        "Calibrate the word-overlap heuristics against an LLM judge and a "
        "small set of human labels before trusting absolute scores",
        "Rerank retrieved chunks with a cross-encoder so evidence containing "
        "the exact policy clause appears in the first two positions",
    )

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        return dict(
            Counter(failure.failure_type or "unclassified" for failure in failures)
        )

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"

        "Multiple issues" is returned when every metric is below the pass
        threshold or when two metrics tie for the lowest score.
        """
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        if all(score < PASS_THRESHOLD for score in scores.values()):
            return self.MULTIPLE_ISSUES
        lowest = min(scores.values())
        weakest = [name for name, score in scores.items() if score - lowest < 1e-9]
        if len(weakest) > 1:
            return self.MULTIPLE_ISSUES
        return self.ROOT_CAUSES[weakest[0]]

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        Each row prefers the suggestion tagged with its failure type (as
        produced by ``generate_improvement_suggestions``), then the suggestion
        at the same index, then the fix mapped to the failure's type.
        """

        def cell(text: str) -> str:
            return str(text).replace("|", "\\|").replace("\n", " ")

        def matched_fix(failure_type: str, index: int) -> str:
            tag = f"[{failure_type.lower()}:"
            for suggestion in suggestions:
                if suggestion.startswith(tag):
                    return suggestion
            if index < len(suggestions):
                return suggestions[index]
            return self.FIXES_BY_TYPE.get(
                failure_type.lower(), "Investigate with 5 Whys"
            )

        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for index, failure in enumerate(failures):
            failure_id = f"F{index + 1:03d}"
            qa_id = failure.qa_pair.metadata.get("id")
            if qa_id:
                failure_id = f"{failure_id} ({qa_id})"
            failure_type = failure.failure_type or "unclassified"
            lines.append(
                f"| {cell(failure_id)} | {cell(failure_type)} | "
                f"{cell(self.find_root_cause(failure))} | "
                f"{cell(matched_fix(failure_type, index))} | Open |"
            )
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).

        Suggestions are ordered by cluster size (largest failure cluster
        first), then padded with pipeline-wide fixes up to three items.
        """
        if not failures:
            return []
        suggestions: list[str] = []
        ranked = Counter(
            (failure.failure_type or "unclassified").lower() for failure in failures
        ).most_common()
        for failure_type, count in ranked:
            fix = self.FIXES_BY_TYPE.get(failure_type)
            if fix is not None:
                suggestions.append(f"[{failure_type}: {count} case(s)] {fix}")
        for fix in self.GENERAL_FIXES:
            if len(suggestions) >= 3:
                break
            suggestions.append(fix)
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
