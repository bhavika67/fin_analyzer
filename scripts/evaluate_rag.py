# scripts/evaluate_rag.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from loguru import logger
from vectorstore.store import VectorStore
from agent.agent import FinancialAgent
from evaluation.evaluator import RAGEvaluator


# ── Ground truth Q&A pairs ────────────────────────────────────────────────────
# These are questions with known correct answers from your ingested documents.
# The agent answers them, RAGAS scores how good the answers are.

EVAL_SAMPLES = [
    {
        "question":    "What is Apple's net profit margin?",
        "ground_truth": "Apple's net profit margin is approximately 26.60%",
    },
    {
        "question":    "What were the key risks mentioned in the annual report?",
        "ground_truth": "Key risks include macroeconomic slowdown affecting SMB segment and rising cloud costs",
    },
    {
        "question":    "What was the total revenue for FY 2023?",
        "ground_truth": "Total revenue for FY 2023 reached $2,847M",
    },
    {
        "question":    "Which company had the highest net income?",
        "ground_truth": "Alphabet Inc. (GOOGL) had the highest net income",
    },
    {
        "question":    "What is the revenue growth outlook for FY 2024?",
        "ground_truth": "Management guides revenue of $3,200M to $3,350M representing 12 to 18 percent growth",
    },
    {
        "question":    "What caused the Q3 2022 revenue dip?",
        "ground_truth": "A supply chain disruption delayed enterprise deal closures pushing three contracts worth $68M into Q4",
    },
    {
        "question":    "What is TCS revenue compared to Infosys?",
        "ground_truth": "TCS has higher revenue than Infosys for the same period",
    },
    {
        "question":    "What is Microsoft's gross margin?",
        "ground_truth": "Microsoft's gross margin is approximately 69 to 70 percent",
    },
]


def build_sample_with_context(question: str, ground_truth: str,
                               agent: FinancialAgent,
                               store: VectorStore) -> dict:
    """Ask the agent, retrieve contexts, build RAGAS sample."""
    # Get agent answer
    result = agent.ask(question)
    answer = result.get("answer") or ""

    # Get retrieved contexts (top 3 chunks)
    chunks  = store.search(question, top_k=3)
    contexts = [c.get("text", "") for c in chunks]

    return {
        "question":     question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": ground_truth,
    }


def main():
    logger.info("Loading vector store...")
    store = VectorStore()
    store.load()
    logger.info(f"Loaded {store.total} vectors")

    logger.info("Initialising agent...")
    agent = FinancialAgent(store)

    logger.info(f"Building {len(EVAL_SAMPLES)} evaluation samples...")
    samples = []
    for i, s in enumerate(EVAL_SAMPLES, 1):
        logger.info(f"  [{i}/{len(EVAL_SAMPLES)}] {s['question'][:60]}...")
        sample = build_sample_with_context(
            s["question"], s["ground_truth"], agent, store
        )
        samples.append(sample)
        logger.info(f"    Answer: {sample['answer'][:80]}...")

    logger.info("Running RAGAS evaluation...")
    evaluator = RAGEvaluator()
    results   = evaluator.evaluate(samples)
    summary   = evaluator.summarize(results)

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\nQ: {r.question[:70]}")
        print(f"   Faithfulness:      {r.faithfulness:.3f}")
        print(f"   Answer Relevancy:  {r.answer_relevancy:.3f}")
        print(f"   Context Precision: {r.context_precision:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples evaluated:      {summary['num_samples']}")
    print(f"  Avg Faithfulness:       {summary['avg_faithfulness']:.3f}")
    print(f"  Avg Answer Relevancy:   {summary['avg_answer_relevancy']:.3f}")
    print(f"  Avg Context Precision:  {summary['avg_context_precision']:.3f}")
    print(f"  Overall Score:          {summary['overall_score']:.3f}")
    print(f"  Quality:                {summary['quality']}")
    print("=" * 60)

    # ── Save results to file ──────────────────────────────────
    import json
    out = ROOT / "reports" / "output" / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "summary": summary,
            "details": [
                {
                    "question":          r.question,
                    "answer":            r.answer,
                    "ground_truth":      r.ground_truth,
                    "faithfulness":      r.faithfulness,
                    "answer_relevancy":  r.answer_relevancy,
                    "context_precision": r.context_precision,
                }
                for r in results
            ]
        }, f, indent=2)
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()