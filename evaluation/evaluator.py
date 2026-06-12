# evaluation/evaluator.py
from dataclasses import dataclass
from loguru import logger


@dataclass
class EvalResult:
    question:          str
    answer:            str
    contexts:          list[str]
    ground_truth:      str
    faithfulness:      float | None = None
    answer_relevancy:  float | None = None
    context_precision: float | None = None


class RAGEvaluator:
    """
    Evaluate RAG pipeline quality using RAGAS (>=0.2 schema).
    Metrics:
      - Faithfulness:       is the answer grounded in the retrieved context?
      - Answer Relevancy:   does the answer address the question?
      - Context Precision:  are the retrieved chunks actually relevant?
    """

    def evaluate(self, samples: list[dict]) -> list[EvalResult]:
        """
        samples: list of dicts with keys:
            question     (str)
            answer       (str)        — agent's response
            contexts     (list[str])  — retrieved chunks used
            ground_truth (str)        — expected correct answer
        """
        try:
            from ragas import evaluate, EvaluationDataset
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            from config import get_settings
            settings = get_settings()

            logger.info(f"Evaluating {len(samples)} samples with RAGAS...")

            # ragas >=0.2 uses: user_input, response, retrieved_contexts, reference
            ragas_records = [
                {
                    "user_input":         s["question"],
                    "response":           s["answer"],
                    "retrieved_contexts": s["contexts"],
                    "reference":          s["ground_truth"],
                }
                for s in samples
            ]
            dataset = EvaluationDataset.from_list(ragas_records)

            # Explicit LLM + embeddings wrappers (required in ragas >=0.2)
            eval_llm = LangchainLLMWrapper(ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0,
                max_tokens=4096,  # avoid truncated faithfulness JSON on long answers
            ))
            eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            ))

            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=eval_llm,
                embeddings=eval_embeddings,
            )
            df = result.to_pandas()

            results = []
            for i, row in df.iterrows():
                original = samples[i]
                results.append(EvalResult(
                    question          = original["question"],
                    answer            = original["answer"],
                    contexts          = original["contexts"],
                    ground_truth      = original["ground_truth"],
                    faithfulness      = round(float(row.get("faithfulness",      0) or 0), 4),
                    answer_relevancy  = round(float(row.get("answer_relevancy",  0) or 0), 4),
                    context_precision = round(float(row.get("context_precision", 0) or 0), 4),
                ))
            logger.info("RAGAS evaluation complete.")
            return results

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise

    def summarize(self, results: list[EvalResult]) -> dict:
        if not results:
            return {}

        def avg(key):
            vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        summary = {
            "num_samples":         len(results),
            "avg_faithfulness":    avg("faithfulness"),
            "avg_answer_relevancy": avg("answer_relevancy"),
            "avg_context_precision": avg("context_precision"),
            "overall_score":       round(
                (avg("faithfulness") + avg("answer_relevancy") + avg("context_precision")) / 3, 4
            ),
        }

        # Quality label
        score = summary["overall_score"]
        summary["quality"] = (
            "Excellent" if score >= 0.8 else
            "Good"      if score >= 0.6 else
            "Fair"      if score >= 0.4 else
            "Poor"
        )
        return summary