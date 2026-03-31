# evaluation/eval_pipeline.py
# Evaluation using RAGAS + ROUGE metrics
from typing import List, Dict
import pandas as pd
from loguru import logger


def run_ragas_eval(questions, answers, contexts, ground_truths=None):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset

        data = {"question": questions, "answer": answers, "contexts": contexts}
        metrics = [faithfulness, answer_relevancy]
        if ground_truths:
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        result = evaluate(Dataset.from_dict(data), metrics=metrics)
        df = result.to_pandas()
        logger.info(f"[Eval] RAGAS:\n{df[['faithfulness', 'answer_relevancy']].mean()}")
        return df
    except ImportError:
        logger.warning("ragas not installed.")
        return pd.DataFrame()


def run_rouge_eval(answers, references):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, a) for r, a in zip(references, answers)]
        avg_f1 = sum(s["rougeL"].fmeasure for s in scores) / len(scores)
        logger.info(f"[Eval] ROUGE-L F1: {avg_f1:.4f}")
        return {"rouge_l_f1": avg_f1, "per_sample": scores}
    except ImportError:
        logger.warning("rouge_score not installed.")
        return {}


def run_full_eval(eval_samples: List[Dict]) -> Dict:
    questions     = [s["question"]     for s in eval_samples]
    answers       = [s["answer"]       for s in eval_samples]
    contexts      = [s.get("contexts", []) for s in eval_samples]
    ground_truths = [s.get("ground_truth", "") for s in eval_samples]

    results = {}
    ragas_df = run_ragas_eval(questions, answers, contexts, ground_truths)
    if not ragas_df.empty:
        results["ragas"] = ragas_df
    rouge = run_rouge_eval(answers, ground_truths)
    if rouge:
        results["rouge"] = rouge
    return results