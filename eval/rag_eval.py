"""
RAG 评估脚本：计算 Retrieval Hit@3 与 Answer Faithfulness。

用法：
    python -m eval.rag_eval
    python -m eval.rag_eval --output eval/report.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path

DATASET_PATH = get_abs_path("eval/dataset.jsonl")
DEFAULT_REPORT_PATH = get_abs_path("eval/report.md")


@dataclass
class EvalItem:
    item_id: str
    query: str
    category: str
    expected_keywords: list[str]


@dataclass
class EvalResult:
    item_id: str
    query: str
    hit_at_3: bool
    faithfulness: bool
    matched_keywords: list[str]
    answer_preview: str


def load_dataset(path: str = DATASET_PATH) -> list[EvalItem]:
    items: list[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                EvalItem(
                    item_id=row["id"],
                    query=row["query"],
                    category=row["category"],
                    expected_keywords=row["expected_keywords"],
                )
            )
    return items


def retrieval_hit_at_3(rag: RagSummarizeService, item: EvalItem) -> tuple[bool, list[str]]:
    docs = rag.retriever_docs(item.query)
    top3_text = "\n".join(doc.page_content for doc in docs[:3])
    matched = [kw for kw in item.expected_keywords if kw in top3_text]
    hit = len(matched) > 0
    return hit, matched


def judge_faithfulness(answer: str, context: str, query: str) -> bool:
    if not answer.strip():
        return False

    prompt = PromptTemplate.from_template(
        "你是RAG评估员。判断「回答」是否主要由「检索上下文」支撑，且没有明显编造。\n"
        "只回答 yes 或 no。\n\n"
        "用户问题：{query}\n"
        "检索上下文：{context}\n"
        "模型回答：{answer}\n"
    )
    chain = prompt | chat_model | StrOutputParser()
    verdict = chain.invoke(
        {"query": query, "context": context[:3000], "answer": answer[:1500]}
    ).strip().lower()
    return verdict.startswith("y") or verdict == "是"


def evaluate_rag(dataset_path: str = DATASET_PATH) -> list[EvalResult]:
    rag = RagSummarizeService()
    dataset = load_dataset(dataset_path)
    results: list[EvalResult] = []

    for item in dataset:
        hit, matched = retrieval_hit_at_3(rag, item)
        docs = rag.retriever_docs(item.query)
        context = "\n".join(doc.page_content for doc in docs[:3])
        answer = rag.rag_summarize(item.query)
        faithful = judge_faithfulness(answer, context, item.query)

        results.append(
            EvalResult(
                item_id=item.item_id,
                query=item.query,
                hit_at_3=hit,
                faithfulness=faithful,
                matched_keywords=matched,
                answer_preview=answer[:120],
            )
        )
        print(f"[{item.item_id}] hit@3={hit} faithfulness={faithful}")

    return results


def summarize_metrics(results: list[EvalResult]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {"hit_at_3": 0.0, "faithfulness": 0.0}
    return {
        "hit_at_3": sum(1 for r in results if r.hit_at_3) / total,
        "faithfulness": sum(1 for r in results if r.faithfulness) / total,
    }


def write_report(results: list[EvalResult], output_path: str) -> None:
    metrics = summarize_metrics(results)
    lines = [
        "# RAG 评估报告",
        "",
        f"- 样本数: {len(results)}",
        f"- Retrieval Hit@3: {metrics['hit_at_3']:.1%}",
        f"- Answer Faithfulness: {metrics['faithfulness']:.1%}",
        "",
        "## 明细",
        "",
        "| ID | Hit@3 | Faithfulness | Matched Keywords | Query |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        matched = ", ".join(r.matched_keywords) if r.matched_keywords else "-"
        lines.append(
            f"| {r.item_id} | {'Y' if r.hit_at_3 else 'N'} | "
            f"{'Y' if r.faithfulness else 'N'} | {matched} | {r.query} |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    results = evaluate_rag(args.dataset)
    metrics = summarize_metrics(results)
    print(
        f"评估完成: Hit@3={metrics['hit_at_3']:.1%}, "
        f"Faithfulness={metrics['faithfulness']:.1%}"
    )
    write_report(results, args.output)


if __name__ == "__main__":
    main()
