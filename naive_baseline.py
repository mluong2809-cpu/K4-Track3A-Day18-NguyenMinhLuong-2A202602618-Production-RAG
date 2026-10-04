"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from src.gemini_backend import chat
from config import NAIVE_COLLECTION


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    from config import GEMINI_API_KEY
    generation_available = bool(GEMINI_API_KEY)

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if generation_available and contexts:
            try:
                context_str = "\n\n".join(contexts)
                answer = chat(
                    "Trả lời trực tiếp bằng tiếng Việt, chỉ dựa trên ngữ cảnh. "
                    "Nếu thiếu dữ kiện thì nói rõ phần thiếu.",
                    f"Ngữ cảnh:\n{context_str}\n\nCâu hỏi: {item['question']}",
                    max_tokens=1024,
                )
            except Exception as error:
                generation_available = False
                print(f"  Gemini baseline generation unavailable: {error}")
                answer = contexts[0]
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths, use_api=True)
    results["predictions"] = [
        {"question": q, "answer": a, "contexts": c, "ground_truth": gt}
        for q, a, c, gt in zip(questions, answers, all_contexts, ground_truths)
    ]
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}" if results.get("status") == "evaluated" else f"  {m}: N/A (RAGAS unavailable)")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if results.get("status") != "evaluated":
        print("\nRAGAS chưa có điểm thật. Cần GEMINI_API_KEY hợp lệ để chấm và so sánh hai pipeline.")
    print("\nBaseline run complete.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
