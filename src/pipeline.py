from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.gemini_backend import chat
from config import RERANK_TOP_K, GEMINI_API_KEY

_GENERATION_AVAILABLE = True


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_lookup = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        source = doc["metadata"]["source"]
        for parent in parents:
            parent_lookup[(source, f"{source}:{parent.parent_id}")] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": f"{source}:{child.parent_id}"}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks, use_api=True)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_lookup = parent_lookup
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    global _GENERATION_AVAILABLE
    results = search.search(query)
    docs = []
    seen_parents = set()
    for result in results:
        parent_key = (result.metadata.get("source"), result.metadata.get("parent_id"))
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        parent_text = getattr(search, "parent_lookup", {}).get(parent_key, result.text)
        docs.append({"text": parent_text, "score": result.score, "metadata": result.metadata})
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [item.text for item in reranked] if reranked else [doc["text"] for doc in docs[:3]]

    if GEMINI_API_KEY and _GENERATION_AVAILABLE and contexts:
        try:
            context_str = "\n\n".join(contexts)
            answer = chat(
                "Trả lời trực tiếp bằng tiếng Việt, chỉ dựa trên ngữ cảnh. "
                "Ưu tiên quy định hiện hành; nếu thiếu dữ kiện thì nói rõ phần thiếu. "
                "Với câu hỏi số học, nêu phép tính ngắn gọn.",
                f"Ngữ cảnh:\n{context_str}\n\nCâu hỏi: {query}",
                max_tokens=1024,
            )
        except Exception as e:
            _GENERATION_AVAILABLE = False
            print(f"  Gemini generation unavailable: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths, use_api=True)
    results["predictions"] = [
        {"question": q, "answer": a, "contexts": c, "ground_truth": gt}
        for q, a, c, gt in zip(questions, answers, all_contexts, ground_truths)
    ]
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {m}: {s:.4f}" if results.get("status") == "evaluated" else f"  {m}: N/A (RAGAS unavailable)")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
