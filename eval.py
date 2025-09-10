# src/eval.py
import json
import time
import random
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional

# Optional, more robust metrics
try:
    import sacrebleu  # reproducible BLEU and chrF
except Exception:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer  # robust ROUGE-1/2/L/Lsum
except Exception:
    rouge_scorer = None

# Fallback libraries kept for compatibility
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except Exception:
    sentence_bleu, SmoothingFunction = None, None

try:
    from nltk.translate.meteor_score import meteor_score
except Exception:
    meteor_score = None

try:
    from rouge import Rouge  # older python-rouge
except Exception:
    Rouge = None

# Optional semantic metric
try:
    from bert_score import score as bertscore_score
except Exception:
    bertscore_score = None

from .retrieval import hybrid_retrieve_pg
from .generator import chat_with_assistant

# === Benchmarks (set only what you truly want to gate on) ===
BENCHMARKS = {
    # Retrieval
    "recall@5": 0.80,
    "recall@10": 0.85,
    "precision@5": 0.80,
    "hit_rate@5": 0.85,
    "mrr@10": 0.75,
    # Generation
    "hallucination_rate": 0.15,
    "actionability": 0.60,
    "rouge1": 0.80,
    "rouge2": 0.70,
    "rougeL": 0.75,   # If preferring ROUGE-Lsum, switch the gate key or compute both.
    "bleu": 0.45,
    "meteor": 0.45,
    "latency_p95": 3.5,   # more realistic default for DB + LLM pipeline
    "fallback_rate": 0.05,
    # Optional/off
    "citation_precision": None,
    "confidence_precision_high": None,
}
METRIC_PRIORITY = [k for k, v in BENCHMARKS.items() if v is not None]

# === Matching thresholds (tune per dataset) ===
F1_THRESHOLDS = {
    "gold_passages": 0.35,
    "supporting_facts": 0.40,
    "degraded_ideal": 0.30,
}
MIN_OVERLAP_TOKENS = {
    "gold_passages": 5,
    "supporting_facts": 3,
    "degraded_ideal": 5,
}

# === Heuristics lists (expand as needed) ===
FALLBACK_PHRASES = [
    "i don't know", "cannot find", "can't find", "not sure",
    "no information found", "unable to", "cannot answer",
    "don't have access", "not available", "insufficient information",
    "no relevant information", "couldn't find",
]
ACTIONABLE_PHRASES = [
    "follow these steps", "step-by-step", "do the following", "click",
    "select", "navigate", "go to", "open", "run", "execute",
    "use", "create", "set", "configure", "install", "download",
]

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

def normalize_text_for_overlap(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.casefold()
    s = re.sub(r"\b(\w+)'s\b", r"\1", s)
    s = re.sub(r"\b(\w+)’s\b", r"\1", s)
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return normalize_text_for_overlap(s).split()

def overlap_stats(pred_text: str, gold_text: str) -> Tuple[int, float, float, float]:
    pt = _tokenize(pred_text)
    gt = _tokenize(gold_text)
    if not pt or not gt:
        return 0, 0.0, 0.0, 0.0
    from collections import Counter as C
    pc, gc = C(pt), C(gt)
    common = sum((pc & gc).values())
    precision = common / len(pt) if pt else 0.0
    recall = common / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return common, precision, recall, f1

def token_f1(pred_text: str, gold_text: str) -> float:
    # overlap_stats returns (common, precision, recall, f1)
    return overlap_stats(pred_text, gold_text)[3]

def mean_reciprocal_rank(ranks: List[int]) -> float:
    return float(np.mean([1.0 / r if r > 0 else 0.0 for r in ranks])) if ranks else 0.0

def has_gt_ids(item):            return isinstance(item.get("positives"), list) and len(item["positives"]) > 0
def has_gt_passages(item):       return isinstance(item.get("gold_passages"), list) and len(item["gold_passages"]) > 0
def has_supporting_facts(item):  return isinstance(item.get("supporting_facts"), list) and len(item["supporting_facts"]) > 0

def relevant_by_passage(doc_text: str, gold: List[str], f1_threshold: float, min_overlap: int) -> bool:
    for gp in gold:
        common, _, _, f1 = overlap_stats(doc_text, gp)
        if f1 >= f1_threshold and common >= min_overlap:
            return True
    return False

def relevant_degraded(doc_text: str, ideal_answer: str, f1_threshold: float, min_overlap: int) -> bool:
    if not ideal_answer:
        return False
    common, _, _, f1 = overlap_stats(doc_text, ideal_answer)
    return f1 >= f1_threshold and common >= min_overlap

def _norm_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    sx = str(x).strip()
    return sx if sx else None

def retrieval_worker(item, candidate_pool=30):
    """Return per-query retrieval metrics with robust recall and per-position relevance flags."""
    query = item["query"]
    if candidate_pool < 10:
        print("⚠️ candidate_pool < 10; MRR@10 and recall@10 may be under-sampled.")

    # Guard: ensure retrieved is a list-like, even if backend returns None
    retrieved = hybrid_retrieve_pg(query, top_k=candidate_pool) or []

    retrieved_ids, retrieved_texts, retrieved_sources = [], [], []

    # Guard: tolerate different shapes from retriever
    for item_r in retrieved:
        if isinstance(item_r, (list, tuple)) and len(item_r) >= 2:
            doc, meta = item_r[0], item_r[1] if isinstance(item_r[1], dict) else {}
        else:
            doc, meta = item_r, {}

        # ✅ prefer URL for ID alignment with testset
        rid = (
            (meta.get("url") if isinstance(meta, dict) else None)
            or (meta.get("source") if isinstance(meta, dict) else None)
            or (meta.get("id") if isinstance(meta, dict) else None)
            or (meta.get("chunk_id") if isinstance(meta, dict) else None)
            or (meta.get("doc_id") if isinstance(meta, dict) else None)
        )

        src = None
        if isinstance(meta, dict):
            src = meta.get("source") or meta.get("retrieval_source") or meta.get("url")

        retrieved_ids.append(_norm_id(rid))
        retrieved_texts.append(str(doc) if doc is not None else "")
        retrieved_sources.append(src)

    flags = [False] * len(retrieved_texts)

    def unique_hits_in_topk(k: int):
        if has_gt_ids(item):
            gt = set(_norm_id(x) for x in item["positives"] if _norm_id(x) is not None)
            hits = set(rid for rid in retrieved_ids[:k] if rid in gt and rid is not None)
            return len(hits), len(gt)
        elif has_gt_passages(item):
            gps = item["gold_passages"]
            hit_gps = set()
            for gp in gps:
                for doc in retrieved_texts[:k]:
                    if relevant_by_passage(doc, [gp], F1_THRESHOLDS["gold_passages"], MIN_OVERLAP_TOKENS["gold_passages"]):
                        hit_gps.add(gp)
                        break
            return len(hit_gps), len(gps)
        elif has_supporting_facts(item):
            sfs = item["supporting_facts"]
            hit_sfs = set()
            for sf in sfs:
                for doc in retrieved_texts[:k]:
                    if relevant_by_passage(doc, [sf], F1_THRESHOLDS["supporting_facts"], MIN_OVERLAP_TOKENS["supporting_facts"]):
                        hit_sfs.add(sf)
                        break
            return len(hit_sfs), len(sfs)
        else:
            return None, None  # degraded: no reliable denominator

    # Fill per-position flags
    if has_gt_ids(item):
        gt_ids = set(_norm_id(x) for x in item["positives"] if _norm_id(x) is not None)
        for i, rid in enumerate(retrieved_ids):
            if rid is not None and rid in gt_ids:
                flags[i] = True
        gt_mode = "ids"
    elif has_gt_passages(item):
        gps = item["gold_passages"]
        for i, doc in enumerate(retrieved_texts):
            flags[i] = relevant_by_passage(doc, gps, F1_THRESHOLDS["gold_passages"], MIN_OVERLAP_TOKENS["gold_passages"])
        gt_mode = "gold_passages"
    elif has_supporting_facts(item):
        sfs = item["supporting_facts"]
        for i, doc in enumerate(retrieved_texts):
            flags[i] = relevant_by_passage(doc, sfs, F1_THRESHOLDS["supporting_facts"], MIN_OVERLAP_TOKENS["supporting_facts"])
        gt_mode = "supporting_facts"
    else:
        ideal = item.get("ideal_answer", "")
        for i, doc in enumerate(retrieved_texts):
            flags[i] = relevant_degraded(doc, ideal, F1_THRESHOLDS["degraded_ideal"], MIN_OVERLAP_TOKENS["degraded_ideal"])
        gt_mode = "degraded_ideal_answer"

    def at_k(arr, k): return arr[:min(k, len(arr))]
    top5, top10 = at_k(flags, 5), at_k(flags, 10)

    precision_at5 = (sum(top5) / len(top5)) if top5 else 0.0
    hit_at5 = 1.0 if any(top5) else 0.0

    rank = 0
    for i, flag in enumerate(top10):
        if flag:
            rank = i + 1
            break
    mrr10 = (1.0 / rank) if rank > 0 else 0.0

    def safe_recall(k):
        hits, denom = unique_hits_in_topk(k)
        if hits is None or denom is None or denom <= 0:
            return float("nan")
        return min(1.0, hits / float(denom))

    recall_at5, recall_at10 = safe_recall(5), safe_recall(10)

    debug = {
        "query_raw": query,
        "gt_mode": gt_mode,
        "gt_ids": item.get("positives", []),
        "retrieved_preview": [
            {
                "id": retrieved_ids[i],
                "source": retrieved_sources[i],
                "snippet": (retrieved_texts[i] or "")[:240].replace("\n", " ")
            }
            for i in range(min(10, len(retrieved_texts)))
        ],
    }

    return {
        "recall@5": recall_at5,
        "recall@10": recall_at10,
        "precision@5": precision_at5,
        "hit_rate@5": hit_at5,
        "mrr@10": mrr10,
        "debug": debug
    }

# === Generation metrics ===

# Reuse a single rouge scorer if available
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True) if rouge_scorer is not None else None

def _compute_bleu(answer: str, ideal: str) -> float:
    if not ideal:
        return 0.0
    # Prefer SacreBLEU for reproducibility; returns percentage, so divide by 100.
    if sacrebleu is not None:
        res = sacrebleu.corpus_bleu([answer], [[ideal]])
        return float(res.score / 100.0)
    if sentence_bleu is not None:
        smoothie = SmoothingFunction().method4 if SmoothingFunction is not None else None
        return float(sentence_bleu([ideal.split()], answer.split(), smoothing_function=smoothie))
    return 0.0

def _compute_chrf(answer: str, ideal: str) -> float:
    if not ideal or sacrebleu is None:
        return 0.0
    try:
        res = sacrebleu.corpus_chrf([answer], [[ideal]])
        return float(res.score / 100.0)
    except Exception:
        return 0.0

def _compute_meteor(answer: str, ideal: str) -> float:
    if not ideal or meteor_score is None:
        return 0.0
    try:
        return float(meteor_score([ideal], answer))
    except Exception:
        return 0.0

def _compute_rouge(answer: str, ideal: str) -> Dict[str, float]:
    if not ideal:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    if _ROUGE_SCORER is not None:
        scores = _ROUGE_SCORER.score(ideal, answer)
        return {
            "rouge1": float(scores["rouge1"].fmeasure),
            "rouge2": float(scores["rouge2"].fmeasure),
            "rougeL": float(scores["rougeL"].fmeasure),
            "rougeLsum": float(scores["rougeLsum"].fmeasure),
        }
    if Rouge is not None:
        try:
            r = Rouge()
            rs_list = r.get_scores(answer, ideal)
            if isinstance(rs_list, list) and rs_list:
                rs = rs_list[0]  # <-- pick the first (and only) result dict
                return {
                    "rouge1": float(rs.get("rouge-1", {}).get("f", 0.0)),
                    "rouge2": float(rs.get("rouge-2", {}).get("f", 0.0)),
                    "rougeL": float(rs.get("rouge-l", {}).get("f", 0.0)),
                    "rougeLsum": 0.0,
                }
        except Exception:
            pass
    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

def _is_fallback(answer: str) -> bool:
    a = answer.casefold()
    return any(p in a for p in FALLBACK_PHRASES)

_LIST_LINE_RE = re.compile(r"^\s*(?:\d+[\).\s]|[-*]\s+)", re.MULTILINE)

def _is_actionable(answer: str) -> bool:
    a = answer.casefold()
    has_keywords = any(p in a for p in ACTIONABLE_PHRASES)
    has_list = len(_LIST_LINE_RE.findall(answer)) >= 2  # two or more enumerated lines
    return has_keywords or has_list

def _hallucination_flag(answer: str, ideal: str) -> Tuple[float, float, float, float, int, int, int]:
    """
    Returns: (flag, f1_overlap, prec, rec, common_tokens, len_ans, len_ref)
    If BERTScore is available, use semantic F1 threshold; else use overlap-based rule.
    """
    ans_toks, ref_toks = _tokenize(answer), _tokenize(ideal)
    common, p, r, f1 = overlap_stats(answer, ideal)
    if not ideal.strip():
        return 0.0, f1, p, r, common, len(ans_toks), len(ref_toks)
    if bertscore_score is not None:
        try:
            P, R, F1 = bertscore_score([answer], [ideal], lang="en", rescale_with_baseline=True)
            bs_f1 = float(F1.mean().item())
            flag = 1.0 if bs_f1 < 0.80 else 0.0
            # Keep overlap F1 in debug for stability; change to bs_f1 if you prefer
            return flag, f1, p, r, common, len(ans_toks), len(ref_toks)
        except Exception:
            pass
    # Overlap fallback: stricter with minimum common tokens depending on ref length
    min_req = 5 if len(ref_toks) >= 8 else 3
    flag = 1.0 if (f1 < 0.25 or common < min_req) else 0.0
    return flag, f1, p, r, common, len(ans_toks), len(ref_toks)

def generation_worker(item, top_k_for_context=3):
    query = item["query"]
    ideal = item.get("ideal_answer", "")
    docs_with_meta = hybrid_retrieve_pg(query, top_k=top_k_for_context) or []
    # Normalize to just the text parts for the generator
    docs = []
    for item_r in docs_with_meta:
        if isinstance(item_r, (list, tuple)) and len(item_r) >= 1:
            docs.append(str(item_r[0]) if item_r[0] is not None else "")
        else:
            docs.append(str(item_r) if item_r is not None else "")

    t0 = time.time()
    answer = chat_with_assistant(query, docs)
    latency = time.time() - t0

    bleu = _compute_bleu(answer, ideal)
    chrf = _compute_chrf(answer, ideal)  # not gated; for debugging/stability
    meteor = _compute_meteor(answer, ideal)
    rouge = _compute_rouge(answer, ideal)
    rouge1, rouge2, rougeL = rouge["rouge1"], rouge["rouge2"], rouge["rougeL"]

    fallback = _is_fallback(answer)
    actionable = _is_actionable(answer)

    hallucination, f1o, prec_o, rec_o, common_o, n_ans, n_ref = _hallucination_flag(answer, ideal)

    return {
        "bleu": bleu,
        "chrf": chrf,
        "meteor": meteor,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "latency": latency,
        "fallback": 1.0 if fallback else 0.0,
        "actionable": 1.0 if actionable else 0.0,
        "hallucination": float(hallucination),
        # debug fields (optional)
        "overlap_f1": f1o,
        "overlap_prec": prec_o,
        "overlap_rec": rec_o,
        "overlap_common_tokens": common_o,
        "answer_tokens": n_ans,
        "ref_tokens": n_ref,
    }

def evaluate(ground_truth,
             eval_size=100,
             seed=42,
             candidate_pool=30,
             top_k_for_context=3,
             progress_every=25,
             show_worst=8):
    rng = random.Random(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass

    items = list(ground_truth)
    if eval_size is not None:
        eval_size = min(eval_size, len(items))
        items = rng.sample(items, eval_size)

    per_query_ret = []
    recall5_vals, recall10_vals = [], []
    prec5_vals, hit5_vals = [], []
    mrr_vals = []
    t_start = time.time()

    gt_mode_counts = {"ids": 0, "gold_passages": 0, "supporting_facts": 0, "degraded_ideal_answer": 0}

    for idx, item in enumerate(items, 1):
        res = retrieval_worker(item, candidate_pool=candidate_pool)
        per_query_ret.append({
            "query": item["query"],
            "ideal": item.get("ideal_answer", ""),
            "metrics": {k: res[k] for k in ["recall@5","recall@10","precision@5","hit_rate@5","mrr@10"]},
            "debug": res["debug"]
        })
        mode = res["debug"]["gt_mode"]
        gt_mode_counts[mode] = gt_mode_counts.get(mode, 0) + 1

        if not np.isnan(res["recall@5"]):  recall5_vals.append(res["recall@5"])
        if not np.isnan(res["recall@10"]): recall10_vals.append(res["recall@10"])
        prec5_vals.append(res["precision@5"])
        hit5_vals.append(res["hit_rate@5"])
        mrr_vals.append(res["mrr@10"])

        if idx % progress_every == 0:
            elapsed = time.time() - t_start
            print(f"[retrieval] processed {idx}/{len(items)} in {elapsed:.1f}s")

    retrieval_metrics = {
        "recall@5":  float(np.mean(recall5_vals))  if recall5_vals  else float("nan"),
        "recall@10": float(np.mean(recall10_vals)) if recall10_vals else float("nan"),
        "precision@5": float(np.mean(prec5_vals)) if prec5_vals else 0.0,
        "hit_rate@5": float(np.mean(hit5_vals)) if hit5_vals else 0.0,
        "mrr@10": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
    }

    print("\nRetrieval metrics (mean over queries):")
    def fmt(x): return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"
    print(f"- recall@5: {fmt(retrieval_metrics['recall@5'])} (n={len(recall5_vals)})")
    print(f"- recall@10: {fmt(retrieval_metrics['recall@10'])} (n={len(recall10_vals)})")
    print(f"- precision@5: {retrieval_metrics['precision@5']:.3f}")
    print(f"- hit_rate@5: {retrieval_metrics['hit_rate@5']:.3f}")
    print(f"- mrr@10: {retrieval_metrics['mrr@10']:.3f}")
    print(f"- supervision mix: {gt_mode_counts}")

    failed_ret = []
    for metric in ["recall@5", "recall@10", "precision@5", "hit_rate@5", "mrr@10"]:
        score = retrieval_metrics.get(metric, float("nan"))
        target = BENCHMARKS.get(metric)
        if target is None:
            continue
        if isinstance(score, float) and np.isnan(score):
            continue  # skip recall gates if undefined
        if score < target:
            failed_ret.append((metric, score, target))

    if failed_ret:
        print("\n❌ Retrieval thresholds not met. Skipping generation metrics.\n")
        for metric, score, target in failed_ret:
            if isinstance(score, float) and np.isnan(score):
                print(f"- {metric}: NA (target {target})")
            else:
                print(f"- {metric}: {score:.3f} (target {target})")
        try:
            def to0(x):
                return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
            def badness(pqr):
                m = pqr["metrics"]
                miss = 1.0 - (1.0 if m["hit_rate@5"] > 0 else 0.0)
                return 2.0 * miss + (1.0 - m["mrr@10"]) + (1.0 - to0(m["recall@5"]))
            worst = sorted(per_query_ret, key=badness, reverse=True)[:show_worst]
            print("\nWorst queries (for debugging):")
            for w in worst:
                m = w["metrics"]
                r5 = "NA" if (isinstance(m["recall@5"], float) and np.isnan(m["recall@5"])) else f"{m['recall@5']:.2f}"
                print(f"- Q: {w['query']}\n  recall@5={r5}, prec@5={m['precision@5']:.2f}, hit@5={m['hit_rate@5']:.2f}, mrr@10={m['mrr@10']:.2f}")
                for r in w["debug"]["retrieved_preview"][:3]:
                    print(f"    · {r['id']} | {r['source']} | {r['snippet']}")
        except Exception:
            pass
        return

    # === Generation stage (only if retrieval passes) ===
    per_query_gen = []
    bleu_s, meteor_s, rouge1_s, rouge2_s, rougeL_s = [], [], [], [], []
    latencies, fallback_flags, actionable_flags, halluc_flags = [], [], [], []
    t_gen_start = time.time()
    for idx, item in enumerate(items, 1):
        res = generation_worker(item, top_k_for_context=top_k_for_context)
        per_query_gen.append({"query": item["query"], "ideal": item.get("ideal_answer", ""), **res})
        bleu_s.append(res["bleu"]); meteor_s.append(res["meteor"])
        rouge1_s.append(res["rouge1"]); rouge2_s.append(res["rouge2"]); rougeL_s.append(res["rougeL"])
        latencies.append(res["latency"])
        fallback_flags.append(res["fallback"]); actionable_flags.append(res["actionable"]); halluc_flags.append(res["hallucination"])
        if idx % progress_every == 0:
            elapsed = time.time() - t_gen_start
            print(f"[generation] processed {idx}/{len(items)} in {elapsed:.1f}s")

    def percentile(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        return float(np.percentile(vals, p))

    gen_metrics = {
        "rouge1": float(np.mean(rouge1_s)) if rouge1_s else 0.0,
        "rouge2": float(np.mean(rouge2_s)) if rouge2_s else 0.0,
        "rougeL": float(np.mean(rougeL_s)) if rougeL_s else 0.0,
        "bleu": float(np.mean(bleu_s)) if bleu_s else 0.0,
        "meteor": float(np.mean(meteor_s)) if meteor_s else 0.0,
        "hallucination_rate": float(np.mean(halluc_flags)) if halluc_flags else 0.0,
        "actionability": float(np.mean(actionable_flags)) if actionable_flags else 0.0,
        "latency_p95": percentile(latencies, 95),
        "fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
    }

    failed_gen = []
    for metric, target in BENCHMARKS.items():
        if target is None:
            continue
        score = gen_metrics.get(metric)
        if score is None:
            continue
        if (metric != "latency_p95" and score < target) or (metric == "latency_p95" and score > target):
            failed_gen.append((metric, score, target))

    if failed_gen:
        print("\n❌ Generation thresholds not met.\n")
        for metric, score, target in failed_gen:
            print(f"- {metric}: {score:.3f} (target {target})")
        return

    print("\n✅ All metrics passed!\n")
    print("{:<28} {:<10} {:<10}".format("Metric", "Score", "Target"))
    print("-" * 52)
    for metric in METRIC_PRIORITY:
        score = gen_metrics.get(metric, retrieval_metrics.get(metric))
        if score is not None:
            if isinstance(score, float) and np.isnan(score):
                print("{:<28} {:<10} {:<10}".format(metric, "NA", BENCHMARKS[metric]))
            else:
                print("{:<28} {:<10.3f} {:<10}".format(metric, score, BENCHMARKS[metric]))

def load_ground_truth(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    # Edit this path or pass your own entry-point around evaluate()
    gt_path = "/Users/srujana/Desktop/RAG-TDD/data/testset_tech_hp.json"
    ground = load_ground_truth(gt_path)
    print(f"Loaded {len(ground)} test cases from {gt_path}")
    evaluate(
        ground,
        eval_size=100,
        seed=42,
        candidate_pool=120,   # ensure your retriever can return this many
        top_k_for_context=5,
        progress_every=25
    )
