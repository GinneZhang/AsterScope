# NovaSearch — HotpotQA Retrieval Rebuild Benchmark Report

## Executive Summary

This report replaces the earlier CLIP-heavy HotpotQA baseline with the results from a full **1,000-question rerun** of NovaSearch's rebuilt **text-first retrieval profile**.

This rerun did **not** exercise the old full-stack multimodal pipeline. It intentionally benchmarked the repaired HotpotQA path with **graph ingestion disabled**, **graph retrieval disabled**, **vision retrieval disabled**, **CLIP text-vision indexing disabled**, and **generation evaluation disabled** so the reported numbers reflect text retrieval quality rather than multimodal or answer-synthesis behavior.

## 1. Benchmark Configuration

| Setting | Rerun Value |
|---|---|
| Dataset | HotpotQA validation (`distractor`) |
| Question count | 1,000 |
| Unique reference pages ingested | 9,803 |
| Retrieval profile | Hybrid text retrieval + cross-encoder reranking |
| Graph ingestion | Disabled |
| Graph retrieval | Disabled |
| Vision retriever | Disabled |
| CLIP text-vision indexing | Disabled |
| Generation evaluation | Disabled |
| Ingestion runtime | 20m 29s |
| Query runtime | 70m 13s |

> [!IMPORTANT]
> This is a **retrieval-first certification run**, not a re-certification of `/ask_vision`, graph-assisted retrieval, or generation faithfulness.

## 2. Retrieval Quality

| Metric | Prior Baseline (2026-03-12) | Retrieval Rebuild Rerun (2026-03-13) | Target | Status |
|---|---|---|---|---|
| **Hit Rate @ 5** | **0.04** | **0.971** | > 0.80 | ✅ EXCEEDS |
| **MRR @ 5** | **0.03** | **0.939** | > 0.70 | ✅ EXCEEDS |

The 4% hit-rate crisis is resolved for this HotpotQA benchmark profile. On the full 1,000-sample rerun, NovaSearch exceeded the original retrieval target by a wide margin.

## 3. Generation & Hallucination Metrics

| Metric | Retrieval Rebuild Rerun | Notes |
|---|---|---|
| **Faithfulness** | **Not evaluated** | `HOTPOT_EVAL_GENERATION=false` |
| **Context Precision** | **Not evaluated** | `HOTPOT_EVAL_GENERATION=false` |

The `0.000` values emitted by the raw benchmark script in earlier runs were a reporting artifact of generation scoring being disabled. They should **not** be interpreted as actual faithfulness or context-precision regressions.

### Supplemental Generation Check (100-sample)

To test these two metrics honestly, I ran a separate **100-question** benchmark with the same text-only retrieval profile, but with benchmark answer generation enabled and LLM-as-judge evaluation turned on.

| Metric | 100-sample Supplemental Run (2026-03-13) | Notes |
|---|---|---|
| **Hit Rate @ 5** | **1.000** | Same retrieval-first profile |
| **MRR @ 5** | **0.985** | Same retrieval-first profile |
| **Faithfulness** | **0.000** | `gpt-4.1-mini` judge, sampled evaluation |
| **Context Precision** | **0.550** | `gpt-4.1-mini` judge, sampled evaluation |

This tells us the retrieval rebuild succeeded, but the **answer synthesis layer is still weak** on HotpotQA-style grounded generation. NovaSearch is now retrieving the right evidence, yet its benchmark answer generation is not producing faithful final answers consistently enough.

## 4. Data & Cost Efficiency

| Metric | Prior Baseline (2026-03-12) | Retrieval Rebuild Rerun (2026-03-13) |
|---|---|---|
| Total Raw Tokens | 1,206,306 | 1,206,672 |
| Retrieved Top-K Tokens | 20,930 | 291,030 |
| Cost Savings | 98.26% | 75.88% |
| Noise Reduction Ratio | 57.6x | 4.1x |

This rerun is **less dehydrated** than the earlier baseline because the system is now retrieving substantially more real evidence. That is the honest tradeoff: NovaSearch is spending more retrieval budget to retrieve the right pages and titles, which is exactly why the retrieval metrics improved so sharply.

## 5. What Changed

- HotpotQA ingestion now preserves original Wikipedia page titles instead of collapsing everything into generic benchmark document names.
- Retrieval benchmarking runs through a text-first path instead of CLIP-centered text retrieval.
- Benchmark mode skips graph and vision subsystems so HotpotQA measures the repaired text retrieval stack directly.
- Chunking and embedding were made thread-safe so large concurrent ingest runs complete reliably instead of silently corrupting the benchmark.

## Conclusion

NovaSearch now clears the HotpotQA retrieval target on a full 1,000-sample rerun for its **retrieval-only text benchmark profile**.

That does **not** mean the entire multimodal production stack has been re-certified. It means the rebuilt text retrieval pipeline is now strong enough to support multi-hop HotpotQA-style evaluation, and future reports must keep multimodal, graph, and generation claims separated unless those subsystems are explicitly rerun.

**Audit Status: RETRIEVAL PROFILE VERIFIED**
*Date: 2026-03-13*
*Samples: 1,000 questions / 9,803 ingested reference pages*
