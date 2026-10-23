import json, sys, torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

def build_evaluator(path="data/eval.jsonl", name="classroom"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    corpus, queries, relevant, doc_id = {}, {}, {}, {}
    for i, r in enumerate(rows):
        d = doc_id.setdefault(r["positive"], f"d{len(doc_id)}")
        corpus[d] = r["positive"]
        queries[f"q{i}"] = r["anchor"]
        relevant.setdefault(f"q{i}", set()).add(d)
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name,
        query_prompt_name="SearchQuery", corpus_prompt_name="Document",
        accuracy_at_k=[1, 3, 5], truncate_dim=None)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"
    model = SentenceTransformer(path, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"vision_config": None, "audio_config": None})
    res = build_evaluator()(model)
    for k in ["accuracy@1", "accuracy@3", "ndcg@10"]:
        print(k, round(res[f"classroom_cosine_{k}"], 3))
