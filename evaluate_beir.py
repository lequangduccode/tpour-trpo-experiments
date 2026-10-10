"""Evaluate a local BEIR dataset using official BEIR metrics and chunked exact search."""
import argparse
import torch
from tpour import encode_all, load_encoder, save_json


def main():
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Extracted dataset folder, e.g. data/scifact")
    parser.add_argument("--checkpoint", help="Omit for original facebook/contriever baseline")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/beir.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-limit", type=int, help="Smoke test only; retains the FULL corpus")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.query_limit is not None and args.query_limit < 1):
        parser.error("batch-size and query-limit must be positive")
    torch.set_num_threads(2)
    corpus, queries, qrels = GenericDataLoader(data_folder=args.data).load(split=args.split)
    encoder = load_encoder(args.checkpoint, args.device)
    ids = sorted(corpus)
    if args.query_limit:
        queries = dict(list(sorted(queries.items()))[:args.query_limit])
    qrels = {qid: r for qid, r in qrels.items() if qid in queries}
    if not qrels:
        raise ValueError("No judged queries in requested split")
    texts = [(corpus[i].get("title", "") + " " + corpus[i]["text"]).strip() for i in ids]
    embeddings = encode_all(encoder, texts, args.batch_size)
    query_ids = sorted(queries)
    results = {}
    for start in range(0, len(query_ids), args.batch_size):
        batch_ids = query_ids[start:start + args.batch_size]
        q = encode_all(encoder, [queries[i] for i in batch_ids], args.batch_size)
        scores = q @ embeddings.T
        for qid, score in zip(batch_ids, scores):
            # BEIR excludes identical query/document IDs. Exclude before top-k selection.
            if qid in corpus:
                score[ids.index(qid)] = float("-inf")
            values, indices = torch.topk(score, min(100, len(ids)))
            results[qid] = {ids[i]: float(v) for i, v in zip(indices.tolist(), values.tolist())
                            if ids[i] != qid}
    evaluator = EvaluateRetrieval(k_values=[5, 10, 100])
    ndcg, mean_ap, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)
    mrr = evaluator.evaluate_custom(qrels, results, evaluator.k_values, metric="mrr")
    output = {"model": args.checkpoint or "facebook/contriever", "split": args.split,
              "partial_queries": args.query_limit is not None, "corpus_size": len(corpus),
              "evaluated_queries": len(qrels), "ndcg": ndcg, "map": mean_ap,
              "recall": recall, "precision": precision, "mrr": mrr}
    save_json(args.output, output)
    print(output)


if __name__ == "__main__":
    main()
