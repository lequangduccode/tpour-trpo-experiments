"""Small, single-device implementation of paper equations 3--5; not an exact reproduction."""
import argparse
import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Encoder(nn.Module):
    def __init__(self, model="toy", max_length=256, normalize=False):
        super().__init__()
        self.model_id, self.max_length, self.normalize = model, max_length, normalize
        if model == "toy":
            self.backbone = nn.Embedding(4096, 64, padding_idx=0)
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.backbone = AutoModel.from_pretrained(model)

    @property
    def dim(self):
        return 64 if self.model_id == "toy" else self.backbone.config.hidden_size

    def forward(self, texts):
        device = next(self.parameters()).device
        if self.model_id == "toy":
            tokens = [re.findall(r"\w+", text.lower())[:self.max_length] for text in texts]
            ids = [[1 + int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little") % 4095
                    for w in words] or [1] for words in tokens]
            batch = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long, device=device)
            for i, row in enumerate(ids):
                batch[i, :len(row)] = torch.tensor(row, device=device)
            mask = batch.ne(0)
            hidden = self.backbone(batch)
        else:
            batch = self.tokenizer(texts, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(device)
            mask = batch["attention_mask"].bool()
            hidden = self.backbone(**batch).last_hidden_state
        emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(emb, dim=-1) if self.normalize else emb


def trpo_loss(q, preferred, rejected, rq, rp, rr, beta=0.3, temperature=0.05):
    gap = (q * (preferred - rejected)).sum(-1) / temperature
    ref_gap = (rq * (rp - rr)).sum(-1).detach() / temperature
    return -F.logsigmoid(beta * (gap - ref_gap)).mean()


class TPOUR(nn.Module):
    def __init__(self, encoder, queue_size=256, momentum=0.9999, temperature=0.05,
                 beta=0.3, ce_weight=0.925):
        super().__init__()
        if queue_size < 1 or not 0 <= momentum < 1 or temperature <= 0 or beta <= 0:
            raise ValueError("Invalid queue, momentum, temperature or beta")
        if not 0 <= ce_weight <= 1:
            raise ValueError("ce_weight must be in [0, 1]")
        self.encoder = encoder
        self.reference = copy.deepcopy(encoder).requires_grad_(False).eval()
        self.momentum, self.temperature, self.beta, self.ce_weight = momentum, temperature, beta, ce_weight
        self.register_buffer("queue", torch.zeros(queue_size, encoder.dim))
        self.register_buffer("pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_reference(self):
        for ref, current in zip(self.reference.parameters(), self.encoder.parameters()):
            ref.mul_(self.momentum).add_(current, alpha=1 - self.momentum)

    @torch.no_grad()
    def enqueue(self, keys):
        size = len(self.queue)
        for chunk in keys.split(size):
            start, n = int(self.pointer), len(chunk)
            first = min(size - start, n)
            self.queue[start:start + first].copy_(chunk[:first])
            if first < n:
                self.queue[:n - first].copy_(chunk[first:])
            self.pointer.fill_((start + n) % size)
            self.count.fill_(min(size, int(self.count) + n))

    def forward(self, queries, preferred, rejected, return_embeddings=False):
        self.reference.eval()
        q, p, r = self.encoder(queries), self.encoder(preferred), self.encoder(rejected)
        with torch.no_grad():
            rq, rp, rr = self.reference(queries), self.reference(preferred), self.reference(rejected)
        positive = (q * p).sum(-1, keepdim=True)
        # Queue contains previous-batch reference document embeddings, as in Eq. 3.
        negatives = q @ self.queue[:int(self.count)].detach().clone().T
        logits = torch.cat([positive, negatives], dim=1) / self.temperature
        ce = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=q.device))
        preference = trpo_loss(q, p, r, rq, rp, rr, self.beta, self.temperature)
        total = self.ce_weight * ce + (1 - self.ce_weight) * preference
        result = (total, {"loss": float(total.detach()), "ce": float(ce.detach()),
                       "trpo": float(preference.detach())}, torch.cat([rp, rr]).detach())
        return result + ((q, p, r),) if return_embeddings else result


def normalize_triplets(rows):
    result = []
    for i, row in enumerate(rows):
        if "query" in row:
            item = {k: row[k] for k in ("query", "preferred", "rejected")}
        else:
            def text(ctx):
                return ctx if isinstance(ctx, str) else ctx["text"]
            if not row.get("positive_ctxs") or not row.get("weak_positive_ctxs"):
                continue
            item = {"query": row["question"], "preferred": text(row["positive_ctxs"][0]),
                    "rejected": text(row["weak_positive_ctxs"][0])}
        if any(not isinstance(x, str) or not x.strip() for x in item.values()):
            raise ValueError(f"Row {i}: empty/non-string text")
        if item["preferred"] == item["rejected"]:
            continue
        result.append(item)
    if len(result) < 2:
        raise ValueError("Need at least two distinct valid triplets")
    return result


def train(encoder, rows, args, output):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder.to(args.device)
    model = TPOUR(encoder, args.queue_size, args.momentum, args.temperature, args.beta, args.ce_weight)
    model.to(args.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    # Warm up with real document keys: avoid a zero-negative first training step.
    with torch.no_grad():
        for start in range(0, min(len(rows), args.queue_size), args.batch_size):
            batch = rows[start:start + args.batch_size]
            model.enqueue(model.reference([r["preferred"] for r in batch] + [r["rejected"] for r in batch]))
    logs = []
    for step in range(args.steps):
        batch = random.sample(rows, min(args.batch_size, len(rows)))
        encoder.train()
        def augment(text):
            words = text.split()
            kept = [w for w in words if random.random() >= args.delete_prob]
            return " ".join(kept) if kept else text
        q = [augment(r["query"]) for r in batch]
        p = [augment(r["preferred"]) for r in batch]
        r = [augment(r["rejected"]) for r in batch]
        optimizer.zero_grad(set_to_none=True)
        loss, stats, keys = model(q, p, r)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        model.update_reference()
        model.enqueue(keys)
        stats["step"] = step + 1
        logs.append(stats)
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(json.dumps(stats), flush=True)
    encoder.eval()
    Path(output).mkdir(parents=True, exist_ok=True)
    torch.save({"model": encoder.model_id, "max_length": encoder.max_length,
                "normalize": encoder.normalize, "state": encoder.state_dict()}, Path(output) / "encoder.pt")
    save_json(Path(output) / "train_log.json", logs)
    save_json(Path(output) / "config.json", vars(args))
    return encoder


def load_encoder(checkpoint, device):
    if checkpoint is None:
        return Encoder("facebook/contriever").to(device).eval()
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder = Encoder(obj["model"], obj["max_length"], obj["normalize"])
    encoder.load_state_dict(obj["state"])
    return encoder.to(device).eval()


@torch.no_grad()
def encode_all(encoder, texts, batch_size=16):
    if not texts:
        raise ValueError("Cannot encode an empty collection")
    return torch.cat([encoder(texts[i:i + batch_size]).cpu()
                      for i in range(0, len(texts), batch_size)])


def metrics(ranked, relevant, k):
    top = ranked[:k]
    gains = [relevant.get(doc, 0) for doc in top]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted([g for g in relevant.values() if g > 0], reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    hits = [i for i, g in enumerate(gains) if g > 0]
    return {f"nDCG@{k}": dcg / idcg if idcg else 0,
            f"Recall@{k}": len(hits) / sum(g > 0 for g in relevant.values()),
            f"MRR@{k}": 1 / (hits[0] + 1) if hits else 0}


def evaluate(encoder, folder, batch_size=16):
    folder = Path(folder)
    corpus, queries = read_jsonl(folder / "corpus.jsonl"), read_jsonl(folder / "queries.jsonl")
    ids = [str(d["_id"]) for d in corpus]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs")
    qids = [str(q["_id"]) for q in queries]
    if len(set(qids)) != len(qids):
        raise ValueError("Duplicate query IDs")
    qrels = {}
    for row in read_jsonl(folder / "qrels.jsonl"):
        if float(row["score"]) > 0:
            qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = float(row["score"])
    if not qrels or set(qrels) - set(qids) or any(set(r) - set(ids) for r in qrels.values()):
        raise ValueError("Empty or invalid qrels")
    docs = encode_all(encoder, [(d.get("title", "") + " " + d["text"]).strip() for d in corpus], batch_size)
    sums, rankings, count = {}, {}, 0
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        scores = encode_all(encoder, [q["text"] for q in batch], batch_size) @ docs.T
        for query, score in zip(batch, scores):
            qid = str(query["_id"])
            ranked = [ids[i] for i in torch.argsort(score, descending=True, stable=True).tolist()]
            rankings[qid] = ranked[:10]
            if qid not in qrels:
                continue
            count += 1
            for k in (5, 10):
                for name, value in metrics(ranked, qrels[qid], k).items():
                    sums[name] = sums.get(name, 0) + value
    return {"metrics": {n: v / count for n, v in sums.items()}, "evaluated_queries": count,
            "corpus_size": len(corpus), "rankings": rankings}


def demo(args):
    output = Path(args.output)
    rows, corpus, queries, qrels = [], [], [], []
    # Fictional entities. Entity-disjoint split; shared templates deliberately make this a smoke test.
    for i in range(40):
        entity = f"Organization{i}"
        old = f"{entity} uses paper forms and provides office desk service."
        new = f"{entity} uses a digital portal and provides online mobile service."
        query = f"How does {entity} currently provide its service?"
        if i < 32:
            rows.append({"query": query, "preferred": new, "rejected": old})
        else:
            corpus.extend([{"_id": f"{i}-old", "text": old, "timestamp": "2018"},
                           {"_id": f"{i}-new", "text": new, "timestamp": "2021"}])
            queries.append({"_id": str(i), "text": query})
            qrels.append({"query_id": str(i), "doc_id": f"{i}-new", "score": 1})
    write_jsonl(output / "data/train.jsonl", rows)
    for name, data in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        write_jsonl(output / f"data/test/{name}.jsonl", data)
    torch.manual_seed(args.seed)
    initial = Encoder("toy", normalize=True)
    initial.eval()
    baseline = evaluate(initial, output / "data/test")
    settings = copy.copy(args)
    settings.ce_weight = 1.0
    ce = train(copy.deepcopy(initial), rows, settings, output / "ce_only")
    tp = train(copy.deepcopy(initial), rows, args, output / "tpour")
    results = {"kind": "synthetic_cpu_smoke_test_not_paper_reproduction", "seed": args.seed,
               "untrained": baseline, "ce_only": evaluate(ce, output / "data/test"),
               "tpour": evaluate(tp, output / "data/test")}
    save_json(output / "results.json", results)
    print(json.dumps({k: v["metrics"] for k, v in results.items() if isinstance(v, dict)}, indent=2))


def prepare(args):
    rows = read_jsonl(args.input)
    if any(not {"_id", "timestamp", "text"} <= r.keys() for r in rows):
        raise ValueError("Raw documents need _id, timestamp, text")
    # Remove every document ID present in held-out eval before mining.
    heldout = {str(r["_id"]) for r in read_jsonl(args.exclude)} if args.exclude else set()
    rows = [r for r in rows if str(r["_id"]) not in heldout]
    enc = Encoder(args.model, normalize=True).to(args.device).eval()
    vectors = encode_all(enc, [r["text"] for r in rows], args.batch_size)
    triples = []
    for i, row in enumerate(rows):
        if str(row["timestamp"]) != args.target_time:
            continue
        words = row["text"].split()
        if len(words) < 20:
            continue
        mid = len(words) // 2
        query, preferred = " ".join(words[:mid]), " ".join(words[mid:])
        candidates = [j for j, r in enumerate(rows) if str(r["timestamp"]) != args.target_time
                      and r["text"] != row["text"]]
        if not candidates:
            continue
        q = encode_all(enc, [query])[0]
        scores = vectors[candidates] @ q
        j = candidates[int(scores.argmax())]
        triples.append({"query": query, "preferred": preferred, "rejected": rows[j]["text"],
                        "source_id": str(row["_id"]), "rejected_id": str(rows[j]["_id"]),
                        "target_time": args.target_time, "mining_score": float(scores.max())})
    normalize_triplets(triples)
    write_jsonl(args.output, triples)
    print(f"Prepared {len(triples)} triplets; excluded {len(heldout)} held-out IDs")


def interpolate(args):
    a, b = [torch.load(p, map_location="cpu", weights_only=True) for p in (args.left, args.right)]
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    for key in ("model", "max_length", "normalize"):
        if a[key] != b[key]:
            raise ValueError("Checkpoints have incompatible metadata")
    if a["state"].keys() != b["state"].keys():
        raise ValueError("Checkpoints have incompatible keys")
    for key, left in a["state"].items():
        right = b["state"][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Incompatible tensor: {key}")
        if left.is_floating_point():
            a["state"][key] = (1 - args.alpha) * left + args.alpha * right
        elif not torch.equal(left, right):
            raise ValueError(f"Non-floating buffer mismatch: {key}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(a, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "train"):
        p = subs.add_parser(command)
        p.add_argument("--output", default=f"outputs/{command}")
        p.add_argument("--steps", type=int, default=150 if command == "demo" else 100)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--lr", type=float, default=0.01 if command == "demo" else 1e-6)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--queue-size", type=int, default=256)
        p.add_argument("--momentum", type=float, default=0.9999)
        p.add_argument("--temperature", type=float, default=0.05)
        p.add_argument("--beta", type=float, default=0.3)
        p.add_argument("--ce-weight", type=float, default=0.925)
        p.add_argument("--delete-prob", type=float, default=0 if command == "demo" else 0.1)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if command == "train":
            p.add_argument("--data", required=True)
            p.add_argument("--model", default="facebook/contriever")
            p.add_argument("--max-length", type=int, default=256)
            p.add_argument("--normalize", action="store_true")
    p = subs.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output", default="outputs/evaluation.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p = subs.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--exclude", help="Held-out corpus.jsonl; all versions of its IDs are excluded")
    p.add_argument("--target-time", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="facebook/contriever")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p = subs.add_parser("interpolate")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if hasattr(args, "steps") and (args.steps < 1 or args.batch_size < 1 or args.lr <= 0
                                 or not 0 <= args.delete_prob < 1):
        parser.error("Invalid steps, batch size, learning rate or deletion probability")
    if args.command == "demo":
        demo(args)
    elif args.command == "train":
        torch.manual_seed(args.seed)
        train(Encoder(args.model, args.max_length, args.normalize),
              normalize_triplets(read_jsonl(args.data)), args, args.output)
    elif args.command == "evaluate":
        results = evaluate(load_encoder(args.checkpoint, args.device), args.data, args.batch_size)
        save_json(args.output, results)
        print(json.dumps(results["metrics"], indent=2))
    elif args.command == "prepare":
        prepare(args)
    else:
        interpolate(args)


if __name__ == "__main__":
    main()
