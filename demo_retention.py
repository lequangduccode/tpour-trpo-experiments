"""Show real top-k results saved by a completed SciFact run (no web UI or invented predictions)."""
import argparse
import csv
import json
from pathlib import Path


def show(root,data,query_id,k):
    root=Path(root); data=Path(data)
    queries={str(r['_id']):r['text'] for r in (json.loads(l) for l in (data/'queries.jsonl').read_text(encoding='utf-8').splitlines())}
    corpus={str(r['_id']):r for r in (json.loads(l) for l in (data/'corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    if query_id not in queries: raise ValueError('Query ID not in SciFact queries')
    gold=set()
    with open(data/'qrels/test.tsv',encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['query-id']==query_id and float(r['score'])>0: gold.add(r['corpus-id'])
    print(f'QUERY {query_id}: {queries[query_id]}\nGold document IDs: {sorted(gold)}')
    for branch in ['baseline','tpour_seed42','ranking_only_seed42','anchored_seed42']:
        path=root/branch/'run.json'
        if not path.exists(): continue
        all_results=json.loads(path.read_text(encoding='utf-8'))
        if query_id not in all_results: raise ValueError('Query was not evaluated in this run')
        results=all_results[query_id]
        print(f'\n{branch} (saved predictions from completed GPU run)')
        for rank,docid in enumerate(sorted(results,key=results.get,reverse=True)[:k],1):
            doc=corpus[docid]; label='RELEVANT' if docid in gold else 'not in positive qrels'
            print(f'{rank}. [{label}] {doc.get("title","")} | ID={docid} | score={results[docid]:.5f}')
            print(doc.get('text','')[:260].replace('\n',' '))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--data',default='data/scifact'); p.add_argument('--query-id',default='1'); p.add_argument('--top-k',type=int,default=3)
    a=p.parse_args()
    if a.top_k<1 or a.top_k>100: p.error('top-k must be 1..100')
    show(a.root,a.data,a.query_id,a.top_k)
