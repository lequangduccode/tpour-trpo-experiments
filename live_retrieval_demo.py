"""Run the trained encoder on a query; cache a real corpus index, with provenance validation."""
import argparse
import json
from pathlib import Path
import torch
from real_experiment import digest,encode,load,read_jsonl,save_json,environment,atomic_save


def demo(args):
    folder=Path(args.data); corpus=sorted(read_jsonl(folder/'corpus.jsonl'),key=lambda r:str(r['_id']))
    ids=[str(r['_id']) for r in corpus]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate corpus IDs')
    query=args.query
    if not query:
        queries={str(r['_id']):r['text'] for r in read_jsonl(folder/'queries.jsonl')}
        query=queries[args.query_id]
    encoder=load(args)
    provenance={'corpus_sha256':digest(folder/'corpus.jsonl'),'checkpoint_sha256':digest(args.checkpoint) if args.checkpoint else None,
                'model':args.model,'max_length':args.max_length,'normalize':args.normalize,
                'pretrained_revision':getattr(getattr(encoder.backbone,'config',None),'_commit_hash',None)}
    cache=Path(args.cache); index=None
    if cache.exists():
        state=torch.load(cache,map_location='cpu',weights_only=True)
        if state['provenance']==provenance and state['ids']==ids: index=state['embeddings']
        else: print('Cache does not match model/corpus/protocol; rebuilding.',flush=True)
    if index is None:
        texts=[(r.get('title','')+' '+r['text']).strip() for r in corpus]
        index=encode(encoder,texts,args.batch_size,'indexing actual corpus')
        atomic_save({'provenance':provenance,'ids':ids,'embeddings':index},cache)
    with torch.no_grad():
        embedding=encoder([query]).float().cpu()
        scores=(embedding@index.T)[0]; ranked=torch.argsort(scores,descending=True,stable=True)[:args.top_k]
    results=[]
    print('LIVE QUERY:',query)
    for rank,i in enumerate(ranked.tolist(),1):
        row=corpus[i]
        result={'rank':rank,'docid':ids[i],'title':row.get('title',''),'excerpt':row['text'][:400],'score':float(scores[i])}
        results.append(result); print(f'{rank}. {result["title"]} | ID={ids[i]} | score={result["score"]:.5f}'); print(result['excerpt'][:220])
    save_json(args.output,{'query':query,'mode':'live encoder inference; not saved benchmark rankings',
                         'provenance':provenance,'environment':environment(),'results':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/scifact'); p.add_argument('--checkpoint'); p.add_argument('--query'); p.add_argument('--query-id',default='1'); p.add_argument('--cache',default='outputs/live_index.pt'); p.add_argument('--output',default='outputs/live_demo.json'); p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--model',default='facebook/contriever'); p.add_argument('--max-length',type=int,default=256); p.add_argument('--normalize',action='store_true'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args(); torch.set_num_threads(2)
    if a.top_k<1 or a.batch_size<1: p.error('top-k and batch-size must be positive')
    demo(a)
