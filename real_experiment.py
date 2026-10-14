"""Real-text experiments: fair CE/TRPO training, checkpoint resume, full retrieval and paired uncertainty."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
import urllib.request
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tpour import Encoder, TPOUR, read_jsonl, save_json, write_jsonl


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def canonical(text):
    return ' '.join(text.casefold().split())


def fingerprint(text):
    return hashlib.sha256(canonical(text).encode('utf-8')).hexdigest()


def environment():
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): commit=None
    versions={}
    for name in ('torch','transformers','numpy'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':versions,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'git_commit':commit,'code_sha256':digest(__file__),'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def clean_rows(rows):
    result=[]; stats={'input':len(rows),'missing_context':0,'invalid_text':0,'identical_pair':0,'duplicate_triplet':0}
    seen=set()
    for number,row in enumerate(rows,1):
        if 'query' in row:
            values=[row.get(k) for k in ('query','preferred','rejected')]
        else:
            pos,neg=row.get('positive_ctxs'),row.get('weak_positive_ctxs')
            if not pos or not neg: stats['missing_context']+=1; continue
            def text(v): return v if isinstance(v,str) else v.get('text') if isinstance(v,dict) else None
            values=[row.get('question'),text(pos[0]),text(neg[0])]
        if any(not isinstance(v,str) or not v.strip() for v in values): stats['invalid_text']+=1; continue
        if canonical(values[1])==canonical(values[2]): stats['identical_pair']+=1; continue
        key=tuple(fingerprint(v) for v in values)
        if key in seen: stats['duplicate_triplet']+=1; continue
        seen.add(key)
        result.append(dict(zip(('query','preferred','rejected'),values),source_line=number))
    stats['valid']=len(result)
    return result,stats


def connected_split(rows,seed=42,test_fraction=.15):
    """Keep every repeated exact text in one split; no claim to remove paraphrase/article overlap."""
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b): parent[find(a)]=find(b)
    seen={}
    for i,row in enumerate(rows):
        for key in ('query','preferred','rejected'):
            h=fingerprint(row[key])
            if h in seen: union(i,seen[h])
            seen[h]=i
    components={}
    for i in range(len(rows)): components.setdefault(find(i),[]).append(i)
    groups=list(components.values()); random.Random(seed).shuffle(groups)
    target=max(2,round(len(rows)*test_fraction)); held=[]
    # Greedily take whole small components while leaving enough training rows.
    for group in groups:
        if len(held)<target and len(group)+len(held)<=len(rows)-2:
            held.extend(group)
    if len(held)<2 or len(rows)-len(held)<2: raise ValueError('Cannot form disjoint splits; provide article IDs or a separate held-out corpus')
    held=set(held)
    train=[r for i,r in enumerate(rows) if i not in held]; test=[r for i,r in enumerate(rows) if i in held]
    a={fingerprint(r[k]) for r in train for k in ('query','preferred','rejected')}
    b={fingerprint(r[k]) for r in test for k in ('query','preferred','rejected')}
    assert not a&b
    return train,test,{'components':len(groups),'largest_component':max(map(len,groups)),
                       'train_rows':len(train),'heldout_rows':len(test),'cross_split_exact_text_overlap':0,
                       'limitation':'Exact-text grouping only; shared articles, excerpts and paraphrases may still overlap.'}


def prepare(args):
    rows,stats=clean_rows(read_jsonl(args.input))
    train,test,split=connected_split(rows,args.seed,args.test_fraction)
    out=Path(args.output)
    write_jsonl(out/'train.jsonl',train); write_jsonl(out/'heldout.jsonl',test)
    save_json(out/'manifest.json',{'source_sha256':digest(args.input),'source':str(args.input),
              'seed':args.seed,'cleaning':stats,'split':split,'synthetic':False,
              'heldout_labels':'Mined preferences from author sample, NOT independently annotated temporal QA gold.',
              'reference_hanh_commit':'bfc75ca9857dd8e0a46440ce1229b22230cdefec'})
    print(json.dumps({'cleaning':stats,'split':split},ensure_ascii=False),flush=True)


def download(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    target=out/'scifact'
    if all((target/f).exists() for f in ('corpus.jsonl','queries.jsonl','qrels/test.tsv')):
        print('SciFact already extracted:',target); return
    archive=out/'scifact.zip'
    url='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
    try:
        with urllib.request.urlopen(url,timeout=30) as response, open(archive,'wb') as dest:
            import shutil
            shutil.copyfileobj(response,dest)
    except OSError as exc:
        print('BEIR mirror unavailable; using BeIR datasets on Hugging Face:',exc,flush=True)
        import pyarrow.parquet as pq
        sources=[]
        for part in ('corpus','queries'):
            source=f'https://huggingface.co/datasets/BeIR/scifact/resolve/main/{part}/{part}-00000-of-00001.parquet'
            path=out/f'scifact_{part}.parquet'
            urllib.request.urlretrieve(source,path)
            write_jsonl(target/f'{part}.jsonl',pq.read_table(path).to_pylist())
            sources.append({'url':source,'sha256':digest(path)})
        source='https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv'
        (target/'qrels').mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(source,target/'qrels/test.tsv')
        sources.append({'url':source,'sha256':digest(target/'qrels/test.tsv')})
        save_json(out/'scifact_download.json',{'sources':sources,'environment':environment()})
        print('Downloaded',target); return
    with zipfile.ZipFile(archive) as z:
        base=out.resolve()
        for member in z.infolist():
            resolved=(out/member.filename).resolve()
            if base not in resolved.parents and resolved!=base: raise ValueError('Archive path escapes output directory')
        z.extractall(out)
    save_json(out/'scifact_download.json',{'source':url,'archive_sha256':digest(archive),'environment':environment()})
    print('Downloaded',target)


def atomic_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train(args):
    if args.steps<1 or args.batch_size<1 or args.lr<=0 or args.warmup>=args.steps: raise ValueError('Invalid training schedule')
    if args.warmup<0 or args.save_every<1 or args.log_every<1 or args.queue_size<1 or not 0<=args.delete_prob<1: raise ValueError('Invalid queue, interval or augmentation settings')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Enable a GPU accelerator first, or pass --device cpu')
    rows=read_jsonl(args.data)
    if len(rows)<2: raise ValueError('Too few training rows')
    seed_all(args.seed)
    encoder=Encoder(args.model,args.max_length,args.normalize).to(args.device)
    if args.gradient_checkpointing and args.model!='toy': encoder.backbone.gradient_checkpointing_enable()
    weight=1.0 if args.objective=='ce' else args.ce_weight
    model=TPOUR(encoder,args.queue_size,args.momentum,args.temperature,args.beta,weight).to(args.device)
    optim=torch.optim.AdamW(encoder.parameters(),lr=args.lr,betas=(.9,.98),eps=1e-6)
    def factor(step):
        if step<args.warmup: return (step+1)/max(1,args.warmup)
        return max(0.,(args.steps-step)/max(1,args.steps-args.warmup))
    schedule=torch.optim.lr_scheduler.LambdaLR(optim,factor)
    amp=args.amp and args.device=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    context=lambda:torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy(); config.pop('resume',None)
    # Resume includes current/reference/queue, optimizer, scheduler and RNG states.
    start=0; history=[]
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['config']!=config or state['data_sha256']!=digest(args.data): raise ValueError('Resume config/data differ; use original settings')
        model.load_state_dict(state['model']); optim.load_state_dict(state['optimizer']); schedule.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']); random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['step']; history=state['history']
    else:
        with torch.no_grad():
            for i in range(0,min(len(rows),args.queue_size),args.batch_size):
                batch=rows[i:i+args.batch_size]
                model.enqueue(model.reference([r['preferred'] for r in batch]+[r['rejected'] for r in batch]))
    save_json(output/'config.json',config)
    save_json(output/'provenance.json',{'environment':environment(),'data_sha256':digest(args.data),
              'initial_model':args.model,'synthetic':args.model=='toy','reference': 'TPOUR compact implementation, not exact paper reproduction'})
    started=time.perf_counter()
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    def checkpoint(step):
        atomic_save({'model':args.model,'max_length':args.max_length,'normalize':args.normalize,'state':encoder.state_dict()},output/'encoder.pt')
        atomic_save({'model':model.state_dict(),'optimizer':optim.state_dict(),'scheduler':schedule.state_dict(),
                     'scaler':scaler.state_dict(),'step':step,'history':history,'config':config,'data_sha256':digest(args.data),
                     'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},output/'training_state.pt')
        save_json(output/'train_log.json',history)
    for step in range(start,args.steps):
        batch=random.sample(rows,min(args.batch_size,len(rows)))
        def augment(text):
            words=text.split(); kept=[w for w in words if random.random()>=args.delete_prob]
            return ' '.join(kept) or text
        values=[[augment(r[k]) for r in batch] for k in ('query','preferred','rejected')]
        encoder.train(); optim.zero_grad(set_to_none=True)
        with context(): loss,stats,keys=model(*values)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step+1}')
        scaler.scale(loss).backward(); scaler.unscale_(optim)
        grad=nn.utils.clip_grad_norm_(encoder.parameters(),1.)
        if not torch.isfinite(grad):
            if not amp: raise RuntimeError('Nonfinite gradient')
            scaler.step(optim); scaler.update()
            history.append({'step':step+1,'skipped_nonfinite_gradient':True})
            continue
        scaler.step(optim); scaler.update(); schedule.step(); model.update_reference(); model.enqueue(keys.float())
        stats.update(step=step+1,lr=optim.param_groups[0]['lr'],seconds=time.perf_counter()-started,gradient_norm=float(grad))
        history.append(stats)
        if (step+1)%args.log_every==0 or step==start: print(json.dumps(stats),flush=True)
        if (step+1)%args.save_every==0 or step+1==args.steps: checkpoint(step+1)
    # Always retain a resumable final checkpoint, including skipped AMP steps.
    checkpoint(args.steps)
    save_json(output/'completed.json',{'steps':args.steps,'optimizer_updates':sum(not r.get('skipped_nonfinite_gradient') for r in history),
              'total_seconds':time.perf_counter()-started,'encoder_sha256':digest(output/'encoder.pt'),
              'peak_gpu_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None})


@torch.no_grad()
def encode(encoder,texts,batch_size,label='encoding'):
    chunks=[]
    for start in range(0,len(texts),batch_size):
        chunks.append(encoder(texts[start:start+batch_size]).float().cpu())
        if start%max(batch_size,20*batch_size)==0: print(f'{label}: {start}/{len(texts)}',flush=True)
    if not chunks: raise ValueError('Empty input')
    return torch.cat(chunks)


def load(args):
    # Every evaluated branch uses the same truncation and normalization settings.
    if not args.checkpoint: return Encoder(args.model,args.max_length,args.normalize).to(args.device).eval()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt['max_length']!=args.max_length or ckpt['normalize']!=args.normalize or ckpt['model']!=args.model:
        raise ValueError('Checkpoint and evaluation protocol do not match')
    enc=Encoder(args.model,args.max_length,args.normalize); enc.load_state_dict(ckpt['state'])
    return enc.to(args.device).eval()


def binary_metrics(ranked,relevant,k):
    relevant={x for x,g in relevant.items() if g>0}
    if not relevant: raise ValueError('No positive qrels')
    hits=[i+1 for i,x in enumerate(ranked[:k]) if x in relevant]
    dcg=sum(1/math.log2(i+1) for i in hits)
    ideal=sum(1/math.log2(i+1) for i in range(1,min(k,len(relevant))+1))
    return {f'NDCG@{k}':dcg/ideal,f'Recall@{k}':len(hits)/len(relevant),
            f'MRR@{k}':1/hits[0] if hits else 0.,
            f'MAP@{k}':sum((j+1)/rank for j,rank in enumerate(hits))/len(relevant)}


def read_beir(folder,split):
    folder=Path(folder)
    corpus={str(r['_id']):r for r in read_jsonl(folder/'corpus.jsonl')}
    queries={str(r['_id']):r['text'] for r in read_jsonl(folder/'queries.jsonl')}
    qrels={}
    with open(folder/'qrels'/f'{split}.tsv',encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            grade=int(r['score'])
            if grade not in (0,1): raise ValueError('This metric implementation is restricted to binary qrels; use BEIR for graded datasets')
            if grade: qrels.setdefault(r['query-id'],{})[r['corpus-id']]=grade
    if not qrels or set(qrels)-set(queries) or any(set(rel)-set(corpus) for rel in qrels.values()): raise ValueError('Invalid qrels')
    return corpus,queries,qrels


def evaluate(args):
    encoder=load(args); corpus,queries,qrels=read_beir(args.data,args.split)
    if args.expected_queries and len(qrels)!=args.expected_queries: raise ValueError('Unexpected judged-query count; do not report a partial dataset as full SciFact')
    query_ids=sorted(qrels); ids=sorted(corpus)
    started=time.perf_counter()
    docs=encode(encoder,[(corpus[i].get('title','')+' '+corpus[i]['text']).strip() for i in ids],args.batch_size,'corpus')
    records=[]; runfile={}
    for start in range(0,len(query_ids),args.batch_size):
        qids=query_ids[start:start+args.batch_size]
        scores=encode(encoder,[queries[q] for q in qids],args.batch_size,'queries') @ docs.T
        for qid,score in zip(qids,scores):
            if qid in corpus: score[ids.index(qid)]=float('-inf')
            order=torch.argsort(score,descending=True,stable=True).tolist()
            ranked=[ids[i] for i in order if ids[i]!=qid][:100]
            metrics={}
            for k in (5,10,100): metrics.update(binary_metrics(ranked,qrels[qid],k))
            records.append({'query_id':qid,**metrics})
            runfile[qid]={ids[i]:float(score[i]) for i in order[:min(101,len(order))] if ids[i]!=qid and float(score[i])!=float('-inf')}
            runfile[qid]=dict(list(runfile[qid].items())[:100])
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    averages={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='query_id'}
    save_json(output/'metrics.json',{'dataset':str(args.data),'split':args.split,'corpus_size':len(corpus),
              'evaluated_queries':len(records),'query_ids':query_ids,'partial_queries':False,'metrics':averages,
              'protocol':{'model':args.model,'max_length':args.max_length,'normalize':args.normalize,'score':'dot',
              'metrics':'binary nDCG/MAP/Recall/MRR; exact stable ranking; zero hits retained'},
              'checkpoint_sha256':digest(args.checkpoint) if args.checkpoint else None,
              'data_hashes':{f:digest(Path(args.data)/f) for f in ('corpus.jsonl','queries.jsonl',f'qrels/{args.split}.tsv')},
              'seconds':time.perf_counter()-started,'environment':environment()})
    write_jsonl(output/'per_query.jsonl',records); save_json(output/'run.json',runfile)
    print(json.dumps(averages),flush=True)


def preference(args):
    rows=read_jsonl(args.data); enc=load(args)
    q=encode(enc,[r['query'] for r in rows],args.batch_size)
    p=encode(enc,[r['preferred'] for r in rows],args.batch_size)
    r=encode(enc,[r['rejected'] for r in rows],args.batch_size)
    gap=(q*(p-r)).sum(-1)
    save_json(args.output,{'examples':len(rows),'pairwise_preference_accuracy':float((gap>0).float().mean()),
              'ties':int((gap==0).sum()),'mean_gap':float(gap.mean()),'per_pair_gap':gap.tolist(),
              'kind':'heldout mined-preference proxy, not gold temporal QA','data_sha256':digest(args.data)})


def paired_bootstrap(a,b,key='NDCG@10',seed=123,iterations=5000):
    a={r['query_id']:r[key] for r in a}; b={r['query_id']:r[key] for r in b}
    if a.keys()!=b.keys(): raise ValueError('Cannot compare results with different judged queries')
    ids=sorted(a); delta=np.array([b[i]-a[i] for i in ids])
    rng=np.random.default_rng(seed); boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(iterations)])
    return {'metric':key,'queries':len(ids),'mean_delta':float(delta.mean()),
            'ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'bootstrap_iterations':iterations,
            'positive_queries':int((delta>0).sum()),'negative_queries':int((delta<0).sum()),'tied_queries':int((delta==0).sum()),
            'interpretation':'Query-level paired uncertainty for this dataset/seed; not proof of temporal alignment.'}


def compare(args):
    a=read_jsonl(Path(args.left)/'per_query.jsonl'); b=read_jsonl(Path(args.right)/'per_query.jsonl')
    left=json.loads((Path(args.left)/'metrics.json').read_text()); right=json.loads((Path(args.right)/'metrics.json').read_text())
    for key in ('protocol','data_hashes','split','corpus_size'):
        if left[key]!=right[key]: raise ValueError(f'Mismatched evaluation protocol: {key}')
    result=paired_bootstrap(a,b,args.metric)
    result['left']=args.left; result['right']=args.right; save_json(args.output,result); print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--input',required=True); a.add_argument('--output',default='data/real_experiment'); a.add_argument('--seed',type=int,default=42); a.add_argument('--test-fraction',type=float,default=.15)
    a=sub.add_parser('download'); a.add_argument('--output',default='data')
    a=sub.add_parser('train')
    for name,default in [('data','data/real_experiment/train.jsonl'),('output','outputs/real/tpour_seed42'),('model','facebook/contriever'),('objective','tpour'),('device','cuda' if torch.cuda.is_available() else 'cpu')]: a.add_argument('--'+name,default=default)
    for name,default in [('steps',1000),('batch-size',4),('queue-size',4096),('max-length',256),('seed',42),('warmup',100),('save-every',200),('log-every',25)]: a.add_argument('--'+name,type=int,default=default)
    for name,default in [('lr',1e-6),('momentum',.9999),('temperature',.05),('beta',.3),('ce-weight',.925),('delete-prob',.1)]: a.add_argument('--'+name,type=float,default=default)
    a.add_argument('--normalize',action='store_true'); a.add_argument('--amp',action='store_true'); a.add_argument('--gradient-checkpointing',action='store_true'); a.add_argument('--resume')
    for command in ('evaluate','preference'):
        a=sub.add_parser(command); a.add_argument('--data',required=True); a.add_argument('--output',required=True)
        a.add_argument('--checkpoint'); a.add_argument('--model',default='facebook/contriever'); a.add_argument('--max-length',type=int,default=256); a.add_argument('--normalize',action='store_true'); a.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a.add_argument('--batch-size',type=int,default=16)
        if command=='evaluate': a.add_argument('--split',default='test'); a.add_argument('--expected-queries',type=int,default=300)
    a=sub.add_parser('compare'); a.add_argument('--left',required=True); a.add_argument('--right',required=True); a.add_argument('--output',required=True); a.add_argument('--metric',default='NDCG@10')
    args=p.parse_args(); torch.set_num_threads(2)
    if args.command=='train' and args.objective not in ('ce','tpour'): p.error('objective must be ce or tpour')
    if hasattr(args,'batch_size') and args.batch_size<1: p.error('batch-size must be positive')
    if args.command=='prepare' and not 0<args.test_fraction<.5: p.error('test-fraction must be in (0,.5)')
    globals()[args.command](args)


if __name__=='__main__': main()
