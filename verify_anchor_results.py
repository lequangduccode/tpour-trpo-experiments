"""Independently recompute exported metrics and enforce the preregistered paired controls."""
import argparse
import json
from pathlib import Path
from real_experiment import read_beir,binary_metrics,read_jsonl,digest,save_json,paired_bootstrap


def verify(root,data):
    root=Path(root); data=Path(data)
    corpus,queries,qrels=read_beir(data,'test')
    hashes={f:digest(data/f) for f in ['corpus.jsonl','queries.jsonl','qrels/test.tsv']}
    expected=['baseline']+[f'{method}_seed{seed}' for seed in [42,43,44] for method in ['tpour','anchored']]+['ranking_only_seed42']
    checks=[]; configurations={}; rankings={}; protocol=None
    for name in expected:
        folder=root/name; m=json.loads((folder/'metrics.json').read_text(encoding='utf-8')); run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        assert not m['partial_queries'] and m['evaluated_queries']==300 and m['corpus_size']==5183
        if protocol is None: protocol=m['protocol']
        assert m['protocol']==protocol
        assert m['data_hashes']==hashes and set(run)==set(qrels)==set(m['query_ids'])
        rankings[name]=run
        rows=[]
        for q,ranking in run.items():
            assert set(ranking)<=set(corpus) and len(ranking)==100
            ranked=sorted(ranking,key=ranking.get,reverse=True); r={'query_id':q}
            for k in [5,10,100]: r.update(binary_metrics(ranked,qrels[q],k))
            rows.append(r)
        for key,value in m['metrics'].items(): assert abs(sum(r[key] for r in rows)/300-value)<1e-10,(name,key)
        saved={r['query_id']:r for r in read_jsonl(folder/'per_query.jsonl')}
        for r in rows:
            for key,value in r.items():
                if key!='query_id': assert abs(saved[r['query_id']][key]-value)<1e-10
        if name!='baseline':
            cfg=json.loads((folder/'config.json').read_text(encoding='utf-8')); configurations[name]=cfg
            done=json.loads((folder/'completed.json').read_text(encoding='utf-8')); logs=json.loads((folder/'train_log.json').read_text(encoding='utf-8'))
            assert len(logs)==cfg['steps']==done['steps']==1000
            assert [r['step'] for r in logs]==list(range(1,1001))
            assert sum(not r.get('skipped_nonfinite_gradient',False) for r in logs)==done['optimizer_updates']
            assert done['encoder_sha256']==m['checkpoint_sha256']
        checks.append({'run':name,'metric_recomputation':'passed','queries':300})
    keys=['data','model','device','steps','batch_size','queue_size','max_length','seed','warmup','lr','momentum','temperature','beta','ce_weight','delete_prob','normalize','amp','gradient_checkpointing']
    for seed in [42,43,44]:
        a=configurations[f'tpour_seed{seed}']; b=configurations[f'anchored_seed{seed}']
        assert all(a[k]==b[k] for k in keys)
        assert b['anchor_weight']==10 and b['anchor_temperature']==.1 and b['representation_weight']==1
        comp=json.loads((root/f'compare_tpour_seed{seed}_anchored_seed{seed}.json').read_text(encoding='utf-8'))
        recalculated=paired_bootstrap(read_jsonl(root/f'tpour_seed{seed}/per_query.jsonl'),read_jsonl(root/f'anchored_seed{seed}/per_query.jsonl'))
        assert abs(comp['mean_delta']-recalculated['mean_delta'])<1e-12
        assert all(abs(x-y)<1e-12 for x,y in zip(comp['ci95'],recalculated['ci95']))
    a=configurations['ranking_only_seed42']; b=configurations['anchored_seed42']
    assert all(a[k]==b[k] for k in keys+['anchor_weight','anchor_temperature']) and a['representation_weight']==0
    agreement={}
    for name,run in rankings.items():
        overlaps=[]
        for q in qrels:
            a=rankings['baseline'][q]; b=run[q]
            ta=set(sorted(a,key=a.get,reverse=True)[:10]); tb=set(sorted(b,key=b.get,reverse=True)[:10])
            overlaps.append(len(ta&tb)/10)
        agreement[name]=sum(overlaps)/len(overlaps)
    save_json(root/'rank_agreement.json',{'mean_top10_overlap_with_baseline':agreement,
              'interpretation':'Agreement with fixed teacher rankings on test corpus; not retrieval accuracy or proof of temporal alignment.'})
    result={'checks':checks,'paired_controls':'passed','paired_bootstrap_recomputation':'passed',
            'checkpoint_note':'Hashes agree between train and evaluation records; actual large checkpoint files not independently downloaded here.',
            'limit':'Confirms exported metrics and protocol; not independent reproduction of training or temporal gold evaluation.'}
    if (root/'live_demo/verification.json').exists():
        for name in ['baseline','tpour_seed42','ranking_only_seed42','anchored_seed42']:
            live=json.loads((root/f'live_demo/{name}_query1.json').read_text(encoding='utf-8'))
            m=json.loads((root/name/'metrics.json').read_text(encoding='utf-8'))
            assert live['provenance']['corpus_sha256']==hashes['corpus.jsonl']
            assert live['provenance']['checkpoint_sha256']==m['checkpoint_sha256']
            ranked=rankings[name]['1']; expected_ids=sorted(ranked,key=ranked.get,reverse=True)[:3]
            assert [r['docid'] for r in live['results']]==expected_ids
            assert max(abs(r['score']-ranked[r['docid']]) for r in live['results'])<1e-4
        result['live_inference_evidence']='Four live runs match saved top3 and corpus/checkpoint hashes; checkpoint hashes were computed in the separate GPU inference job.'
    save_json(root/'verification.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--data',default='data/scifact')
    a=p.parse_args(); verify(a.root,a.data)
