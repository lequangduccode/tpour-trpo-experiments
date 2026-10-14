"""Static, reproducible result demo from real rankings, with positive AND negative examples."""
import argparse
import csv
import json
import statistics
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def build(root,data,output):
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    groups={}; protocol=None; hashes=None
    for folder in sorted(root.iterdir()):
        path=folder/'metrics.json'
        if not path.exists(): continue
        result=read(path)
        if result['evaluated_queries']!=300 or result['corpus_size']!=5183 or result['partial_queries']: raise ValueError('Incomplete SciFact test')
        if protocol is None: protocol=result['protocol']; hashes=result['data_hashes']
        if result['protocol']!=protocol or result['data_hashes']!=hashes: raise ValueError('Different protocols/data')
        name=folder.name.split('_seed')[0]
        groups.setdefault(name,[]).append((folder,result))
    summary={}
    for name,runs in groups.items():
        vals=[r['metrics']['NDCG@10'] for _,r in runs]
        summary[name]={'runs':[p.name for p,_ in runs],'n':len(vals),'mean':statistics.mean(vals),
                       'sample_sd':statistics.stdev(vals) if len(vals)>1 else None}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    names=[n for n in ['baseline','tpour','ranking_only','anchored'] if n in summary]
    fig,ax=plt.subplots(figsize=(8,4.5))
    values=[summary[n]['mean'] for n in names]; errs=[summary[n]['sample_sd'] or 0 for n in names]
    bars=ax.bar(names,values,yerr=errs,capsize=5,color=['#777777','#355c7d','#b08854','#9b3946'][:len(names)])
    ax.set_ylim(0,1); ax.set_ylabel('nDCG@10'); ax.set_title('SciFact: 300 queries / 5,183 documents; error bars = SD across seeds')
    for b,n,v in zip(bars,names,values): ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.4f}\nn={summary[n]["n"]}',ha='center')
    fig.tight_layout(); fig.savefig(output/'01_full_test.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name in ['tpour','ranking_only','anchored']:
        if name not in groups: continue
        for folder,_ in groups[name]:
            rows=read(folder/'train_log.json'); good=[r for r in rows if 'loss' in r]
            axs[0].plot([r['step'] for r in good],[r['loss'] for r in good],label=folder.name,alpha=.7)
            if name=='anchored': axs[1].plot([r['step'] for r in good],[r['anchor_loss'] for r in good],label=folder.name,alpha=.7)
    axs[0].set_title('Training objectives (not directly comparable)'); axs[1].set_title('Retention loss during anchored training')
    for ax in axs: ax.set_xlabel('Step'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'02_training.png',dpi=180); plt.close(fig)
    corpus={str(r['_id']):r for r in (json.loads(l) for l in (Path(data)/'corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    queries={str(r['_id']):r['text'] for r in (json.loads(l) for l in (Path(data)/'queries.jsonl').read_text(encoding='utf-8').splitlines())}
    gold={}
    with open(Path(data)/'qrels/test.tsv',encoding='utf-8') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if float(row['score'])>0: gold.setdefault(row['query-id'],set()).add(row['corpus-id'])
    a={r['query_id']:r for r in (json.loads(l) for l in (root/'tpour_seed42/per_query.jsonl').read_text().splitlines())}
    b={r['query_id']:r for r in (json.loads(l) for l in (root/'anchored_seed42/per_query.jsonl').read_text().splitlines())}
    ids=sorted(a); deltas={q:b[q]['NDCG@10']-a[q]['NDCG@10'] for q in ids}
    chosen=[('fixed first query',ids[0]),('largest improvement (selected example)',max(ids,key=deltas.get)),
            ('largest decrease / smallest change (selected example)',min(ids,key=deltas.get))]
    demos=[]
    for index,(reason,q) in enumerate(chosen,3):
        fig,axs=plt.subplots(1,2,figsize=(13,6)); fig.suptitle(textwrap.fill(f'{reason} | query {q}: {queries[q]}',110),fontsize=11)
        entry={'selection':reason,'query_id':q,'query':queries[q],'ndcg_delta':deltas[q],'results':{}}
        for ax,method in zip(axs,['tpour','anchored']):
            ranked=read(root/f'{method}_seed42/run.json')[q]
            top=sorted(ranked,key=ranked.get,reverse=True)[:3]
            lines=[]; entry['results'][method]=[]
            for rank,docid in enumerate(top,1):
                doc=corpus[docid]; excerpt=doc.get('text','')[:330]
                item={'rank':rank,'docid':docid,'title':doc.get('title',''),'excerpt':excerpt,'score':ranked[docid],
                      'gold_relevant':docid in gold[q]}
                entry['results'][method].append(item)
                label='GOLD RELEVANT' if item['gold_relevant'] else 'not in positive qrels'
                lines.append(textwrap.fill(f'{rank}. {item["title"]} [ID {docid}; {label}]',63)+'\n'+textwrap.fill(excerpt,70))
            metric=a[q]['NDCG@10'] if method=='tpour' else b[q]['NDCG@10']
            ax.set_title(f'{method} | nDCG@10={metric:.4f}')
            ax.text(0,1,'\n\n'.join(lines),va='top',fontsize=9,transform=ax.transAxes); ax.axis('off')
        fig.tight_layout(rect=[0,0,1,.9]); fig.savefig(output/f'{index:02d}_ranking_demo.png',dpi=160); plt.close(fig)
        demos.append(entry)
    (output/'ranking_examples.json').write_text(json.dumps(demos,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); build(a.root,a.data,a.output)
