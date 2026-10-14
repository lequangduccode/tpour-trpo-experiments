"""Aggregate actual SciFact runs and draw ordinary Matplotlib figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(root,output):
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    groups={}; protocol=None; hashes=None
    for folder in sorted(root.iterdir()):
        path=folder/'metrics.json'
        if not path.exists(): continue
        if folder.name!='baseline' and not folder.name.startswith(('ce_seed','tpour_seed')): continue
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['partial_queries'] or result['evaluated_queries']!=300: raise ValueError('Expected all 300 SciFact test queries')
        if protocol is None: protocol=result['protocol']; hashes=result['data_hashes']
        if result['protocol']!=protocol or result['data_hashes']!=hashes: raise ValueError('Mismatched evaluation protocols/data')
        name='baseline' if folder.name=='baseline' else folder.name.split('_seed')[0]
        groups.setdefault(name,[]).append((folder.name,result))
    if not groups: raise ValueError('No completed evaluation runs')
    summary={}
    for name,items in groups.items():
        values=[r['metrics']['NDCG@10'] for _,r in items]
        summary[name]={'runs':[f for f,_ in items],'n':len(items),'mean_NDCG@10':float(np.mean(values)),
                       'std_NDCG@10':float(np.std(values,ddof=1)) if len(values)>1 else None}
    (output/'summary.json').write_text(json.dumps({'groups':summary,'protocol':protocol,
        'data_hashes':hashes,'interpretation':'Sample SD across seeds; absent for a single run. No invented scores.'},indent=2),encoding='utf-8')
    names=list(summary); vals=[summary[n]['mean_NDCG@10'] for n in names]
    errs=[summary[n]['std_NDCG@10'] or 0 for n in names]
    fig,ax=plt.subplots(figsize=(7,4)); bars=ax.bar(names,vals,yerr=errs,capsize=4,color=['#888888','#355c7d','#96364a'][:len(names)])
    ax.set_ylim(0,1); ax.set_ylabel('SciFact nDCG@10'); ax.set_title('Full test: 300 queries / 5,183 documents')
    for bar,n,v in zip(bars,names,vals): ax.text(bar.get_x()+bar.get_width()/2,v+.02,f'{v:.4f}\nn={summary[n]["n"]}',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(output/'scifact_ndcg.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); count=0
    for folder in sorted(root.iterdir()):
        log=folder/'train_log.json'
        if not log.exists(): continue
        rows=[r for r in json.loads(log.read_text(encoding='utf-8')) if 'loss' in r]
        if rows: ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],label=folder.name,alpha=.8); count+=1
    if count:
        ax.set_xlabel('Step'); ax.set_ylabel('Training objective'); ax.legend(fontsize=8)
        ax.set_title('Loss traces (different objectives: compare retrieval metrics)')
        fig.tight_layout(); fig.savefig(output/'training_loss.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); summarize(a.root,a.output)
