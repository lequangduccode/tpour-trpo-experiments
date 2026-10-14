"""GPU inference demonstration using checkpoint outputs of the completed retention experiment."""
from pathlib import Path
import nbformat as nb
md=nb.v4.new_markdown_cell; code=nb.v4.new_code_cell
book=nb.v4.new_notebook(cells=[md('''# Demo inference thực từ checkpoint TPOUR

Input là output của notebook kiểm chứng retention. Chạy model trên query thật và tạo corpus index, không chỉ đọc run.json. Đối chiếu top-3 query cố định với rankings đã đánh giá; sau đó thử query nhập mới, chưa có nhãn gold nên không báo accuracy cho query mới.
'''),code('''import os,sys,json,subprocess,zipfile
from pathlib import Path
ROOT=Path('/kaggle/working/tpour-trpo-experiments')
subprocess.run(['git','clone','https://github.com/lequangduccode/tpour-trpo-experiments.git',str(ROOT)],check=True)
os.chdir(ROOT)
PIN='main' # Replaced with fixed commit at upload.
subprocess.run(['git','checkout',PIN],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-experiment.txt'],check=True)
subprocess.run([sys.executable,'real_experiment.py','download'],check=True)
import torch
assert torch.cuda.is_available()
OUT=Path('/kaggle/working/live_demo'); OUT.mkdir(exist_ok=True)
paths=list(Path('/kaggle/input').rglob('encoder.pt'))
print('Input checkpoint folders:',[p.parent.name for p in paths])
def checkpoint(name):
    found=[p for p in paths if p.parent.name==name]
    assert len(found)==1,(name,found)
    return found[0]
'''),code('''verification=[]
for name in ['baseline','tpour_seed42','ranking_only_seed42','anchored_seed42']:
    ckpt=None if name=='baseline' else checkpoint(name)
    args=[sys.executable,'live_retrieval_demo.py','--data','data/scifact','--query-id','1',
          '--cache',str(OUT/f'{name}_index.pt'),'--output',str(OUT/f'{name}_query1.json'),'--device','cuda']
    if ckpt: args+=['--checkpoint',str(ckpt)]
    subprocess.run(args,check=True)
    live=json.loads((OUT/f'{name}_query1.json').read_text())
    source_folder=ckpt.parent if ckpt else checkpoint('tpour_seed42').parent.parent/'baseline'
    if ckpt:
        metric=json.loads((source_folder/'metrics.json').read_text())
        assert live['provenance']['checkpoint_sha256']==metric['checkpoint_sha256']
    stored=json.loads((source_folder/'run.json').read_text())['1']
    expected=sorted(stored,key=stored.get,reverse=True)[:3]
    actual=[r['docid'] for r in live['results']]
    assert actual==expected,(name,actual,expected)
    error=max(abs(r['score']-stored[r['docid']]) for r in live['results'])
    assert error<1e-4,(name,error)
    verification.append({'model':name,'top3_matches_saved_run':True,'max_score_error':error,
                         'checkpoint_sha256_matches_evaluation':True if ckpt else None})
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
'''),md('''## Nhập query mới

Đổi biến QUERY rồi chạy ô này. Dùng index của anchored đã tạo; output là dự đoán thực, không có accuracy khi chưa có qrels cho query đó.'''),code('''QUERY='What influences the mechanical properties of biological materials?'
subprocess.run([sys.executable,'live_retrieval_demo.py','--checkpoint',str(checkpoint('anchored_seed42')),
               '--query',QUERY,'--cache',str(OUT/'anchored_seed42_index.pt'),
               '--output',str(OUT/'anchored_custom_query.json'),'--device','cuda'],check=True)
with zipfile.ZipFile('/kaggle/working/TPOUR_live_demo_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.glob('*.json'): z.write(p,p.name)
print('Demo finished; export JSON evidence and retain the notebook outputs.')
''')],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.validate(book); nb.write(book,Path('notebooks/TPOUR_live_checkpoint_demo.ipynb'))
