"""Build preregistered anchored-vs-original experiments for Kaggle."""
import nbformat as nb
from pathlib import Path
md=nb.v4.new_markdown_cell; code=nb.v4.new_code_cell
cells=[md('''# TPOUR với ràng buộc giữ kiến thức — kiểm chứng cải tiến

Đề xuất của nhóm: thêm fixed-teacher ranking KL và relative embedding MSE vào CE+TRPO. Knowledge distillation là kỹ thuật có trước; không tuyên bố đây là thuật toán hoàn toàn mới.

Cố định trước khi chạy: gamma=10, distillation temperature=0.1, representation weight=1; 1.000 bước, seed 42/43/44. Không chọn gamma hoặc checkpoint theo SciFact test. So sánh TRPO gốc với bản anchored ở từng seed; thêm ablation chỉ KL ở seed42. SciFact chỉ đo truy xuất tổng quát; preference giữ riêng chưa phải temporal gold.

[Learning without Forgetting](https://arxiv.org/abs/1606.09282), [Weighted KL for ranking](https://arxiv.org/abs/2406.05977).'''),
code('''import os,sys,json,subprocess,hashlib,zipfile,shutil
from pathlib import Path
ROOT=Path('/kaggle/working/tpour-trpo-experiments')
if not ROOT.exists(): subprocess.run(['git','clone','https://github.com/lequangduccode/tpour-trpo-experiments.git',str(ROOT)],check=True)
os.chdir(ROOT)
PIN='main' # Replaced with a fixed tested commit in the uploaded notebook.
subprocess.run(['git','checkout',PIN],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-experiment.txt'],check=True)
import torch
assert torch.cuda.is_available(), 'GPU required'
print('GPU count',torch.cuda.device_count(),torch.cuda.get_device_name(0))
OUT=Path('/kaggle/working/anchor_results'); OUT.mkdir(exist_ok=True)
SEEDS=[42,43,44]; STEPS=1000
def run(*args): subprocess.run([sys.executable,'real_experiment.py',*map(str,args)],check=True)
run('prepare','--input','upstream/TPOUR/2018_sample_train_trimmed.jsonl')
run('download')
subprocess.run([sys.executable,'-m','unittest','test_tpour','test_real_experiment','test_anchored_tpour','-v'],check=True)
'''),
md('''## Baseline và đối chứng

Mọi nhánh dùng cùng tokenizer, max length 256, dot product, không normalize. Hai nhánh mỗi seed chạy song song nếu có hai GPU; mỗi nhánh dùng đúng một GPU. Khác biệt số cập nhật do AMP được lưu trong completed.json.
Loss giữ kiến thức cần thêm một teacher cố định nên chi phí train tăng. Sau khi đánh giá một run thành công, bỏ training_state lớn để tiết kiệm đĩa; vẫn giữ encoder, cấu hình, log và hash. Resume còn dùng được cho run đang dang dở.'''),
code('''def evaluate(folder,checkpoint=None):
    args=['evaluate','--data','data/scifact','--output',folder,'--device','cuda']
    if checkpoint: args+=['--checkpoint',checkpoint]
    run(*args)
def preference(folder,checkpoint=None):
    args=['preference','--data','data/real_experiment/heldout.jsonl','--output',folder/'heldout_preference.json','--device','cuda']
    if checkpoint: args+=['--checkpoint',checkpoint]
    run(*args)
evaluate(OUT/'baseline'); preference(OUT/'baseline')
'''),
code('''def train_pair(seed,variants):
    jobs=[]
    for slot,(name,objective,rep) in enumerate(variants):
        folder=OUT/f'{name}_seed{seed}'; folder.mkdir(exist_ok=True)
        args=['train','--objective',objective,'--output',folder,'--device','cuda','--seed',seed,
              '--steps',STEPS,'--batch-size',4,'--queue-size',4096,'--max-length',256,
              '--warmup',100,'--save-every',500,'--log-every',25,'--amp','--gradient-checkpointing',
              '--anchor-weight',10,'--anchor-temperature',.1,'--representation-weight',rep]
        state=folder/'training_state.pt'
        if state.exists(): args+=['--resume',state]
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(slot%torch.cuda.device_count())
        log=open(folder/'console_train.txt','w',encoding='utf-8')
        job=subprocess.Popen([sys.executable,'real_experiment.py',*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((job,log,folder))
        if torch.cuda.device_count()==1:
            status=job.wait(); log.close()
            if status: raise RuntimeError((folder/'console_train.txt').read_text()[-6000:])
    for job,log,folder in jobs:
        status=job.wait(); log.close()
        print(folder.name,(folder/'console_train.txt').read_text()[-1800:])
        if status: raise RuntimeError((folder/'console_train.txt').read_text()[-6000:])
        evaluate(folder,folder/'encoder.pt'); preference(folder,folder/'encoder.pt')
        (folder/'training_state.pt').unlink() # Only after successful train AND both evaluations; retain encoder.
for seed in SEEDS:
    train_pair(seed,[('tpour','tpour',1),('anchored','anchored',1)])
train_pair(42,[('ranking_only','anchored',0)])
'''),
md('''## Thống kê và demo

Bootstrap ghép cặp theo query cho mỗi seed; trung bình và SD qua ba seed. Báo cả mức giảm so với Contriever gốc.
Demo hiển thị một query cố định và các ví dụ tốt/xấu được chọn tự động; ví dụ minh họa không thay thế kết quả toàn bộ test.
Biểu đồ, bảng và ảnh demo được tạo bằng Python/Matplotlib, không có ảnh AI hoặc giao diện web giả.'''),
code('''for seed in SEEDS:
    for left,right in [('baseline',f'tpour_seed{seed}'),('baseline',f'anchored_seed{seed}'),(f'tpour_seed{seed}',f'anchored_seed{seed}')]:
        run('compare','--left',OUT/left,'--right',OUT/right,'--output',OUT/f'compare_{left}_{right}.json')
run('compare','--left',OUT/'ranking_only_seed42','--right',OUT/'anchored_seed42','--output',OUT/'compare_ranking_only_full_seed42.json')
subprocess.run([sys.executable,'tools/demo_anchored_results.py','--root',str(OUT),'--data','data/scifact','--output',str(OUT/'demo')],check=True)
from IPython.display import Image,display
for p in sorted((OUT/'demo').glob('*.png')): display(Image(filename=str(p)))
archive=Path('/kaggle/working/TPOUR_anchor_evidence.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in {'.json','.jsonl','.png','.txt','.md'}: z.write(p,'results/'+str(p.relative_to(OUT)))
    for p in [Path('data/real_experiment/manifest.json'),Path('data/scifact_download.json')]: z.write(p,str(p))
print('Evidence archive',archive)
''')]
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.validate(book)
path=Path('notebooks/TPOUR_anchored_Kaggle.ipynb'); nb.write(book,path); print(path)
