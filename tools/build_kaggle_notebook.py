"""Generate the executable Kaggle notebook without output cells or fabricated results."""
from pathlib import Path
import nbformat as nb

md=nb.v4.new_markdown_cell
code=nb.v4.new_code_cell
cells=[md('''# TPOUR/TRPO — thử nghiệm văn bản thật

Bản nâng cấp từ code thực hành và sửa lọc context rỗng của Hạnh (commit `bfc75ca`).
Chọn **GPU** và bật **Internet** trong Kaggle trước khi chạy. Notebook chưa được chạy trên Kaggle bởi người tạo; các ô kết quả để trống.

So sánh Contriever gốc, chỉ CE, và CE+TRPO. Dùng mẫu Wikipedia do tác giả cung cấp; không dùng data mock cho kết luận chất lượng.
SciFact đo truy xuất tổng quát, không chứng minh độ chính xác theo thời gian. Bộ held-out mined preferences chỉ là phép kiểm tra bổ sung.'''),
code('''import os, subprocess, sys
from pathlib import Path
ROOT=Path('/kaggle/working/tpour-trpo-experiments')
if not ROOT.exists():
    subprocess.run(['git','clone','https://github.com/lequangduccode/tpour-trpo-experiments.git',str(ROOT)],check=True)
os.chdir(ROOT)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-experiment.txt'],check=True)
import torch
assert torch.cuda.is_available(), 'Hãy bật GPU trong Notebook Settings rồi chạy lại.'
print(torch.cuda.get_device_name(0), 'VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30,1))
print(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
def run(*args):
    subprocess.run([sys.executable,'real_experiment.py',*map(str,args)],check=True)
OUT=Path('/kaggle/working/real_results'); OUT.mkdir(exist_ok=True)
MODEL='facebook/contriever'
STEPS=1000
SEEDS=[42] # Chạy thử một seed trước; đổi [42,43,44] cho kết quả nhiều seed.
BATCH=4 # Nếu OOM: khởi động lại kernel, giảm 2 và chạy lại cả CE lẫn TRPO.
MAXLEN=256
QUEUE=4096
'''),
md('''## 1. Làm sạch và tách dữ liệu

Chia cố định cùng bộ dữ liệu cho mọi seed huấn luyện. Nhóm các bộ ba có cùng văn bản (sau chuẩn hóa khoảng trắng/chữ hoa), tránh đưa văn bản đó vào cả train và held-out.
Chưa loại được mọi đoạn trích hoặc diễn đạt lại từ cùng bài Wikipedia vì chưa có article ID đáng tin cậy.
Không tự tạo nhãn thời gian. Tệp gốc có 3.000 dòng; bộ lọc hiện giữ 2.992 dòng.'''),
code('''run('prepare','--input','upstream/TPOUR/2018_sample_train_trimmed.jsonl','--output','data/real_experiment')
run('download')
import json
print(json.loads(Path('data/real_experiment/manifest.json').read_text()))
subprocess.run([sys.executable,'-m','unittest','test_tpour','test_real_experiment','-v'],check=True)
'''),
md('''## 2. Contriever gốc: đánh giá đủ SciFact

5.183 tài liệu, 300 truy vấn test; không có tùy chọn giới hạn truy vấn. Dùng cùng truncation 256 token, dot product và không normalize cho tất cả nhánh.
Lưu nDCG, Recall, MAP, MRR, từng truy vấn và top-100. Không chọn siêu tham số theo test.'''),
code('''def evaluate(folder, checkpoint=None):
    args=['evaluate','--data','data/scifact','--output',folder,'--model',MODEL,'--max-length',MAXLEN,'--device','cuda','--batch-size',16]
    if checkpoint: args+=['--checkpoint',checkpoint]
    run(*args)
def preference(folder, checkpoint=None):
    args=['preference','--data','data/real_experiment/heldout.jsonl','--output',folder/'heldout_preference.json','--model',MODEL,'--max-length',MAXLEN,'--device','cuda','--batch-size',16]
    if checkpoint: args+=['--checkpoint',checkpoint]
    run(*args)
evaluate(OUT/'baseline')
preference(OUT/'baseline')
'''),
md('''## 3. Train cùng ngân sách: CE và CE+TRPO

Cùng model khởi tạo, seed, queue, warmup, augmentation, learning rate và số bước. AMP và gradient checkpointing giảm bộ nhớ; chưa xác nhận mức VRAM trên từng GPU Kaggle.
Mặc định 1.000 bước là thử nghiệm mở rộng, **chưa phải cấu hình tái lập đầy đủ bài báo**. Một seed không đủ kết luận vững chắc.
Thời gian phụ thuộc GPU; nhật ký có thời gian và bộ nhớ đỉnh để báo cáo đúng máy chạy.

Mỗi lần chạy lưu encoder để đánh giá và training_state để resume (reference, queue, optimizer, scheduler, RNG). Tệp lớn; theo dõi dung lượng đĩa.'''),
code('''for seed in SEEDS:
    for objective in ['ce','tpour']:
        folder=OUT/f'{objective}_seed{seed}'
        args=['train','--data','data/real_experiment/train.jsonl','--output',folder,'--model',MODEL,
              '--objective',objective,'--device','cuda','--steps',STEPS,'--batch-size',BATCH,
              '--queue-size',QUEUE,'--max-length',MAXLEN,'--seed',seed,'--warmup',min(100,STEPS//10),
              '--save-every',200,'--log-every',25,'--amp','--gradient-checkpointing']
        # Chạy lại sau gián đoạn: phải giữ nguyên cấu hình, thư mục, dữ liệu và tổng số bước.
        state=folder/'training_state.pt'
        if state.exists(): args+=['--resume',state]
        run(*args)
        evaluate(folder,folder/'encoder.pt')
        preference(folder,folder/'encoder.pt')
'''),
md('''## 4. So sánh và vẽ biểu đồ bằng code

Bootstrap ghép cặp theo đúng query ID, 5.000 lần. Khoảng tin cậy bao gồm 0: chưa đủ bằng chứng tăng trên bộ này.
Độ lệch chuẩn qua seed chỉ có khi chạy ít nhất hai seed. Báo cả trường hợp giảm hoặc không đổi.
Không so độ lớn loss CE với loss tổng để kết luận nhánh nào tốt hơn.'''),
code('''for seed in SEEDS:
    pairs=[('baseline',f'ce_seed{seed}'),('baseline',f'tpour_seed{seed}'),(f'ce_seed{seed}',f'tpour_seed{seed}')]
    for left,right in pairs:
        run('compare','--left',OUT/left,'--right',OUT/right,'--output',OUT/f'compare_{left}_{right}.json')
subprocess.run([sys.executable,'tools/summarize_real_runs.py','--root',str(OUT),'--output',str(OUT/'summary')],check=True)
from IPython.display import display, Image
display(Image(filename=str(OUT/'summary/scifact_ndcg.png')))
if (OUT/'summary/training_loss.png').exists(): display(Image(filename=str(OUT/'summary/training_loss.png')))
'''),
md('''## 5. Lưu bằng chứng thực nghiệm

Zip dưới đây chứa số liệu, biểu đồ, manifest và notebook; không chứa checkpoint lớn. Checkpoint tải riêng từ Kaggle Output khi cần.
Trước khi đóng phiên, dùng chức năng lưu phiên bản notebook và kiểm tra output đã được lưu. Phiên mới không tự có checkpoint phiên cũ: gắn output cũ làm input và chép lại đúng thư mục chạy.

Để thuyết trình: đưa biểu đồ nDCG, CI so sánh CE–TRPO, loss, cấu hình và hạn chế; chỉ cập nhật báo cáo sau khi có log thực.'''),
code('''import zipfile
archive=Path('/kaggle/working/TPOUR_ket_qua_thuc.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in {'.json','.jsonl','.png'}: z.write(p,'results/'+str(p.relative_to(OUT)))
    for p in [Path('data/real_experiment/manifest.json'),Path('data/scifact_download.json'),Path('notebooks/TPOUR_real_data_Kaggle.ipynb')]:
        z.write(p,str(p))
print('Tải kết quả:',archive)
print('Checkpoint riêng:', list(OUT.glob('*/encoder.pt')))
''')]
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.validate(book)
path=Path('notebooks/TPOUR_real_data_Kaggle.ipynb'); path.parent.mkdir(exist_ok=True)
nb.write(book,path)
print(path)
