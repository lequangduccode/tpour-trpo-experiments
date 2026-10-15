"""Plain academic defense deck; editable shapes and charts from measured JSON only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from docx import Document
from docx.shared import Pt as WP

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/anchored_kaggle'
OUT=ROOT/'reports/presentation_final'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
NAVY='233E57';BLUE='39779B';RED='A44C47';GREY='59636D';INK='202A32';LIGHT='EEF2F4'
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
notes=[]
def box(s,text,x,y,w,h,size=24,bold=False,color=INK):
    sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=sh.text_frame;tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.03)
    tf.margin_top=tf.margin_bottom=Inches(.02)
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
        p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
        p.space_after=Pt(10)
    return sh
def rect(s,x,y,w,h,fill=LIGHT):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill);sh.line.fill.background();return sh
def slide(title,note,source='Báo cáo cập nhật · results/anchored_kaggle',appendix=False):
    s=prs.slides.add_slide(prs.slide_layouts[6]);rect(s,.55,.5,.08,.5,NAVY)
    box(s,title,.8,.42,11.8,.85,29,True,NAVY)
    box(s,source,.65,7.08,11.7,.25,10,color=GREY)
    box(s,str(len(prs.slides)),12.25,7.04,.5,.3,12,color=GREY)
    s.notes_slide.notes_text_frame.text=note
    notes.append((title,note,appendix));return s
def bullets(s,items,x=.85,y=1.6,w=11.6,size=25,gap=1.03):
    for i,t in enumerate(items):
        box(s,'•',x,y+i*gap,.25,.5,size,color=BLUE)
        box(s,t,x+.35,y+i*gap,w-.35,gap-.1,size)
def takeaway(s,text,y=6.35):
    rect(s,.8,y,11.7,.52);box(s,text,1,y+.06,11.3,.42,18,True,NAVY)
def table(s,headers,rows,x=.85,y=1.7,widths=None,rowh=.64,size=20):
    widths=widths or [11.6/len(headers)]*len(headers)
    sh=s.shapes.add_table(len(rows)+1,len(headers),Inches(x),Inches(y),Inches(sum(widths)),Inches(rowh*(len(rows)+1)))
    for c,w in zip(sh.table.columns,widths):c.width=Inches(w)
    for i,row in enumerate([headers]+rows):
        for j,v in enumerate(row):
            c=sh.table.cell(i,j);c.text=str(v);c.margin_left=Inches(.12);c.margin_top=Inches(.09)
            c.fill.solid();c.fill.fore_color.rgb=RGBColor.from_string(NAVY if i==0 else ('F0F3F5' if i%2 else 'FFFFFF'))
            for p in c.text_frame.paragraphs:
                p.font.name='Arial';p.font.size=Pt(size);p.font.bold=i==0;p.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else INK)
    return sh
def picture(s,path,x,y,w,h):
    from PIL import Image
    iw,ih=Image.open(path).size;scale=min(w/iw,h/ih)
    return s.shapes.add_picture(str(path),Inches(x+(w-iw*scale)/2),Inches(y+(h-ih*scale)/2),width=Inches(iw*scale),height=Inches(ih*scale))
def mathfig(text,name):
    fig=plt.figure(figsize=(12,1.1));fig.text(.5,.5,text,fontsize=26,ha='center',va='center');fig.savefig(OUT/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return OUT/name

g=read(R/'demo/summary.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8.6,4.3));vals=[g[k]['mean'] for k in ['baseline','tpour','anchored']]
bars=ax.bar(['Contriever gốc','CE + TRPO','Thêm retention'],vals,color=['#727A82','#39779B','#A44C47'],width=.55,yerr=[0,g['tpour']['sample_sd'],g['anchored']['sample_sd']],capsize=6)
ax.set_ylim(0,.75);ax.set_ylabel('nDCG@10');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.5f}',ha='center',fontsize=16)
fig.tight_layout();fig.savefig(OUT/'scifact.png',dpi=180);plt.close(fig)
pref=[]
for k in ['tpour','anchored']:pref.append(sum(read(R/f'{k}_seed{s}/heldout_preference.json')['pairwise_preference_accuracy'] for s in [42,43,44])/3)
fig,ax=plt.subplots(figsize=(7.6,4.2));bars=ax.bar(['CE + TRPO','Thêm retention'],pref,color=['#39779B','#A44C47'],width=.5);ax.set_ylim(0,.6);ax.set_ylabel('Pairwise preference accuracy');ax.axhline(.5,color='#999999',ls='--',lw=1)
for b,v in zip(bars,pref):ax.text(b.get_x()+b.get_width()/2,v+.016,f'{v:.5f}',ha='center',fontsize=17)
fig.tight_layout();fig.savefig(OUT/'preference.png',dpi=180);plt.close(fig)

s=slide('TPOUR / TRPO', 'Mở đầu: Nhóm tìm hiểu truy xuất thông tin có yếu tố thời gian. Em trình bày phương pháp của bài báo, phần nhóm đã triển khai và một biến thể giữ kiến thức đã được kiểm chứng. Kết quả có lợi trên SciFact so với TRPO nhưng còn đánh đổi. Thời lượng gợi ý 12–15 phút cho slide 1–19; slide 20–23 dùng khi hỏi.',source='Học phần Tìm kiếm và Truy xuất thông tin')
box(s,'Thực nghiệm dữ liệu thật\nvà biến thể giữ kiến thức',.85,1.8,11.5,1.65,37,True)
box(s,'Bài báo → Cài đặt → Đối chứng → Demo → Kết luận',.9,4.05,11.5,.6,24,color=GREY)
box(s,'Nhóm 15 · M25CQHT02-B · Mẫu Wikipedia / SciFact / Kaggle T4',.9,5.35,11.5,.6,21,color=NAVY)

s=slide('1. Vì sao cần quan tâm đến thời gian?', 'Ví dụ là tình huống giả định, không phải dữ liệu benchmark. Tài liệu giống chủ đề chưa chắc đáp ứng đúng mốc. Một quy định cũ vẫn có nhiều từ giống truy vấn về quy định mới. Bài toán cần học sự ưu tiên giữa tài liệu đúng mốc và lệch mốc.',source='Ví dụ minh họa của nhóm · Không dùng để đánh giá mô hình')
box(s,'“Quy trình của tổ chức X năm 2020 là gì?”',.9,1.55,11.5,.8,29,True)
table(s,['Tài liệu','Nội dung','Vấn đề'],[['Bản năm 2018','Cùng chủ đề, phiên bản cũ','Có thể rất gần ngữ nghĩa'],['Bản năm 2020','Cùng chủ đề, đúng mốc','Cần được ưu tiên']],y=2.8,widths=[2.2,4.3,5.1],rowh=.8,size=22)
takeaway(s,'Đúng chủ đề chưa đủ; còn phải phù hợp mốc truy vấn.')

s=slide('2. Bài báo nhóm chọn đề xuất gì?', 'Đọc đúng tên tác giả. Bản PDF nhóm chọn ghi ICML 2026/PMLR 306, lưu trong upstream. TRPO ở đây là Temporal Retrieval Preference Optimization, không phải Trust Region Policy Optimization. Bài báo còn có nội suy time embedding; nhóm chưa tái lập toàn bộ.',source='[1] Temporal Preference Optimization for Unsupervised Retrieval · PDF nhóm chọn, mục 3')
box(s,'Temporal Preference Optimization for Unsupervised Retrieval',.9,1.45,11.5,.8,25,True)
box(s,'HyunJin Kim · Jaejun Shim · Young Jin Kim · JinYeong Bak',.9,2.3,11.5,.5,20,color=GREY)
bullets(s,['Dùng tín hiệu ưu tiên giữa tài liệu khớp và lệch mốc.','Kết hợp contrastive loss với TRPO.','Dùng reference momentum và negative queue.'],y=3.2,size=24,gap=.8)
takeaway(s,'Nhóm triển khai bản gọn; chưa tái lập toàn bộ temporal QA và time interpolation.')

s=slide('3. Nhóm đã làm đến đâu?', 'Phân biệt rõ ba lớp công việc. Chạy toy chỉ xác nhận pipeline; bản hiện tại đã chuyển dữ liệu thật và thêm biến thể objective. Không lấy nDCG bằng 1 trên toy hoặc 0,74472 trên subset 30 query làm bằng chứng cho full SciFact.',source='Báo cáo Chương 3–5 · tpour.py / real_experiment.py / anchored_tpour.py')
table(s,['Phần việc','Đã làm'],[['Cài đặt nền tảng','Encoder, CE/TRPO, EMA reference, queue'],['Thực nghiệm thật','Làm sạch dữ liệu, train Kaggle, đủ 300 query SciFact'],['Biến thể của nhóm','Teacher cố định + ranking KL + relative MSE'],['Kiểm chứng','Ba seed, ablation, metrics và demo checkpoint']],widths=[3.1,8.5],rowh=.82,size=22)

s=slide('4. Dữ liệu: đã thay mock bằng mẫu thật', 'Nguồn là mẫu Wikipedia tác giả phát hành, không phải tự tái khai phá toàn bộ snapshot. Trong 3.000 dòng loại 4 thiếu context và 4 cặp giống nhau. Exact-text connected split giảm trùng nguyên văn nhưng không bảo đảm loại trùng bài hoặc diễn đạt lại. Held-out preference có nhãn mined, chưa phải temporal gold.',source='results/anchored_kaggle/data_manifest.json · reports/anchored_improvement/input_audit.json')
table(s,['Bước','Số lượng','Dùng để làm gì?'],[['Mẫu tác giả','3.000 dòng','Nguồn train, có provenance'],['Sau làm sạch','2.992 bộ ba','Query / preferred / rejected'],['Train','2.543','Cập nhật mô hình'],['Held-out preference','449','Đo khớp nhãn mined']],widths=[3.2,2.3,6.1],rowh=.75,size=22)
takeaway(s,'Split không trùng exact text; chưa loại mọi trùng bài hoặc paraphrase.')

s=slide('5. Từ TPOUR gọn đến biến thể retention', 'Student là mạng được optimizer cập nhật. Reference không nhận gradient và theo student bằng EMA, tham gia preference gap và queue. Teacher là bản Contriever khởi tạo đóng băng, không EMA, chỉ có trong biến thể nhóm. Cả ba nhận các đầu vào theo protocol; suy luận chỉ dùng student.',source='tpour.py · AnchoredTPOUR trong anchored_tpour.py')
for x,title,desc in [(1,'Student','Nhận gradient\nHọc CE + TRPO + retention'),(5,'Reference','Cập nhật EMA\nTham chiếu TRPO / queue'),(9,'Teacher','Đóng băng\nGiữ hành vi ban đầu')]:
    rect(s,x,2,3.2,2.1);box(s,title,x+.15,2.2,2.9,.5,27,True,NAVY);box(s,desc,x+.15,2.85,2.9,1.25,19)
box(s,'Cùng query / preferred / rejected đã augmentation',1,4.65,11.3,.6,25)
takeaway(s,'Khi truy xuất: chỉ tải student, không cộng điểm teacher.')

s=slide('6. Vì sao nhóm thêm teacher cố định?', 'Thử nghiệm dữ liệu thật lần đầu cho thấy TRPO giảm SciFact so với Contriever. Giả thuyết hẹp là thêm mốc hành vi ban đầu sẽ hạn chế suy giảm retrieval tổng quát. Chưa chứng minh catastrophic forgetting là nguyên nhân duy nhất. Teacher cũng có thể giữ thông tin cũ, nên phải đo đánh đổi.',source='results/kaggle_real_seed42 · ANCHORED_METHOD.md')
bullets(s,['Fine-tune mẫu nhỏ làm giảm SciFact so với Contriever gốc.','EMA reference đi theo student, không giữ nguyên mốc ban đầu.','Teacher cố định tạo mục tiêu giữ xếp hạng và biểu diễn.'],size=26)
takeaway(s,'Giả thuyết cần đo: giữ retrieval tổng quát có làm cản học ưu tiên mới không?')

s=slide('7. Nhóm thay đổi hàm mục tiêu thế nào?', 'Đọc công thức theo từng phần. Hai hạng đầu là CE và TRPO của bản thực hành; nhóm thêm retention gồm ranking KL và relative embedding MSE. KL dùng cosine softmax trên 2B candidates, temperature 0,1; MSE giữ hướng và norm. Các tham số cố định trước lần chạy mới, nhưng SciFact đã được biết trước khi thiết kế nên vẫn có thiên lệch benchmark.',source='anchored_tpour.py · λ=0,925; γ=10; η=1; temperature retention=0,1')
picture(s,mathfig(r'$\mathcal{L}_{new}=\lambda\mathcal{L}_{CE}+(1-\lambda)\mathcal{L}_{TRPO}+\gamma(\mathcal{L}_{KL}+\eta\mathcal{L}_{rep})$','objective.png'),.85,1.6,11.6,1.2)
table(s,['Thành phần thêm','Mục tiêu','Giới hạn'],[['Ranking KL','Giữ phân bố ưu tiên của teacher','Chỉ trên candidate list của batch'],['Relative MSE','Giữ hướng và độ lớn embedding','Cần ablation để biết lợi ích riêng']],y=3.1,widths=[2.5,4.4,4.7],rowh=.85,size=21)
takeaway(s,'Đây là biến thể nhóm thử nghiệm; knowledge distillation là kỹ thuật đã có.')

s=slide('8. Một bước train và một lần truy xuất', 'Tóm tắt luồng dữ liệu để cô thấy nhóm hiểu code. Batch size 4 tạo ma trận ranking distillation 4×8. AMP có 5 bước không hợp lệ bị bỏ; chỉ cập nhật scheduler/EMA/queue khi optimizer step hợp lệ. Inference kiểm tra cache theo hash và tính dot product exact trên corpus.',source='real_experiment.py · live_retrieval_demo.py')
box(s,'TRAIN',.9,1.5,2,.5,23,True,NAVY)
box(s,'Bộ ba → augmentation → embeddings → loss → gradient',.9,2.25,11.5,.8,25)
box(s,'Bước hợp lệ → optimizer → scheduler → EMA → queue',.9,3.15,11.5,.7,24)
box(s,'INFERENCE',.9,4.3,2.5,.5,23,True,NAVY)
box(s,'Student checkpoint → corpus index → query mới → top-k',.9,5.1,11.5,.8,25)

s=slide('9. Thiết kế đối chứng trên Kaggle', 'Mỗi nhánh dùng một T4; hai GPU có thể chạy hai nhánh song song, không phải một run distributed. Cùng dữ liệu, tokenizer, dot product và training budget. Mọi run retention mới và TRPO đối chứng có 995 cập nhật thực trên 1.000 lần thử. Baseline không fine-tune; only KL ablation chỉ seed42.',source='results/anchored_kaggle/*/config.json, completed.json, provenance.json')
table(s,['Hạng mục','Thiết lập'],[['Đối chứng','Contriever gốc / CE+TRPO / thêm retention'],['Seed train','42, 43, 44; thêm only-KL ở seed 42'],['Ngân sách','1.000 lần thử; 995 cập nhật hợp lệ mỗi run'],['Cấu hình','Batch 4 · queue 4.096 · max length 256 · lr 10⁻⁶'],['Đánh giá','300 query / 5.183 tài liệu SciFact; cùng protocol']],widths=[3,8.6],rowh=.69,size=21)

s=slide('10. Kết quả: giảm suy giảm so với TRPO', 'nDCG@10 đánh giá thứ tự các tài liệu liên quan trong top 10, không phải tỷ lệ phần trăm trả lời đúng. Các error bar là sample SD giữa ba seed; baseline cố định. Retention tăng 0,02872 so với TRPO nhưng vẫn thấp hơn baseline 0,01028. Đây là kết luận chính phải giữ cả hai vế.',source='results/anchored_kaggle/demo/summary.json · Mean ± sample SD; baseline cố định')
picture(s,OUT/'scifact.png',.9,1.4,8,4.7)
box(s,'+0,02872',9.05,2.2,3.2,.7,34,True,RED)
box(s,'so với CE + TRPO',9.05,3.05,3.3,.8,23)
box(s,'Chưa vượt\nContriever gốc',9.05,4.15,3.3,1.2,24,True)
takeaway(s,'Retention 0,62907 ± 0,00204; TRPO 0,60035 ± 0,00432.')

s=slide('11. Kết quả có lặp lại qua các seed không?', 'Ba chênh lệch cùng dương. CI là paired bootstrap 5.000 lần theo query, không phải CI trên mọi lần huấn luyện hay mọi dataset. Việc CI dương ở ba seed hỗ trợ kết luận trong cấu hình này, chưa loại bias thiết kế do đã biết SciFact.',source='compare_tpour_seed{42,43,44}_anchored_seed{42,43,44}.json · Bootstrap 5.000 lần')
rows=[]
for seed in [42,43,44]:
    c=read(R/f'compare_tpour_seed{seed}_anchored_seed{seed}.json');rows.append([str(seed),f'{c["mean_delta"]:+.5f}',f'[{c["ci95"][0]:+.5f}; {c["ci95"][1]:+.5f}]'])
table(s,['Seed','Delta nDCG@10','CI 95% theo query'],rows,widths=[2,3.5,6.1],rowh=.85,size=25)
takeaway(s,'Cả ba CI dương; cần thêm benchmark độc lập để kiểm tra tính tổng quát.')

s=slide('12. Đánh đổi: preference proxy giảm', 'Held-out pairwise accuracy kiểm tra score preferred lớn hơn rejected trên 449 bộ ba mined. Cả hai thấp hơn 0,5; cần kiểm tra nhãn, độ khó và pipeline thay vì coi proxy là temporal gold. Kết quả này ngăn nhóm khẳng định cải thiện toàn diện về thời gian.',source='*/heldout_preference.json · Trung bình ba seed, 449 bộ ba mined giữ riêng')
picture(s,OUT/'preference.png',.85,1.5,7.5,4.6)
box(s,'0,44915 → 0,42910',8.65,2.4,3.8,.8,26,True,RED)
box(s,'SciFact tăng,\nnhưng khớp nhãn mined giảm.',8.65,3.55,3.8,1.9,23)
takeaway(s,'Chưa có temporal gold test nên chưa kết luận cải thiện temporal alignment.')

s=slide('13. Ablation và chi phí: chưa phải mọi phần đều tốt', 'Only KL seed42 đạt 0,62870, full 0,63082; delta 0,00212 nhưng CI chứa 0. Không chứng minh MSE vô dụng, cũng chưa chứng minh cần thiết. Train time không gồm tải model, warmup queue, evaluate; peak allocated không phải toàn VRAM. Chi phí tăng trong train do teacher.',source='compare_ranking_only_full_seed42.json · */completed.json')
c=read(R/'compare_ranking_only_full_seed42.json')
table(s,['Kiểm tra','Kết quả','Cách đọc'],[['MSE thêm vào KL','Delta +0,00212; CI chứa 0','Chưa rõ lợi ích riêng'],['Train trung bình','332,2 → 385,3 giây','Tăng khoảng 16%'],['Peak allocated','2,51 → 2,92 GiB','Chỉ bộ nhớ được PyTorch cấp phát']],widths=[3,4.3,4.3],rowh=.9,size=22)
takeaway(s,'Ablation mới một seed; chi phí trên chưa tính tải model và đánh giá.')

s=slide('14. Demo checkpoint: đã chạy inference thật', 'Đây là kết quả JSON từ job inference đã chạy, không phải đang trực tiếp gọi GPU trong slide. Muốn demo trực tiếp mở notebook TPOUR_live_checkpoint_demo và chạy cell query. Bốn nhánh khớp top3 query1 với ranking đã lưu, max score error 4,77e-7; ba checkpoint fine-tune có hash khớp. Query mới dưới đây chưa có gold, chỉ minh họa sử dụng.',source='results/anchored_kaggle/live_demo/anchored_custom_query.json · verification.json')
live=read(R/'live_demo/anchored_custom_query.json')
box(s,'Query mới: '+live['query'],.9,1.45,11.5,1.05,23,True)
table(s,['Hạng','Tiêu đề tài liệu trả về'],[[r['rank'],r['title']] for r in live['results']],y=2.85,widths=[1.2,10.4],rowh=.8,size=21)
takeaway(s,'Query mới chạy bằng student; chưa có gold labels nên không báo accuracy.')

s=slide('15. Demo tốt và xấu đều phải trình bày', 'Hai ví dụ chọn sau đánh giá: query1282 cải thiện lớn nhất và query314 giảm lớn nhất seed42. Không dùng các ví dụ được chọn để đại diện toàn test. Baseline gốc cũng đưa gold của query1282 lên hạng1. Các câu SciFact là benchmark truy xuất, không phải tư vấn y tế.',source='results/anchored_kaggle/demo/ranking_examples.json · Seed 42 · Ví dụ chọn sau đánh giá')
examples=read(R/'demo/ranking_examples.json')
for y,e in [(1.55,examples[1]),(3.8,examples[2])]:
    box(s,f'Query {e["query_id"]}',.9,y,2.2,.5,23,True,NAVY)
    box(s,e['query'],3.1,y,9.3,.95,20)
    box(s,f'Delta nDCG@10: {e["ndcg_delta"]:+.5f}',3.1,y+1,9.3,.5,24,True,RED)
takeaway(s,'Ví dụ minh họa không thay thế số liệu của đủ 300 truy vấn.')

s=slide('16. Đóng góp và phần chưa hoàn tất', 'Nhấn mạnh phần triển khai có thể kiểm tra và phần phương pháp thử thêm. Không nói nhóm phát minh KD hoặc đã chứng minh paper. Chưa có temporal gold, chưa full scale, chưa kiểm tra benchmark mới. 14 tests là kiểm tra các trường hợp cụ thể chứ không bảo đảm mọi đầu vào đúng.',source='Báo cáo Chương 3.10 và 5 · verification.json · 14 kiểm tra code')
table(s,['Đã thực hiện','Chưa có bằng chứng đầy đủ'],[['Dữ liệu thật, runner và provenance','Tái lập toàn bộ quy mô bài báo'],['Teacher + retention, ba seed và ablation','Vượt Contriever gốc'],['Full SciFact và checkpoint inference','Cải thiện temporal QA độc lập'],['Báo cả kết quả tăng và giảm','Tính mới học thuật của phương pháp']],widths=[5.8,5.8],rowh=.84,size=23)

s=slide('17. Hướng tiếp theo: kiểm tra đúng mục tiêu trước', 'Đây là kế hoạch, chưa thực hiện. Ưu tiên thẩm định nhãn và test thời gian, tiếp đến benchmark chưa dùng khi thiết kế và ablation nhiều seed. Sau đó mới thử retention có điều kiện; phải chọn cấu hình trên validation và giữ test độc lập. Không thêm độ phức tạp chỉ để có tên phương pháp mới.',source='Báo cáo Chương 5.5–5.6 · Các bước đề xuất chưa thực hiện')
bullets(s,['Thẩm định mined labels và xây temporal gold test độc lập.','Đánh giá benchmark mới; lặp ablation trên nhiều seed.','Thử retention có điều kiện để giảm cản trở cập nhật mới.','Đo đồng thời temporal quality, retrieval tổng quát và chi phí.'],size=25,gap=1)

s=slide('18. Kết luận', 'Kết thúc bằng ba ý. Đồ án đã vượt demo mock để có thực nghiệm thật và bằng chứng. Biến thể làm giảm mất chất lượng SciFact so với TRPO. Nhưng chưa vượt baseline, proxy giảm, nên kết luận chỉ trong phạm vi đã đo. Dừng ở slide này; các slide sau dành cho hỏi đáp.',source='Kết luận từ số liệu đã chạy · Không bổ sung kết quả chưa thực hiện')
bullets(s,['Đã triển khai và kiểm chứng một biến thể giữ kiến thức.','SciFact tốt hơn TRPO đối chứng trong ba seed đã chạy.','Chưa vượt Contriever gốc; preference proxy còn giảm.'],y=1.75,size=28,gap=1.15)
takeaway(s,'Kết quả có giá trị kiểm chứng, nhưng chưa phải cải tiến toàn diện TPOUR.')

s=slide('Hỏi đáp · Vì sao có cả reference và teacher?', 'Nếu cô hỏi: reference EMA phục vụ preference objective và di chuyển theo student; teacher cố định bảo tồn hành vi khởi tạo. Cả hai không nhận gradient trực tiếp nhưng cách cập nhật khác nhau. Khi inference chỉ student. Không cần mở slide này nếu không được hỏi.',appendix=True)
table(s,['','Reference','Teacher'],[['Khởi tạo','Từ encoder ban đầu','Từ encoder ban đầu'],['Cập nhật','EMA theo student','Không cập nhật'],['Mục đích','Mốc tương đối cho TRPO','Mục tiêu giữ kiến thức'],['Inference','Không dùng','Không dùng']],widths=[2.8,4.4,4.4],rowh=.84,size=23)

s=slide('Hỏi đáp · Chưa vượt baseline thì có ý nghĩa gì?', 'Câu trả lời ngắn: có ý nghĩa như thí nghiệm hạn chế suy giảm so với TRPO dưới cùng protocol, không phải giải pháp tốt nhất cho SciFact. Muốn khẳng định tác dụng thời gian phải có temporal gold. Phải thừa nhận biết SciFact trước khi thiết kế nên cần benchmark mới.',appendix=True)
bullets(s,['Câu hỏi đã kiểm chứng: retention có giảm suy giảm so với TRPO?','Câu trả lời hiện tại: có trên SciFact, trong ba seed này.','Nếu chỉ cần SciFact, chưa có căn cứ chọn thay Contriever gốc.','Muốn dùng cho thời gian: cần đo temporal gold và đánh đổi.'],size=25,gap=1.05)

s=slide('Hỏi đáp · Demo thế nào khi thuyết trình?', 'Chuẩn bị checkpoint đúng run và corpus trước buổi nói để tránh chờ lập chỉ mục. Chạy query1 để đối chiếu ranking, rồi thử query mới. Nếu không có GPU/kết nối, dùng JSON đã lưu và nói rõ đó là replay. Không giả vờ video/ảnh là phiên inference đang chạy.',source='live_retrieval_demo.py · notebooks/TPOUR_live_checkpoint_demo.ipynb',appendix=True)
bullets(s,['Mở notebook live; nạp encoder.pt và corpus SciFact.','Chạy query 1 để đối chiếu top-k đã lưu.','Nhập query mới; nói rõ chưa có gold để chấm độ chính xác.','Phương án dự phòng: replay JSON, ghi rõ kết quả đã chạy.'],size=24,gap=.95)
box(s,'Kaggle: vananh1911/tpour-live-checkpoint-verification',1.2,6,11,.4,18,color=GREY)

s=slide('Nguồn để đối chiếu', 'Tài liệu gốc nằm trong repo, cùng PDF đã chọn. Notebook Kaggle có thể cần quyền tài khoản; JSON/log/notebook source công khai trong GitHub cho phép đọc bằng chứng. Slide 1–19 là phần trình bày chính; 20–23 là phụ lục.',source='Tài liệu và mã nguồn của báo cáo',appendix=True)
bullets(s,['Bài TPOUR: upstream/TPOUR/ICML_2026_TPOUR.pdf','Code: tpour.py, real_experiment.py, anchored_tpour.py','Kết quả: results/anchored_kaggle/','Báo cáo đúng mẫu PTIT: reports/ptit_verified/'],size=23,gap=.87)
box(s,'github.com/lequangduccode/tpour-trpo-experiments',1.2,5.6,11,.5,22,True,NAVY)

# Restore paper equations that a concise first deck omitted. Reorder together with notes.
def move_last(index):
    ids=prs.slides._sldIdLst;item=ids[-1];ids.remove(item);ids.insert(index,item)
    note=notes.pop();notes.insert(index,note)

s=slide('Công thức nền: CE và TRPO', 's là dot product chưa chia nhiệt độ; Δ là chênh lệch score đã chia T. CE hiển thị cho một query, code trung bình theo batch. k_j là negative key reference trong queue. TRPO có reference gap detach và β=0,3, T=0,05. Khi hai gap bằng nhau, loss ln2 nhưng gradient student chưa bằng0. Đây là công thức bản gọn đối chiếu mục3.1, không phải weighted KL hoặc PPO.',source='Bài TPOUR [1], phương trình (3)–(4) · tpour.py:trpo_loss và TPOUR.forward')
picture(s,mathfig(r'$\mathcal{L}_{CE}(q)=-\log\frac{e^{q^\top p/T}}{e^{q^\top p/T}+\sum_{j\in\mathcal{Q}} e^{q^\top k_j/T}}$','ce_formula.png'),1,1.45,11.3,1.25)
picture(s,mathfig(r'$\Delta_\theta=\frac{s_\theta(Q,D^+)-s_\theta(Q,D^-)}{T}$','gap_formula.png'),1,3,11.3,.9)
picture(s,mathfig(r'$\mathcal{L}_{TRPO}=-\frac{1}{B}\sum_{i=1}^{B}\log\sigma\!\left[\beta(\Delta_\theta^{(i)}-\Delta_{ref}^{(i)})\right]$','trpo_formula.png'),1,4.3,11.3,1.15)
takeaway(s,'s(Q,D) là dot product; T=0,05; reference không nhận gradient trực tiếp.')
move_last(3)

s=slide('Time vector: nội dung của bài báo, chưa tái lập đủ', 'Nội suy trọng số giữa hai encoder chuyên biệt thời gian cùng nền, không phải chèn year embedding vào query. Hai checkpoint toy từng nội suy chỉ là kiểm tra thao tác tensor, không đại diện 2018 và2021. Điều kiện là cùng kiến trúc, pretrained base, tokenizer và metadata tương thích. Phải mã hóa lại corpus sau khi đổi encoder; nhóm chưa kiểm chứng chất lượng nội suy trên temporal gold.',source='Bài TPOUR [1], mục 3.2, phương trình (5) · Báo cáo PTIT mục 2.6')
picture(s,mathfig(r'$\tau_t=\theta_t-\theta_{base}$','time_vector.png'),1,1.8,11.3,.95)
picture(s,mathfig(r'$\theta_{mid}=\theta_{base}+(1-\alpha)\tau_{left}+\alpha\tau_{right}$','time_interpolation.png'),1,3.1,11.3,1)
box(s,'Hai checkpoint phải cùng nền; đổi encoder cần mã hóa lại corpus.',1,4.8,11.3,.85,24)
takeaway(s,'Đã có kiểm tra phép nội suy tensor; chưa có kết quả temporal interpolation thật.')
move_last(4)

s=slide('Hỏi đáp · KL và relative MSE tính trên gì?', 'Đây là công thức chính xác theo anchored_tpour.retention_loss. B query;2B preferred/rejected trong batch;3B embedding gồm query/preferred/rejected. KL(teacher||student) lấy batchmean; MSE tương đối lấy mean3B. Cùng input augmentation, teacher detach. Tr=0,1; epsilon=1e-6. Không có nhãn qrels test hoặc dữ liệu truy xuất ngoài batch đưa vào hai hạng này.',source='anchored_tpour.py:retention_loss · Báo cáo PTIT mục 3.4',appendix=True)
picture(s,mathfig(r'$P_{aij}=\frac{\exp(\cos(q_{ai},d_{aj})/T_r)}{\sum_{\ell=1}^{2B}\exp(\cos(q_{ai},d_{a\ell})/T_r)}$','distribution.png'),1,1.45,11.3,1.25)
picture(s,mathfig(r'$\mathcal{L}_{KL}=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{2B}P_{tij}\log\frac{P_{tij}}{P_{sij}}$','kl_formula.png'),1,3.05,11.3,1)
picture(s,mathfig(r'$\mathcal{L}_{rep}=\frac{1}{3B}\sum_{u=1}^{3B}\frac{\|z_{s,u}-z_{t,u}\|_2^2}{\max(\|z_{t,u}\|_2^2,10^{-6})}$','mse_formula.png'),1,4.45,11.3,1.15)
takeaway(s,'a∈{student, teacher}; B=4; Tr=0,1. Teacher targets được detach.')
move_last(len(prs.slides)-2)

s=slide('Hỏi đáp · Có dùng context ngoài vào train không?', 'Rà code và provenance:train đọc train.jsonl xuất từ mẫu tác giả; queue và teacher targets tạo trên cùng dữ liệu đó. SciFact corpus/query/qrels là nguồn đánh giá riêng. Demo chỉ truy xuất corpus SciFact, không gọi web search hay LLM sinh context. Chỉ khẳng định pipeline đã rà soát; chưa thể kiểm toán toàn bộ dữ liệu tiền huấn luyện hoặc mọi trùng ngữ nghĩa. SciFact đã được xem khi thiết kế retention nên vẫn có design bias.',source='real_experiment.py:clean_rows/train · live_retrieval_demo.py · data_manifest.json',appendix=True)
table(s,['Nguồn','Vai trò','Có cập nhật gradient?'],[['Mẫu Wikipedia tác giả','Train split 2.543 bộ ba','Có'],['Held-out mined','449 bộ ba đo proxy','Không'],['SciFact / qrels','Đánh giá và truy xuất demo','Không'],['Contriever tiền huấn luyện','Khởi tạo model / teacher','Student được fine-tune']],widths=[3.5,4.6,3.5],rowh=.8,size=22)
takeaway(s,'Không thấy bước gọi web/LLM để bổ sung context trong pipeline đã chạy.')
move_last(len(prs.slides)-2)

# Fix footer numbers after insertion; keep main/appendix guidance consistent.
for i,s in enumerate(prs.slides,1):
    for sh in s.shapes:
        if sh.has_text_frame and sh.left>Inches(12) and sh.top>Inches(7):sh.text=str(i)
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if sh.left>Inches(12) and sh.top>Inches(7):r.font.name='Arial';r.font.size=Pt(12)
main_count=sum(not n[2] for n in notes)
for i,(title,note,ap) in enumerate(notes):
    note=note.replace('slide 1–19; slide 20–23',f'slide 1–{main_count}; slide {main_count+1}–{len(notes)}').replace('Slide 1–19',f'Slide 1–{main_count}').replace('20–23',f'{main_count+1}–{len(notes)}').replace('19–22',f'{main_count+1}–{len(notes)}')
    notes[i]=(title,note,ap);prs.slides[i].notes_slide.notes_text_frame.text=note
prs.save(OUT/'Slide_thuyet_trinh_TPOUR_TRPO_cap_nhat.pptx')

doc=Document()
for st in ['Normal','Title','Heading 1','Heading 2']:doc.styles[st].font.name='Arial'
doc.styles['Normal'].font.size=WP(11)
doc.add_heading('GỢI Ý THUYẾT TRÌNH TPOUR / TRPO',0)
doc.add_paragraph(f'Slide 1–{main_count}: phần trình bày chính khoảng 14–17 phút. Slide {main_count+1}–{len(notes)}: mở khi cần hỏi đáp. Đây là gợi ý để nói bằng lời của mình, không cần đọc nguyên văn.')
doc.add_paragraph('Trước buổi nói: điền thông tin nhóm nếu cần; kiểm tra quyền notebook, tải checkpoint, tạo corpus cache trước và thử mở PowerPoint trên máy trình chiếu. Các biểu đồ lấy từ kết quả thực; không dùng ảnh sinh bằng AI hoặc giao diện web giả.')
for i,(title,note,ap) in enumerate(notes,1):
    doc.add_heading(f'{i:02d}. {title}'+(' [phụ lục]' if ap else ''),2);doc.add_paragraph(note)
doc.add_heading('Câu chốt nên nhớ',1)
doc.add_paragraph('Nhóm đã triển khai và kiểm chứng một biến thể hạn chế suy giảm retrieval sau fine-tune. Kết quả tốt hơn TRPO trên SciFact, nhưng chưa vượt Contriever gốc và mined-preference accuracy giảm; chưa đủ bằng chứng nói cải thiện toàn diện temporal retrieval.')
doc.save(OUT/'Goi_y_thuyet_trinh_TPOUR_TRPO.docx')
print(f'Saved {len(prs.slides)} slides and speaker notes to {OUT}')
