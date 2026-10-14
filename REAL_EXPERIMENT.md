# Thử nghiệm TPOUR/TRPO với văn bản thật

Notebook: [TPOUR_real_data_Kaggle.ipynb](notebooks/TPOUR_real_data_Kaggle.ipynb). Bật GPU và Internet, rồi chạy các ô theo thứ tự. Mặc định một seed, 1.000 bước cho mỗi nhánh; đổi `SEEDS=[42,43,44]` để kiểm tra nhiều seed khi còn ngân sách GPU. Chương trình dùng một GPU, kể cả khi Kaggle cấp hai T4.

Đây là bản nâng cấp code thực hành, tiếp nhận sửa lọc context rỗng của Hạnh tại commit `bfc75ca9857dd8e0a46440ce1229b22230cdefec`. Chưa tái lập đầy đủ cấu hình và kết quả bài báo. Chưa có kết quả GPU mới khi tạo tài liệu này.

## Dữ liệu và phép đo

- Mẫu Wikipedia thật từ tác giả trong `upstream/TPOUR/2018_sample_train_trimmed.jsonl`, không phải bộ giả lập của lệnh `demo`.
- 3.000 dòng gốc; bỏ 4 dòng thiếu context và 4 dòng có preferred/rejected giống nhau. Lấy context đầu tiên của mỗi loại như bản gọn hiện tại. Còn 2.992 bộ ba: 2.543 train, 449 held-out (seed chia 42).
- Nhóm các bộ ba có cùng văn bản sau chuẩn hóa; không trùng văn bản nguyên dạng giữa hai phần. Chưa bảo đảm độc lập ở cấp bài Wikipedia hoặc đoạn diễn đạt lại. Không dùng held-out để train.
- Held-out chứa nhãn preference được khai thác từ dữ liệu tác giả, chưa phải nhãn thời gian được con người kiểm chứng. Accuracy này chỉ là phép đo phụ.
- SciFact test: 5.183 tài liệu, toàn bộ 300 truy vấn có nhãn. SciFact đo truy xuất tổng quát, không đủ chứng minh temporal alignment. Không điều chỉnh siêu tham số theo test.

Ba nhánh là Contriever gốc, chỉ CE và CE+TRPO. Hai nhánh train dùng cùng khởi tạo, seed, ngân sách bước, batch, queue, augmentation, learning rate và lịch warmup/decay. Mọi nhánh đánh giá dùng cùng max length 256, dot product và không normalize. CE-only đặt trọng số CE bằng 1; nhánh kết hợp dùng 0,925. Không so sánh độ lớn hai objective khác nhau để kết luận chất lượng.

## Chạy ở terminal

```powershell
python -m pip install -r requirements-experiment.txt
python real_experiment.py prepare --input upstream/TPOUR/2018_sample_train_trimmed.jsonl
python real_experiment.py download
python real_experiment.py evaluate --data data/scifact --output outputs/real/baseline --device cuda
python real_experiment.py train --objective ce --device cuda --amp --gradient-checkpointing --output outputs/real/ce_seed42
python real_experiment.py train --objective tpour --device cuda --amp --gradient-checkpointing --output outputs/real/tpour_seed42
python real_experiment.py evaluate --data data/scifact --checkpoint outputs/real/ce_seed42/encoder.pt --output outputs/real/ce_seed42 --device cuda
python real_experiment.py evaluate --data data/scifact --checkpoint outputs/real/tpour_seed42/encoder.pt --output outputs/real/tpour_seed42 --device cuda
python real_experiment.py compare --left outputs/real/ce_seed42 --right outputs/real/tpour_seed42 --output outputs/real/ce_vs_tpour.json
python tools/summarize_real_runs.py --root outputs/real --output outputs/real/summary
```

Nếu thiếu GPU, dùng `--device cpu` cho kiểm tra nhỏ; huấn luyện/evaluate toàn bộ sẽ chậm. Chưa đo VRAM thực tế trên Kaggle. Nếu OOM, giảm batch, giữ điều kiện giống nhau cho cả hai nhánh và chạy lại; không ghép kết quả từ các cấu hình khác nhau.

## Checkpoint và bằng chứng

`encoder.pt` dùng cho đánh giá; `training_state.pt` chứa current/reference encoder, queue, optimizer, scheduler, scaler và RNG. Resume bằng cách thêm `--resume <training_state.pt>` vào đúng lệnh train ban đầu. Giữ nguyên tổng số bước, dữ liệu và thư mục output. Checkpoint lớn; tải riêng hoặc gắn output phiên cũ làm input khi chạy lại Kaggle. Resume đã kiểm tra cho kết quả giống hệt trên bài kiểm tra CPU nhỏ; chưa xác nhận tính tương đương bitwise trên GPU/AMP.

Lưu hash dữ liệu/checkpoint, phiên bản phần mềm, commit, cấu hình, log loss, thời gian, GPU và bộ nhớ đỉnh. Mỗi evaluate lưu metrics tổng, metrics từng query và top-100. So sánh bootstrap ghép cặp 5.000 lần; CI chứa 0 nghĩa là chưa đủ bằng chứng cải thiện trên bộ/seed này. Biểu đồ dùng Matplotlib, chỉ đọc số liệu thực. SD giữa các seed để trống nếu chỉ chạy một seed.

10 kiểm tra tự động đã qua, gồm loss/gradient, queue/momentum, detach reference, tách dữ liệu, metric nhiều positive và resume. Baseline CPU toàn bộ đã dừng trước khi hoàn tất để chuyển sang Kaggle; chưa có metric mới để đưa vào báo cáo.

Nguồn: [BEIR](https://github.com/beir-cellar/beir), [SciFact](https://github.com/allenai/scifact), [Kaggle CLI](https://github.com/Kaggle/kaggle-cli).
