# Thử nghiệm TPOUR / TRPO

## Báo cáo và slide thuyết trình

Bộ tài liệu hoàn thiện ngày 12/10/2026 nằm tại [reports/](reports/README.md): [báo cáo Word chi tiết](reports/Bao_cao_chi_tiet_TPOUR_TRPO.docx), [bản DOC](reports/Bao_cao_chi_tiet_TPOUR_TRPO.doc), [PDF](reports/Bao_cao_chi_tiet_TPOUR_TRPO.pdf), [PowerPoint 19 slide](reports/Slide_thuyet_trinh_TPOUR_TRPO.pptx), [kịch bản thuyết trình](reports/Kich_ban_thuyet_trinh_TPOUR_TRPO.docx) và ảnh chụp demo. Có kết quả chạy bổ sung ba seed và log gốc; chưa đánh giá BEIR đầy đủ. Điền thông tin sinh viên/giảng viên trên bìa trước khi nộp.

Tải repository cùng mã nguồn tác giả bằng `git clone --recurse-submodules <URL-repository>`. Nếu đã clone thông thường, chạy `git submodule update --init --recursive`.

Bộ mã thử nghiệm một thiết bị, có thể chạy trên CPU hoặc GPU. Mã chính thức được giữ tại `upstream/TPOUR`; bản triển khai gọn nằm trong `tpour.py`. Đây là bản thực hành, chưa phải tái lập toàn bộ kết quả công bố.

Đã chạy thành công demo, 5 kiểm tra tự động và 2 bước huấn luyện `facebook/contriever` trên 8 mẫu từ dữ liệu tác giả. Checkpoint thật ở `outputs/contriever_smoke/encoder.pt`. Báo cáo nằm trong `BAO_CAO_THUC_NGHIEM.txt`; BEIR đầy đủ chưa chạy.

Các file JSON kết quả và cấu hình đã chạy được lưu tại `results/` để xem trực tiếp trên GitHub. Checkpoint và môi trường `.venv` nằm trên máy thực nghiệm; chạy lại các lệnh bên dưới để tạo lại chúng. Trên máy mới, cài Python và chạy `python -m pip install -r requirements.txt` trước khi chạy demo hoặc kiểm tra.

## Chạy ngay trên máy này

```powershell
python tpour.py demo --output outputs/demo
python -m unittest test_tpour -v
```

Demo dùng encoder embedding nhỏ, dữ liệu giả lập bằng tiếng Anh và không tải mô hình. Kết quả, cấu hình, lịch sử loss và checkpoint được lưu trong `outputs/demo`. Train/test tách theo tổ chức giả định, nhưng dùng chung mẫu câu; bài toán này chỉ kiểm tra luồng xử lý.

## Chạy Contriever

Môi trường `.venv` dùng lại PyTorch đã có trên máy; cài phần còn thiếu:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tpour.py train --data upstream/TPOUR/2018_sample_train_trimmed.jsonl --model facebook/contriever --steps 100 --batch-size 2 --queue-size 256 --device cpu --output outputs/contriever
```

Lần đầu cần tải tokenizer và trọng số từ Hugging Face. GPU dùng `--device cuda` khi PyTorch hỗ trợ CUDA. Chạy đối chứng với cùng dữ liệu, seed và số bước bằng cách thêm `--ce-weight 1 --output outputs/contriever_ce`. Không dùng `--normalize` nếu muốn giữ pooling và điểm tích vô hướng mặc định của nhánh Contriever trong bản này.

## Chuẩn bị dữ liệu riêng

Đầu vào JSONL: mỗi dòng chứa `_id`, `timestamp`, `text`. Timestamp là mốc của snapshot/corpus, không tự động suy ra năm từ chuỗi văn bản. ID phải thống nhất qua các phiên bản của cùng tài liệu.

```json
{"_id":"article-1","timestamp":"2018","text":"A sufficiently long paragraph from the earlier corpus..."}
{"_id":"article-1","timestamp":"2021","text":"An updated paragraph from the later corpus..."}
```

```powershell
.venv\Scripts\python tpour.py prepare --input data/raw.jsonl --target-time 2021 --exclude data/test/corpus.jsonl --output data/train_2021.jsonl
.venv\Scripts\python tpour.py train --data data/train_2021.jsonl --output outputs/2021
```

`prepare` chia đôi tài liệu ở mốc đích để lấy query giả và positive; negative là tài liệu khác mốc có độ tương đồng cao nhất. Đây là cách dựng dữ liệu đơn giản của bộ thực hành; pipeline khai thác của tác giả nằm trong `upstream/TPOUR/preprocessing`. Công cụ hiện giữ toàn bộ embeddings trong RAM và phù hợp tập nhỏ. Cần loại trùng nội dung và chia train/test theo tài liệu hoặc thực thể trước khi tạo cặp. Tùy chọn `--exclude` loại mọi phiên bản có ID trùng tập đánh giá; không tự phát hiện ID khác nhưng trùng nội dung.

File `data/raw_example.jsonl` là dữ liệu giả định đủ dài để thử lệnh ngay. Dùng `--model toy` để kiểm tra pipeline mà không tải Contriever. Tài liệu dưới 20 từ bị bỏ qua; phải còn ít nhất hai cặp hợp lệ để huấn luyện.

## Đánh giá retrieval và BEIR

Đánh giá tập riêng cần ba file: `corpus.jsonl` (`_id`, `text`, `title` tùy chọn), `queries.jsonl` (`_id`, `text`), `qrels.jsonl` (`query_id`, `doc_id`, `score`). Chỉ qrels đánh giá mới cần nhãn; các nhãn này không dùng để huấn luyện.

```powershell
python tpour.py evaluate --data outputs/demo/data/test --checkpoint outputs/demo/tpour/encoder.pt --output outputs/demo/recheck.json
```

BEIR dùng định dạng gốc với `qrels/test.tsv`; đặt dataset đã giải nén vào `data/scifact`:

```powershell
.venv\Scripts\python -m pip install -r requirements-beir.txt
.venv\Scripts\python evaluate_beir.py --data data/scifact --output outputs/beir_baseline.json
.venv\Scripts\python evaluate_beir.py --data data/scifact --checkpoint outputs/contriever/encoder.pt --output outputs/beir_tpour.json
```

Có thể thêm `--query-limit 5` để kiểm tra nhanh; kết quả được gắn cờ đánh giá một phần. Corpus vẫn giữ đầy đủ. Tìm kiếm chính xác bằng dot product, tính chỉ số qua thư viện BEIR. Với tập lớn cần thay bằng index hoặc phân khối corpus để giảm RAM. BEIR đánh giá retrieval tổng quát; để kiểm chứng ưu tiên thời gian cần thêm tập temporal retrieval với qrels đúng mốc.

## Nội suy checkpoint

Hai mô hình phải được fine-tune từ cùng một checkpoint nền và dùng cùng tokenizer. Không nội suy các mô hình khởi tạo độc lập.

```powershell
python tpour.py interpolate --left outputs/2018/encoder.pt --right outputs/2021/encoder.pt --alpha 0.333333 --output outputs/2019/encoder.pt
```

Code tính `(1-alpha)*theta_left + alpha*theta_right`, tương đương nội suy hai độ lệch so với cùng mô hình nền. Kiểm tra metadata và kích thước tensor không thể chứng minh hai checkpoint cùng nguồn; người chạy phải bảo đảm điều này.

## Phạm vi kiểm chứng

TRPO có phần chênh lệch điểm của encoder tham chiếu; reference không nhận gradient và cập nhật bằng momentum. Queue chứa embeddings reference của positive và negative. Loss tổng là `lambda*CE + (1-lambda)*TRPO`. Kiểm tra tự động gồm chiều gradient ưu tiên, tách gradient reference, wrap queue, momentum, số đo retrieval và định dạng dữ liệu tác giả.

Các lựa chọn riêng của bản gọn: reference ở eval mode; CE dùng positive từ encoder hiện tại và queue cũ; queue khởi động bằng dữ liệu thật; không bổ sung in-batch negatives; optimizer learning rate cố định; checkpoint chỉ phục vụ inference, không resume toàn bộ trạng thái train. Vì thế không xem kết quả bản gọn là kết quả của code upstream. `src/moco.py` upstream có thêm in-batch positives/negatives và cách chia temperature khác giữa CE và TRPO.

Không thể xác nhận TPOUR dễ nhất trong 11 bài vì thư mục ban đầu trống. Cũng chưa có phép đo GPU để khẳng định cấu hình RTX 3090 chạy được thiết lập tái lập đầy đủ.

## Nguồn

- [Bài báo gốc](https://arxiv.org/abs/2606.17664), mục 3 và phụ lục A.
- [Repository chính thức](https://github.com/agwaBom/TPOUR), commit `d1d8e70326deae6ef2ea03a8cb2c0cbc60db7a78`.
- [BEIR](https://github.com/beir-cellar/beir) và [Contriever](https://github.com/facebookresearch/contriever).
