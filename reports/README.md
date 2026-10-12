# Bộ tài liệu báo cáo TPOUR / TRPO

Hoàn thiện ngày **12/10/2026**. Chủ đề là TPOUR/TRPO trong repository này.

## File để nộp và thuyết trình

- [Báo cáo Word (.docx)](Bao_cao_chi_tiet_TPOUR_TRPO.docx) — 22 trang, lý thuyết, công thức, thiết kế, thực nghiệm, ảnh demo, giới hạn và hỏi đáp.
- [Báo cáo Word định dạng cũ (.doc)](Bao_cao_chi_tiet_TPOUR_TRPO.doc) — xuất bằng Microsoft Word.
- [Báo cáo PDF](Bao_cao_chi_tiet_TPOUR_TRPO.pdf) — xem bố cục trước khi nộp.
- [Slide PowerPoint](Slide_thuyet_trinh_TPOUR_TRPO.pptx) — 19 slide, khung 16:9, có Notes cho từng slide.
- [Slide PDF](Slide_thuyet_trinh_TPOUR_TRPO.pdf) — phương án dự phòng khi máy không có PowerPoint.
- [Kịch bản thuyết trình Word](Kich_ban_thuyet_trinh_TPOUR_TRPO.docx) — gợi ý lời nói, thời lượng và trình diễn.
- [Kịch bản PDF](Kich_ban_thuyet_trinh_TPOUR_TRPO.pdf).

Điền **tên, lớp, mã sinh viên và giảng viên** trên bìa Word/slide đầu trước khi nộp. Nội dung kết quả đã có số liệu; không cần tự điền điểm thực nghiệm.

## Ảnh chụp demo

Ảnh được chụp bằng Chrome từ viewer có dữ liệu đo thực tế, không dùng công cụ sinh ảnh:

1. [Tổng quan kết quả](assets/demo_01_overview.png).
2. [Truy xuất và thứ hạng](assets/demo_02_retrieval.png).
3. [Log và kiểm tra](assets/demo_03_checks.png).

Hai ảnh `*_detail.png` là ảnh chụp vùng chi tiết để dễ đọc trên slide. [Biểu đồ chỉ số](assets/metrics.png), [biểu đồ loss](assets/loss.png) và [sơ đồ kiến trúc](assets/architecture.png) được tạo bằng code, không phải screenshot.

Mở [demo_viewer.html](demo_viewer.html) bằng Chrome/Edge sau khi clone hoặc tải cả thư mục `reports`. Các bảng xếp hạng được tính bằng checkpoint trước khi tạo viewer; viewer hiển thị các kết quả và hỗ trợ đổi query trong 8 query test. Trang HTML không chạy huấn luyện hay suy luận mới trong trình duyệt.

## Bằng chứng và giới hạn

- Ba lần chạy toy demo với seed **42, 43, 44**, mỗi nhánh **150 bước**; 32 bộ ba train, 8 query và 16 tài liệu test.
- Hai nhánh chỉ CE và CE + TRPO đều đạt điểm tối đa trên demo. Chưa có bằng chứng TRPO vượt chỉ CE.
- **5/5** kiểm tra tự động đạt trong lần chạy mới.
- Contriever thật được kiểm tra **2 bước** ngày **10/10/2026**, lấy 8 mẫu tác giả. Log lịch sử nằm tại `../results/contriever_smoke`.
- Chưa đánh giá BEIR hoặc temporal QA đầy đủ; đây chưa phải tái lập kết quả công bố.
- [Provenance](evidence/provenance.json) lưu timestamp Asia/Bangkok, phiên bản môi trường, commit nền và SHA-256 của code.
- [Tổng hợp ba seed](evidence/summary.json), [điểm retrieval](evidence/retrieval_rankings.json), log lệnh và lịch sử loss nằm trong `evidence/`.
- `previews/` chứa ảnh render trực tiếp từ Word/PDF và PowerPoint cùng contact sheet để kiểm tra bố cục.

## Tạo lại tài liệu

Từ thư mục gốc của repository:

```powershell
python -m pip install -r requirements-report.txt
python tools/prepare_report_evidence.py
python tools/build_report.py
python tools/export_office.py
```

Script chụp ảnh cần Chrome cài sẵn. `export_office.py` cần Windows có Microsoft Word và PowerPoint; app chạy ẩn, xuất DOC/PDF/ảnh preview. Hai script đầu dùng thư viện Python để tạo DOCX và PPTX. Dùng `--capture-only` với script bằng chứng nếu chỉ muốn tạo lại viewer/ảnh từ bằng chứng đã lưu.

Nguồn phương pháp: [TPOUR](https://arxiv.org/abs/2606.17664) và [code tác giả](https://github.com/agwaBom/TPOUR). Báo cáo có tài liệu tham khảo và phân biệt kết quả demo với phương pháp gốc.
