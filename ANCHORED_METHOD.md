# Biến thể TPOUR có ràng buộc giữ kiến thức

## Vấn đề và giả thuyết

Ở lần chạy seed42 trước, TRPO đạt SciFact nDCG@10=0,5956, thấp hơn Contriever gốc 0,6394. Chưa thể xác định mọi nguyên nhân của mức giảm này. Giả thuyết của nhóm: dùng Contriever gốc làm teacher cố định để hạn chế drift khi tối ưu temporal preferences trên một mẫu Wikipedia nhỏ. Đây là cải tiến đối với bản thực hành hiện tại, không phải tuyên bố phát minh knowledge distillation hoặc tính mới đã được kiểm chứng trong toàn bộ literature.

## Thành phần được thêm vào thuật toán

Giữ nguyên CE, TRPO, reference momentum và queue. Thêm một encoder teacher khởi tạo từ Contriever gốc, đóng băng vĩnh viễn. Teacher không cập nhật momentum và không nhận gradient.

Với mỗi batch, candidate list là các tài liệu preferred và rejected của batch đó. Student và teacher mã hóa cùng văn bản đã augmentation. Tính softmax của cosine similarity / 0,1 trên cùng candidate list; retention ranking loss là KL(teacher || student). Thành phần thứ hai là squared L2 distance embedding chia squared teacher norm (epsilon 1e-6), lấy trung bình trên query/preferred/rejected. MSE giữ cả hướng lẫn độ lớn embedding vì inference dùng dot product.

Objective: **L_new = 0,925 L_CE + 0,075 L_TRPO + gamma (L_KL + eta L_rep)**.

Teacher là mô hình cố định ban đầu, khác reference momentum của TRPO. Không thêm fusion với baseline lúc inference: đánh giá chỉ dùng student sau train. Tính retention bằng FP32 cho ổn định softmax/KL, kể cả khi train bằng AMP. Teacher làm tăng thời gian, bộ nhớ và checkpoint; phải báo chi phí bên cạnh chất lượng.

## Kế hoạch cố định trước lần chạy mới

- gamma=10, eta=1, retention temperature=0,1. Giá trị là thiết kế thử ban đầu, chưa tối ưu; không lựa chọn dựa trên SciFact test của lần chạy này.
- 1.000 bước, lr1e-6, warmup100, batch4, queue4096, max_length256, delete probability0,1; cùng dữ liệu sạch và cùng seed 42/43/44 cho original TRPO và anchored.
- Ablation seed42: eta=0 (chỉ KL), so với eta=1. Đây chỉ là ablation một seed, không khẳng định độ ổn định giữa seed.
- Contriever gốc làm baseline chung. Cả ba seed đánh giá đủ 300 SciFact test queries; không chọn checkpoint tốt nhất theo test.
- Báo nDCG/Recall/MAP/MRR, mean/SD qua seed, query-level paired bootstrap, số cập nhật thực/skipped AMP, thời gian và peak memory. Ngân sách bước được giữ bằng nhau; số cập nhật có thể khác nếu AMP skip, phải ghi rõ.
- Held-out mined preference accuracy chỉ là proxy; chưa dùng làm temporal gold. Chia theo văn bản nguyên dạng, chưa loại mọi trùng article/excerpt. SciFact là general retrieval, không đủ chứng minh temporal alignment.
- Demo gồm query đầu tiên cố định, query cải thiện nhiều nhất và giảm nhiều nhất; ghi rõ hai ví dụ sau được chọn từ kết quả, không dùng chúng để thay thế thống kê toàn bộ.

14 kiểm tra hiện đã qua: gồm teacher đóng băng, gradient retention đi đúng hướng, retention bằng 0 khi embedding giống nhau, gamma0 cho gradient giống TRPO gốc, cùng các kiểm tra loss/queue/dữ liệu/metric/resume trước đó. Kiểm tra toy chỉ xác nhận toán và code, không chứng minh chất lượng.

Code: `anchored_tpour.py`; notebook: `notebooks/TPOUR_anchored_Kaggle.ipynb`; demo ảnh PNG: `tools/demo_anchored_results.py`. Kế hoạch ban đầu được lưu trước lúc chạy tại commit `219a21a`. [Kết quả đủ ba seed và ablation](results/anchored_kaggle/README.md) cho thấy SciFact tăng so với TRPO nhưng vẫn thấp hơn baseline và preference accuracy giảm. Phần MSE chưa có bằng chứng lợi ích riêng trong ablation một seed; không gọi đây là cải thiện toàn diện.

## Nguồn phương pháp

Knowledge distillation để giữ tri thức đã có trong [Learning without Forgetting, Li & Hoiem](https://arxiv.org/abs/1606.09282). KL trên phân bố tài liệu và hạn chế phụ thuộc teacher đã được nghiên cứu trong [Weighted KL-Divergence for Document Ranking Model Refinement](https://arxiv.org/abs/2406.05977). Đóng góp dự kiến của nhóm là tích hợp và kiểm chứng retention trong bản TPOUR thực hành, chưa phải thuật toán KD hoàn toàn mới.
