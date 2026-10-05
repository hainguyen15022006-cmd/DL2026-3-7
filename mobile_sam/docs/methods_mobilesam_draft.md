# Bản nháp: Methods (MobileSAM) và gợi ý Related Work

> Bản nháp tiếng Việt, bạn dịch sang tiếng Anh nếu báo cáo viết bằng tiếng Anh. Mọi con số dưới đây lấy từ bài báo/README
> chính thức của MobileSAM; hãy mở lại nguồn để kiểm tra trước khi nộp. **Đừng thêm số liệu thực nghiệm khi chưa chạy.**

## 1. Đoạn Methods — MobileSAM (baseline)

MobileSAM (Zhang et al., 2023) là phiên bản nhẹ của Segment Anything Model (SAM). SAM gồm image encoder, prompt encoder
và mask decoder; trong đó image encoder (ViT-H) chiếm phần lớn tham số và thời gian tính. MobileSAM thay image encoder này
bằng một encoder nhỏ (TinyViT, khoảng 5M tham số so với khoảng 611M của encoder SAM gốc theo README của tác giả), được huấn
luyện bằng *decoupled distillation*: encoder nhỏ học tái tạo embedding mà encoder ViT-H tạo ra, nên có thể dùng chung
mask decoder gốc của SAM mà không cần huấn luyện lại decoder. Theo README, mô hình được huấn luyện trên khoảng 100k ảnh
(1% dữ liệu SA-1B) với một GPU trong chưa đầy một ngày. Vì prompt encoder và mask decoder giữ nguyên kiến trúc, MobileSAM
nhận cùng loại prompt (điểm, box) và cùng giao diện `SamPredictor` như SAM.

Trong dự án, chúng tôi dùng checkpoint `mobile_sam.pt` công bố trong repo chính thức (`model_type = vit_t`), không
fine-tune. Với mỗi ảnh, encoder chạy một lần; mỗi prompt (điểm dương hoặc box dạng `[x_min, y_min, x_max, y_max]`) được
đưa qua decoder với `multimask_output = False` để nhận đúng một mask. Ảnh được đưa vào ở dạng RGB; mask trả về có cùng kích
thước với ảnh gốc. Mask không được chọn dựa trên ground truth. Thời gian encode và decode được đo riêng trên cùng một máy
với SAM ViT-B.

## 2. Lý do chọn baseline và giới hạn của phép so sánh (đề nhóm yêu cầu nêu)

**Lý do chọn.**
- Cùng họ SAM, cùng loại prompt và cùng API, nên Setup 1 so sánh hai model dưới *đúng cùng instance và cùng prompt*.
- Nhẹ hơn nhiều (encoder khoảng 5M so với khoảng 611M tham số theo README của tác giả), phù hợp với phần cứng của nhóm, và
  cho phép hỏi liệu mô hình nhẹ có giữ được chất lượng dưới prompt chuẩn và prompt lệch hay không.

**Giới hạn (nên đưa vào Limitations).**
- MobileSAM được chưng cất từ encoder **ViT-H**, không phải ViT-B. Vì vậy "MobileSAM so với SAM ViT-B" không phải phép so
  sánh học sinh–giáo viên cùng cỡ.
- Hai checkpoint khác nhau ở cả encoder lẫn bộ trọng số decoder (mỗi checkpoint SAM là một mô hình đầy đủ, còn MobileSAM
  dùng decoder của SAM gốc theo bài báo). Do đó chênh lệch IoU không thể quy riêng cho encoder.
- Chỉ dùng 50 instance (hoặc 20 nếu thiếu tài nguyên): chênh lệch nhỏ có thể không đủ để kết luận; báo cáo hiệu theo cặp và
  mô tả mức biến thiên thay vì nói "tốt hơn" khi chênh lệch nhỏ.
- Prompt được mô phỏng từ GT (điểm sâu nhất, box khít, dịch 10%/20%), không phải thao tác của người dùng thật.
- Tốc độ chỉ có ý nghĩa trên máy nhóm đã đo, không suy ra từ bài báo.

## 3. Gợi ý nguồn cho Related Work (viết chung với Hoàng Công Dương)

Đã kiểm tra mã định danh:
- Kirillov et al., *Segment Anything*, 2023 — https://arxiv.org/abs/2304.02643 (task phân vùng theo prompt, mô hình SAM, dữ liệu SA-1B).
- Zhang et al., *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications*, 2023 — https://arxiv.org/abs/2306.14289.

Các nguồn khác (cần tự tìm và kiểm tra thông tin trích dẫn trước khi dùng): phân vùng tương tác bằng click
(vd. RITM — Sofiiuk et al.), COCO (Lin et al., 2014) cho dữ liệu, và một nguồn về độ nhạy của SAM với chất lượng prompt nếu nhóm tìm được.
