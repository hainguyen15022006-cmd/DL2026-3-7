# MobileSAM (baseline) — hướng dẫn checkpoint và hợp đồng I/O

Người phụ trách: Nguyễn Sỹ Chúc · Nhóm 3 / Đề tài 7 · vai trò: baseline của Setup 1 (MobileSAM vs SAM ViT-B)

## 1. Nguồn chính thức

| Mục | Giá trị |
|---|---|
| Code | https://github.com/ChaoningZhang/MobileSAM |
| Checkpoint | `weights/mobile_sam.pt` trong repo trên (https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) |
| `model_type` | `vit_t` |
| Bài báo | Zhang et al., *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications*, https://arxiv.org/abs/2306.14289 |
| Không dùng | MobileSAMv2 (dự án khác nằm trong cùng repo), SAM 2, checkpoint ViT-H/ViT-L |

## 2. Cài đặt và tải (Windows cmd)

```bat
pip install torch torchvision numpy opencv-python pillow timm pytest
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM.git %TEMP%\MobileSAM
mkdir weights
copy %TEMP%\MobileSAM\weights\mobile_sam.pt weights\mobile_sam.pt
dir weights\mobile_sam.pt
certutil -hashfile weights\mobile_sam.pt SHA256
```

(Linux/macOS: `cp`, `ls -l`, `sha256sum`.) Không commit file `.pt` lên GitHub; README chỉ hướng dẫn tải về `weights/`.

## 3. Kiểm tra đúng file

| Kiểm tra | Kết quả mong đợi |
|---|---|
| Kích thước | **40.728.226 byte** (~40,7 MB). Chỉ vài trăm byte = tải sai / con trỏ Git LFS |
| SHA-256 (file dùng để test adapter) | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |

Đây là hash của file đã được dùng để kiểm thử adapter. Nếu máy bạn ra hash khác, hãy báo cả nhóm **trước khi chạy thí nghiệm**: mọi thành viên phải dùng cùng một checkpoint cho Setup 1.

## 4. Ghi lại phiên bản (điền sau khi chạy trên máy của nhóm)

```bat
pip freeze | findstr /i "mobile torch timm numpy opencv"
```

`pip freeze` in dòng dạng `mobile_sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@<commit>`; hãy chép commit đó vào bảng.

| Mục | Máy nhóm (điền) | Ghi chú: môi trường thử của tôi (không phải kết quả nhóm) |
|---|---|---|
| OS / CPU / GPU | | Linux, CPU 2 luồng, không GPU |
| Python / PyTorch | | 3.13.16 / 2.14.1 |
| Commit MobileSAM | | `f706ad9c4eb7f219c00d9050e46328518ffb65d2` |
| SHA-256 checkpoint | | như mục 3 |
| encode / decode mỗi lần | | ≈ 1,2 s / ≈ 60 ms trên CPU nói trên |

Tốc độ của nhóm phải **tự đo trên cùng một máy** cho cả hai model (adapter trả `encode_seconds` và `decode_seconds` riêng). Bỏ lần chạy đầu tiên khi đo (khởi động). Không dùng số tốc độ trong bài báo thay cho số đo của nhóm.

## 5. Hợp đồng I/O (SAM ViT-B cần giống hệt)

| Đối tượng | Quy ước |
|---|---|
| Ảnh | `uint8`, shape `(H, W, 3)`, **RGB** (cv2.imread trả BGR: phải đổi; PIL `.convert("RGB")` là RGB) |
| Point | `(x, y)` pixel ảnh gốc, nhãn dương = 1; phải nằm trong ảnh |
| Box | `[x_min, y_min, x_max, y_max]` pixel ảnh gốc, trong ảnh, `x_max > x_min`, `y_max > y_min`. Bbox COCO `[x,y,w,h]` đổi bằng `coco_xywh_to_xyxy` |
| Đầu ra | `SegResult(mask: bool (H, W), score, encode_seconds, decode_seconds)` |
| Chọn mask | `multimask_output=False` → đúng 1 mask/prompt. **Không dùng GT để chọn mask.** `score` là IoU do model tự dự đoán |
| API | `set_image(img)` một lần/ảnh → `predict(point=… hoặc box=…)` nhiều lần; hoặc `segment(img, point=…)` |

Adapter **từ chối** (báo lỗi rõ) ảnh xám/RGBA/float, point hoặc box nằm ngoài ảnh, box đảo ngược hoặc nhầm định dạng COCO, và việc truyền cả point lẫn box. Nó không tự sửa prompt: việc clip vào ảnh thuộc về runner, và point rơi ngoài GT được giữ nguyên để ghi `point_inside_gt`.

Để Hoàng Công Dương làm `sam_vit_b.py` giống hệt: sao chép file này, đổi `from mobile_sam import …` thành `from segment_anything import …` (hoặc `pip install git+https://github.com/facebookresearch/segment-anything.git`), `MODEL_TYPE = "vit_b"`, `name = "sam_vit_b"`, checkpoint `sam_vit_b_01ec64.pth`. Hai thư viện có cùng API `SamPredictor`.

Tiền xử lý do `SamPredictor` đảm nhiệm: resize cạnh dài về 1024, chuẩn hoá, padding, rồi đưa mask về kích thước ảnh gốc. Đã kiểm tra: ảnh 640×480 và ảnh không vuông 150×210 đều cho mask đúng `(H, W)`.

## 6. Chạy smoke test

```bat
python scripts\smoke_mobile_sam.py --checkpoint weights\mobile_sam.pt
python scripts\smoke_mobile_sam.py --checkpoint weights\mobile_sam.pt --image anh.jpg --point 320 240 --box 100 80 500 400 --gt gt_mask.png
python -m pytest -q
```

Lệnh thứ nhất dùng ảnh tổng hợp có GT. Lệnh thứ hai dùng ảnh thật của nhóm, nên hãy chạy đúng các ảnh/prompt mà người làm SAM ViT-B đã dùng để so sánh. `tests/` có nhóm kiểm tra đầu vào (không cần checkpoint) và nhóm kiểm tra với model thật (đặt `MOBILE_SAM_CKPT` hoặc để file ở `weights/mobile_sam.pt`, nếu không sẽ tự bỏ qua).
