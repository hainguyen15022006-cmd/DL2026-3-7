# Dữ liệu đánh giá - Lê Tấn Thành

## Phạm vi bàn giao

COCO 2017 **val2017**, 50 ảnh và 50 instance, một instance trên mỗi ảnh,
`seed=2026`. Mười dòng đầu của manifest là tập smoke test. Danh sách này dùng
chung cho cả ba setup. Đây là **tập đánh giá**, không có bước train/fine-tune
và không tạo train/validation split riêng cho nhóm.

Phần dữ liệu gồm ảnh gốc, GT masks, manifest, script chuẩn bị, ba overlay kiểm
tra và tài liệu này. Prompt, mô hình, IoU và phân tích kết quả mô hình thuộc
phần việc của các thành viên khác. Không có kết quả suy luận trong bộ bàn giao.

## Nguồn và điều kiện sử dụng

- Trang tải COCO: <https://cocodataset.org/#download>
- Annotation chính thức: <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>
  (252,907,541 bytes khi tải ngày 05/10/2026, giờ Việt Nam).
  Script chỉ giải nén `annotations/instances_val2017.json`.
- Toàn bộ ảnh validation: <http://images.cocodataset.org/zips/val2017.zip>.
  Để tránh tải ảnh ngoài tập đánh giá, script tải trực tiếp 50 ảnh đã chốt từ
  `http://images.cocodataset.org/val2017/{file_name}`; không resize hoặc nén lại.
- COCO API: <https://github.com/cocodataset/cocoapi>;
  giải mã polygon/RLE bằng `pycocotools.COCO.annToMask`.
- Điều khoản gốc: <https://cocodataset.org/#termsofuse>;
  [bản nội dung trên repo chính thức](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).

Annotation thuộc COCO Consortium, theo CC BY 4.0:
<https://creativecommons.org/licenses/by/4.0/>. GT PNG là bản chuyển đổi từ
annotation COCO do Lê Tấn Thành chuẩn bị cho nhóm 3/đề tài 7; không sửa nhãn
bằng tay. Bản quyền ảnh thuộc chủ sở hữu ảnh, không phải COCO Consortium;
phải xem điều khoản Flickr và giấy phép riêng của từng ảnh. Không gán chung
CC BY 4.0 cho toàn bộ ảnh. `preparation_stats.json` giữ `flickr_url`, URL
nguồn, ID/tên/URL giấy phép của từng ảnh theo metadata COCO, cùng checksum.
Overlay là hình minh họa có thêm lớp màu GT trên ảnh gốc.

## Quy tắc chọn mẫu đã khóa

1. Đọc toàn bộ annotation của val2017. Duyệt `image_id` tăng dần, trong mỗi
   ảnh duyệt `annotation_id` tăng dần để không phụ thuộc thứ tự JSON.
2. Loại `iscrowd != 0`; bbox không hữu hạn/không đủ bốn phần tử/rộng hoặc cao
   không dương; area không hữu hạn/không dương; segmentation không giải mã
   được; mask sai H×W hoặc rỗng. Không đặt ngưỡng kích thước vật thể, không
   lọc theo category hoặc độ dễ của ảnh. Các lý do được đếm theo thứ tự này,
   mỗi annotation bị loại được đếm một lần.
3. Khởi tạo **một** `random.Random(2026)` của Python. Với mỗi ảnh còn instance
   hợp lệ, dùng `rng.choice` chọn một instance từ danh sách đã sắp theo ID.
4. Sau khi đã chọn một instance trên mọi ảnh hợp lệ, dùng cùng RNG gọi
   `rng.sample(candidates, 50)`. Giữ nguyên thứ tự trả về trong manifest.
5. Ghi manifest trước khi tải ảnh hoặc tạo overlay. Nếu chạy lại cho ra danh
   sách khác manifest hiện có, script dừng thay vì tự đổi danh sách.
6. Dùng `manifest[:10]` cho smoke test và `manifest[:3]` cho overlay kiểm tra.
   Không chọn lại mẫu dựa trên mask dự đoán hay IoU. Bản bàn giao này dùng đủ
   50 mẫu; không sử dụng phương án dự phòng 20 mẫu.

Chỉ ghi seed là chưa đủ để tái tạo: cần giữ đúng thứ tự xử lý, thuật toán
trên, dữ liệu nguồn và phiên bản môi trường. Manifest đã xuất là danh sách
chính thức cho các thành viên sử dụng.

## Số lượng thực tế sau chuẩn bị

| Hạng mục | Số lượng |
|---|---:|
| Ảnh trong annotation val2017 | 5,000 |
| Annotation ban đầu | 36,781 |
| Loại vì `iscrowd != 0` | 446 |
| Loại vì bbox/area không hợp lệ | 0 |
| Loại vì lỗi giải mã, sai kích thước hoặc mask rỗng | 0 |
| Annotation hợp lệ còn lại | 36,335 |
| Ảnh có ít nhất một instance hợp lệ | 4,952 |
| Ảnh không có instance hợp lệ | 48 |
| Ảnh / instance được chọn | 50 / 50 |
| Category có mặt trong 50 instance | 30 |
| Smoke instances / overlay kiểm tra | 10 / 3 |

Đây là thống kê dữ liệu, không phải kết quả mô hình. Tập nhỏ không được cân
bằng theo lớp và không đại diện đầy đủ cho COCO. Thống kê máy đọc được nằm
trong `data/coco/preparation_stats.json`; lý do loại không xuất hiện trong
dictionary `rejected_annotations` có số lượng bằng 0.

## File và giao diện bàn giao

```text
scripts/prepare_data.py
configs/eval_manifest.json
DATA.md
data/coco/annotations/instances_val2017.json
data/coco/val2017/{file_name}
data/coco/gt_masks/{annotation_id}.png
data/coco/preparation_stats.json
results/examples/data_{image_id}_{annotation_id}.png  (3 ảnh)
data/coco_eval_seed2026.zip
```

Manifest là JSON **list**, mỗi dòng có đúng tám trường đã thống nhất:

| Trường | Ý nghĩa / định dạng |
|---|---|
| `image_id` | ID ảnh COCO, số nguyên |
| `annotation_id` | ID instance COCO, số nguyên |
| `file_name` | Tên JPEG gốc để ghép với `data/coco/val2017/` |
| `width`, `height` | Kích thước ảnh gốc, pixel |
| `category_id` | ID category gốc COCO, không remap |
| `bbox_xywh` | `[x, y, width, height]` từ annotation, không đổi định dạng |
| `area` | Diện tích do annotation COCO cung cấp, không thay bằng diện tích bbox |

`area` gốc có thể khác số pixel 1 sau rasterize polygon. GT mask lưu bằng PNG
8-bit một kênh, giá trị **0 hoặc 255**; khi nạp phải chuyển `> 0` thành bool.
Mask đánh dấu đúng instance của dòng manifest, không gom các vật thể cùng
category. Tất cả mask có shape `(height, width)` của ảnh gốc. Không resize,
crop, augmentation hay chuẩn hóa pixel ở bước chuẩn bị dữ liệu.

Ví dụ nạp dữ liệu từ thư mục gốc repo:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path('.')
manifest = json.loads((root / 'configs/eval_manifest.json').read_text(encoding='utf-8'))
row = manifest[0]  # manifest[:10] cho smoke test
image = np.asarray(Image.open(root / 'data/coco/val2017' / row['file_name']).convert('RGB'))
gt = np.asarray(Image.open(root / 'data/coco/gt_masks' / f"{row['annotation_id']}.png")) > 0
assert image.shape[:2] == gt.shape == (row['height'], row['width'])
assert gt.dtype == np.bool_
```

Người làm prompt tạo positive point và tight box từ GT mask theo giao thức
của nhóm. `bbox_xywh` trong manifest giữ nguyên COCO; nếu dùng nó làm box
input thì phải đổi sang `[x, y, x+w, y+h]`. Không truyền nhầm xywh vào SAM.
GT được dùng để tạo prompt chuẩn và chấm điểm, không dùng để chọn mask dự
đoán tốt nhất.

Mười mẫu smoke, theo đúng thứ tự manifest:

| STT | image_id | annotation_id |
|---|---:|---:|
| 1 | 521282 | 1152444 |
| 2 | 260925 | 144786 |
| 3 | 54592 | 613127 |
| 4 | 455085 | 206430 |
| 5 | 220310 | 1161337 |
| 6 | 474854 | 1042325 |
| 7 | 261732 | 658626 |
| 8 | 134689 | 595505 |
| 9 | 32811 | 37495 |
| 10 | 378284 | 506923 |

## Cài đặt và chạy lại

Môi trường đã dùng: Python **3.12.14**, NumPy **2.5.3**, Pillow **12.3.0**,
pycocotools **2.0.11**. Chỉ cần CPU cho bước chuẩn bị dữ liệu.
Các lệnh dưới đây chạy trong thư mục gốc repo trên PowerShell.

Máy mới có Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.3 Pillow==12.3.0 pycocotools==2.0.11
.\.venv\Scripts\python.exe scripts\prepare_data.py --download
```

Trên máy Thành đã có `.venv`, chạy trực tiếp lệnh cuối. Môi trường này được
tạo từ Python đi kèm Codex; không chia sẻ `.venv` cho các máy khác. Các phiên
bản trên là dependencies của phần dữ liệu để trưởng nhóm ghép vào
`requirements.txt` chung.

Script chỉ tải file còn thiếu. Nếu một ảnh tải lỗi, giữ nguyên danh sách ID
và dừng; chạy lại cùng lệnh sẽ tiếp tục từ các file đã có. Không tự thay ảnh.
Nếu đã tải thủ công, đặt JSON và ảnh vào đúng đường dẫn trên rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

Kiểm tra lại không sửa dữ liệu:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --verify-only
```

Lệnh này sinh lại danh sách từ annotation, so với manifest đã khóa, kiểm tra
đủ 50 cặp ảnh/mask, giá trị PNG nhị phân, đối chiếu từng pixel với
`annToMask`, kiểm tra checksum và ba file overlay. Không dùng mạng.

Tạo lại ZIP sau khi đã chuẩn bị hoặc kiểm tra:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --verify-only --package
```

## Kiểm tra overlay và giới hạn nhãn

Ba hình lấy từ ba dòng đầu, không chọn theo chất lượng mô hình:

- `data_521282_1152444.png`: category **vase**, GT đánh dấu bình hoa.
- `data_260925_144786.png`: category **car**, GT đánh dấu xe; vùng gắn nhãn
  COCO khá thô và có phần phủ qua con mèo phía trước. Giữ nguyên annotation
  gốc, ghi nhận là giới hạn GT khi diễn giải IoU sau này.
- `data_54592_613127.png`: category **skis**, GT đánh dấu ván trượt được chọn.

Đã xem trực quan cả ba: hình bên trái là ảnh gốc, bên phải là GT màu đỏ với
độ đục 45%; không thấy lỗi dịch hệ tọa độ/đảo kích thước do xử lý dữ liệu.
Kiểm tra trực quan ba mẫu không chứng minh mọi nhãn COCO đều hoàn hảo.

SHA256 của `instances_val2017.json` đã tải:
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

SHA256 của manifest đã khóa:
`de36a8b929c27bc14872969d091b72ad54f2bd62013b2c342d80d7ac0bd0b4a2`.

Đây là checksum ghi nhận tại lần chuẩn bị, không phải chữ ký xác thực của
nhà phát hành. Checksum từng ảnh và mask nằm trong `preparation_stats.json`.

## Gói dữ liệu xử lý để bàn giao

File tại máy: **`data/coco_eval_seed2026.zip`**. Gói chứa 50 ảnh gốc, 50 GT
masks, manifest, thống kê/nguồn/giấy phép và ba overlay, kèm `DATA.md`.
Giải nén tại thư mục gốc repo để dùng ngay các ảnh và mask. Gói không chứa
toàn bộ annotation COCO; để chạy `--verify-only` trên máy mới, cần tải và
đặt `instances_val2017.json` đúng đường dẫn, hoặc chạy `--download` trước.

- [Tải ZIP trực tiếp](https://drive.google.com/uc?export=download&id=1WbxCsxLFAPNl1HjxBa-_02kNHiUH-zu8)
- [Mở file trên Google Drive](https://drive.google.com/file/d/1WbxCsxLFAPNl1HjxBa-_02kNHiUH-zu8/view?usp=sharing)

Đã kiểm tra ngày 05/10/2026: tải được không cần đăng nhập; ZIP có
9,856,902 bytes, 106 file và vượt qua kiểm tra toàn vẹn. SHA256 của ZIP:
`10366a6fe77cf343bb23f704ed575d53151f7ce83ce78cfc9077a4d20c55d6fa`.

ZIP trên Drive giữ nguyên bản dữ liệu đã kiểm tra. Bản `DATA.md` bên trong
ZIP được tạo trước khi có link chia sẻ; dùng `DATA.md` trên nhánh
`thanh-data` làm tài liệu cập nhật. Ảnh, masks và manifest không thay đổi.

Thư mục `/data/` được bỏ qua bởi Git, gồm ảnh, masks, annotation và ZIP.
Code, manifest, ba overlay và `DATA.md` là các file nhẹ để bàn giao qua repo
trên nhánh `thanh-data`. Không cần đưa ảnh COCO hay checkpoint vào Git.

## Dataset and Data Preparation - nội dung bàn giao cho báo cáo

We constructed a fixed evaluation subset from the COCO 2017 validation
split (val2017), using the official instance segmentation annotations.
The source contains 5,000 images and 36,781 annotations. We excluded 446
crowd annotations (`iscrowd != 0`), leaving 36,335 valid instances across
4,952 images. We also checked bounding-box and area validity, segmentation
decoding, non-empty masks and mask dimensions; these checks rejected no
additional annotations. No category-specific or minimum-size filtering
was applied.

Using Python's `random.Random(2026)`, we selected one valid instance per
eligible image in ascending image-ID order, with annotations sorted by ID,
and then sampled 50 image-instance pairs using the same random generator.
The subset contains 30 categories and is not class-balanced. All three
experimental setups must use this fixed manifest; its first ten entries
form the smoke-test subset. Selection was completed before model inference
or inspection of IoU scores.

Ground-truth segmentations were decoded with pycocotools 2.0.11 and saved
as binary PNG masks at the original image resolution (0 for background,
255 for the selected instance). Images were not resized or augmented;
RGB conversion is performed when loading them. Three overlays were
visually inspected to check image-mask alignment. We retained the original
COCO annotations, including coarse boundaries or occlusion-related label
limitations. The dataset is used only for evaluation of pretrained models,
with no training, fine-tuning or additional train/validation split. The
small sample and annotation imperfections limit generalization. Sources,
selection rules, software versions and checksums are documented in DATA.md
and the accompanying preparation statistics.

Nguồn cho phần Dataset: COCO download/terms và COCO API ở đầu tài liệu.
Trưởng nhóm ghép đoạn này vào báo cáo chung và sử dụng link tải gói xử lý
ở mục bàn giao phía trên.
