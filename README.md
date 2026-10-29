# Computer Vision: Road Damage Detect 🚗🛣️

Dự án Hệ thống Nhận diện Hư hỏng Mặt đường (**Road Damage Detection**) thuộc môn học Hệ thống Nhúng AI (CE340). Hệ thống huấn luyện mô hình YOLOv8 trên tập dữ liệu SVRDD và tối ưu hóa chuyển đổi sang **TensorFlow Lite (TFLite)** để chạy suy luận thời gian thực trên thiết bị nhúng **Raspberry Pi Zero 2W**.

---

## 📁 Cấu trúc Thư mục Dự án

```text
Road_Damage_Detect/
├── .gitignore                      # File cấu hình bỏ qua tài nguyên nặng & file rác Git
├── README.md                       # Tài liệu hướng dẫn & giải thích dự án
├── Final_TRAIN/                    # Thư mục chứa mã nguồn huấn luyện
│   └── FinalProject.ipynb          # Notebook quy trình chuẩn bị data, train & convert TFLite
└── FINAL_TEST/                     # Thư mục mã nguồn suy luận trên Raspberry Pi Zero 2W
    ├── main.py                     # Entrypoint chính chạy suy luận & tính điểm FPS/mAP
    ├── model.py                    # Class nạp mô hình TFLite Interpreter & tiền xử lý
    ├── nms.py                      # Thuật toán Non-Max Suppression (NMS) tối ưu cho YOLOv8
    ├── metrics.py                  # Module tính toán các chỉ số mAP50, mAP50-95, F1
    ├── preProcessData.py           # Script chuẩn hóa đường dẫn file test
    ├── requirements.txt            # Thư viện phụ thuộc tối ưu cho Raspberry Pi Zero 2W
    └── Best_FP32.tflite            # Mô hình TFLite FP32 đã qua chuyển đổi
```

---

## 🛠️ Quy trình Huấn luyện & Chuyển đổi Mô hình (`Final_TRAIN`)

Thư mục `Final_TRAIN/FinalProject.ipynb` bao gồm 3 phần chính được tối ưu sạch sẽ:

### Phần 1: Môi trường & Thư viện (Environment & Libraries)
- **Mount Google Drive**: Kết nối Google Drive để lưu trữ weights và không bị mất dữ liệu khi làm việc trên Google Colab.
- **Cài đặt thư viện**: Cài đặt `ultralytics`, `tensorflow`, `onnx`, `opencv-python`, `Pillow`.

### Phần 2: Tiền xử lý dữ liệu & Cấu hình YAML (Preprocessing & YAML Config)
- **Tại sao phải chuyển đường dẫn về tuyệt đối (Absolute Path)?**
  Khi di chuyển giữa các thư mục hoặc chạy trên các môi trường khác nhau (Windows, Linux, Colab), đường dẫn tương đối trong các file list (`train.txt`, `val.txt`, `test.txt`) dễ bị lỗi `No valid images found` do sai làm việc tại working directory gốc. Việc chuyển đổi thành đường dẫn tuyệt đối giúp YOLO truy cập chính xác 100% vị trí file ảnh và nhãn.
- **Tại sao phải chuẩn hóa dấu gạch chéo (`\` ➔ `/`)?**
  Hệ điều hành Windows mặc định dùng dấu `\`, trong khi Linux/Colab/Raspberry Pi OS sử dụng `/`. Chuyển về `/` giúp mã nguồn tương thích đa nền tảng.
- **Tạo file `svrdd.yaml`**: Cấu hình đường dẫn và 7 nhãn hư hỏng mặt đường:
  1. `Alligator Crack` (Nứt da cá)
  2. `Longitudinal Crack` (Nứt dọc)
  3. `Longitudinal Patch` (Vết vá dọc)
  4. `Manhole Cover` (Nắp hố ga)
  5. `Pothole` (Ổ gà)
  6. `Transverse Crack` (Nứt ngang)
  7. `Transverse Patch` (Vết vá ngang)

### Phần 3: Huấn luyện, Đánh giá & Chuyển đổi Mô hình
- **Lựa chọn Backbone YOLOv8n (`yolov8n.pt`)**: YOLOv8 Nano có số lượng tham số rất nhỏ (~3.2 triệu tham số), cực kỳ tối ưu cho các thiết bị phần cứng hạn chế như Raspberry Pi Zero 2W (RAM 512MB, CPU Cortex-A53).
- **Kích thước đầu vào `imgsz=320`**: Giảm kích thước ảnh từ 640x640 xuống 320x320 làm giảm số lượng phép tính FLOPs đi 4 lần, giúp tăng tốc độ suy luận (FPS) đáng kể trên phần cứng nhúng.
- **Xuất định dạng TFLite (`.tflite`)**: Định dạng TFLite nhẹ, cho phép suy luận bằng `tflite-runtime` chuyên biệt mà không cần cài đặt thư viện PyTorch/TensorFlow nặng.

---

## ⚡ Luồng suy luận trên Raspberry Pi Zero 2W (`FINAL_TEST`)

### 1. Định dạng mô hình
Mô hình chạy dưới định dạng **TensorFlow Lite (`.tflite`)**, nạp qua `tensorflow.lite.Interpreter` hoặc `tflite_runtime.interpreter.Interpreter`.

### 2. Tiền xử lý ảnh đầu vào
- Đọc ảnh bằng Pillow: `Image.open(path)`
- Resize ảnh về kích thước `320x320`.
- Chuyển thành mảng NumPy và mở rộng chiều batch: shape `(1, 320, 320, 3)`.
- **Chuẩn hóa (Normalization)**: Đối với mô hình Float32 (`floating_model == True`), giá trị pixel được chuẩn hóa về khoảng `[0.0, 1.0]` theo công thức:
  $$\text{input\_data} = \frac{\text{np.float32}(\text{input}) - 0.0}{255.0}$$

### 3. Suy luận & Hậu xử lý (Inference & Post-processing)
- Thực thi suy luận qua `interpreter.invoke()`.
- Đầu ra raw tensor dạng `[1, 11, 2100]` được transpose thành `[2100, 11]`.
- Lọc các box candidates có điểm số lớn hơn threshold (`score_threshold = 0.2`).
- Áp dụng thuật toán **Weighted Non-Max Suppression (NMS)** trong `nms.py` để loại bỏ các bounding box đè lấp.
- Quy đổi tọa độ box normalized center `(x, y, w, h)` về dạng góc `[left_x, top_y, right_x, bottom_y]` ứng với ảnh `320x320`.

### 4. Đánh giá hiệu năng
- Đo thời gian xử lý từng frame, tính chỉ số **FPS** trung bình.
- Tính toán chỉ số mAP50, mAP50-95, Precision, Recall và điểm tổng hợp Score.

---

## 🚀 Hướng dẫn Cài đặt & Chạy trên Raspberry Pi Zero 2W

### 1. Tạo môi trường ảo & Cài đặt thư viện
```bash
# Tạo môi trường ảo Python
python3 -m venv CE340
source CE340/bin/activate

# Cài đặt các thư viện cần thiết từ FINAL_TEST/requirements.txt
pip install -r FINAL_TEST/requirements.txt
```

### 2. Chạy suy luận
```bash
cd FINAL_TEST
python main.py
```
