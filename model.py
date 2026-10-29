
import os
import numpy as np

from ultralytics import YOLO

class Model:
    
    def __init__(self, model_path):
        """
        Hàm khởi tạo mô hình.

        Args:
            model_path (str): Đường dẫn đến file trọng số mô hình (.pt).
        """
        print(f"Đang khởi tạo Model với mô hình tại: {model_path}")
        self.model_path = model_path
        self.model = None # Khởi tạo mô hình sau trong prepare hoặc predict

    def prepare(self):
        """
        Thiết lập các thao tác cần thiết. Hàm này chỉ thực thi một lần.
        Thường dùng để tải mô hình hoặc thiết lập device nếu cần.
        """
        if self.model is None:
            print("Đang chuẩn bị Model (tải mô hình)...")
            try:
                # Tải mô hình YOLO từ đường dẫn đã cung cấp
                self.model = YOLO(self.model_path)
                print("Model đã chuẩn bị xong.")
                # ultralytics thường tự động chọn device (GPU nếu có)
                # Nếu muốn chỉ định rõ, có thể thêm: self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            except Exception as e:
                 print(f"Lỗi khi chuẩn bị model (tải file {self.model_path}): {e}")
                 # Có thể raise lỗi hoặc xử lý tùy yêu cầu đề tài
                 raise e # Báo lỗi để dừng nếu không tải được

    def predict(self, image):
        """
        Thực hiện dự đoán phát hiện đối tượng trên một ảnh.

        Args:
            image (numpy.ndarray): Ảnh đầu vào là một numpy array.
                                   Đề tài quy định kích thước 3x320x320.
                                   (Lưu ý: ultralytics predict thường nhận HxWxC hoặc PIL Image)
                                   Chúng ta sẽ xử lý chuyển đổi format nếu cần.

        Returns:
            list: Danh sách các đối tượng phát hiện được.
                  Mỗi đối tượng là list 6 phần tử:
                  [left_x, top_y, right_x, bottom_y, score, class_index]
                  Tọa độ ở đây là tọa độ pixel trên ảnh gốc (hoặc ảnh đã resize 320x320).
        """
        if self.model is None:
            # Nếu prepare chưa được gọi hoặc bị lỗi, tải lại mô hình ở đây
            print("Model chưa được chuẩn bị. Đang tải mô hình trong predict...")
            self.prepare() # Cố gắng chuẩn bị lại

        if self.model is None:
             print("Lỗi: Không thể tải mô hình. Không thể dự đoán.")
             return [] # Trả về rỗng nếu model không sẵn sàng

        # --- Xử lý định dạng ảnh đầu vào ---
        # Đề tài quy định input là numpy array kích thước 3x320x320
        # Ultralytics predict thường nhận numpy array kích thước HxWxC (ví dụ: 320x320x3)
        # Chúng ta sẽ chuyển đổi từ C x H x W sang H x W x C
        # Cần kiểm tra dtype và channel order (BGR/RGB)
        # Giả định input là uint8 và cần chuyển từ C,H,W sang H,W,C
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[1:] == (320, 320):
             # Input đúng định dạng 3x320x320 như đề bài quy định
             # Chuyển sang định dạng HxWxC cho ultralytics predict
             # Cần xác định thứ tự kênh (RGB hay BGR). Ultralytics dùng OpenCV (BGR).
             # Nếu input là RGB (thường từ PIL): Transpose (1,2,0) rồi BGR2RGB
             # Nếu input là BGR: Chỉ cần Transpose (1,2,0)
             # Giả định input sau transpose (1,2,0) sẽ có thứ tự kênh mà ultralytics hiểu (thường là BGR)
             img_processed = np.transpose(image, (1, 2, 0)) # Kết quả: 320x320x3

             # Đảm bảo dtype là uint8 nếu chưa phải (cần cho OpenCV/ultralytics)
             if img_processed.dtype != np.uint8:
                 img_processed = img_processed.astype(np.uint8)

        elif image.ndim == 3 and image.shape[:2] == (320, 320) and image.shape[2] == 3:
             # Input đã là HxWxC 320x320x3
             img_processed = image
             print("Info: Input image đã có định dạng HxWxC (320x320x3), không cần transpose.")
             if img_processed.dtype != np.uint8:
                 img_processed = img_processed.astype(np.uint8)
        else:
             print(f"Cảnh báo: Định dạng ảnh đầu vào không đúng như quy định (mong muốn 3x320x320 hoặc 320x320x3). Định dạng nhận được: {image.shape}. Cố gắng dự đoán với ảnh gốc...")
             img_processed = image # Thử dùng ảnh gốc nếu không đúng format quy định
             if img_processed.dtype != np.uint8 and img_processed.ndim == 3:
                  img_processed = img_processed.astype(np.uint8)
             # Lưu ý: Nếu ảnh gốc không phải 320x320, kết quả có thể không đúng
             # Bạn có thể thêm logic resize ở đây nếu cần

        # --- Thực hiện dự đoán ---
        # Predict trên 1 ảnh (chỉ cần truyền numpy array hoặc list chứa 1 numpy array)
        # imgsz=320 để đảm bảo mô hình xử lý ảnh ở kích thước 320x320
        # verbose=False để không in log dự đoán cho mỗi lần gọi predict
        results = self.model(img_processed, imgsz=320, verbose=False)

        # --- Xử lý kết quả dự đoán ---
        detections = []
        if results and results[0].boxes: # Kiểm tra nếu có kết quả và có bounding box
            # results là list (vì predict có thể nhận list ảnh), lấy kết quả cho ảnh đầu tiên (index 0)
            boxes = results[0].boxes # Lấy object Boxes chứa các bounding box

            # Duyệt qua từng bounding box được phát hiện
            for box in boxes:
                # Lấy tọa độ xyxy (góc trên trái, góc dưới phải) ở dạng pixel
                xyxy = box.xyxy[0].tolist() # Tensor to list [x1, y1, x2, y2]

                # Lấy confidence score và class index
                score = box.conf[0].item() # Tensor to float
                class_index = int(box.cls[0].item()) # Tensor to int

                # Định dạng kết quả theo yêu cầu: [left_x, top_y, right_x, bottom_y, score, class_index]
                detection_info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], score, class_index]
                detections.append(detection_info)

        # Trả về danh sách các đối tượng phát hiện được
        return detections

# --- Ghi chú: Cách sử dụng lớp Model này sẽ ở một script khác ---
# Ví dụ script sử dụng lớp Model sẽ nằm ở dưới trong Colab Notebook