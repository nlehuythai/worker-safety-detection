# Ứng dụng Xử lý Ảnh: Phát hiện và Đếm số lượng công nhân mặc/không mặc đồ bảo hộ lao động

Đồ án môn học **Xử lý ảnh** ứng dụng mô hình Hybrid kết hợp **Xử lý ảnh truyền thống bằng OpenCV (Python)** với **Deep Learning (YOLOv8)** để phát hiện, định vị và đếm số lượng công nhân có hoặc không mặc đồ bảo hộ lao động (mũ, áo phản quang) tại công trường xây dựng, tích hợp giao diện web trực quan.

---

## Các tính năng chính
* **Tiền xử lý ảnh (Preprocessing):** 
  * Khử nhiễu hạt bằng bộ lọc **Gaussian Blur**.
  * Chuyển đổi không gian màu sang **LAB** kết hợp thuật toán **CLAHE (Contrast Limited Adaptive Histogram Equalization)** để tối ưu hóa độ tương phản và chi tiết vùng áo phản quang, mũ bảo hộ dưới điều kiện ánh sáng phức tạp.
* **Huấn luyện mô hình AI (YOLOv8 Object Detection):** Huấn luyện mô hình Deep Learning trên tập dữ liệu chuẩn hóa để nhận diện chính xác 2 nhóm đối tượng: công nhân có mặc đồ bảo hộ và không mặc đồ bảo hộ.
* **Pipeline kết hợp (Hybrid Approach):** Tận dụng xử lý ảnh truyền thống ở khâu tiền hóa dữ liệu đầu vào kết hợp cùng YOLOv8 ở tầng suy luận (Inference) giúp nâng cao độ chính xác.
* **Giao diện Web trực quan (Streamlit / FastAPI UI):** Cho phép người dùng tải ảnh lên, xem trực quan các bounding box kết quả và thống kê số lượng đối tượng được phát hiện.

---

## Cấu trúc thư mục dự án
```text
worker-safety-detection/
│
├── data/                    # Thư mục chứa dữ liệu
│   ├── raw_img/             # Ảnh gốc thu thập (hoặc raw_dataset từ Roboflow)
│   ├── processed_img/       # Ảnh sau khi qua pipeline tiền xử lý (CLAHE + Blur)
│   └── data.yaml            # File cấu hình tập dữ liệu cho YOLO
│
├── src/                     # Mã nguồn chính của dự án
│   ├── Preprocessing/       # Module tiền xử lý (preprocessing.py)
│   ├── detection/           # Module huấn luyện và inference mô hình YOLO (train.py)
│   ├── segmentation/        # Module phân đoạn hỗ trợ (nếu có)
│   └── UI/                  # File giao diện ứng dụng web
│
├── requirements.txt         # Danh sách các thư viện Python cần thiết
└── README.md                # Tài liệu hướng dẫn sử dụng dự án