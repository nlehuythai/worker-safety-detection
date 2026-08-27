# Ứng dụng Xử lý Ảnh: Phát hiện và Đếm phát hiện công nhân mặc đồ bảo hộ lao động tại công trường xây dựng.

Đồ án môn học **Xử lý ảnh** sử dụng các phương pháp xử lý ảnh truyền thống bằng OpenCV (Python) kết hợp giao diện trực quan với Streamlit để phát hiện, phân đoạn và đếm số lượng công nhân/thiết bị mặc áo/mũ phản quang trên công trường.

---

## Các tính năng chính
* **Tiền xử lý ảnh (Preprocessing):** Chuyển đổi không gian màu (BGR sang HSV/LAB) để tăng độ thích ứng với ánh sáng phức tạp và áp dụng bộ lọc khử nhiễu (Gaussian Blur).
* **Phân đoạn màu sắc (Color Thresholding):** Tách biệt vùng màu đặc trưng của áo/mũ phản quang dựa trên dải màu HSV (`cv2.inRange`).
* **Xử lý hình thái học (Morphological Operations):** Sử dụng các phép toán Erosion, Dilation, Opening và Closing để loại bỏ nhiễu và làm sạch đối tượng.
* **Nhận diện đường viền (Contour Detection):** Tìm kiếm và lọc các đối tượng dựa trên diện tích (Area) và tỷ lệ khung hình (Aspect Ratio).
* **Giao diện Web trực quan (Streamlit UI):** Cho phép người dùng tải ảnh lên, xem kết quả trực quan và theo dõi thống kê số lượng đối tượng phát hiện.
## cấu trúc thư mục dự án
```text
Construction_Image_Processing/
│
├── data/                    # Thư mục chứa ảnh test đầu vào
│   ├── raw/                 # Ảnh gốc thu thập
│   └── processed/           # Ảnh sau khi qua các bước xử lý
│
├── src/                     # Mã nguồn chính của dự án
│   ├── preprocessing.py     # Module xử lý màu và lọc nhiễu
│   ├── segmentation.py      # Module tách màu và hình thái học
│   ├── detection.py         # Module tìm contours và vẽ bounding box
│   └── app.py               # File giao diện ứng dụng Streamlit
│
├── requirements.txt         # Danh sách các thư viện Python cần thiết
└── README.md                # Tài liệu hướng dẫn sử dụng dự án
```
## Công Nghệ & Thư Viện Sử Dụng
Python (Phiên bản 3.9 trở lên)
OpenCV (cv2): Thư viện xử lý ảnh cốt lõi.
NumPy: Thao tác và xử lý mảng ma trận pixel.
Streamlit: Xây dựng giao diện web demo trực quan.
## Hướng Dẫn Cài Đặt & Chạy Ứng Dụng
1. Clone repository về máy
2. Cài đặt các thư viện phụ thuộc
Đảm bảo bạn đã cài đặt Python, sau đó chạy lệnh sau để cài đặt các thư viện trong file requirements.txt:

pip install -r requirements.txt