
# SafetyGuard AI
### Hệ thống giám sát và phát hiện vi phạm an toàn lao động bằng YOLO

SafetyGuard AI là hệ thống hỗ trợ phát hiện các hành vi vi phạm an toàn lao động thông qua hình ảnh. Hệ thống sử dụng mô hình YOLO để nhận diện đối tượng và các hành vi vi phạm, kết hợp với OpenCV để tiền xử lý ảnh nhằm hỗ trợ cải thiện chất lượng đầu vào.

## 1. Công nghệ sử dụng

- **Python:** Ngôn ngữ lập trình chính.
- **Flask:** Xây dựng Backend và xử lý yêu cầu từ giao diện.
- **OpenCV:** Đọc ảnh và tiền xử lý ảnh.
- **Ultralytics YOLO:** Nhận diện và phát hiện vi phạm.
- **HTML, Tailwind CSS, JavaScript:** Xây dựng giao diện người dùng.

## 2. Chức năng chính

- Tải ảnh lên hệ thống để phân tích.
- Tiền xử lý ảnh bằng không gian màu LAB.
- Tăng cường độ tương phản bằng CLAHE.
- Làm mượt ảnh bằng Gaussian Blur.
- Phát hiện vi phạm an toàn lao động bằng mô hình YOLO.
- Điều chỉnh ngưỡng Confidence.
- Hiển thị ảnh kết quả cùng Bounding Box, nhãn nhận diện và độ tin cậy.

## 3. Yêu cầu hệ thống

Trước khi chạy dự án, cần chuẩn bị:

- Python 3.9 trở lên.
- pip (Python Package Manager).
- Visual Studio Code hoặc IDE tương đương.
- Trình duyệt web.
- GPU NVIDIA hỗ trợ CUDA (không bắt buộc, có thể chạy bằng CPU).

## 4. Cài đặt và chạy dự án

### Bước 1: Tải mã nguồn

Clone repository về máy:

```bash
git clone <repository-url>
```

Di chuyển vào thư mục dự án:

```bash
cd SafetyGuard-AI
```

Nếu đã có mã nguồn trên máy, chỉ cần mở thư mục dự án bằng VS Code.

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

Trên macOS/Linux:

```bash
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Nếu dự án chưa có file `requirements.txt`, có thể cài đặt các thư viện cơ bản bằng lệnh:

```bash
pip install flask opencv-python ultralytics numpy pillow
```

### Bước 4: Kiểm tra mô hình YOLO

Đảm bảo file trọng số mô hình đã được đặt đúng vị trí mà chương trình đang cấu hình.

Ví dụ:

```text
SafetyGuard-AI/
│
├── best.pt
├── app.py
├── requirements.txt
└── ...
```

**Lưu ý:** Nếu file `best.pt` nằm trong thư mục khác, cần cập nhật đường dẫn đến file trọng số trong mã nguồn.

### Bước 5: Khởi chạy hệ thống

Tại thư mục gốc của dự án, chạy:

```bash
python app.py
```

Nếu chương trình khởi động thành công, Flask sẽ hiển thị địa chỉ máy chủ trên terminal.

Truy cập địa chỉ sau bằng trình duyệt:

```text
http://127.0.0.1:5000
```

## 5. Hướng dẫn sử dụng

1. Mở giao diện SafetyGuard AI trên trình duyệt.
2. Chọn hình ảnh cần kiểm tra.
3. Điều chỉnh ngưỡng Confidence nếu cần.
4. Bật hoặc tắt chức năng tiền xử lý ảnh.
5. Nhấn nút phân tích để hệ thống thực hiện nhận diện.
6. Theo dõi kết quả hiển thị trên giao diện.

## 6. Quy trình xử lý

```text
Người dùng tải ảnh lên
          |
          v
     Flask Backend
          |
          v
     Đọc và kiểm tra ảnh
          |
          v
      Tiền xử lý ảnh
          |
          v
       Mô hình YOLO
          |
          v
   Phân tích kết quả nhận diện
          |
          v
  Vẽ Bounding Box và nhãn
          |
          v
    Hiển thị kết quả
```

## 7. Một số lỗi thường gặp

**Lỗi không tìm thấy module:**

```bash
pip install -r requirements.txt
```

**Lỗi không tìm thấy file best.pt:**

Kiểm tra lại vị trí file trọng số và đường dẫn được khai báo trong mã nguồn.

**Lỗi không thể kích hoạt môi trường ảo trên Windows:**

Mở PowerShell và chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó kích hoạt lại:

```powershell
.\venv\Scripts\Activate.ps1
```

**Lỗi cổng 5000 đã được sử dụng:**

Kiểm tra tiến trình đang sử dụng cổng hoặc thay đổi cổng chạy Flask trong mã nguồn.

## 8. Lưu ý

- Đảm bảo file trọng số YOLO được cung cấp đầy đủ trước khi chạy hệ thống.
- Kết quả nhận diện phụ thuộc vào chất lượng ảnh đầu vào và mô hình đã huấn luyện.
- Thời gian xử lý phụ thuộc vào cấu hình phần cứng.
- Phiên bản hiện tại tập trung vào phân tích hình ảnh tĩnh.

---

**SafetyGuard AI – Hỗ trợ giám sát và nâng cao an toàn lao động.**
