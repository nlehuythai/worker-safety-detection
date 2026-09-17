import os
import cv2
import numpy as np
from tqdm import tqdm

def enhance_image(img):
    """
    Hàm tiền xử lý ảnh truyền thống: Khử nhiễu + Cân bằng sáng CLAHE
    """
    # 1. Khử nhiễu nhẹ
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 2. Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 3. Cân bằng tương phản CLAHE trên kênh độ sáng
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Gộp kênh và chuyển về BGR
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def process_yolo_dataset(raw_base_dir, processed_base_dir):
    """
    Duyệt qua các tập train, valid/val, test của YOLO dataset
    """
    splits = ['train', 'valid', 'val', 'test']
    
    for split in splits:
        raw_img_dir = os.path.join(raw_base_dir, split, 'images')
        raw_lbl_dir = os.path.join(raw_base_dir, split, 'labels')
        
        # Nếu thư mục split này không tồn tại trong dataset thì bỏ qua
        if not os.path.exists(raw_img_dir):
            continue
            
        proc_img_dir = os.path.join(processed_base_dir, split, 'images')
        proc_lbl_dir = os.path.join(processed_base_dir, split, 'labels')
        
        os.makedirs(proc_img_dir, exist_ok=True)
        os.makedirs(proc_lbl_dir, exist_ok=True)
        
        print(f"\n[+] Đang xử lý tập dữ liệu: [{split}]")
        image_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(image_files, desc=f"Processing {split}"):
            img_path = os.path.join(raw_img_dir, filename)
            
            # 1. Đọc và xử lý ảnh
            img = cv2.imread(img_path)
            if img is not None:
                processed_img = enhance_image(img)
                cv2.imwrite(os.path.join(proc_img_dir, filename), processed_img)
            
            # 2. Copy file nhãn (.txt) tương ứng sang thư mục mới mà không làm thay đổi nội dung
            base_name, _ = os.path.splitext(filename)
            lbl_filename = base_name + ".txt"
            src_lbl_path = os.path.join(raw_lbl_dir, lbl_filename)
            dst_lbl_path = os.path.join(proc_lbl_dir, lbl_filename)
            
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                    f.write(content)

    print("\n[+] Hoàn tất toàn bộ quá trình tiền xử lý dataset cho YOLO!")

if __name__ == "__main__":
    # Đường dẫn thư mục dataset thô bạn vừa tải về và giải nén
    # (Hãy chỉnh lại tên thư mục cho khớp với thực tế của bạn)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw_dataset","construction PPE.v1i.yolov8")         # Hoặc tên thư mục dataset thô của bạn
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_dataset")
    print(f"[DEBUG] Kiểm tra thư mục nguồn: {RAW_DIR}")
    print(f"[DEBUG] Tồn tại không?: {os.path.exists(RAW_DIR)}")
    
    if os.path.exists(RAW_DIR):
        print(f"[DEBUG] Các file/thư mục con bên trong RAW_DIR: {os.listdir(RAW_DIR)}")
    process_yolo_dataset(RAW_DIR, PROCESSED_DIR)