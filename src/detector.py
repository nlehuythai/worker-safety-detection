import os
import torch
import cv2

from ultralytics import YOLO


class YOLOSafetyDetector:

    def __init__(self, model_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}"
            )

        print(
            f"Loading YOLO model from: {model_path}"
        )

        self.model = YOLO(model_path)

        # =============================================
        # Tự động chọn GPU hoặc CPU
        # =============================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        print(
            f"Model successfully moved to "
            f"{self.device.upper()}"
        )

        # =============================================
        # Hiển thị class của model
        # =============================================

        print(
            "MODEL CLASSES:",
            self.model.names
        )


    def detect(
        self,
        image,
        conf_threshold: float = 0.5
    ):
        """
        Thực hiện YOLO inference.

        Input:
            image: ảnh OpenCV dạng BGR

        Output:
            YOLO Results
        """

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        # =============================================
        # OpenCV sử dụng BGR
        # YOLO nhận RGB
        # =============================================

        if len(image.shape) == 3:

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        else:

            image_rgb = image

        # =============================================
        # YOLO inference
        # =============================================

        results = self.model(
            image_rgb,
            conf=conf_threshold,
            verbose=False,
            device=self.device
        )

        return results[0]