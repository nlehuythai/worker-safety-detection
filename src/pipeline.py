import os
import cv2
import time
import uuid
import numpy as np

from detector import YOLOSafetyDetector
from config import VIOLATION_CLASSES


class SafetyViolationPipeline:

    def __init__(self, model_path: str):

        self.detector = YOLOSafetyDetector(model_path)

        self.violation_classes = VIOLATION_CLASSES

    # =====================================================
    # PREPROCESSING
    # =====================================================

    def preprocess(self, image):

        processing = {}

        # -------------------------------------------------
        # 1. RGB/BGR -> LAB
        # -------------------------------------------------

        start = time.perf_counter()

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        processing["color_space"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # 2. CLAHE
        # -------------------------------------------------

        start = time.perf_counter()

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l_channel = clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel
            )
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        processing["clahe"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # 3. Gaussian Filtering
        # -------------------------------------------------

        start = time.perf_counter()

        denoised = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0
        )

        processing["noise_filter"] = (
            time.perf_counter() - start
        )

        return denoised, processing

    # =====================================================
    # YOLO DETECTION
    # =====================================================

    def inference(self, image, confidence):

        start = time.perf_counter()

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.detector.detect(
            rgb,
            conf_threshold=confidence
        )

        inference_time = (
            time.perf_counter() - start
        )

        return results, inference_time

    # =====================================================
    # POST PROCESSING
    # =====================================================

    def postprocess(self, results):

        start = time.perf_counter()

        detections = []

        if results is None:
            return [], 0

        if results.boxes is None:
            return [], 0

        boxes = results.boxes

        for box, cls, conf in zip(
            boxes.xyxy,
            boxes.cls,
            boxes.conf
        ):

            class_id = int(
                cls.item()
            )

            confidence = float(
                conf.item()
            )

            class_name = results.names.get(
                class_id,
                str(class_id)
            )

            # Chỉ lấy những class người dùng chọn
            if class_name not in self.violation_classes:
                continue

            coords = box.tolist()

            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

            detections.append({

                "class_id": class_id,

                "class_name": class_name,

                "confidence": confidence,

                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]

            })

        processing_time = (
            time.perf_counter() - start
        )

        return detections, processing_time

    # =====================================================
    # STATISTICS
    # =====================================================

    def calculate_statistics(
        self,
        detections
    ):

        total = len(detections)

        unsafe = len(detections)

        safe = 0

        return {

            "total": total,

            "safe": safe,

            "unsafe": unsafe

        }

    # =====================================================
    # REPORT
    # =====================================================

    def create_report(
        self,
        detections,
        confidence_threshold
    ):

        breakdown = {}

        for detection in detections:

            class_name = detection[
                "class_name"
            ]

            if class_name not in breakdown:

                breakdown[class_name] = {

                    "count": 0,

                    "confidence_sum": 0

                }

            breakdown[class_name][
                "count"
            ] += 1

            breakdown[class_name][
                "confidence_sum"
            ] += detection[
                "confidence"
            ]

        violation_breakdown = {}

        for class_name, value in breakdown.items():

            count = value["count"]

            avg_confidence = (
                value["confidence_sum"]
                / count
            )

            violation_breakdown[
                class_name
            ] = count

        report_id = (
            "SG-"
            + uuid.uuid4().hex[:8].upper()
        )

        details = []

        for detection in detections:

            class_name = detection[
                "class_name"
            ]

            info = self.violation_classes[
                class_name
            ]

            details.append({

                "class": class_name,

                "label": {
                    "en": info["en"]
                },

                "emoji": info["emoji"],

                "confidence": round(
                    detection["confidence"],
                    4
                ),

                "bbox": {

                    "x1": detection[
                        "bbox"
                    ][0],

                    "y1": detection[
                        "bbox"
                    ][1],

                    "x2": detection[
                        "bbox"
                    ][2],

                    "y2": detection[
                        "bbox"
                    ][3]
                }

            })

        return {

            "report_id": report_id,

            "summary": {

                "total_violations": len(
                    detections
                ),

                "violation_breakdown":
                    violation_breakdown

            },

            "details": details,

            "confidence_threshold":
                confidence_threshold

        }

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def run(
        self,
        image_path: str,
        conf_thresh: float = 0.5,
        apply_preprocessing: bool = True
    ):

        total_start = time.perf_counter()

        # -------------------------------------------------
        # CHECK FILE
        # -------------------------------------------------

        if not os.path.exists(image_path):

            return {

                "error":
                    f"Không tìm thấy ảnh: {image_path}"

            }

        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            return {

                "error":
                    "Không thể đọc file ảnh."

            }

        # -------------------------------------------------
        # CREATE STATIC FOLDERS
        # -------------------------------------------------

        os.makedirs(
            "static/results",
            exist_ok=True
        )

        # =================================================
        # ORIGINAL IMAGE
        # =================================================

        image_id = uuid.uuid4().hex[:10]

        original_filename = (
            f"original_{image_id}.jpg"
        )

        original_path = os.path.join(
            "static",
            "results",
            original_filename
        )

        cv2.imwrite(
            original_path,
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        # =================================================
        # PREPROCESSING
        # =================================================

        if apply_preprocessing:

            processed_image, processing = (
                self.preprocess(image)
            )

        else:

            processed_image = image.copy()

            processing = {

                "color_space": 0,

                "clahe": 0,

                "noise_filter": 0

            }

        # =================================================
        # SAVE PREPROCESSED IMAGE
        # =================================================

        preprocessing_filename = (
            f"preprocessed_{image_id}.jpg"
        )

        preprocessing_path = os.path.join(
            "static",
            "results",
            preprocessing_filename
        )

        cv2.imwrite(
            preprocessing_path,
            processed_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        # =================================================
        # YOLO
        # =================================================

        results, yolo_time = self.inference(
            processed_image,
            conf_thresh
        )

        processing[
            "yolo"
        ] = yolo_time

        # =================================================
        # POST PROCESSING
        # =================================================

        detections, post_time = (
            self.postprocess(results)
        )

        processing[
            "post_processing"
        ] = post_time

        # =================================================
        # STATISTICS
        # =================================================

        statistics = (
            self.calculate_statistics(
                detections
            )
        )

        # =================================================
        # REPORT
        # =================================================

        report = self.create_report(
            detections,
            conf_thresh
        )

        # =================================================
        # ANNOTATED IMAGE
        # =================================================

        annotated_image = image.copy()

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]

            color = (
                0,
                0,
                255
            )

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            label = (
                f"{self.violation_classes[class_name]['en']} "
                f"{confidence:.1%}"
            )

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            text_y = max(
                y1 - 8,
                text_height + 5
            )

            cv2.rectangle(
                annotated_image,
                (
                    x1,
                    text_y - text_height - baseline
                ),
                (
                    x1 + text_width + 8,
                    text_y + 3
                ),
                color,
                -1
            )

            cv2.putText(
                annotated_image,
                label,
                (
                    x1 + 4,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # =================================================
        # DRAW STATISTICS
        # =================================================

        overlay = annotated_image.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (245, 115),
            (20, 20, 20),
            -1
        )

        annotated_image = cv2.addWeighted(
            overlay,
            0.75,
            annotated_image,
            0.25,
            0
        )

        cv2.putText(
            annotated_image,
            f"Total: {statistics['total']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_image,
            f"Safe: {statistics['safe']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_image,
            f"Unsafe: {statistics['unsafe']}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # =================================================
        # SAVE RESULT
        # =================================================

        result_filename = (
            f"result_{image_id}.jpg"
        )

        result_path = os.path.join(
            "static",
            "results",
            result_filename
        )

        cv2.imwrite(
            result_path,
            annotated_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        # =================================================
        # TOTAL TIME
        # =================================================

        processing["total"] = (
            time.perf_counter()
            - total_start
        )

        return {

            "success": True,

            "detections": detections,

            "violations": report[
                "details"
            ],

            "statistics": statistics,

            "summary_report": report,

            "processing": processing,

            "confidence_threshold_used":
                conf_thresh,

            "original_image":
                f"/static/results/{original_filename}",

            "preprocessed_image":
                f"/static/results/{preprocessing_filename}",

            "annotated_image":
                f"/static/results/{result_filename}",

            "total_detections":
                len(detections),

            "has_violations":
                len(detections) > 0,

            "message":
                (
                    f"Phát hiện {len(detections)} "
                    "vi phạm"
                    if len(detections) > 0
                    else
                    "Không phát hiện vi phạm"
                )

        }