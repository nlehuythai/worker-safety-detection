import cv2


class PPEVisualizer:

    def __init__(self):

        self.violation_color = (
            0,
            0,
            255
        )

        self.safe_color = (
            0,
            255,
            0
        )

    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    def draw(
        self,
        image,
        detections
    ):

        output = image.copy()

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            class_name = detection.get(
                "class_name",
                "unknown"
            )

            confidence = float(
                detection.get(
                    "confidence",
                    0
                )
            )

            # Tất cả detection hiện tại
            # đều là nhóm vi phạm
            color = self.violation_color

            label = (
                f"{class_name} "
                f"{confidence:.1%}"
            )

            # Bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label
            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            label_y = max(
                y1,
                text_height + 5
            )

            cv2.rectangle(
                output,
                (
                    x1,
                    label_y - text_height - baseline
                ),
                (
                    x1 + text_width + 8,
                    label_y + 3
                ),
                color,
                -1
            )

            cv2.putText(
                output,
                label,
                (
                    x1 + 4,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return output

    # =====================================================
    # DRAW STATISTICS
    # =====================================================

    def draw_statistics(
        self,
        image,
        statistics
    ):

        output = image.copy()

        overlay = output.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (250, 120),
            (15, 15, 15),
            -1
        )

        output = cv2.addWeighted(
            overlay,
            0.75,
            output,
            0.25,
            0
        )

        cv2.putText(
            output,
            f"Total: {statistics.get('total', 0)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Safe: {statistics.get('safe', 0)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Unsafe: {statistics.get('unsafe', 0)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        return output