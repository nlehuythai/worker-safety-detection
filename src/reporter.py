import os
import json
from datetime import datetime


class DetectionReporter:

    def __init__(self, report_dir="reports"):

        self.report_dir = report_dir

        os.makedirs(
            self.report_dir,
            exist_ok=True
        )


    def generate_report(
        self,
        image_name,
        violations,
        confidence_threshold=0.5
    ):

        timestamp = datetime.now()

        report = {

            "image":
                image_name,

            "timestamp":
                timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "confidence_threshold":
                confidence_threshold,

            "total_violations":
                len(violations),

            "violations":
                violations
        }


        filename = (
            f"report_"
            f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
            f".json"
        )


        report_path = os.path.join(
            self.report_dir,
            filename
        )


        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=4
            )


        return {

            "file":
                filename,

            "path":
                report_path,

            "total_violations":
                len(violations)
        }