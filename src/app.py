from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

import os
import uuid


from pipeline import SafetyViolationPipeline


app = Flask(__name__)


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = os.path.join(
    "Model Weights",
    "best.pt"
)


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_PATH}"
    )


# =========================================================
# PIPELINE
# =========================================================

pipeline = SafetyViolationPipeline(
    MODEL_PATH
)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# ANALYZE
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        # -------------------------------------------------
        # CHECK FILE
        # -------------------------------------------------

        if "file" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "Không tìm thấy file ảnh."

            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "Chưa chọn file ảnh."

            }), 400

        # -------------------------------------------------
        # EXTENSION
        # -------------------------------------------------

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        allowed = [

            ".jpg",
            ".jpeg",
            ".png",
            ".jfif",
            ".bmp",
            ".webp"

        ]

        if extension not in allowed:

            return jsonify({

                "success": False,

                "error":
                    "Định dạng ảnh không được hỗ trợ."

            }), 400

        # -------------------------------------------------
        # UPLOAD
        # -------------------------------------------------

        upload_folder = "uploads"

        os.makedirs(
            upload_folder,
            exist_ok=True
        )

        filename = (
            uuid.uuid4().hex
            + extension
        )

        image_path = os.path.join(
            upload_folder,
            filename
        )

        file.save(
            image_path
        )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = request.form.get(
            "confidence",
            "0.5"
        )

        try:

            confidence = float(
                confidence
            )

        except ValueError:

            confidence = 0.5

        confidence = max(
            0.1,
            min(
                confidence,
                0.95
            )
        )

        # -------------------------------------------------
        # PREPROCESSING
        # -------------------------------------------------

        preprocessing_value = (
            request.form.get(
                "preprocessing",
                "true"
            )
        )

        apply_preprocessing = (
            preprocessing_value.lower()
            == "true"
        )

        # -------------------------------------------------
        # RUN PIPELINE
        # -------------------------------------------------

        result = pipeline.run(

            image_path,

            conf_thresh=confidence,

            apply_preprocessing=
                apply_preprocessing

        )

        # -------------------------------------------------
        # PIPELINE ERROR
        # -------------------------------------------------

        if result.get("error"):

            return jsonify({

                "success": False,

                "error":
                    result["error"]

            }), 500

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                result.get(
                    "message",
                    ""
                ),

            "has_violations":
                result.get(
                    "has_violations",
                    False
                ),

            "total_detections":
                result.get(
                    "total_detections",
                    0
                ),

            "violations":
                result.get(
                    "violations",
                    []
                ),

            "statistics":
                result.get(
                    "statistics",
                    {

                        "total": 0,

                        "safe": 0,

                        "unsafe": 0

                    }
                ),

            "summary_report":
                result.get(
                    "summary_report",
                    {}
                ),

            "processing":
                result.get(
                    "processing",
                    {}
                ),

            "confidence_threshold_used":
                result.get(
                    "confidence_threshold_used",
                    confidence
                ),

            "original_image":
                result.get(
                    "original_image"
                ),

            "preprocessed_image":
                result.get(
                    "preprocessed_image"
                ),

            "annotated_image":
                result.get(
                    "annotated_image"
                )

        })

    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )