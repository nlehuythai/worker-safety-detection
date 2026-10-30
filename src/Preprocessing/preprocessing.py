import cv2


class ImagePreprocessor:
    """
    Module tiền xử lý ảnh cho hệ thống phát hiện PPE.

    Pipeline:
        BGR
        -> LAB
        -> CLAHE trên kênh L
        -> Gaussian Filter
    """

    def __init__(
        self,
        clahe_clip_limit=2.0,
        clahe_grid_size=(8, 8),
        gaussian_kernel=(5, 5),
        gaussian_sigma=0
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def bgr_to_lab(self, image):
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

    def lab_to_bgr(self, image):
        return cv2.cvtColor(
            image,
            cv2.COLOR_LAB2BGR
        )

    def apply_clahe(self, image):
        """
        Cải thiện độ tương phản cục bộ
        thông qua CLAHE trên kênh L của LAB.
        """

        lab = self.bgr_to_lab(image)

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )

        enhanced_l = clahe.apply(l_channel)

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel
            )
        )

        return self.lab_to_bgr(
            enhanced_lab
        )

    def apply_gaussian_filter(self, image):
        """
        Gaussian Blur để giảm nhiễu.
        """

        return cv2.GaussianBlur(
            image,
            self.gaussian_kernel,
            self.gaussian_sigma
        )

    def apply_median_filter(self, image):
        """
        Median Filter - có thể sử dụng
        trong các thí nghiệm về nhiễu.
        """

        return cv2.medianBlur(
            image,
            5
        )

    def process(
        self,
        image,
        use_clahe=True,
        use_gaussian=True,
        use_median=False
    ):
        """
        Chạy pipeline tiền xử lý.
        """

        if image is None:
            raise ValueError(
                "Ảnh đầu vào không hợp lệ."
            )

        processed = image.copy()

        if use_clahe:
            processed = self.apply_clahe(
                processed
            )

        if use_gaussian:
            processed = self.apply_gaussian_filter(
                processed
            )

        if use_median:
            processed = self.apply_median_filter(
                processed
            )

        return processed

    def compare(self, image):
        """
        Trả về ảnh gốc và ảnh sau preprocessing.
        """

        processed = self.process(
            image,
            use_clahe=True,
            use_gaussian=True,
            use_median=False
        )

        return image.copy(), processed