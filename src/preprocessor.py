# src/preprocessor.py
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(128, 128), use_clahe=True):
        """
        Khởi tạo bộ tiền xử lý ảnh.
        :param target_size: Kích thước ảnh đầu ra (width, height).
        :param use_clahe: Có sử dụng cân bằng sáng thích ứng hay không (Nên bật).
        """
        self.target_size = target_size
        self.use_clahe = use_clahe
        # Khởi tạo bộ CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clipLimit=2.0 là ngưỡng tương phản, tileGridSize=(8,8) là lưới cục bộ
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, image):
        """
        Thực hiện chuỗi xử lý: Resize -> Khử nhiễu -> Cân bằng sáng -> Chuẩn hóa.
        :param image: Ảnh gốc (BGR format).
        :return: Ảnh đã xử lý sạch đẹp.
        """
        if image is None:
            return None

        # 1. Resize về kích thước cố định
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # 2. Khử nhiễu nhẹ (Gaussian Blur) để loại bỏ hạt nhiễu của camera
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # 3. Cân bằng sáng (Enhancement)
        if self.use_clahe:
            # Chuyển sang LAB color space để chỉ chỉnh độ sáng (Channel L), giữ nguyên màu
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)  # Áp dụng CLAHE lên kênh sáng
            lab = cv2.merge((l2, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image