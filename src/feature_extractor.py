# src/feature_extractor.py
import cv2
import numpy as np
import mahotas # <-- Thư viện mới, cực mạnh cho Haralick
from skimage.feature import hog, local_binary_pattern

class FeatureExtractor:
    def __init__(self):
        # HOG configuration (giữ nguyên)
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # LBP configuration (tối ưu hóa)
        self.lbp_points = 24
        self.lbp_radius = 8 # Tăng bán kính để nhìn được kết cấu lớn hơn

    def compute_color_moments(self, image):
        """
        [NÂNG CẤP VƯỢT TRỘI] Tính Color Moments trên không gian màu HSV.
        Nắm bắt các thuộc tính thống kê cốt lõi của màu sắc.
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Tách các kênh Hue, Saturation, Value
        h, s, v = cv2.split(hsv)
        channels = [h, s, v]
        
        features = []
        for channel in channels:
            # Chỉ tính toán trên các pixel không phải màu đen (vật thể thực sự)
            mask = channel > 0
            if np.any(mask):
                pixels = channel[mask]
                # Tính 4 moments: Mean, Std Dev, Skewness, Kurtosis
                mean = np.mean(pixels)
                std = np.std(pixels)
                
                # Để tránh lỗi chia cho 0, tính skew và kurtosis một cách an toàn
                skew = np.mean(((pixels - mean) / (std + 1e-6))**3) if std > 1e-6 else 0
                kurtosis = np.mean(((pixels - mean) / (std + 1e-6))**4) if std > 1e-6 else 0
                
                features.extend([mean, std, skew, kurtosis])
            else:
                # Nếu kênh rỗng (ví dụ ảnh xám), thêm 4 số 0
                features.extend([0, 0, 0, 0])
        
        # Trả về 12 đặc trưng (4 moments x 3 kênh)
        return np.array(features)
        
    def compute_haralick(self, gray_img):
        """
        [MỚI - ĐẲNG CẤP] Haralick Texture Features.
        Cực kỳ mạnh để phân biệt bề mặt nhám (giấy), bóng (nhựa), gồ ghề (hữu cơ).
        """
        # Tính toán Haralick textures, trả về 13 đặc trưng
        haralick_features = mahotas.features.haralick(gray_img).mean(axis=0)
        return haralick_features

    def compute_hog(self, gray_img):
        """HOG Shape descriptor (giữ nguyên)"""
        fd = hog(gray_img, 
                 orientations=self.hog_orientations,
                 pixels_per_cell=self.hog_pixels_per_cell,
                 cells_per_block=self.hog_cells_per_block,
                 block_norm='L2-Hys', 
                 visualize=False)
        return fd

    def compute_lbp(self, gray_img):
        """LBP Texture descriptor (giữ nguyên logic, tinh chỉnh tham số ở __init__)"""
        lbp = local_binary_pattern(gray_img, self.lbp_points, self.lbp_radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def extract(self, image):
        """
        [PIPELINE TRÍCH XUẤT ĐẶC TRƯNG PHIÊN BẢN MỚI]
        """
        if image is None: return None
        
        # Tạo ảnh xám chỉ một lần
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. HOG (Biên dạng)
        hog_feat = self.compute_hog(gray)
        
        # 2. [MỚI] Color Moments (Màu sắc Thống kê)
        color_feat = self.compute_color_moments(image)
        
        # 3. LBP (Kết cấu Cục bộ)
        lbp_feat = self.compute_lbp(gray)
        
        # 4. [MỚI] Haralick (Kết cấu Toàn cục)
        haralick_feat = self.compute_haralick(gray)
        
        # Hợp nhất tất cả (Feature Fusion)
        # Bỏ Hu Moments vì HOG đã làm tốt hơn về hình dáng và Haralick mạnh hơn về kết cấu
        final_vector = np.hstack([hog_feat, color_feat, lbp_feat, haralick_feat])
        
        return final_vector