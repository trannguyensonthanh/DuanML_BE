# src/data_loader.py
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from .utils import setup_logger, ensure_dir

logger = setup_logger("DataLoader")

class YoloDataLoader:
    def __init__(self, input_dir, output_dir, preprocessor):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.preprocessor = preprocessor
        self.classes = ["metal", "organic", "paper", "plastic"]

    def _parse_polygon(self, line, img_w, img_h):
        """
        Đọc dòng text Polygon YOLO và chuyển thành mảng tọa độ.
        Format: class_id x1 y1 x2 y2 ... xn yn
        """
        parts = list(map(float, line.strip().split()))
        # Bỏ phần tử đầu tiên (class_id)
        coords = parts[1:]
        
        # Polygon phải có ít nhất 3 điểm (6 số)
        if len(coords) < 6:
            return None
            
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_w)
            y = int(coords[i+1] * img_h)
            points.append([x, y])
            
        return np.array(points, dtype=np.int32)

    def _crop_and_mask(self, img, polygon):
        """
        KỸ THUẬT ĐỈNH CAO: Masking (Tách nền).
        Chỉ giữ lại pixel bên trong polygon, phần còn lại tô đen (0,0,0).
        """
        # 1. Tạo mặt nạ đen (Mask) cùng kích thước ảnh
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # 2. Vẽ polygon màu trắng lên mặt nạ
        cv2.fillPoly(mask, [polygon], 255)
        
        # 3. Áp dụng mặt nạ lên ảnh gốc (Bitwise AND)
        # Những chỗ mask đen -> Ảnh thành đen. Mask trắng -> Giữ nguyên ảnh.
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # 4. Cắt vùng chứa vật thể (Bounding Rect của Polygon)
        # Để loại bỏ phần đen thừa thãi xung quanh, giúp ảnh tập trung vào vật thể
        x, y, w, h = cv2.boundingRect(polygon)
        crop = masked_img[y:y+h, x:x+w]
        
        return crop

    def run_pipeline(self):
        logger.info(f"🚀 Bắt đầu xử lý dữ liệu POLYGON từ: {self.input_dir}")
        total_count = 0
        stats = {c: 0 for c in self.classes}

        for class_name in self.classes:
            class_path = os.path.join(self.input_dir, class_name)
            save_path = os.path.join(self.output_dir, class_name)
            ensure_dir(save_path)

            if not os.path.exists(class_path):
                logger.warning(f"Không tìm thấy thư mục: {class_path}")
                continue

            # Lấy list ảnh
            types = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            image_files = []
            for t in types:
                image_files.extend(glob.glob(os.path.join(class_path, t)))
                image_files.extend(glob.glob(os.path.join(class_path, t.upper())))

            logger.info(f"📂 Đang xử lý lớp '{class_name}' - {len(image_files)} ảnh gốc.")

            for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                txt_path = os.path.join(class_path, base_name + ".txt")

                if not os.path.exists(txt_path):
                    continue

                img = cv2.imread(img_path)
                if img is None: continue
                h_img, w_img = img.shape[:2]

                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                for idx, line in enumerate(lines):
                    # Parse Polygon
                    polygon = self._parse_polygon(line, w_img, h_img)
                    if polygon is None: continue

                    # --- QUAN TRỌNG: CẮT VÀ TÁCH NỀN ---
                    try:
                        crop_masked = self._crop_and_mask(img, polygon)
                    except Exception as e:
                        continue

                    # Bỏ qua ảnh lỗi hoặc quá nhỏ
                    if crop_masked.size == 0 or crop_masked.shape[0] < 10 or crop_masked.shape[1] < 10:
                        continue

                    # --- GỌI PREPROCESSOR ---
                    # Resize, khử nhiễu, cân bằng sáng
                    processed_img = self.preprocessor.process(crop_masked)

                    # Lưu ảnh (Lúc này ảnh sẽ có nền đen thui, rất đẹp cho model học)
                    out_name = f"{base_name}_poly_{idx}.jpg"
                    cv2.imwrite(os.path.join(save_path, out_name), processed_img)
                    
                    stats[class_name] += 1
                    total_count += 1

        logger.info("="*40)
        logger.info(f"✅ HOÀN THÀNH TÁCH NỀN POLYGON! Tổng ảnh sạch: {total_count}")
        logger.info(f"Thống kê: {stats}")
        logger.info(f"Dữ liệu đã lưu tại: {self.output_dir}")