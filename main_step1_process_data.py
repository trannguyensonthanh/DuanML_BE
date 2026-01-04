# main_step1_process_data.py
import sys
sys.path.append("./src")

from src.preprocessor import ImagePreprocessor
from src.data_loader import YoloDataLoader
from src.utils import setup_logger

def main():
    logger = setup_logger("MainProcess")
    logger.info("🚀 KHỞI ĐỘNG DỰ ÁN PHÂN LOẠI RÁC THẢI (ML PIPELINE)")

    # --- CẤU HÌNH ---
    RAW_DATA_DIR = "data/raw"         
    PROCESSED_DATA_DIR = "data/processed" 
    
    # 1. Khởi tạo bộ tiền xử lý (Target size 128x128 là chuẩn vàng cho HOG/SVM)
    preprocessor = ImagePreprocessor(target_size=(128, 128), use_clahe=True)

    # 2. Khởi tạo bộ nạp dữ liệu
    loader = YoloDataLoader(
        input_dir=RAW_DATA_DIR, 
        output_dir=PROCESSED_DATA_DIR, 
        preprocessor=preprocessor
    )

    # 3. Chạy Pipeline
    loader.run_pipeline()

if __name__ == "__main__":
    main()