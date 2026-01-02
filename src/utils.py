# src/utils.py
import os
import logging

def setup_logger(name="ProjectLogger"):
    """
    Thiết lập hệ thống log chuyên nghiệp.
    Giúp theo dõi tiến trình dự án rõ ràng hơn print().
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Tạo handler để in ra màn hình console
    c_handler = logging.StreamHandler()
    
    # Định dạng log: [Thời gian] - [Mức độ] - [Nội dung]
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    c_handler.setFormatter(c_format)
    
    if not logger.handlers:
        logger.addHandler(c_handler)
        
    return logger

def ensure_dir(directory):
    """Đảm bảo thư mục tồn tại, nếu chưa có thì tạo mới."""
    if not os.path.exists(directory):
        os.makedirs(directory)