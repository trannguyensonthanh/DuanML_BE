# main_step2_train.py
import sys
sys.path.append("./src")
from src.model_trainer import TrashClassifier

def main():
    # --- CẤU HÌNH ---
    PROCESSED_DATA_DIR = "data/processed"
    
    print("🤖 KHỞI ĐỘNG TRAINER AI...")
    
    # Khởi tạo và huấn luyện
    classifier = TrashClassifier(data_dir=PROCESSED_DATA_DIR)
    classifier.train()

if __name__ == "__main__":
    main()