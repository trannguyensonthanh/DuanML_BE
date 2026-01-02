# main_step3_evaluate.py
import sys
sys.path.append("./src")
from src.evaluator import GoldEvaluator

def main():
    # --- CẤU HÌNH ---
    # MODEL_PATH = "models/best_model.pkl"
    MODEL_PATH = "models/best_tuned_ensemble_model.pkl"
    ENCODER_PATH = "models/label_encoder.pkl"
    TEST_DATA_DIR = "test_gold"       # Folder chứa ảnh Polygon test
    OUTPUT_REPORT_DIR = "evaluation_results"
    
    print("🕵️ KHỞI ĐỘNG HỆ THỐNG KIỂM THỬ ĐẲNG CẤP (POLYGON MODE)...")
    
    try:
        evaluator = GoldEvaluator(
            model_path=MODEL_PATH,
            encoder_path=ENCODER_PATH,
            test_dir=TEST_DATA_DIR,
            output_dir=OUTPUT_REPORT_DIR
        )
        evaluator.run()
        
        print("\n✅ HOÀN THÀNH XUẤT SẮC!")
        print(f"👉 File CSV chi tiết: {OUTPUT_REPORT_DIR}/FULL_Evaluation_Report.csv")
        print(f"👉 Ảnh đoán sai: {OUTPUT_REPORT_DIR}/errors_gallery")
        print(f"👉 Biểu đồ phân tích: {OUTPUT_REPORT_DIR}/analysis_plots")
        
    except Exception as e:
        print(f"❌ CÓ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()