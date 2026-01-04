# src/evaluator.py
import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score

# Import từ các module đã tối ưu trước đó
from .feature_extractor import FeatureExtractor
from .utils import setup_logger, ensure_dir

# Cấu hình giao diện biểu đồ chuyên nghiệp
sns.set_theme(style="whitegrid")
logger = setup_logger("EvaluatorPro")


class GoldEvaluator:
    def __init__(self, model_path, encoder_path, test_dir, output_dir):
        """
        Khởi tạo bộ đánh giá cao cấp.
        Tương thích với: PyTorch CLAHE + Logistic Regression
        """
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.error_dir = os.path.join(output_dir, "errors_gallery")
        self.plots_dir = os.path.join(output_dir, "analysis_plots")

        ensure_dir(self.output_dir)
        ensure_dir(self.error_dir)
        ensure_dir(self.plots_dir)

        # 1. Load Model Pipeline & Label Encoder
        logger.info(f"⏳ Đang tải 'brain' (Model) từ: {model_path}")
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError("CRITICAL ERROR: Không tìm thấy model hoặc label encoder. Hãy train trước!")

        self.model = joblib.load(model_path)
        self.le = joblib.load(encoder_path)
        self.classes = self.le.classes_

        logger.info("⏳ Đang khởi tạo Feature Extractor (CLAHE)...")
        self.extractor = FeatureExtractor()

    def _parse_polygon(self, line, img_w, img_h):
        """Đọc chuỗi tọa độ Polygon YOLO và chuẩn hóa."""
        try:
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]  # Bỏ class_id
            if len(coords) < 6: return None
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])
            return np.array(points, dtype=np.int32)
        except Exception:
            return None

    def _crop_and_mask(self, img, polygon):
        """
        KỸ THUẬT TÁCH NỀN (MASKING):
        Tô đen toàn bộ background, chỉ giữ lại vật thể trong polygon.
        """
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Bitwise AND để xóa nền
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # Cắt khung hình chữ nhật bao quanh polygon
        x, y, w, h = cv2.boundingRect(polygon)
        crop = masked_img[y:y + h, x:x + w]
        return crop

    def run(self):
        logger.info(f"🚀 Bắt đầu quy trình kiểm thử trên tập: {self.test_dir}")

        results_data = []  # Lưu data để xuất CSV
        y_true_all = []
        y_probs_all = []  # Lưu xác suất để tính Top-K

        folder_classes = sorted(os.listdir(self.test_dir))

        # Duyệt qua từng folder class thật (Ground Truth)
        for true_label_str in folder_classes:
            class_path = os.path.join(self.test_dir, true_label_str)
            if not os.path.isdir(class_path): continue

            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in tqdm(image_files, desc=f"Testing {true_label_str}"):
                img_path = os.path.join(class_path, img_name)
                txt_path = os.path.splitext(img_path)[0] + ".txt"

                # Bắt buộc phải có file nhãn Polygon
                if not os.path.exists(txt_path): continue

                img = cv2.imread(img_path)
                if img is None: continue
                h_img, w_img = img.shape[:2]

                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                # Xử lý từng object trong ảnh (thường là 1)
                for idx, line in enumerate(lines):
                    # 1. Parse Polygon & Masking
                    polygon = self._parse_polygon(line, w_img, h_img)
                    if polygon is None: continue

                    try:
                        crop_masked = self._crop_and_mask(img, polygon)
                    except Exception:
                        continue

                    if crop_masked.size == 0 or crop_masked.shape[0] < 10: continue

                    # 2. Feature Extraction (Trực tiếp từ ảnh crop)
                    # FeatureExtractor sẽ tự động Resize -> CLAHE -> ResNet
                    features = self.extractor.extract(crop_masked)

                    if features is None: continue

                    # 3. Prediction (Dự đoán)
                    features = features.reshape(1, -1)

                    # Lấy xác suất
                    probs = self.model.predict_proba(features)[0]

                    # Lấy class có xác suất cao nhất
                    pred_idx = np.argmax(probs)
                    pred_label_str = self.le.inverse_transform([pred_idx])[0]
                    confidence = probs[pred_idx]

                    # Lấy Top-2 Prediction (Để xem nếu sai thì có suýt đúng không)
                    top2_idx = np.argsort(probs)[-2:][::-1]
                    top2_labels = self.le.inverse_transform(top2_idx)

                    # 4. Ghi nhận dữ liệu
                    is_correct = (true_label_str == pred_label_str)

                    y_true_all.append(true_label_str)
                    y_probs_all.append(probs)

                    # Tạo record chi tiết cho CSV
                    record = {
                        "Image": img_name,
                        "Ground_Truth": true_label_str,
                        "Prediction": pred_label_str,
                        "Confidence": round(confidence * 100, 2),
                        "Is_Correct": is_correct,
                        "Top_2_Guess": f"{top2_labels[1]} ({round(probs[top2_idx[1]] * 100, 2)}%)"
                    }
                    # Thêm xác suất từng class vào CSV
                    for i, cls_name in enumerate(self.classes):
                        record[f"Prob_{cls_name}"] = round(probs[i], 4)

                    results_data.append(record)

                    # 5. Lưu ảnh sai (Error Analysis Gallery)
                    if not is_correct:
                        err_fname = f"Err_True[{true_label_str}]_Pred[{pred_label_str}]_Conf[{int(confidence * 100)}]_{img_name}"

                        # Vẽ thêm text lên ảnh để debug
                        debug_img = crop_masked.copy()  # Dùng ảnh crop để dễ nhìn vật thể
                        debug_img = cv2.resize(debug_img, (256, 256))

                        # Apply CLAHE lên ảnh debug để người xem dễ nhìn chi tiết như máy nhìn
                        lab = cv2.cvtColor(debug_img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
                        debug_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

                        cv2.putText(debug_img, f"True: {true_label_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                        cv2.putText(debug_img, f"Pred: {pred_label_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)

                        cv2.imwrite(os.path.join(self.error_dir, err_fname), debug_img)

        # --- TỔNG HỢP VÀ BÁO CÁO ---
        if not results_data:
            logger.error("❌ Không tìm thấy dữ liệu test nào hợp lệ (hoặc lỗi file txt)!")
            return

        df = pd.DataFrame(results_data)

        # 1. Tính toán Metrics
        y_pred_all = df["Prediction"].values
        acc = accuracy_score(y_true_all, y_pred_all)

        # Top-2 Accuracy
        y_true_indices = self.le.transform(y_true_all)
        top2_acc = top_k_accuracy_score(y_true_indices, np.array(y_probs_all), k=2, labels=np.arange(len(self.classes)))

        print("\n" + "═" * 60)
        print(f"📊 REPORT KẾT QUẢ KIỂM THỬ (POLYGON DATASET)")
        print("═" * 60)
        print(f"🏆 Top-1 Accuracy (Chính xác tuyệt đối):  {acc * 100:.2f}%")
        print(f"🥈 Top-2 Accuracy (Đáp án đúng nằm trong Top 2): {top2_acc * 100:.2f}%")
        print("-" * 60)
        print(classification_report(y_true_all, y_pred_all))
        print("═" * 60)

        # 2. Xuất CSV
        csv_path = os.path.join(self.output_dir, "FULL_Evaluation_Report.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"📄 Đã xuất báo cáo chi tiết tại: {csv_path}")

        # 3. Vẽ biểu đồ
        self._visualize_performance(y_true_all, y_pred_all, df)

    def _visualize_performance(self, y_true, y_pred, df):
        """Vẽ các biểu đồ phân tích chuyên sâu."""

        # A. Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "1_Confusion_Matrix.png"))
        plt.close()

        # B. Confidence Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="Confidence", hue="Is_Correct", multiple="stack", bins=20, kde=True)
        plt.title("Phân phối độ tin cậy của dự đoán (Đúng vs Sai)")
        plt.xlabel("Độ tin cậy (Confidence Score %)")
        plt.ylabel("Số lượng ảnh")
        plt.savefig(os.path.join(self.plots_dir, "2_Confidence_Distribution.png"))
        plt.close()

        logger.info(f"📊 Đã lưu các biểu đồ phân tích tại: {self.plots_dir}")